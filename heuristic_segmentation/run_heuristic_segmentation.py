#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from heuristic_segmentation import HeuristicConfig, make_overlay, segment_image  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
Image.MAX_IMAGE_PIXELS = None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the standalone heuristic ore segmentation baseline."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Single image to segment")
    source.add_argument("--input-dir", type=Path, help="Directory of images to segment")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/heuristic_segmentation"))
    parser.add_argument("--recursive", action="store_true", help="Recurse when --input-dir is used")
    parser.add_argument("--max-images", type=int, default=0, help="0 means no limit")
    parser.add_argument("--max-side", type=int, default=2400, help="Resize analysis copy; 0 keeps original size")
    parser.add_argument("--min-component-area", type=int, default=64)
    parser.add_argument("--fine-max-area-px", type=int, default=450)
    parser.add_argument("--fine-min-replacement-ratio", type=float, default=0.22)
    parser.add_argument("--disable-talc-candidate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = HeuristicConfig(
        min_component_area=args.min_component_area,
        fine_max_area_px=args.fine_max_area_px,
        fine_min_replacement_ratio=args.fine_min_replacement_ratio,
        enable_talc_candidate=not args.disable_talc_candidate,
    )

    image_paths = [args.image.resolve()] if args.image else _list_images(args.input_dir.resolve(), args.recursive)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        print("no images found", file=sys.stderr)
        return 2

    rows = []
    for index, image_path in enumerate(image_paths, start=1):
        sample_id = _sample_id(image_path, index if args.input_dir else None)
        sample_dir = output_dir if args.image else output_dir / "samples" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        row = run_one(
            image_path=image_path,
            sample_dir=sample_dir,
            config=config,
            max_side=args.max_side,
        )
        row["sample_id"] = sample_id
        rows.append(row)
        print(
            f"{sample_id}: {row['ore_class_candidate']} "
            f"sulfide={row['sulfide_fraction']} talc={row['talc_candidate_fraction']}"
        )

    if args.input_dir:
        _write_summary_csv(output_dir / "summary.csv", rows)
    batch_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.image.resolve() if args.image else args.input_dir.resolve()),
        "image_count": len(rows),
        "max_side": args.max_side,
        "config": config.__dict__,
        "samples": rows,
    }
    _write_json(output_dir / "batch_summary.json", batch_summary)
    print(f"wrote {output_dir}")
    return 0


def run_one(
    *,
    image_path: Path,
    sample_dir: Path,
    config: HeuristicConfig,
    max_side: int,
) -> dict:
    rgb, load_meta = _load_rgb(image_path, max_side=max_side)
    result = segment_image(rgb, config=config)
    overlay = make_overlay(rgb, result.class_mask)

    _save_rgb(sample_dir / "analysis_image.jpg", rgb, quality=92)
    _save_gray(sample_dir / "class_mask.png", result.class_mask)
    _save_gray(sample_dir / "sulfide_mask.png", result.sulfide_mask * 255)
    _save_gray(sample_dir / "talc_candidate_mask.png", result.talc_candidate_mask * 255)
    _save_gray(sample_dir / "analyzed_mask.png", result.analyzed_mask * 255)
    _save_rgb(sample_dir / "overlay.png", overlay, quality=95)
    _write_components_csv(sample_dir / "components.csv", result.components)

    run_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_image": str(image_path),
        "source_sha256": _sha256(image_path),
        "load": load_meta,
        "metrics": result.metrics,
        "config": result.config,
        "artifacts": {
            "analysis_image": "analysis_image.jpg",
            "class_mask": "class_mask.png",
            "sulfide_mask": "sulfide_mask.png",
            "talc_candidate_mask": "talc_candidate_mask.png",
            "analyzed_mask": "analyzed_mask.png",
            "overlay": "overlay.png",
            "components": "components.csv",
        },
    }
    _write_json(sample_dir / "run_summary.json", run_summary)
    _write_json(sample_dir / "metrics.json", result.metrics)
    row = {
        "source_image": str(image_path),
        "analysis_width": result.metrics["image_width"],
        "analysis_height": result.metrics["image_height"],
        **result.metrics,
    }
    return row


def _list_images(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    paths = [
        path
        for path in sorted(input_dir.glob(pattern))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return paths


def _load_rgb(path: Path, max_side: int) -> tuple[np.ndarray, dict]:
    with Image.open(path) as image:
        original_width, original_height = image.size
        image = image.convert("RGB")
        if max_side > 0 and max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        analysis_width, analysis_height = image.size
        rgb = np.asarray(image, dtype=np.uint8).copy()
    scale_x = analysis_width / original_width
    scale_y = analysis_height / original_height
    return rgb, {
        "original_width": original_width,
        "original_height": original_height,
        "analysis_width": analysis_width,
        "analysis_height": analysis_height,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "max_side": max_side,
        "note": "Masks are emitted at analysis size. Use --max-side 0 to keep original dimensions.",
    }


def _sample_id(path: Path, index: int | None) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in path.stem)[:60]
    return f"{index:04d}_{stem}_{digest}" if index is not None else f"{stem}_{digest}"


def _write_components_csv(path: Path, components: list[dict]) -> None:
    fieldnames = [
        "component_id",
        "class_id",
        "class_label",
        "area_px",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "centroid_x",
        "centroid_y",
        "perimeter_px",
        "solidity",
        "compactness",
        "footprint_area_px",
        "internal_dark_area_px",
        "replacement_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in components:
            writer.writerow(row)


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = [
        "sample_id",
        "source_image",
        "ore_class_candidate",
        "component_count",
        "ordinary_component_count",
        "fine_component_count",
        "sulfide_fraction",
        "ordinary_fraction",
        "fine_fraction",
        "talc_candidate_fraction",
        "analysis_width",
        "analysis_height",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _save_rgb(path: Path, array: np.ndarray, quality: int = 92) -> None:
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path, quality=quality, optimize=True)


def _save_gray(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8), mode="L").save(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
