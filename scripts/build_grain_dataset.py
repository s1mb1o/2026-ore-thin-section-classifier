#!/usr/bin/env python3
"""Build a per-grain dataset for the grain-level human-in-the-loop classifier (path B).

Input: a COMPLETED official batch directory produced by `run_official_batch.py`
(or `evaluate_official_pipeline.py`), i.e. one with `summary.csv` at the root and
`runs/<label>/<run_id>/ore_analysis/component_features.csv` per image. Each grain
is a connected sulfide component the segmentation model already found; its bbox,
centroid, morphology features and a heuristic ordinary/fine label are read
straight from `component_features.csv` (same coordinate space as the source
image — verified 1:1).

Output (under --out-dir):
  - `grains_manifest.csv`  one row per exported grain (features + provenance +
                           heuristic pre-label + crop path + specimen group)
  - `crops/<grade>/<grain_uid>.png`  small crop of each grain from the ORIGINAL
                           image, for the labeling UI (`apps/grain_review_web.py`)
  - `dataset_summary.json` counts and build parameters

Because a full batch has ~69k grains (median ~147/image), we do NOT export all
of them for human labeling. We keep the largest, most informative grains per
image (`--max-grains-per-image` after an `--min-grain-area-px` floor). The
downstream classifier still runs on ALL grains at aggregation time by reading
`component_features.csv` directly; only the human-labelable subset is exported
here.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.specimen import specimen_group  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

# Numeric per-grain features copied verbatim from component_features.csv.
FEATURE_COLUMNS = [
    "area_px",
    "footprint_area_px",
    "dark_inside_area_px",
    "dark_inside_ratio",
    "solidity",
    "compactness",
    "boundary_complexity",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "centroid_x",
    "centroid_y",
]

MANIFEST_COLUMNS = [
    "grain_uid",
    "run_id",
    "grade_label",
    "expected_ore_class",
    "image_rel_path",
    "source_dataset_path",
    "specimen_group",
    "component_id",
    "heuristic_label",
    "crop_path",
    *FEATURE_COLUMNS,
]

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-dir", type=Path, required=True, help="Completed official batch dir (has summary.csv + runs/).")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-grain-area-px", type=int, default=300, help="Drop grains smaller than this (noise).")
    parser.add_argument("--max-grains-per-image", type=int, default=48, help="Keep the N largest grains per image for labeling.")
    parser.add_argument("--crop-pad-px", type=int, default=10, help="Padding around each grain bbox when cropping.")
    parser.add_argument("--crop-max-side", type=int, default=256, help="Downscale crops so the longest side is at most this.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary_csv = args.batch_dir / "summary.csv"
    if not summary_csv.exists():
        raise SystemExit(f"batch summary not found: {summary_csv}")

    out_dir = args.out_dir
    crops_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(summary_csv.open(encoding="utf-8")))
    manifest_rows: list[dict[str, Any]] = []
    per_grade: dict[str, int] = {}
    per_heuristic: dict[str, int] = {}
    images_used = 0
    images_skipped = 0
    grains_skipped_bad_bbox = 0

    for image_index, row in enumerate(rows, start=1):
        grade_label = str(row.get("source_label", ""))
        expected = str(row.get("expected_ore_class", ""))
        rel_path = str(row.get("source_rel_path", ""))
        source_path = Path(str(row.get("source_dataset_path", "")))
        run_dir = Path(str(row.get("run_dir", "")))
        run_id = run_dir.name or str(row.get("run_id", ""))
        component_csv = run_dir / "ore_analysis" / "component_features.csv"
        if not component_csv.exists() or not source_path.exists():
            images_skipped += 1
            continue

        grains = read_and_select_grains(
            component_csv,
            min_area=args.min_grain_area_px,
            max_grains=args.max_grains_per_image,
        )
        if not grains:
            images_skipped += 1
            continue

        group_id = specimen_group(rel_path)
        try:
            image = Image.open(source_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - a single unreadable image must not abort the build.
            print(f"[warn] cannot open {source_path}: {exc}", flush=True)
            images_skipped += 1
            continue

        grade_crop_dir = crops_dir / (grade_label or "unknown")
        grade_crop_dir.mkdir(parents=True, exist_ok=True)
        for grain in grains:
            grain_uid = f"{run_id}__c{grain.get('component_id', '')}"
            # A single malformed bbox cell (empty/non-numeric) must skip that grain,
            # not abort the whole build after partial crops are written.
            try:
                crop = crop_grain(image, grain, pad=args.crop_pad_px, max_side=args.crop_max_side)
            except (ValueError, TypeError, KeyError) as exc:
                grains_skipped_bad_bbox += 1
                print(f"[warn] skip grain {grain_uid}: bad bbox ({exc})", flush=True)
                continue
            crop_path = grade_crop_dir / f"{grain_uid}.png"
            crop.save(crop_path, format="PNG", compress_level=3)
            heuristic = str(grain.get("label", ""))
            manifest_rows.append(
                {
                    "grain_uid": grain_uid,
                    "run_id": run_id,
                    "grade_label": grade_label,
                    "expected_ore_class": expected,
                    "image_rel_path": rel_path,
                    "source_dataset_path": str(source_path),
                    "specimen_group": group_id,
                    "component_id": grain["component_id"],
                    "heuristic_label": heuristic,
                    "crop_path": str(crop_path.relative_to(out_dir)),
                    **{col: grain.get(col, "") for col in FEATURE_COLUMNS},
                }
            )
            per_grade[grade_label] = per_grade.get(grade_label, 0) + 1
            per_heuristic[heuristic] = per_heuristic.get(heuristic, 0) + 1
        images_used += 1
        if image_index % 25 == 0:
            print(f"[{image_index}/{len(rows)}] images processed, {len(manifest_rows)} grains exported", flush=True)

    write_manifest(out_dir / "grains_manifest.csv", manifest_rows)
    summary = {
        "schema_version": "grain-dataset-v0.1",
        "batch_dir": str(args.batch_dir),
        "images_used": images_used,
        "images_skipped": images_skipped,
        "grains_skipped_bad_bbox": grains_skipped_bad_bbox,
        "grains_exported": len(manifest_rows),
        "grains_per_grade": per_grade,
        "grains_per_heuristic_label": per_heuristic,
        "specimen_groups": len({r["specimen_group"] for r in manifest_rows}),
        "params": {
            "min_grain_area_px": args.min_grain_area_px,
            "max_grains_per_image": args.max_grains_per_image,
            "crop_pad_px": args.crop_pad_px,
            "crop_max_side": args.crop_max_side,
        },
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def read_and_select_grains(component_csv: Path, *, min_area: int, max_grains: int) -> list[dict[str, Any]]:
    grains: list[dict[str, Any]] = []
    with component_csv.open(encoding="utf-8", newline="") as f:
        for record in csv.DictReader(f):
            try:
                area = float(record.get("area_px", 0) or 0)
            except ValueError:
                area = 0.0
            if area < min_area:
                continue
            grains.append(record)
    grains.sort(key=lambda r: float(r.get("area_px", 0) or 0), reverse=True)
    return grains[:max_grains]


def crop_grain(image: Image.Image, grain: dict[str, Any], *, pad: int, max_side: int) -> Image.Image:
    x = int(float(grain["bbox_x"]))
    y = int(float(grain["bbox_y"]))
    w = int(float(grain["bbox_w"]))
    h = int(float(grain["bbox_h"]))
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(image.width, x + w + pad)
    bottom = min(image.height, y + h + pad)
    if right <= left or bottom <= top:
        right, bottom = min(image.width, left + 1), min(image.height, top + 1)
    crop = image.crop((left, top, right, bottom))
    if max(crop.size) > max_side:
        crop.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return crop


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
