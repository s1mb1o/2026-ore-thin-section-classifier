#!/usr/bin/env python3
"""Build a from-scratch talc annotation workspace for apps/talc_review_web.py.

Unlike the blue-line conversion path, this workspace starts from arbitrary
clean images (any class folder, any mix of directories) with empty talc and
ignore masks, so a reviewer can annotate talc regions manually with the web
canvas tools (brush / lasso / polygon / rectangle / SAM2).

The generated manifest.json is compatible with the review app:

    python3 scripts/prepare_talc_scratch_workspace.py \
      --images "data/Фото руд по сортам. ч1/Оталькованные руды" \
      --images "data/Фото руд по сортам. ч1/Рядовые руды" \
      --per-dir-limit 10 --shuffle-seed 7 \
      --output-dir outputs/talc_scratch_workspace

    python3 apps/talc_review_web.py --conversion-dir outputs/talc_scratch_workspace
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SCHEMA_VERSION = "talc-scratch-workspace-v0.1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def collect_images(inputs: list[Path], *, per_dir_limit: int | None, shuffle_seed: int | None) -> list[Path]:
    images: list[Path] = []
    for entry in inputs:
        entry = entry.expanduser()
        if entry.is_file():
            if entry.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(entry.resolve())
            continue
        if not entry.is_dir():
            raise FileNotFoundError(f"input does not exist: {entry}")
        found = sorted(
            p.resolve()
            for p in entry.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(found)
        if per_dir_limit is not None:
            found = found[:per_dir_limit]
        images.extend(found)
    # Preserve order but drop duplicates.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in images:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def unique_sample_id(stem: str, taken: set[str]) -> str:
    sample_id = stem
    counter = 2
    while sample_id in taken:
        sample_id = f"{stem}_{counter}"
        counter += 1
    taken.add(sample_id)
    return sample_id


def build_workspace(
    images: list[Path],
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict:
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    taken_ids: set[str] = set()
    samples: list[dict] = []
    for image_path in images:
        sample_id = unique_sample_id(image_path.stem, taken_ids)
        sample_dir = samples_dir / sample_id
        summary_path = sample_dir / "conversion_summary.json"
        if summary_path.exists() and not overwrite:
            samples.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        sample_dir.mkdir(parents=True, exist_ok=True)

        source_copy = sample_dir / image_path.name
        if not source_copy.exists() or overwrite:
            shutil.copy2(image_path, source_copy)
        with Image.open(source_copy) as img:
            width, height = img.size

        empty_mask = np.zeros((height, width), dtype=np.uint8)
        final_mask_path = sample_dir / "final_talc_mask.png"
        ignore_mask_path = sample_dir / "ignore_mask.png"
        if not final_mask_path.exists() or overwrite:
            Image.fromarray(empty_mask, mode="L").save(final_mask_path)
        if not ignore_mask_path.exists() or overwrite:
            Image.fromarray(empty_mask, mode="L").save(ignore_mask_path)

        summary = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "image_id": sample_id,
            "image_path": str(image_path),
            "original_path": str(source_copy),
            "source_folder": str(image_path.parent),
            "width": int(width),
            "height": int(height),
            "candidate_talc_pixels": 0,
            "final_talc_pixels": 0,
            "overlap_pixels": 0,
            "status": "scratch_unlabeled",
            "paths": {
                "source_image": str(source_copy),
                "final_talc_mask": str(final_mask_path),
                "ignore_mask": str(ignore_mask_path),
            },
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        samples.append(summary)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "input_path": str(images[0].parent) if images else str(output_dir),
        "mode": "scratch_annotation",
        "sample_count": len(samples),
        "samples": samples,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--images",
        type=Path,
        action="append",
        required=True,
        help="Image file or directory; repeat the flag to mix several folders/classes.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Workspace directory for the review app.")
    parser.add_argument("--per-dir-limit", type=int, default=None, help="Take at most N images from each input directory.")
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Shuffle each directory with this seed before applying --per-dir-limit (stratified random pick).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Global cap on the number of samples.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing samples (keeps nothing).")
    args = parser.parse_args()

    images = collect_images(args.images, per_dir_limit=args.per_dir_limit, shuffle_seed=args.shuffle_seed)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise SystemExit("no images found for the given --images inputs")

    manifest = build_workspace(images, args.output_dir, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "workspace": str(args.output_dir),
                "sample_count": manifest["sample_count"],
                "manifest": str(args.output_dir / "manifest.json"),
                "launch": f"python3 apps/talc_review_web.py --conversion-dir {args.output_dir}",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
