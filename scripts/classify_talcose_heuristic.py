#!/usr/bin/env python3
"""Model-free heuristic talcose classifier — CLI.

Detects talc zones (scattered dark flakes in the non-ore matrix) with a
brightness+morphology pipeline and classifies an optical-microscopy ore image as
talcose / not-talcose by talc-zone area. No neural model required.

Algorithm and approved production parameters:
`src/ore_classifier/talc_zone_heuristic.py` and
`docs/notes/2026-07-04-heuristic-talcose-classifier.md`.

Single image:

    python3 scripts/classify_talcose_heuristic.py \
      --image "data/Фото руд по сортам. ч1/Оталькованные руды/DSCN4718.JPG" \
      --out-dir outputs/talcose_heuristic_demo

Batch over a folder (or several) with an optional CSV of results:

    python3 scripts/classify_talcose_heuristic.py \
      --images-dir "data/Фото руд по сортам. ч2/оталькованные" \
      --out-dir outputs/talcose_heuristic_batch \
      --summary-csv outputs/talcose_heuristic_batch/summary.csv

Provide `--ore-mask` (bool PNG) to use the trained sulfide model's mask instead
of the built-in brightness ore heuristic.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.talc_zone_heuristic import (  # noqa: E402
    OpaqueMaskConfig,
    TalcZoneConfig,
    detect_talc_zones,
    result_to_dict,
)

Image.MAX_IMAGE_PIXELS = None
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def make_overlay(rgb: np.ndarray, zone: np.ndarray, flake: np.ndarray, max_side: int = 1600) -> np.ndarray:
    img = Image.fromarray(rgb.astype(np.uint8), "RGB")
    scale = min(1.0, max_side / float(max(img.size)))
    if scale < 1.0:
        size = (max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale)))
        img = img.resize(size, Image.Resampling.BILINEAR)
        zone = np.asarray(Image.fromarray(zone.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)) > 0
        flake = np.asarray(Image.fromarray(flake.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)) > 0
    base = np.asarray(img).astype(np.float32)
    color = np.zeros_like(base)
    color[zone] = (255, 140, 0)
    alpha = (zone.astype(np.float32) * 0.40)[..., None]
    out = np.clip(base * (1 - alpha) + color * alpha, 0, 255)
    out[flake] = (10, 10, 10)
    return out.astype(np.uint8)


def process_one(image_path: Path, out_dir: Path, cfg: TalcZoneConfig, ore_cfg: OpaqueMaskConfig,
                ore_mask_path: Path | None, write_overlay: bool) -> dict:
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    ore_mask = None
    if ore_mask_path is not None:
        ore = np.asarray(Image.open(ore_mask_path).convert("L"))
        if ore.shape[:2] != rgb.shape[:2]:
            ore = np.asarray(Image.fromarray(ore).resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST))
        ore_mask = ore > 0
    result = detect_talc_zones(rgb, ore_mask=ore_mask, config=cfg, opaque_config=ore_cfg)
    record = result_to_dict(result, cfg)
    record["image"] = str(image_path)

    sample_dir = out_dir / image_path.stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.zone_mask.astype(np.uint8) * 255, "L").save(sample_dir / "talc_zone_mask.png")
    if write_overlay:
        Image.fromarray(make_overlay(rgb, result.zone_mask, result.flake_mask), "RGB").save(
            sample_dir / "overlay.jpg", quality=88
        )
    (sample_dir / "talcose_result.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record


def gather_images(image: Path | None, images_dir: list[Path] | None) -> list[Path]:
    paths: list[Path] = []
    if image is not None:
        paths.append(image)
    for d in images_dir or []:
        paths.extend(sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, action="append", default=None,
                        help="Folder of images; repeat for several folders.")
    parser.add_argument("--ore-mask", type=Path, default=None,
                        help="Optional opaque/ore mask PNG (e.g. from the sulfide model). Single-image only.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--k-threshold", type=float, default=None, help="Override dark threshold factor (default 0.85).")
    parser.add_argument("--classify-threshold", type=float, default=None, help="Override talcose zone-fraction cutoff.")
    parser.add_argument("--no-overlay", action="store_true")
    args = parser.parse_args()

    cfg = TalcZoneConfig()
    if args.k_threshold is not None:
        cfg.k_threshold = args.k_threshold
    if args.classify_threshold is not None:
        cfg.classify_threshold = args.classify_threshold
    ore_cfg = OpaqueMaskConfig()

    images = gather_images(args.image, args.images_dir)
    if not images:
        raise SystemExit("no images given: pass --image or --images-dir")
    if args.ore_mask is not None and len(images) != 1:
        raise SystemExit("--ore-mask is only valid with a single --image")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, path in enumerate(images, 1):
        rec = process_one(path, args.out_dir, cfg, ore_cfg, args.ore_mask, not args.no_overlay)
        records.append(rec)
        print(f"[{i}/{len(images)}] {path.name}: {rec['ore_class']} "
              f"(talc {rec['talc_fraction']*100:.1f}%)", flush=True)

    if args.summary_csv or len(images) > 1:
        csv_path = args.summary_csv or (args.out_dir / "summary.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "ore_class", "is_talcose", "talc_fraction",
                                                   "matrix_median_luma", "dark_threshold"], extrasaction="ignore")
            writer.writeheader()
            for rec in records:
                writer.writerow(rec)
        print(f"summary -> {csv_path}")
    n_talcose = sum(r["is_talcose"] for r in records)
    print(f"done: {len(records)} images, {n_talcose} talcose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
