#!/usr/bin/env python3
"""Model-free heuristic talcose classifier — CLI.

Detects talc zones (scattered dark flakes in the non-ore matrix) with a
brightness+morphology pipeline and classifies an optical-microscopy ore image as
talcose / not-talcose by talc-zone area. No neural model required.

Algorithm and approved production parameters:
`src/ore_classifier/talc_zone_heuristic.py`. Talc Review browser integration is
documented in `docs/ui/v2/specs/talc-mask-review-web-app-v0.1.md`.

PRODUCTION REQUIREMENT: pass `--ore-mask` from the trained sulfide model. Without
it, the built-in brightness fallback is used, which over-calls talcose on unseen
folders. The fixed 87.8% accuracy assumes a model-quality ore mask.

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
    make_talc_zone_overlay,
    save_talc_zone_outputs,
)

Image.MAX_IMAGE_PIXELS = None
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def make_overlay(rgb: np.ndarray, zone: np.ndarray, flake: np.ndarray, max_side: int = 1600) -> np.ndarray:
    return make_talc_zone_overlay(rgb, zone, flake, max_side=max_side)


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
    sample_dir = out_dir / image_path.stem
    saved = save_talc_zone_outputs(
        sample_dir,
        rgb,
        result,
        cfg,
        image_path=image_path,
        ore_mask_source=str(ore_mask_path) if ore_mask_path is not None else "brightness_fallback",
        write_overlay=write_overlay,
    )
    return saved["record"]


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
    if args.ore_mask is None:
        print("WARNING: no --ore-mask given; using the BRIGHTNESS FALLBACK ore mask. "
              "For production pass the trained sulfide model's mask (it over-calls "
              "talcose on unseen folders otherwise).", file=sys.stderr, flush=True)
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
