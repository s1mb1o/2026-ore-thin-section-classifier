#!/usr/bin/env python3
"""Build a tiled talc/not-talc training dataset from human-reviewed masks.

Positives come from `reviewed/reviewed_talc_mask.png` in the blue-line
conversion workspace, ignore pixels from `reviewed_ignore_mask.png` plus
non-analyzed (black border) pixels, and negatives from everything else in the
talcose images. Optional pure-negative tiles can be sampled from non-talcose
official folders once the talc-poor audit passes.

The output manifest matches `build_binary_sulfide_dataset.py`, so
`BinarySulfideTileDataset` and `scripts/train_binary_sulfide.py` consume it
unchanged. Splits are assigned per source image (never per tile), stratified
by series/magnification groups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.analyzed_area import build_analyzed_mask  # noqa: E402
from ore_classifier.tiling import crop_array_with_pad, iter_tiles, save_gray, save_rgb  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DEFAULT_CONVERSION_DIR = Path("outputs/talc_blue_line_conversion")
DEFAULT_CLEAN_IMAGE_DIR = Path("dataset/Фото руд по сортам. ч1/Оталькованные руды")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--conversion-dir", type=Path, default=DEFAULT_CONVERSION_DIR)
    parser.add_argument("--clean-image-dir", type=Path, default=DEFAULT_CLEAN_IMAGE_DIR)
    parser.add_argument(
        "--negative-dir",
        type=Path,
        action="append",
        default=None,
        help="Optional folder with talc-free images used as pure-negative sources; repeatable.",
    )
    parser.add_argument(
        "--max-negative-images",
        type=int,
        default=0,
        help="Cap on negative source images per --negative-dir; 0 disables negative sources.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/talc_dataset_v0"))
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--val-samples",
        default="",
        help="Comma-separated sample ids forced into val (overrides the stratified draw).",
    )
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--max-tiles-per-source", type=int, default=40)
    parser.add_argument("--min-positive-fraction", type=float, default=0.002)
    parser.add_argument("--min-valid-fraction", type=float, default=0.30)
    parser.add_argument("--negative-keep-fraction", type=float, default=0.25)
    parser.add_argument("--analyzed-min-value", type=int, default=8)
    parser.add_argument("--downscale-max-side", type=int, default=0, help="0 disables resizing before tiling")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = build_dataset(
        conversion_dir=args.conversion_dir.resolve(),
        clean_image_dir=args.clean_image_dir.resolve(),
        negative_dirs=[p.resolve() for p in (args.negative_dir or [])],
        max_negative_images=args.max_negative_images,
        out_dir=args.out_dir.resolve(),
        tile_size=args.tile_size,
        stride=args.stride,
        val_fraction=args.val_fraction,
        val_samples={s.strip() for s in args.val_samples.split(",") if s.strip()},
        seed=args.seed,
        max_tiles_per_source=args.max_tiles_per_source,
        min_positive_fraction=args.min_positive_fraction,
        min_valid_fraction=args.min_valid_fraction,
        negative_keep_fraction=args.negative_keep_fraction,
        analyzed_min_value=args.analyzed_min_value,
        downscale_max_side=args.downscale_max_side,
        overwrite=args.overwrite,
    )
    split_counts = defaultdict(int)
    positive_tiles = 0
    for item in manifest["items"]:
        split_counts[item["split"]] += 1
        positive_tiles += int(item["positive_fraction"] > 0)
    print(f"wrote {Path(manifest['out_dir']) / 'manifest.json'}")
    print(
        f"tiles: {len(manifest['items'])} train={split_counts['train']} val={split_counts['val']} "
        f"with_positive={positive_tiles}"
    )
    print(f"stats: {manifest['stats']}")
    return 0


def build_dataset(
    *,
    conversion_dir: Path,
    clean_image_dir: Path,
    negative_dirs: list[Path],
    max_negative_images: int,
    out_dir: Path,
    tile_size: int,
    stride: int,
    val_fraction: float,
    val_samples: set[str],
    seed: int,
    max_tiles_per_source: int,
    min_positive_fraction: float,
    min_valid_fraction: float,
    negative_keep_fraction: float,
    analyzed_min_value: int,
    downscale_max_side: int,
    overwrite: bool,
) -> dict:
    rng = random.Random(seed)
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    stats: defaultdict[str, int] = defaultdict(int)

    samples = list_reviewed_samples(conversion_dir, clean_image_dir, stats)
    splits = assign_sample_splits(samples, val_fraction=val_fraction, forced_val=val_samples, rng=rng)

    for sample in samples:
        sample_id = sample["sample_id"]
        added = add_source_tiles(
            items=items,
            stats=stats,
            source_type="talc_reviewed",
            label_hint="talcose_reviewed",
            sample_id=sample_id,
            group=sample["group"],
            image_path=sample["image_path"],
            talc_mask_path=sample["talc_mask_path"],
            ignore_mask_path=sample["ignore_mask_path"],
            split=splits[sample_id],
            out_dir=out_dir,
            tile_size=tile_size,
            stride=stride,
            max_tiles_per_source=max_tiles_per_source,
            min_positive_fraction=min_positive_fraction,
            min_valid_fraction=min_valid_fraction,
            negative_keep_fraction=negative_keep_fraction,
            analyzed_min_value=analyzed_min_value,
            downscale_max_side=downscale_max_side,
            rng=rng,
        )
        stats["talc_reviewed_tiles"] += added

    if max_negative_images > 0:
        for negative_dir in negative_dirs:
            for image_path in select_negative_images(negative_dir, max_negative_images, rng, stats):
                sample_id = f"neg_{image_path.stem}"
                added = add_source_tiles(
                    items=items,
                    stats=stats,
                    source_type="negative_official",
                    label_hint=infer_official_label(negative_dir),
                    sample_id=sample_id,
                    group="negative",
                    image_path=image_path,
                    talc_mask_path=None,
                    ignore_mask_path=None,
                    split=choose_split(rng, val_fraction),
                    out_dir=out_dir,
                    tile_size=tile_size,
                    stride=stride,
                    max_tiles_per_source=max_tiles_per_source,
                    min_positive_fraction=min_positive_fraction,
                    min_valid_fraction=min_valid_fraction,
                    negative_keep_fraction=1.0,
                    analyzed_min_value=analyzed_min_value,
                    downscale_max_side=downscale_max_side,
                    rng=rng,
                )
                stats["negative_tiles"] += added

    if not any(item["split"] == "val" for item in items) and len(items) > 1:
        items[-1]["split"] = "val"
    if not any(item["split"] == "train" for item in items) and len(items) > 1:
        items[0]["split"] = "train"

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "binary_talc",
        "tile_size": tile_size,
        "stride": stride,
        "seed": seed,
        "val_fraction": val_fraction,
        "analyzed_min_value": analyzed_min_value,
        "sample_splits": splits,
        "out_dir": str(out_dir),
        "stats": dict(stats),
        "items": items,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def list_reviewed_samples(conversion_dir: Path, clean_image_dir: Path, stats: defaultdict) -> list[dict]:
    samples_dir = conversion_dir / "samples"
    if not samples_dir.exists():
        raise FileNotFoundError(f"missing samples directory: {samples_dir}")
    samples: list[dict] = []
    for sample_dir in sorted(samples_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        sample_id = sample_dir.name
        talc_mask_path = sample_dir / "reviewed" / "reviewed_talc_mask.png"
        if not talc_mask_path.exists():
            stats["samples_without_reviewed_mask"] += 1
            print(f"skip {sample_id}: no reviewed talc mask", file=sys.stderr)
            continue
        image_path = find_clean_image(clean_image_dir, sample_id)
        if image_path is None:
            stats["samples_without_clean_image"] += 1
            print(f"skip {sample_id}: no clean original in {clean_image_dir}", file=sys.stderr)
            continue
        ignore_mask_path = sample_dir / "reviewed" / "reviewed_ignore_mask.png"
        samples.append(
            {
                "sample_id": sample_id,
                "group": sample_group(sample_id),
                "image_path": image_path,
                "talc_mask_path": talc_mask_path,
                "ignore_mask_path": ignore_mask_path if ignore_mask_path.exists() else None,
            }
        )
    stats["reviewed_samples"] = len(samples)
    return samples


def find_clean_image(clean_image_dir: Path, sample_id: str) -> Path | None:
    for suffix in (".JPG", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"):
        candidate = clean_image_dir / f"{sample_id}{suffix}"
        if candidate.exists():
            return candidate
    matches = sorted(clean_image_dir.glob(f"{sample_id}.*"))
    return matches[0] if matches else None


def sample_group(sample_id: str) -> str:
    lower = sample_id.lower()
    series = "dscn" if lower.startswith("dscn") else "scan"
    if "10x" in lower or "10х" in lower:
        magnification = "10x"
    elif "5x" in lower or "5х" in lower:
        magnification = "5x"
    else:
        magnification = "na"
    return f"{series}_{magnification}"


def assign_sample_splits(
    samples: list[dict],
    *,
    val_fraction: float,
    forced_val: set[str],
    rng: random.Random,
) -> dict[str, str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        groups[sample["group"]].append(sample["sample_id"])
    splits: dict[str, str] = {}
    for group_ids in groups.values():
        free = [sid for sid in sorted(group_ids) if sid not in forced_val]
        rng.shuffle(free)
        n_forced = len(group_ids) - len(free)
        n_val = max(0, math.ceil(len(group_ids) * val_fraction) - n_forced)
        for i, sid in enumerate(free):
            splits[sid] = "val" if i < n_val else "train"
    for sid in forced_val:
        splits[sid] = "val"
    return splits


def select_negative_images(negative_dir: Path, max_images: int, rng: random.Random, stats: defaultdict) -> list[Path]:
    if not negative_dir.exists():
        print(f"missing negative dir: {negative_dir}", file=sys.stderr)
        return []
    paths = [
        p
        for p in sorted(negative_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    rng.shuffle(paths)
    selected: list[Path] = []
    seen_hashes: set[str] = set()
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen_hashes:
            stats["negative_duplicates_skipped"] += 1
            continue
        seen_hashes.add(digest)
        selected.append(path)
        if len(selected) >= max_images:
            break
    return selected


def infer_official_label(path: Path) -> str:
    lower = str(path).lower()
    if "труднообогат" in lower or "тонкие" in lower:
        return "fine_intergrowth"
    if "рядовые" in lower:
        return "ordinary_intergrowth"
    return path.name


def choose_split(rng: random.Random, val_fraction: float) -> str:
    return "val" if rng.random() < val_fraction else "train"


def add_source_tiles(
    *,
    items: list[dict],
    stats: defaultdict,
    source_type: str,
    label_hint: str,
    sample_id: str,
    group: str,
    image_path: Path,
    talc_mask_path: Path | None,
    ignore_mask_path: Path | None,
    split: str,
    out_dir: Path,
    tile_size: int,
    stride: int,
    max_tiles_per_source: int,
    min_positive_fraction: float,
    min_valid_fraction: float,
    negative_keep_fraction: float,
    analyzed_min_value: int,
    downscale_max_side: int,
    rng: random.Random,
) -> int:
    try:
        rgb = load_rgb(image_path, downscale_max_side)
        talc = (
            load_mask(talc_mask_path, rgb.shape[:2], downscale_max_side) > 0
            if talc_mask_path is not None
            else np.zeros(rgb.shape[:2], dtype=bool)
        )
        reviewed_ignore = (
            load_mask(ignore_mask_path, rgb.shape[:2], downscale_max_side) > 0
            if ignore_mask_path is not None
            else np.zeros(rgb.shape[:2], dtype=bool)
        )
    except Exception as exc:
        stats[f"{source_type}_source_errors"] += 1
        print(f"skip {image_path}: {exc}", file=sys.stderr)
        return 0

    analyzed = build_analyzed_mask(rgb, min_value=analyzed_min_value).astype(bool)
    # The reviewed talc mask is authoritative: border/markup exclusion never
    # removes human-confirmed positives.
    ignore = (reviewed_ignore | ~analyzed) & ~talc

    h, w = rgb.shape[:2]
    tiles = iter_tiles(w, h, tile_size=tile_size, stride=stride)
    rng.shuffle(tiles)
    added = 0
    source_hash = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:12]
    for tile in tiles:
        image_tile = crop_array_with_pad(rgb, tile, fill_value=(0, 0, 0))
        mask_tile = crop_array_with_pad(talc.astype(np.uint8), tile, fill_value=0)
        ignore_tile = crop_array_with_pad(ignore.astype(np.uint8), tile, fill_value=1)
        valid = ignore_tile == 0
        valid_fraction = float(valid.mean())
        if valid_fraction < min_valid_fraction:
            continue
        pos_fraction = float(((mask_tile > 0) & valid).sum() / max(valid.sum(), 1))
        if pos_fraction < min_positive_fraction and rng.random() > negative_keep_fraction:
            continue

        name = f"{source_type}_{source_hash}_{tile.x}_{tile.y}"
        image_rel = Path("tiles") / split / "images" / f"{name}.jpg"
        mask_rel = Path("tiles") / split / "masks" / f"{name}.png"
        ignore_rel = Path("tiles") / split / "ignore" / f"{name}.png"
        save_rgb(out_dir / image_rel, image_tile)
        save_gray(out_dir / mask_rel, mask_tile * 255)
        save_gray(out_dir / ignore_rel, ignore_tile * 255)

        items.append(
            {
                "image": str(image_rel),
                "mask": str(mask_rel),
                "ignore": str(ignore_rel),
                "split": split,
                "source_type": source_type,
                "label_hint": label_hint,
                "sample_id": sample_id,
                "group": group,
                "source_image": str(image_path),
                "source_mask": str(talc_mask_path) if talc_mask_path else None,
                "x": tile.x,
                "y": tile.y,
                "w": tile.width,
                "h": tile.height,
                "positive_fraction": round(pos_fraction, 6),
                "valid_fraction": round(valid_fraction, 6),
            }
        )
        added += 1
        if max_tiles_per_source and added >= max_tiles_per_source:
            break
    stats[f"{source_type}_sources"] += 1
    return added


def load_rgb(path: Path, downscale_max_side: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if downscale_max_side and max(image.size) > downscale_max_side:
            image.thumbnail((downscale_max_side, downscale_max_side), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8).copy()


def load_mask(path: Path, target_hw: tuple[int, int], downscale_max_side: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != (target_hw[1], target_hw[0]):
            image = image.resize((target_hw[1], target_hw[0]), Image.Resampling.NEAREST)
        return np.asarray(image, dtype=np.uint8).copy()


if __name__ == "__main__":
    raise SystemExit(main())
