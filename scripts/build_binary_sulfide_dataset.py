#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
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

from ore_classifier.pseudo_labels import (  # noqa: E402
    DEFAULT_SULFIDE_CLASS_IDS,
    brightness_sulfide_pseudo_mask,
    lumenstone_binary_mask,
    parse_class_ids,
)
from ore_classifier.tiling import crop_array_with_pad, iter_tiles, save_gray, save_rgb  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DEFAULT_LUMENSTONE_ROOTS = (
    ROOT / "data/external/lumenstone/full/S1_v1/S1_v1",
    ROOT / "data/external/lumenstone/full/S2_v2/S2_v2",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build tiled sulfide/not_sulfide training data from LumenStone masks and official pseudo masks."
    )
    parser.add_argument("--official-root", type=Path, default=Path("dataset"))
    parser.add_argument("--lumenstone-root", type=Path, action="append", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/binary_sulfide_dataset"))
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--sulfide-class-ids", default=",".join(map(str, DEFAULT_SULFIDE_CLASS_IDS)))
    parser.add_argument("--max-lumenstone-images", type=int, default=0, help="0 means no limit")
    parser.add_argument("--max-official-images-per-label", type=int, default=80, help="0 disables official images")
    parser.add_argument("--max-tiles-per-source", type=int, default=24)
    parser.add_argument("--max-total-tiles", type=int, default=0, help="0 means no limit")
    parser.add_argument("--min-positive-fraction", type=float, default=0.002)
    parser.add_argument("--min-valid-fraction", type=float, default=0.30)
    parser.add_argument("--negative-keep-fraction", type=float, default=0.10)
    parser.add_argument("--downscale-max-side", type=int, default=0, help="0 disables resizing before tiling")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sulfide_class_ids = parse_class_ids(args.sulfide_class_ids)
    items: list[dict] = []
    stats = defaultdict(int)

    lumenstone_roots = args.lumenstone_root or [p for p in DEFAULT_LUMENSTONE_ROOTS if p.exists()]
    for root in lumenstone_roots:
        root = root.resolve()
        if not root.exists():
            print(f"missing LumenStone root: {root}", file=sys.stderr)
            continue
        for image_path, mask_path, source_split in list_lumenstone_pairs(root, args.max_lumenstone_images, rng):
            split = "val" if source_split == "test" else choose_split(rng, args.val_fraction)
            added = add_source_tiles(
                items=items,
                stats=stats,
                source_type="lumenstone",
                label_hint=root.parent.name,
                image_path=image_path,
                mask_path=mask_path,
                split=split,
                out_dir=out_dir,
                tile_size=args.tile_size,
                stride=args.stride,
                max_tiles_per_source=args.max_tiles_per_source,
                min_positive_fraction=args.min_positive_fraction,
                min_valid_fraction=args.min_valid_fraction,
                negative_keep_fraction=args.negative_keep_fraction,
                downscale_max_side=args.downscale_max_side,
                rng=rng,
                sulfide_class_ids=sulfide_class_ids,
            )
            stats["lumenstone_tiles"] += added
            if args.max_total_tiles and len(items) >= args.max_total_tiles:
                break
        if args.max_total_tiles and len(items) >= args.max_total_tiles:
            break

    if not args.max_total_tiles or len(items) < args.max_total_tiles:
        official_groups = list_official_images(args.official_root.resolve())
        for label_hint, paths in sorted(official_groups.items()):
            if args.max_official_images_per_label <= 0:
                break
            rng.shuffle(paths)
            for image_path in paths[: args.max_official_images_per_label]:
                split = choose_split(rng, args.val_fraction)
                added = add_source_tiles(
                    items=items,
                    stats=stats,
                    source_type="official_heuristic",
                    label_hint=label_hint,
                    image_path=image_path,
                    mask_path=None,
                    split=split,
                    out_dir=out_dir,
                    tile_size=args.tile_size,
                    stride=args.stride,
                    max_tiles_per_source=args.max_tiles_per_source,
                    min_positive_fraction=args.min_positive_fraction,
                    min_valid_fraction=args.min_valid_fraction,
                    negative_keep_fraction=args.negative_keep_fraction,
                    downscale_max_side=args.downscale_max_side,
                    rng=rng,
                    sulfide_class_ids=sulfide_class_ids,
                )
                stats["official_tiles"] += added
                if args.max_total_tiles and len(items) >= args.max_total_tiles:
                    break
            if args.max_total_tiles and len(items) >= args.max_total_tiles:
                break

    if not any(item["split"] == "val" for item in items) and len(items) > 1:
        items[-1]["split"] = "val"
    if not any(item["split"] == "train" for item in items) and len(items) > 1:
        items[0]["split"] = "train"

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tile_size": args.tile_size,
        "stride": args.stride,
        "sulfide_class_ids": list(sulfide_class_ids),
        "stats": dict(stats),
        "items": items,
    }
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    split_counts = defaultdict(int)
    for item in items:
        split_counts[item["split"]] += 1
    print(f"wrote {manifest_path}")
    print(f"tiles: {len(items)} train={split_counts['train']} val={split_counts['val']}")
    print(f"stats: {dict(stats)}")
    return 0


def list_lumenstone_pairs(root: Path, max_images: int, rng: random.Random):
    pairs = []
    for split in ("train", "test"):
        mask_dir = root / "masks" / split
        image_dir = root / "imgs" / split
        if not mask_dir.exists() or not image_dir.exists():
            continue
        for mask_path in sorted(mask_dir.iterdir()):
            if not mask_path.is_file() or mask_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image_path = find_matching_image(image_dir, mask_path.stem)
            if image_path is not None:
                pairs.append((image_path, mask_path, split))
    rng.shuffle(pairs)
    return pairs if max_images <= 0 else pairs[:max_images]


def find_matching_image(image_dir: Path, stem: str) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    matches = sorted(image_dir.glob(f"{stem}.*"))
    return matches[0] if matches else None


def list_official_images(root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_hint = infer_official_label(path.relative_to(root))
        if label_hint in {"panorama", "talc_annotation", "unknown"}:
            continue
        groups[label_hint].append(path)
    return groups


def infer_official_label(rel_path: Path) -> str:
    lower = "/".join(rel_path.parts).lower()
    if "панорамы" in lower:
        return "panorama"
    if "области оталькования" in lower:
        return "talc_annotation"
    if "отальк" in lower:
        return "talcose"
    if "труднообогат" in lower or "/тонкие/" in lower:
        return "fine_intergrowth"
    if "рядовые" in lower:
        return "ordinary_intergrowth"
    return "unknown"


def choose_split(rng: random.Random, val_fraction: float) -> str:
    return "val" if rng.random() < val_fraction else "train"


def add_source_tiles(
    *,
    items: list[dict],
    stats: defaultdict,
    source_type: str,
    label_hint: str,
    image_path: Path,
    mask_path: Path | None,
    split: str,
    out_dir: Path,
    tile_size: int,
    stride: int,
    max_tiles_per_source: int,
    min_positive_fraction: float,
    min_valid_fraction: float,
    negative_keep_fraction: float,
    downscale_max_side: int,
    rng: random.Random,
    sulfide_class_ids: tuple[int, ...],
) -> int:
    try:
        rgb = load_rgb(image_path, downscale_max_side)
        if mask_path is None:
            pseudo = brightness_sulfide_pseudo_mask(rgb)
        else:
            mask_array = load_mask(mask_path, rgb.shape[:2], downscale_max_side)
            pseudo = lumenstone_binary_mask(mask_array, sulfide_class_ids=sulfide_class_ids)
    except Exception as exc:
        stats[f"{source_type}_source_errors"] += 1
        print(f"skip {image_path}: {exc}", file=sys.stderr)
        return 0

    h, w = rgb.shape[:2]
    tiles = iter_tiles(w, h, tile_size=tile_size, stride=stride)
    rng.shuffle(tiles)
    added = 0
    source_hash = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:12]
    source_rel = str(image_path)
    for tile in tiles:
        image_tile = crop_array_with_pad(rgb, tile, fill_value=(0, 0, 0))
        mask_tile = crop_array_with_pad(pseudo.mask, tile, fill_value=0)
        ignore_tile = crop_array_with_pad(pseudo.ignore, tile, fill_value=1)
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
                "source_image": source_rel,
                "source_mask": str(mask_path) if mask_path else None,
                "x": tile.x,
                "y": tile.y,
                "w": tile.width,
                "h": tile.height,
                "positive_fraction": round(pos_fraction, 6),
                "valid_fraction": round(valid_fraction, 6),
                "threshold": pseudo.threshold,
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
        image = image.convert("RGB")
        if downscale_max_side and max(image.size) > downscale_max_side:
            image.thumbnail((downscale_max_side, downscale_max_side), Image.Resampling.NEAREST)
        if image.size != (target_hw[1], target_hw[0]):
            image = image.resize((target_hw[1], target_hw[0]), Image.Resampling.NEAREST)
        return np.asarray(image, dtype=np.uint8).copy()


if __name__ == "__main__":
    raise SystemExit(main())
