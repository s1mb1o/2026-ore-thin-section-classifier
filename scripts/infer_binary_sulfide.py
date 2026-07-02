#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.datasets import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from ore_classifier.model_io import (  # noqa: E402
    forward_logits,
    load_binary_segmentation_checkpoint,
    resolve_device,
)
from ore_classifier.tiling import Tile, iter_tiles, save_gray  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run tiled binary sulfide inference on one image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--preview-max-side", type=int, default=2048)
    parser.add_argument("--save-full-overlay", action="store_true")
    args = parser.parse_args()

    if args.stride > args.tile_size:
        raise ValueError("--stride must be <= --tile-size")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = resolve_device(args.device)
    model, checkpoint_meta = load_binary_segmentation_checkpoint(args.checkpoint, device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    width, height = image.size
    tiles = iter_tiles(width=width, height=height, tile_size=args.tile_size, stride=args.stride)
    weight = tile_weight(args.tile_size)

    with tempfile.TemporaryDirectory(prefix="sulfide_infer_", dir=str(args.out_dir)) as tmp:
        prob_sum = np.memmap(Path(tmp) / "prob_sum.dat", mode="w+", dtype=np.float32, shape=(height, width))
        weight_sum = np.memmap(Path(tmp) / "weight_sum.dat", mode="w+", dtype=np.float32, shape=(height, width))
        processed = 0
        with torch.no_grad():
            for batch_tiles in batched(tiles, args.batch_size):
                tensor = torch.stack([preprocess_tile(image, tile) for tile in batch_tiles]).to(device)
                logits = forward_logits(model, tensor, (args.tile_size, args.tile_size))
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().astype(np.float32)
                for tile, prob in zip(batch_tiles, probs, strict=True):
                    valid_h = min(tile.height, height - tile.y)
                    valid_w = min(tile.width, width - tile.x)
                    tile_weight_valid = weight[:valid_h, :valid_w]
                    y_slice = slice(tile.y, tile.y + valid_h)
                    x_slice = slice(tile.x, tile.x + valid_w)
                    prob_sum[y_slice, x_slice] += prob[:valid_h, :valid_w] * tile_weight_valid
                    weight_sum[y_slice, x_slice] += tile_weight_valid
                    processed += 1

        prob = np.asarray(prob_sum / np.maximum(weight_sum, 1e-6), dtype=np.float32)
        confidence = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
        mask = (prob >= args.threshold).astype(np.uint8) * 255
        mask_path = args.out_dir / "sulfide_mask.png"
        confidence_path = args.out_dir / "confidence.png"
        save_gray(mask_path, mask)
        save_gray(confidence_path, confidence)

        overlay_preview_path = args.out_dir / "overlay_preview.jpg"
        save_overlay(
            image=image,
            mask=mask,
            confidence=confidence,
            path=overlay_preview_path,
            max_side=args.preview_max_side,
        )
        full_overlay_path = None
        if args.save_full_overlay:
            full_overlay_path = args.out_dir / "overlay_full.jpg"
            save_overlay(image=image, mask=mask, confidence=confidence, path=full_overlay_path, max_side=0)

    sulfide_fraction = float((mask > 0).mean())
    summary = {
        "schema_version": "binary-sulfide-inference-v0.1",
        "image": str(args.image),
        "checkpoint": str(args.checkpoint),
        "checkpoint_meta": checkpoint_meta,
        "width": width,
        "height": height,
        "tile_size": args.tile_size,
        "stride": args.stride,
        "tiles": len(tiles),
        "tiles_processed": processed,
        "threshold": args.threshold,
        "device": str(device),
        "seconds": round(time.time() - started, 3),
        "sulfide_fraction": sulfide_fraction,
        "paths": {
            "sulfide_mask": str(mask_path),
            "confidence": str(confidence_path),
            "overlay_preview": str(overlay_preview_path),
            "overlay_full": str(full_overlay_path) if full_overlay_path else None,
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


def preprocess_tile(image: Image.Image, tile: Tile) -> torch.Tensor:
    crop = image.crop((tile.x, tile.y, tile.x + tile.width, tile.y + tile.height))
    if crop.size != (tile.width, tile.height):
        padded = Image.new("RGB", (tile.width, tile.height), (0, 0, 0))
        padded.paste(crop, (0, 0))
        crop = padded
    tensor = TF.to_tensor(crop)
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def tile_weight(tile_size: int) -> np.ndarray:
    if tile_size <= 2:
        return np.ones((tile_size, tile_size), dtype=np.float32)
    one_d = np.hanning(tile_size).astype(np.float32)
    one_d = np.maximum(one_d, 0.05)
    weight = np.outer(one_d, one_d)
    return (weight / weight.max()).astype(np.float32)


def save_overlay(image: Image.Image, mask: np.ndarray, confidence: np.ndarray, path: Path, max_side: int) -> None:
    base = image.copy()
    mask_img = Image.fromarray(mask, mode="L")
    conf_img = Image.fromarray(confidence, mode="L")
    if max_side and max(base.size) > max_side:
        scale = max_side / float(max(base.size))
        new_size = (max(1, int(base.size[0] * scale)), max(1, int(base.size[1] * scale)))
        base = base.resize(new_size, Image.Resampling.BILINEAR)
        mask_img = mask_img.resize(new_size, Image.Resampling.NEAREST)
        conf_img = conf_img.resize(new_size, Image.Resampling.BILINEAR)

    base_arr = np.asarray(base).astype(np.float32)
    mask_arr = np.asarray(mask_img) > 0
    conf_arr = np.asarray(conf_img).astype(np.float32) / 255.0
    color = np.zeros_like(base_arr)
    color[..., 0] = 255.0
    color[..., 1] = 216.0
    alpha = np.where(mask_arr, 0.25 + 0.45 * conf_arr, 0.0)[..., None]
    overlay = base_arr * (1.0 - alpha) + color * alpha
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").save(path, quality=92, optimize=True)


def batched(items: list[Tile], batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


if __name__ == "__main__":
    raise SystemExit(main())
