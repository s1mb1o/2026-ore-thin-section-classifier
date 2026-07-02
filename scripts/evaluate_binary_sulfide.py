#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.datasets import BinarySulfideTileDataset  # noqa: E402
from ore_classifier.model_io import (  # noqa: E402
    forward_logits,
    load_binary_segmentation_checkpoint,
    resolve_device,
)
from ore_classifier.segmentation_metrics import BinarySegmentationAccumulator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate binary sulfide segmentation checkpoint.")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--auc-bins", type=int, default=512)
    parser.add_argument("--hausdorff-max-items", type=int, default=512)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dataset = BinarySulfideTileDataset(args.dataset_manifest, split=args.split, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    model, checkpoint_meta = load_binary_segmentation_checkpoint(args.checkpoint, device)
    model.eval()

    accumulator = BinarySegmentationAccumulator(auc_bins=args.auc_bins)
    started = time.time()
    seen_tiles = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits = forward_logits(model, images, masks.shape[-2:])
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            masks_np = masks.cpu().numpy()
            preds_np = preds.cpu().numpy().astype(np.uint8)
            probs_np = probs.cpu().numpy().astype(np.float32)
            for idx in range(masks_np.shape[0]):
                target = masks_np[idx]
                valid = target != 255
                target_binary = (target == 1).astype(np.uint8)
                accumulator.update_confusion(
                    target=target_binary,
                    pred=preds_np[idx],
                    valid=valid,
                    prob_sulfide=probs_np[idx],
                )
                if args.hausdorff_max_items <= 0 or accumulator.summary().hausdorff_items < args.hausdorff_max_items:
                    accumulator.update_hausdorff(target=target_binary, pred=preds_np[idx], valid=valid)
                seen_tiles += 1

            if args.max_batches and batch_idx >= args.max_batches:
                break

    metrics = accumulator.summary().to_dict()
    output = {
        "schema_version": "binary-sulfide-eval-v0.1",
        "dataset_manifest": str(args.dataset_manifest),
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_meta": checkpoint_meta,
        "tiles_evaluated": seen_tiles,
        "hausdorff_max_items": args.hausdorff_max_items,
        "auc_bins": args.auc_bins,
        "device": str(device),
        "seconds": round(time.time() - started, 3),
        "metrics": metrics,
    }

    text = json.dumps(output, ensure_ascii=False, indent=2, default=str)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
