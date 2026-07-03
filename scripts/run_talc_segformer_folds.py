#!/usr/bin/env python3
"""Run image-level SegFormer folds for non-sulfide talc segmentation.

This script wraps the talc dataset builder and trainer so validation never
leaks tiles from the same source image into training. Each fold:

1. selects validation sample ids at image level;
2. rebuilds a non-sulfide talc dataset with those ids forced into val;
3. trains a SegFormer talc model via `scripts/train_talc_segmentation.py`;
4. calibrates the probability threshold on validation tiles.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_talc_dataset import (  # noqa: E402
    DEFAULT_CLEAN_IMAGE_DIR,
    DEFAULT_CONVERSION_DIR,
    build_dataset,
    list_reviewed_samples,
)
from ore_classifier.datasets import BinarySulfideTileDataset  # noqa: E402
from ore_classifier.model_io import (  # noqa: E402
    forward_logits,
    load_binary_segmentation_checkpoint,
    resolve_device,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion-dir", type=Path, default=DEFAULT_CONVERSION_DIR)
    parser.add_argument("--clean-image-dir", type=Path, default=DEFAULT_CLEAN_IMAGE_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/talc_segformer_folds"))
    parser.add_argument("--model", choices=("segformer_b0", "segformer_b1", "segformer_b2"), default="segformer_b0")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--folds-to-run", default="all", help="Comma-separated fold ids or 'all'.")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--tile-size", type=int, default=384)
    parser.add_argument("--stride", type=int, default=288)
    parser.add_argument("--max-tiles-per-source", type=int, default=36)
    parser.add_argument("--min-positive-fraction", type=float, default=0.001)
    parser.add_argument("--min-valid-fraction", type=float, default=0.30)
    parser.add_argument("--negative-keep-fraction", type=float, default=0.20)
    parser.add_argument("--analyzed-min-value", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pretrained-model", default=None)
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--calibration-batch-size", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.folds < 2:
        raise ValueError("--folds must be >= 2")
    thresholds = parse_thresholds(args.thresholds)

    stats: defaultdict[str, int] = defaultdict(int)
    samples = list_reviewed_samples(args.conversion_dir.resolve(), args.clean_image_dir.resolve(), stats)
    if not samples:
        raise RuntimeError("no reviewed talc samples found")
    folds = make_stratified_folds(samples, k=args.folds, seed=args.seed)
    fold_ids = parse_folds_to_run(args.folds_to_run, args.folds)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    folds_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "folds": folds,
        "samples": [
            {"sample_id": sample["sample_id"], "group": sample["group"], "image_path": str(sample["image_path"])}
            for sample in samples
        ],
    }
    write_json(args.out_dir / "folds.json", folds_payload)

    fold_summaries = []
    for fold_id in fold_ids:
        fold_summary = run_fold(args=args, fold_id=fold_id, val_samples=set(folds[str(fold_id)]), thresholds=thresholds)
        fold_summaries.append(fold_summary)

    summary = {
        "schema_version": "talc-segformer-folds-v0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "folds": args.folds,
        "folds_run": fold_ids,
        "thresholds": thresholds,
        "sample_count": len(samples),
        "fold_summaries": fold_summaries,
        "mean_best_iou_talc": mean([fold["best_threshold_metrics"]["iou_talc"] for fold in fold_summaries]),
        "mean_best_f1_talc": mean([fold["best_threshold_metrics"]["f1_talc"] for fold in fold_summaries]),
    }
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


def make_stratified_folds(samples: list[dict], *, k: int, seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    grouped: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        grouped[sample["group"]].append(sample["sample_id"])
    folds = {str(i): [] for i in range(k)}
    for group_ids in grouped.values():
        ids = sorted(group_ids)
        rng.shuffle(ids)
        for index, sample_id in enumerate(ids):
            folds[str(index % k)].append(sample_id)
    for ids in folds.values():
        ids.sort()
    return folds


def parse_folds_to_run(raw: str, total_folds: int) -> list[int]:
    if raw.strip().lower() == "all":
        return list(range(total_folds))
    fold_ids = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        fold_id = int(part)
        if fold_id < 0 or fold_id >= total_folds:
            raise ValueError(f"fold id out of range: {fold_id}")
        fold_ids.append(fold_id)
    if not fold_ids:
        raise ValueError("--folds-to-run selected no folds")
    return fold_ids


def parse_thresholds(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("--thresholds cannot be empty")
    for value in values:
        if value <= 0 or value >= 1:
            raise ValueError(f"threshold must be in (0,1): {value}")
    return sorted(set(values))


def run_fold(args, fold_id: int, val_samples: set[str], thresholds: list[float]) -> dict:
    fold_dir = args.out_dir / f"fold_{fold_id:02d}"
    if args.overwrite and fold_dir.exists():
        shutil.rmtree(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = fold_dir / "dataset"
    train_dir = fold_dir / args.model

    manifest = build_dataset(
        conversion_dir=args.conversion_dir.resolve(),
        clean_image_dir=args.clean_image_dir.resolve(),
        negative_dirs=[],
        max_negative_images=0,
        out_dir=dataset_dir.resolve(),
        tile_size=args.tile_size,
        stride=args.stride,
        val_fraction=0.0,
        val_samples=val_samples,
        seed=args.seed + fold_id,
        max_tiles_per_source=args.max_tiles_per_source,
        min_positive_fraction=args.min_positive_fraction,
        min_valid_fraction=args.min_valid_fraction,
        negative_keep_fraction=args.negative_keep_fraction,
        analyzed_min_value=args.analyzed_min_value,
        sulfide_as_ignore=True,
        downscale_max_side=0,
        overwrite=True,
    )

    train_command = [
        sys.executable,
        str(ROOT / "scripts" / "train_talc_segmentation.py"),
        "--dataset-manifest",
        str(dataset_dir / "manifest.json"),
        "--out-dir",
        str(train_dir),
        "--model",
        args.model,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--device",
        args.device,
        "--max-steps-per-epoch",
        str(args.max_steps_per_epoch),
        "--seed",
        str(args.seed + fold_id),
    ]
    if args.amp:
        train_command.append("--amp")
    if args.pretrained_model:
        train_command.extend(["--pretrained-model", args.pretrained_model])
    if args.allow_random_init:
        train_command.append("--allow-random-init")

    (fold_dir / "train_command.txt").write_text(" ".join(train_command) + "\n", encoding="utf-8")
    subprocess.run(train_command, cwd=ROOT, check=True)

    threshold_metrics = calibrate_thresholds(
        manifest_path=dataset_dir / "manifest.json",
        checkpoint_path=train_dir / "best.pt",
        thresholds=thresholds,
        batch_size=args.calibration_batch_size,
        num_workers=args.num_workers,
        device_raw=args.device,
    )
    write_threshold_csv(fold_dir / "threshold_metrics.csv", threshold_metrics)
    best = max(threshold_metrics, key=lambda row: (row["iou_talc"], row["f1_talc"], row["threshold"]))
    fold_summary = {
        "fold_id": fold_id,
        "val_samples": sorted(val_samples),
        "dataset_manifest": str(dataset_dir / "manifest.json"),
        "train_dir": str(train_dir),
        "checkpoint": str(train_dir / "best.pt"),
        "dataset_items": len(manifest["items"]),
        "dataset_stats": manifest["stats"],
        "best_threshold_metrics": best,
    }
    write_json(fold_dir / "summary.json", fold_summary)
    return fold_summary


@torch.no_grad()
def calibrate_thresholds(
    *,
    manifest_path: Path,
    checkpoint_path: Path,
    thresholds: list[float],
    batch_size: int,
    num_workers: int,
    device_raw: str,
) -> list[dict]:
    device = resolve_device(device_raw)
    model, _checkpoint_meta = load_binary_segmentation_checkpoint(checkpoint_path, device)
    model.eval()
    dataset = BinarySulfideTileDataset(manifest_path, split="val", augment=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    counts = {
        threshold: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "valid": 0}
        for threshold in thresholds
    }
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = forward_logits(model, images, masks.shape[-2:])
        probs = torch.softmax(logits, dim=1)[:, 1]
        valid = masks != 255
        target_pos = masks == 1
        target_neg = masks == 0
        for threshold in thresholds:
            pred_pos = probs >= threshold
            pred_neg = ~pred_pos
            counts[threshold]["tp"] += int((pred_pos & target_pos & valid).sum().detach().cpu())
            counts[threshold]["fp"] += int((pred_pos & target_neg & valid).sum().detach().cpu())
            counts[threshold]["fn"] += int((pred_neg & target_pos & valid).sum().detach().cpu())
            counts[threshold]["tn"] += int((pred_neg & target_neg & valid).sum().detach().cpu())
            counts[threshold]["valid"] += int(valid.sum().detach().cpu())
    return [threshold_row(threshold, counts[threshold]) for threshold in thresholds]


def threshold_row(threshold: float, count: dict[str, int]) -> dict:
    tp = count["tp"]
    fp = count["fp"]
    fn = count["fn"]
    tn = count["tn"]
    valid = count["valid"]
    iou_talc = safe_div(tp, tp + fp + fn)
    iou_not_talc = safe_div(tn, tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "threshold": threshold,
        "iou_talc": iou_talc,
        "iou_not_talc": iou_not_talc,
        "precision_talc": precision,
        "recall_talc": recall,
        "f1_talc": f1,
        "pixel_acc": safe_div(tp + tn, valid),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "valid": valid,
    }


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def write_threshold_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
