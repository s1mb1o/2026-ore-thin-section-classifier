#!/usr/bin/env python3
"""Train binary talc segmentation over non-sulfide pixels.

The input manifest is produced by `scripts/build_talc_dataset.py` with its
default sulfide-as-ignore behavior. Class 1 is talc; class 0 is non-talc
non-sulfide material; ignore index 255 covers sulfides, reviewed-ignore pixels,
and non-analyzed borders.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ore_classifier.datasets import BinarySulfideTileDataset  # noqa: E402
from ore_classifier.tracking import add_mlflow_args, mlflow_run  # noqa: E402
from train_binary_sulfide import (  # noqa: E402
    append_csv,
    create_model,
    evaluate,
    resolve_device,
    save_checkpoint,
    train_one_epoch,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-manifest", type=Path, default=Path("outputs/talc_non_sulfide_dataset_v0/manifest.json"))
    parser.add_argument("--model", choices=("resunet", "segformer_b0", "segformer_b1", "segformer_b2"), default="resunet")
    parser.add_argument("--out-dir", type=Path, default=Path("models/talc_segmentation/resunet_non_sulfide"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pretrained-model", default=None)
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260703)
    add_mlflow_args(parser, default_experiment="talc-segmentation")
    args = parser.parse_args()
    args.task = "binary_talc_non_sulfide"

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = BinarySulfideTileDataset(args.dataset_manifest, split="train", augment=True)
    val_ds = BinarySulfideTileDataset(args.dataset_manifest, split="val", augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = create_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log_path = args.out_dir / "train_log.csv"
    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("epoch", "train_loss", "val_loss", "val_iou_talc", "val_iou_not_talc", "val_pixel_acc", "seconds"),
        )
        writer.writeheader()

    with (args.out_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args) | {"device_resolved": str(device)}, f, ensure_ascii=False, indent=2, default=str)

    best_iou = -1.0
    with mlflow_run(args, params=vars(args) | {"device_resolved": str(device)}) as run:
        for epoch in range(1, args.epochs + 1):
            started = time.time()
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                max_steps=args.max_steps_per_epoch,
            )
            val_metrics = evaluate_talc(model, val_loader, criterion, device)
            row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_metrics["loss"], 6),
                "val_iou_talc": round(val_metrics["iou_talc"], 6),
                "val_iou_not_talc": round(val_metrics["iou_not_talc"], 6),
                "val_pixel_acc": round(val_metrics["pixel_acc"], 6),
                "seconds": round(time.time() - started, 2),
            }
            append_csv(log_path, row)
            run.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
            print(row, flush=True)

            is_best = val_metrics["iou_talc"] > best_iou
            if is_best:
                best_iou = val_metrics["iou_talc"]
            save_checkpoint(args.out_dir / "last.pt", model, optimizer, epoch, best_iou, args)
            if is_best:
                save_checkpoint(args.out_dir / "best.pt", model, optimizer, epoch, best_iou, args)

        metrics = {
            "task": args.task,
            "best_val_iou_talc": best_iou,
            "note": "checkpoint keeps best_iou_sulfide for loader compatibility; it is talc IoU for this task",
        }
        (args.out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        run.log_metrics({"best_val_iou_talc": best_iou})
        run.log_artifact(log_path)
        run.log_artifact(args.out_dir / "metrics.json")
    return 0


def evaluate_talc(model, loader, criterion, device) -> dict[str, float]:
    with torch.no_grad():
        metrics = evaluate(model, loader, criterion, device, max_steps=0)
    return {
        "loss": metrics["loss"],
        "iou_not_talc": metrics["iou_bg"],
        "iou_talc": metrics["iou_sulfide"],
        "pixel_acc": metrics["pixel_acc"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
