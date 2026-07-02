#!/usr/bin/env python3
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
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.datasets import BinarySulfideTileDataset  # noqa: E402
from ore_classifier.models import create_resunet  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a binary sulfide segmentation model.")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--model", choices=("resunet", "segformer_b0", "segformer_b1"), default="resunet")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/train_binary_sulfide"))
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
    args = parser.parse_args()

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

    best_iou = -1.0
    log_path = args.out_dir / "train_log.csv"
    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("epoch", "train_loss", "val_loss", "val_iou_sulfide", "val_iou_bg", "val_pixel_acc", "seconds"),
        )
        writer.writeheader()

    with (args.out_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args) | {"device_resolved": str(device)}, f, ensure_ascii=False, indent=2, default=str)

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
        val_metrics = evaluate(model, val_loader, criterion, device, max_steps=0)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_iou_sulfide": round(val_metrics["iou_sulfide"], 6),
            "val_iou_bg": round(val_metrics["iou_bg"], 6),
            "val_pixel_acc": round(val_metrics["pixel_acc"], 6),
            "seconds": round(time.time() - started, 2),
        }
        append_csv(log_path, row)
        print(row, flush=True)

        is_best = val_metrics["iou_sulfide"] > best_iou
        if is_best:
            best_iou = val_metrics["iou_sulfide"]
        save_checkpoint(args.out_dir / "last.pt", model, optimizer, epoch, best_iou, args)
        if is_best:
            save_checkpoint(args.out_dir / "best.pt", model, optimizer, epoch, best_iou, args)

    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"best_val_iou_sulfide": best_iou}, f, indent=2)
    return 0


def resolve_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_model(args):
    if args.model == "resunet":
        return create_resunet(base_channels=args.base_channels)
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    pretrained = args.pretrained_model
    if pretrained in {"", "none", "random"}:
        pretrained = None
        args.allow_random_init = True
    if pretrained is None:
        if args.allow_random_init:
            return SegformerForSemanticSegmentation(segformer_config(args.model))
        pretrained = "nvidia/mit-b0" if args.model == "segformer_b0" else "nvidia/mit-b1"
    try:
        return SegformerForSemanticSegmentation.from_pretrained(
            pretrained,
            num_labels=2,
            id2label={0: "not_sulfide", 1: "sulfide"},
            label2id={"not_sulfide": 0, "sulfide": 1},
            ignore_mismatched_sizes=True,
        )
    except Exception as exc:
        if not args.allow_random_init:
            raise
        print(f"pretrained SegFormer load failed, using random init: {exc}", file=sys.stderr)
        return SegformerForSemanticSegmentation(segformer_config(args.model))


def segformer_config(model_name: str):
    from transformers import SegformerConfig

    if model_name == "segformer_b1":
        return SegformerConfig(
            num_labels=2,
            depths=[2, 2, 2, 2],
            hidden_sizes=[64, 128, 320, 512],
            decoder_hidden_size=256,
            id2label={0: "not_sulfide", 1: "sulfide"},
            label2id={"not_sulfide": 0, "sulfide": 1},
        )
    return SegformerConfig(
        num_labels=2,
        id2label={0: "not_sulfide", 1: "sulfide"},
        label2id={"not_sulfide": 0, "sulfide": 1},
    )


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp: bool, max_steps: int) -> float:
    model.train()
    total_loss = 0.0
    total_seen = 0
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        context = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
        with context:
            logits = forward_logits(model, images, masks.shape[-2:])
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu())
        total_seen += 1
        if max_steps and step >= max_steps:
            break
    return total_loss / max(total_seen, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_steps: int) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_seen = 0
    conf = torch.zeros((2, 2), dtype=torch.float64)
    correct = 0
    valid_total = 0
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = forward_logits(model, images, masks.shape[-2:])
        loss = criterion(logits, masks)
        total_loss += float(loss.detach().cpu())
        total_seen += 1
        preds = logits.argmax(dim=1)
        valid = masks != 255
        correct += int((preds[valid] == masks[valid]).sum().detach().cpu())
        valid_total += int(valid.sum().detach().cpu())
        for target in (0, 1):
            for pred in (0, 1):
                conf[target, pred] += int(((masks == target) & (preds == pred) & valid).sum().detach().cpu())
        if max_steps and step >= max_steps:
            break
    iou_bg = class_iou(conf, 0)
    iou_sulfide = class_iou(conf, 1)
    return {
        "loss": total_loss / max(total_seen, 1),
        "iou_bg": iou_bg,
        "iou_sulfide": iou_sulfide,
        "pixel_acc": correct / max(valid_total, 1),
    }


def forward_logits(model, images: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    outputs = model(images)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    if logits.shape[-2:] != target_hw:
        logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
    return logits


def class_iou(conf: torch.Tensor, class_id: int) -> float:
    tp = conf[class_id, class_id]
    fp = conf[:, class_id].sum() - tp
    fn = conf[class_id, :].sum() - tp
    denom = tp + fp + fn
    if float(denom) == 0.0:
        return 0.0
    return float(tp / denom)


def append_csv(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tuple(row.keys()))
        writer.writerow(row)


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_iou: float, args) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_iou_sulfide": best_iou,
            "args": vars(args),
        },
        path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
