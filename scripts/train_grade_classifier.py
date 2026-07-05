#!/usr/bin/env python3
"""Train the grade-classifier CNN branch.

A supervised image-level EfficientNet-B3 classifier over ore-grade folder labels,
added as a PARALLEL branch to our segmentation-first pipeline. By default it
classifies the two data-rich grades — ordinary vs fine intergrowth — because the
talcose grade is scarce (all deconflicted talcose sits in the eval split) and is
deferred to the talc-segmentation branch. See docs/plans/37_grade-classifier-cnn-branch.md.

Training data = official manifest images with the selected labels, EXCLUDING the
fixed evaluation split (and its sha256 duplicates and label-conflict paths), so
the held-out 345 split stays a clean test set. Deps: torch + torchvision only
(no timm/albumentations/sklearn).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B3_Weights,
    ResNet50_Weights,
    convnext_tiny,
    efficientnet_b3,
    resnet50,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.tracking import add_mlflow_args, mlflow_run  # noqa: E402
from ore_classifier.preprocessing import (  # noqa: E402
    apply_preprocessing,
    default_preprocess_settings,
    normalize_preprocess_settings,
)
from ore_classifier.augmentation import (  # noqa: E402
    apply_augmentation,
    normalize_augmentation_settings,
)

Image.MAX_IMAGE_PIXELS = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPECIMEN_RE = re.compile(r"^\s*(\d{3,})")

# Moderate acquisition/surface-artifact profile for train-time augmentation
# (grinding scratches, polishing haze, pits, mild blur/noise). Seed is randomized
# per sample so each crop gets a distinct artifact pattern.
DEFAULT_TRAIN_AUGMENT: dict[str, Any] = {
    "enabled": True,
    "acquisition": {"blur_radius": 1.5, "gaussian_noise_std": 8.0},
    "surface_artifacts": {
        "scratch_count": 30,
        "scratch_intensity_pct": 35.0,
        "polishing_haze_pct": 20.0,
        "pit_count": 100,
        "pit_intensity_pct": 30.0,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/official_manifest.json")
    parser.add_argument("--audit-json", type=Path, default=ROOT / "outputs/official_label_audit/summary.json")
    parser.add_argument("--eval-split-json", type=Path, default=ROOT / "outputs/official_balanced_eval_split_deconflicted.json")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", default=["ordinary_intergrowth", "fine_intergrowth"])
    parser.add_argument("--backbone", default="efficientnet_b3", choices=list(BACKBONES))
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Cap total training-pool images (smoke).")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None, help="Cap optimizer steps per epoch (smoke).")
    parser.add_argument(
        "--preprocess-aug-prob",
        type=float,
        default=0.0,
        help="Probability of applying UI preprocessing to each training crop (train/serve robustness). 0 = off.",
    )
    parser.add_argument(
        "--preprocess-json",
        default=None,
        help="Path to JSON file or inline JSON of the preprocessing preset used for --preprocess-aug-prob (default: UI default preset).",
    )
    parser.add_argument(
        "--augment-aug-prob",
        type=float,
        default=0.0,
        help="Probability of applying acquisition/surface augmentation (scratches/haze/pits/blur/noise) to each training crop. 0 = off.",
    )
    parser.add_argument(
        "--augment-json",
        default=None,
        help="Path to JSON file or inline JSON of augmentation settings for --augment-aug-prob (default: a moderate acquisition profile).",
    )
    parser.add_argument(
        "--four-class",
        action="store_true",
        help="Benchmark mode: 4-class grade (ordinary/thin/talc/refractory) with grouped train/val over ALL labelled images (no eval-split holdout).",
    )
    add_mlflow_args(parser, default_experiment="grade-classifier")
    args = parser.parse_args()

    if args.four_class and args.classes == ["ordinary_intergrowth", "fine_intergrowth"]:
        args.classes = ["ordinary", "thin", "talc", "refractory"]

    preprocess_preset = None
    if args.preprocess_aug_prob > 0.0:
        if args.preprocess_json is not None:
            candidate = Path(args.preprocess_json)
            payload = json.loads(candidate.read_text(encoding="utf-8")) if candidate.exists() else json.loads(args.preprocess_json)
            preprocess_preset = normalize_preprocess_settings(payload)
        else:
            preprocess_preset = default_preprocess_settings()

    augment_settings = None
    if args.augment_aug_prob > 0.0:
        if args.augment_json is not None:
            candidate = Path(args.augment_json)
            payload = json.loads(candidate.read_text(encoding="utf-8")) if candidate.exists() else json.loads(args.augment_json)
            augment_settings = normalize_augmentation_settings(payload)
        else:
            augment_settings = normalize_augmentation_settings(DEFAULT_TRAIN_AUGMENT)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    samples = build_four_class_pool(args) if args.four_class else build_sample_pool(args)
    if args.limit is not None:
        samples = samples[: args.limit]
    train_items, val_items = grouped_split(samples, val_fraction=args.val_fraction, seed=args.seed)
    class_to_idx = {name: i for i, name in enumerate(args.classes)}
    print(f"pool={len(samples)} train={len(train_items)} val={len(val_items)} classes={args.classes}", flush=True)
    print(f"train class counts: {dict(Counter(s['label'] for s in train_items))}", flush=True)
    print(f"val class counts:   {dict(Counter(s['label'] for s in val_items))}", flush=True)

    if preprocess_preset is not None:
        print(f"preprocessing augmentation ON: prob={args.preprocess_aug_prob} preset={preprocess_preset}", flush=True)
    if augment_settings is not None:
        print(f"acquisition augmentation ON: prob={args.augment_aug_prob} settings={augment_settings}", flush=True)
    train_ds = GradeDataset(
        train_items,
        args.dataset_root,
        class_to_idx,
        train_transforms(
            args.img_size,
            preprocess_preset=preprocess_preset,
            preprocess_prob=args.preprocess_aug_prob,
            augment_settings=augment_settings,
            augment_prob=args.augment_aug_prob,
        ),
    )
    val_ds = GradeDataset(val_items, args.dataset_root, class_to_idx, eval_transforms(args.img_size))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = build_model(args.backbone, len(args.classes)).to(device)
    class_weights = inverse_frequency_weights(train_items, args.classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, epochs=args.epochs, warmup=args.warmup)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    mlflow_params = {
        "classes": args.classes,
        "img_size": args.img_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup": args.warmup,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "device": str(device),
        "pool_size": len(samples),
        "train_size": len(train_items),
        "val_size": len(val_items),
    }
    best_f1 = -1.0
    history: list[dict[str, Any]] = []
    with mlflow_run(args, params=mlflow_params) as run:
        for epoch in range(1, args.epochs + 1):
            train_loss = run_train_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp, args.max_steps_per_epoch)
            scheduler.step()
            metrics = evaluate(model, val_loader, device, args.classes)
            row = {"epoch": epoch, "train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"], **metrics}
            history.append(row)
            run.log_metrics(
                {
                    "train_loss": train_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "val_accuracy": metrics["accuracy"],
                    "val_macro_f1": metrics["macro_f1"],
                    **{f"val_f1_{name}": val for name, val in metrics["per_class_f1"].items()},
                },
                step=epoch,
            )
            print(
                f"[epoch {epoch}/{args.epochs}] loss={train_loss:.4f} val_acc={metrics['accuracy']:.4f} "
                f"val_macro_f1={metrics['macro_f1']:.4f} per_class_f1={metrics['per_class_f1']}",
                flush=True,
            )
            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                save_checkpoint(args.out_dir / "best.pt", model, args, class_to_idx, best_f1, epoch)
        save_checkpoint(args.out_dir / "last.pt", model, args, class_to_idx, best_f1, args.epochs)

        result = {
            "schema_version": "grade-classifier-train-v0.1",
            "classes": args.classes,
            "img_size": args.img_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup": args.warmup,
            "device": str(device),
            "pool_size": len(samples),
            "train_size": len(train_items),
            "val_size": len(val_items),
            "class_weights": class_weights.detach().cpu().tolist(),
            "preprocess_aug_prob": args.preprocess_aug_prob,
            "preprocess_preset": preprocess_preset,
            "augment_aug_prob": args.augment_aug_prob,
            "augment_settings": augment_settings,
            "best_val_macro_f1": best_f1,
            "history": history,
        }
        (args.out_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.out_dir / "train_val_split.json").write_text(
            json.dumps({"train": train_items, "val": val_items}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        run.log_metrics({"best_val_macro_f1": best_f1})
        run.log_artifact(args.out_dir / "metrics.json")
        run.log_artifact(args.out_dir / "best.pt")
    print(json.dumps({"best_val_macro_f1": best_f1, "out_dir": str(args.out_dir)}, ensure_ascii=False), flush=True)
    return 0


def build_sample_pool(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    eval_split = json.loads(args.eval_split_json.read_text(encoding="utf-8"))
    classes = set(args.classes)

    eval_paths = {item["path"] for item in eval_split.get("items", [])}
    conflict_paths = set(audit.get("conflict_paths", []))
    path_to_sha = audit.get("path_to_sha256", {})
    eval_hashes = {path_to_sha.get(p) for p in eval_paths if path_to_sha.get(p)}

    pool: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    excluded = Counter()
    for item in manifest["items"]:
        label = item.get("label_hint")
        if label not in classes:
            continue
        path = item["path"]
        if path in eval_paths:
            excluded["eval_split"] += 1
            continue
        if path in conflict_paths:
            excluded["conflict"] += 1
            continue
        digest = path_to_sha.get(path)
        if digest and digest in eval_hashes:
            excluded["eval_duplicate"] += 1
            continue
        if digest and digest in seen_hashes:
            excluded["train_duplicate"] += 1
            continue
        if not (args.dataset_root / path).exists():
            excluded["missing_file"] += 1
            continue
        if digest:
            seen_hashes.add(digest)
        pool.append({"path": path, "label": label, "group": specimen_group(path)})
    print(f"pool excluded: {dict(excluded)}", flush=True)
    pool.sort(key=lambda s: s["path"])
    return pool


def four_class_label(label_hint: str, path: str) -> str | None:
    # 4-class grade schema: split fine_intergrowth into thin (ч2/тонкие)
    # vs refractory (труднообогатимые). talc = the talcose grade only, NOT the 42
    # "Области оталькования" blue-contour annotations (label_hint talc_annotation).
    if label_hint == "ordinary_intergrowth":
        return "ordinary"
    if label_hint == "talcose":
        return "talc"
    if label_hint == "fine_intergrowth":
        low = "/".join(re.split(r"[\\/]", path)).lower()
        return "thin" if "тонкие" in low else "refractory"
    return None


def build_four_class_pool(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    path_to_sha = audit.get("path_to_sha256", {})
    # Group by content to (a) drop label-conflicting content and (b) dedupe.
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    no_hash: list[dict[str, Any]] = []
    for item in manifest["items"]:
        label = four_class_label(item.get("label_hint", ""), item["path"])
        if label is None:
            continue
        if not (args.dataset_root / item["path"]).exists():
            continue
        rec = {"path": item["path"], "label": label, "group": specimen_group(item["path"])}
        digest = path_to_sha.get(item["path"])
        (by_hash[digest].append(rec) if digest else no_hash.append(rec))
    pool: list[dict[str, Any]] = list(no_hash)
    excluded = Counter()
    for digest, recs in by_hash.items():
        labels = {r["label"] for r in recs}
        if len(labels) > 1:
            excluded["conflict"] += len(recs)  # same content, different grade → drop all
            continue
        excluded["duplicate"] += len(recs) - 1
        pool.append(recs[0])  # dedupe to one representative
    print(f"four-class pool excluded: {dict(excluded)}", flush=True)
    pool.sort(key=lambda s: s["path"])
    return pool


def specimen_group(path: str) -> str:
    name = Path(path).name
    match = SPECIMEN_RE.match(name)
    if match:
        return f"spec:{match.group(1)}"
    return f"file:{path}"  # DSCN camera names: no specimen id -> group is the file


def grouped_split(samples: list[dict[str, Any]], *, val_fraction: float, seed: int) -> tuple[list, list]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by_group[s["group"]].append(s)
    groups = list(by_group.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    target_val = int(round(len(samples) * val_fraction))
    val_items: list[dict[str, Any]] = []
    train_items: list[dict[str, Any]] = []
    val_labels: set[str] = set()
    for group in groups:
        members = by_group[group]
        if len(val_items) < target_val:
            val_items.extend(members)
            val_labels.update(m["label"] for m in members)
        else:
            train_items.extend(members)
    # Guarantee both classes appear in val; if not, move one train group over.
    all_labels = {s["label"] for s in samples}
    if not all_labels.issubset(val_labels):
        missing = all_labels - val_labels
        for label in missing:
            for group in groups:
                members = by_group[group]
                if any(m["label"] == label for m in members) and members[0] in train_items:
                    for m in members:
                        train_items.remove(m)
                    val_items.extend(members)
                    break
    return train_items, val_items


class GradeDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]], dataset_root: Path, class_to_idx: dict[str, int], transform):
        self.items = items
        self.dataset_root = dataset_root
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        image = Image.open(self.dataset_root / item["path"]).convert("RGB")
        tensor = self.transform(image)
        return tensor, self.class_to_idx[item["label"]]


class RandomPreprocess:
    """Stochastically apply the shared UI preprocessing to a PIL crop.

    Exposes the grade classifier to preprocessed inputs during training so it stays
    accurate when a user enables preprocessing at inference (train/serve match).
    Uses the same ``apply_preprocessing`` as the UI and the robustness harness.
    """

    def __init__(self, preset: dict[str, Any], prob: float):
        self.preset = preset
        self.prob = float(prob)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.prob > 0.0 and float(torch.rand(1).item()) < self.prob:
            return apply_preprocessing(image, self.preset)
        return image


class RandomTrainAug:
    """Stochastically apply acquisition/surface augmentation (scratches, haze, pits,
    blur, noise) via the shared ``apply_augmentation``. A fresh random seed per call
    gives each crop a distinct artifact pattern (the augmentation is seed-driven)."""

    def __init__(self, settings: dict[str, Any], prob: float):
        self.settings = settings
        self.prob = float(prob)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.prob > 0.0 and float(torch.rand(1).item()) < self.prob:
            settings = {**self.settings, "enabled": True,
                        "runtime": {**self.settings.get("runtime", {}),
                                    "random_seed": int(torch.randint(0, 2**31 - 1, (1,)).item())}}
            return apply_augmentation(image, settings)
        return image


def train_transforms(
    img_size: int,
    preprocess_preset: dict[str, Any] | None = None,
    preprocess_prob: float = 0.0,
    augment_settings: dict[str, Any] | None = None,
    augment_prob: float = 0.0,
):
    steps: list[Any] = [
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
    ]
    # Acquisition/surface artifacts then preprocessing, both on the 384x384 crop (fast).
    if augment_settings is not None and augment_prob > 0.0:
        steps.append(RandomTrainAug(augment_settings, augment_prob))
    if preprocess_preset is not None and preprocess_prob > 0.0:
        steps.append(RandomPreprocess(preprocess_preset, preprocess_prob))
    steps += [
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(steps)


def eval_transforms(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


BACKBONES = ("efficientnet_b3", "convnext_tiny", "resnet50")


def build_model(backbone: str, num_classes: int) -> nn.Module:
    if backbone == "efficientnet_b3":
        model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif backbone == "convnext_tiny":
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif backbone == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"unknown backbone: {backbone}")
    return model


def inverse_frequency_weights(items: list[dict[str, Any]], classes: list[str]) -> torch.Tensor:
    counts = Counter(s["label"] for s in items)
    total = sum(counts.values())
    weights = [total / (len(classes) * max(counts.get(name, 0), 1)) for name in classes]
    return torch.tensor(weights, dtype=torch.float32)


def build_scheduler(optimizer, *, epochs: int, warmup: int):
    if warmup > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup))
        return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_sched, cosine], milestones=[warmup])
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))


def run_train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, max_steps) -> float:
    model.train()
    total_loss = 0.0
    count = 0
    for step, (images, targets) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * images.size(0)
        count += images.size(0)
    return total_loss / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, device, classes: list[str]) -> dict[str, Any]:
    model.eval()
    num_classes = len(classes)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        for true, pred in zip(targets.numpy(), preds, strict=True):
            confusion[int(true), int(pred)] += 1
    return macro_f1_from_confusion(confusion, classes)


def macro_f1_from_confusion(confusion: np.ndarray, classes: list[str]) -> dict[str, Any]:
    num_classes = len(classes)
    per_class_f1 = {}
    f1_values = []
    for i in range(num_classes):
        tp = int(confusion[i, i])
        fp = int(confusion[:, i].sum() - tp)
        fn = int(confusion[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class_f1[classes[i]] = round(f1, 4)
        f1_values.append(f1)
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class_f1": per_class_f1,
        "confusion_matrix": confusion.tolist(),
    }


def save_checkpoint(path: Path, model, args, class_to_idx: dict[str, int], best_f1: float, epoch: int) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": args.backbone,
            "classes": list(class_to_idx.keys()),
            "class_to_idx": class_to_idx,
            "img_size": args.img_size,
            "normalize": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
            "best_val_macro_f1": best_f1,
            "epoch": epoch,
        },
        path,
    )


def resolve_device(choice: str) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    sys.exit(main())
