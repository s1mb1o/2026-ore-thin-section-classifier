#!/usr/bin/env python3
"""Evaluate the trained grade-classifier CNN branch on the fixed eval split.

Runs the EfficientNet-B3 grade classifier over the images of the deconflicted
345 eval split whose labels are in the model's class set (ordinary/fine by
default — the 230 held-out ordinary/fine images that were excluded from
training), and reports binary macro-F1, per-class precision/recall/F1, and the
confusion matrix. Prints a comparison against our feature-CV. Talcose is NOT
scored here — it is deferred to the talc-segmentation branch (see docs/plans/37).

Deps: torch + torchvision only. Intended to run where the checkpoint + dataset
live (e.g. gx10 with the train-models venv).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_tiny, efficientnet_b3, resnet50


def build_model_by_arch(arch: str, num_classes: int) -> nn.Module:
    if arch == "efficientnet_b3":
        m = efficientnet_b3(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "convnext_tiny":
        m = convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif arch == "resnet50":
        m = resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"unknown arch: {arch}")
    return m

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
Image.MAX_IMAGE_PIXELS = None

from ore_classifier.augmentation import (  # noqa: E402
    apply_augmentation,
    augmentation_enabled,
    normalize_augmentation_settings,
)
from ore_classifier.preprocessing import (  # noqa: E402
    normalize_preprocess_settings,
    preprocess_image,
    preprocessing_enabled,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-split-json", type=Path, default=ROOT / "outputs/official_balanced_eval_split_deconflicted.json")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    # Robustness perturbation (same JSON schema/transforms as the UI + the seg harness).
    parser.add_argument("--augmentation-json", default=None, help="Path to JSON file or inline JSON of augmentation settings.")
    parser.add_argument("--preprocess-json", default=None, help="Path to JSON file or inline JSON of preprocessing settings.")
    args = parser.parse_args()

    augmentation = load_settings(args.augmentation_json, normalize_augmentation_settings)
    preprocess = load_settings(args.preprocess_json, normalize_preprocess_settings)
    aug_on = bool(augmentation and augmentation_enabled(augmentation))
    pre_on = bool(preprocess and preprocessing_enabled(preprocess))

    device = resolve_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    img_size = int(ckpt["img_size"])
    norm = ckpt["normalize"]
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_model_by_arch(ckpt.get("arch", "efficientnet_b3"), len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    tfm = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(tuple(norm["mean"]), tuple(norm["std"])),
    ])

    condition = "baseline"
    if aug_on or pre_on:
        parts = (["augmentation"] if aug_on else []) + (["preprocessing"] if pre_on else [])
        condition = "perturbed: " + " + ".join(parts)
    split = json.loads(args.eval_split_json.read_text(encoding="utf-8"))
    items = [it for it in split.get("items", []) if it["label"] in class_to_idx]
    print(f"eval images (classes {classes}): {len(items)} | {dict(Counter(i['label'] for i in items))} | condition: {condition}", flush=True)

    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    per_image: list[dict[str, Any]] = []
    with torch.no_grad():
        for it in items:
            image = Image.open(args.dataset_root / it["path"]).convert("RGB")
            if aug_on:
                image = apply_augmentation(image, augmentation)
            if pre_on:
                image = preprocess_image(image, preprocess)
            tensor = tfm(image).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(probs.argmax())
            true_idx = class_to_idx[it["label"]]
            confusion[true_idx, pred_idx] += 1
            per_image.append({
                "path": it["path"],
                "true": it["label"],
                "pred": idx_to_class[pred_idx],
                "probs": {classes[i]: float(probs[i]) for i in range(len(classes))},
            })

    metrics = macro_f1_from_confusion(confusion, classes)
    result = {
        "schema_version": "grade-branch-eval-v0.1",
        "checkpoint": str(args.checkpoint),
        "eval_split": str(args.eval_split_json),
        "classes": classes,
        "n_images": int(confusion.sum()),
        "condition": condition,
        "perturbation": {"augmentation_applied": aug_on, "preprocessing_applied": pre_on,
                          "augmentation": augmentation, "preprocess": preprocess},
        **metrics,
        "note": (
            "CNN grade branch scored on the held-out ordinary/fine images of the deconflicted "
            "345 split (excluded from training). Talcose deferred to the talc-segmentation branch."
        ),
        "per_image": per_image,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = render_md(result)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
    print("\n" + md)
    return 0


def macro_f1_from_confusion(confusion: np.ndarray, classes: list[str]) -> dict[str, Any]:
    per_class = {}
    f1_values = []
    for i in range(len(classes)):
        tp = int(confusion[i, i])
        fp = int(confusion[:, i].sum() - tp)
        fn = int(confusion[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[classes[i]] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "support": int(confusion[i, :].sum())}
        f1_values.append(f1)
    total = int(confusion.sum())
    return {
        "accuracy": round(int(np.trace(confusion)) / total, 4) if total else 0.0,
        "macro_f1": round(float(np.mean(f1_values)), 4) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def render_md(result: dict[str, Any]) -> str:
    lines = [
        "# Grade CNN branch evaluation (ordinary vs fine)",
        "",
        f"- Checkpoint: `{Path(result['checkpoint']).parent.name}/{Path(result['checkpoint']).name}`",
        f"- Condition: {result.get('condition', 'baseline')}",
        f"- Held-out images: {result['n_images']}",
        f"- Accuracy: {result['accuracy']:.4f}",
        f"- **Macro F1: {result['macro_f1']:.4f}**",
        "",
        "| Class | Support | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, pc in result["per_class"].items():
        lines.append(f"| {name} | {pc['support']} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} |")
    cm = result["confusion_matrix"]
    lines += [
        "",
        "Confusion (rows=true, cols=pred): " + "; ".join(
            f"{result['classes'][i]}=[{', '.join(str(x) for x in cm[i])}]" for i in range(len(cm))
        ),
        "",
        "Reference: our feature-CV ordinary 0.72 / fine 0.72.",
        f"Note: {result['note']}",
        "",
    ]
    return "\n".join(lines)


def load_settings(value, normalizer):
    if value is None:
        return None
    candidate = Path(value)
    if candidate.exists():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"could not read settings (not a file, not valid JSON): {value}") from exc
    return normalizer(payload)


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
