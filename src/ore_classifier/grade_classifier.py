"""Inference for the grade-classifier CNN branch (efficientnet_b3, ordinary↔fine).

Reusable load + predict shared by the offline pipeline (`scripts/run_ore_pipeline.py`)
and the browser app (`apps/ore_pipeline_web.py`). This is a PARALLEL grade opinion
(image-level, learned) alongside the deterministic segmentation-rule ore class; it
covers the two data-rich grades (ordinary vs fine intergrowth). Talcose is not
predicted here — it is handled by the segmentation/rule + (future) talc branch.

Deps: torch + torchvision + PIL only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3

# Map the classifier's folder-label classes to the pipeline's ore-class vocabulary.
LABEL_TO_ORE_CLASS = {
    "ordinary_intergrowth": "row_ore",
    "fine_intergrowth": "hard_to_process_ore",
    "talcose": "talcose_ore",
}
ORE_CLASS_RU = {
    "row_ore": "рядовая руда",
    "hard_to_process_ore": "труднообогатимая руда",
    "talcose_ore": "оталькованная руда",
}


@dataclass
class GradeModel:
    model: nn.Module
    classes: list[str]
    img_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    device: torch.device
    checkpoint: str

    @property
    def transform(self):
        return transforms.Compose([
            transforms.Resize(int(self.img_size * 1.15)),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])


def resolve_device(choice: str = "auto") -> torch.device:
    if choice and choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_grade_model(checkpoint: str | Path, device: str | torch.device = "auto") -> GradeModel:
    dev = device if isinstance(device, torch.device) else resolve_device(device)
    ckpt = torch.load(Path(checkpoint), map_location=dev, weights_only=False)
    classes = list(ckpt["classes"])
    model = efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()
    norm = ckpt.get("normalize", {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)})
    return GradeModel(
        model=model,
        classes=classes,
        img_size=int(ckpt.get("img_size", 384)),
        mean=tuple(norm["mean"]),
        std=tuple(norm["std"]),
        device=dev,
        checkpoint=str(checkpoint),
    )


@torch.no_grad()
def predict_grade(grade: GradeModel, image: Image.Image) -> dict[str, Any]:
    tensor = grade.transform(image.convert("RGB")).unsqueeze(0).to(grade.device)
    probs = torch.softmax(grade.model(tensor), dim=1)[0].cpu().tolist()
    ranked = sorted(zip(grade.classes, probs, strict=True), key=lambda kv: kv[1], reverse=True)
    top_label, top_prob = ranked[0]
    ore_class = LABEL_TO_ORE_CLASS.get(top_label, top_label)
    return {
        "schema_version": "grade-branch-prediction-v0.1",
        "checkpoint": grade.checkpoint,
        "classes": grade.classes,
        "scope": "ordinary_vs_fine",
        "predicted_label": top_label,
        "predicted_ore_class": ore_class,
        "predicted_ore_class_ru": ORE_CLASS_RU.get(ore_class, ore_class),
        "confidence": float(top_prob),
        "probabilities": {label: float(p) for label, p in zip(grade.classes, probs, strict=True)},
    }
