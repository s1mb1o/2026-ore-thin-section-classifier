from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from ore_classifier.models import create_resunet


def resolve_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_binary_segmentation_checkpoint(path: Path, device: torch.device):
    # Checkpoints are produced by this repository's training script.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})
    model_name = train_args.get("model", "resunet")
    if model_name == "resunet":
        model = create_resunet(base_channels=int(train_args.get("base_channels", 32)))
    elif model_name in {"segformer_b0", "segformer_b1"}:
        model = create_segformer_for_eval(model_name)
    else:
        raise ValueError(f"Unsupported checkpoint model={model_name!r}")
    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as exc:
        raise RuntimeError(_checkpoint_load_error(path, checkpoint, exc)) from exc
    model.to(device)
    return model, {
        "model": model_name,
        "epoch": checkpoint.get("epoch"),
        "best_iou_sulfide": checkpoint.get("best_iou_sulfide"),
        "train_args": train_args,
    }


def create_segformer_for_eval(model_name: str):
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    if model_name == "segformer_b1":
        config = SegformerConfig(
            num_labels=2,
            depths=[2, 2, 2, 2],
            hidden_sizes=[64, 128, 320, 512],
            decoder_hidden_size=256,
            id2label={0: "not_sulfide", 1: "sulfide"},
            label2id={"not_sulfide": 0, "sulfide": 1},
        )
    else:
        config = SegformerConfig(
            num_labels=2,
            id2label={0: "not_sulfide", 1: "sulfide"},
            label2id={"not_sulfide": 0, "sulfide": 1},
        )
    return SegformerForSemanticSegmentation(config)


def forward_logits(model, images: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    outputs = model(images)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    if logits.shape[-2:] != target_hw:
        logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
    return logits


def _checkpoint_load_error(path: Path, checkpoint: dict[str, Any], exc: RuntimeError) -> str:
    try:
        import transformers

        transformers_version = transformers.__version__
    except Exception:
        transformers_version = "unavailable"
    keys = checkpoint.get("model", {}).keys()
    first_key = next(iter(keys), "<empty>")
    return (
        f"Could not load checkpoint {path}. The checkpoint appears to use model key namespace "
        f"{first_key!r}, but installed transformers={transformers_version} created a different "
        "SegFormer module layout. Use the same environment that trained the checkpoint, or install "
        "the pinned requirements used by this project. Original load error follows:\n"
        f"{exc}"
    )
