from __future__ import annotations

from pathlib import Path
import re
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
    elif model_name in {"segformer_b0", "segformer_b1", "segformer_b2"}:
        model = create_segformer_for_eval(model_name)
    elif model_name == "mask2former":
        model = create_mask2former_for_eval(train_args)
    else:
        raise ValueError(f"Unsupported checkpoint model={model_name!r}")
    state_dict = checkpoint["model"]
    compatibility_note = "exact"
    try:
        compatibility_note = _load_state_dict_strictly(model, state_dict)
    except RuntimeError as exc:
        raise RuntimeError(_checkpoint_load_error(path, checkpoint, exc)) from exc
    model.to(device)
    return model, {
        "model": model_name,
        "epoch": checkpoint.get("epoch"),
        "best_iou_sulfide": checkpoint.get("best_iou_sulfide"),
        "state_dict_compatibility": compatibility_note,
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
    elif model_name == "segformer_b2":
        config = SegformerConfig(
            num_labels=2,
            depths=[3, 4, 6, 3],
            hidden_sizes=[64, 128, 320, 512],
            decoder_hidden_size=768,
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


def create_mask2former_for_eval(train_args: dict[str, Any] | None = None):
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

    train_args = train_args or {}
    pretrained = train_args.get("pretrained_model")
    allow_random = bool(train_args.get("allow_random_init", False))
    if pretrained in {"", "none", "random"}:
        pretrained = None
        allow_random = True
    if pretrained is None and not allow_random:
        pretrained = "facebook/mask2former-swin-tiny-ade-semantic"

    if pretrained is None:
        config = mask2former_config()
    else:
        try:
            config = Mask2FormerConfig.from_pretrained(pretrained)
        except Exception:
            config = mask2former_config()
        config.num_labels = 2
        config.id2label = {0: "not_sulfide", 1: "sulfide"}
        config.label2id = {"not_sulfide": 0, "sulfide": 1}
        config.ignore_value = 255
    return Mask2FormerForUniversalSegmentation(config)


def mask2former_config():
    from transformers import Mask2FormerConfig

    return Mask2FormerConfig(
        num_labels=2,
        id2label={0: "not_sulfide", 1: "sulfide"},
        label2id={"not_sulfide": 0, "sulfide": 1},
        ignore_value=255,
    )


def _load_state_dict_strictly(model: torch.nn.Module, state_dict: dict[str, Any]) -> str:
    try:
        model.load_state_dict(state_dict)
        return "exact"
    except RuntimeError as original_exc:
        remapped = remap_segformer_state_dict_for_model(state_dict, model.state_dict())
        if remapped is state_dict:
            raise original_exc
        try:
            model.load_state_dict(remapped)
        except RuntimeError:
            raise original_exc
        return "segformer_transformers_namespace_remap"


def remap_segformer_state_dict_for_model(
    checkpoint_state: dict[str, Any], model_state: dict[str, Any]
) -> dict[str, Any]:
    """Remap between Transformers SegFormer module namespaces when shapes match."""
    checkpoint_keys = set(checkpoint_state)
    model_keys = set(model_state)
    if checkpoint_keys == model_keys:
        return checkpoint_state

    if any(key.startswith("segformer.stages.") for key in checkpoint_keys) and any(
        key.startswith("segformer.encoder.") for key in model_keys
    ):
        remapped = {_segformer_stages_to_encoder_key(key): value for key, value in checkpoint_state.items()}
        if _state_dict_keys_and_shapes_match(remapped, model_state):
            return remapped

    if any(key.startswith("segformer.encoder.") for key in checkpoint_keys) and any(
        key.startswith("segformer.stages.") for key in model_keys
    ):
        remapped = {_segformer_encoder_to_stages_key(key): value for key, value in checkpoint_state.items()}
        if _state_dict_keys_and_shapes_match(remapped, model_state):
            return remapped

    return checkpoint_state


def _state_dict_keys_and_shapes_match(candidate: dict[str, Any], model_state: dict[str, Any]) -> bool:
    if set(candidate) != set(model_state):
        return False
    for key, value in candidate.items():
        candidate_shape = getattr(value, "shape", None)
        model_shape = getattr(model_state[key], "shape", None)
        if candidate_shape != model_shape:
            return False
    return True


def _segformer_stages_to_encoder_key(key: str) -> str:
    key = re.sub(
        r"^segformer\.stages\.(\d+)\.patch_embeddings\.",
        r"segformer.encoder.patch_embeddings.\1.",
        key,
    )
    key = re.sub(
        r"^segformer\.stages\.(\d+)\.blocks\.(\d+)\.",
        r"segformer.encoder.block.\1.\2.",
        key,
    )
    key = re.sub(
        r"^segformer\.stages\.(\d+)\.layer_norm\.",
        r"segformer.encoder.layer_norm.\1.",
        key,
    )
    key = re.sub(
        r"^decode_head\.linear_projections\.(\d+)\.",
        r"decode_head.linear_c.\1.",
        key,
    )
    replacements = (
        ("layernorm_before.", "layer_norm_1."),
        ("layernorm_after.", "layer_norm_2."),
        ("attention.q_proj.", "attention.self.query."),
        ("attention.k_proj.", "attention.self.key."),
        ("attention.v_proj.", "attention.self.value."),
        ("attention.o_proj.", "attention.output.dense."),
        ("attention.sequence_reduction.sequence_reduction.", "attention.self.sr."),
        ("attention.sequence_reduction.layer_norm.", "attention.self.layer_norm."),
        ("mlp.fc1.", "mlp.dense1."),
        ("mlp.fc2.", "mlp.dense2."),
    )
    for old, new in replacements:
        key = key.replace(old, new)
    return key


def _segformer_encoder_to_stages_key(key: str) -> str:
    key = re.sub(
        r"^segformer\.encoder\.patch_embeddings\.(\d+)\.",
        r"segformer.stages.\1.patch_embeddings.",
        key,
    )
    key = re.sub(
        r"^segformer\.encoder\.block\.(\d+)\.(\d+)\.",
        r"segformer.stages.\1.blocks.\2.",
        key,
    )
    key = re.sub(
        r"^segformer\.encoder\.layer_norm\.(\d+)\.",
        r"segformer.stages.\1.layer_norm.",
        key,
    )
    key = re.sub(
        r"^decode_head\.linear_c\.(\d+)\.",
        r"decode_head.linear_projections.\1.",
        key,
    )
    replacements = (
        ("layer_norm_1.", "layernorm_before."),
        ("layer_norm_2.", "layernorm_after."),
        ("attention.self.query.", "attention.q_proj."),
        ("attention.self.key.", "attention.k_proj."),
        ("attention.self.value.", "attention.v_proj."),
        ("attention.output.dense.", "attention.o_proj."),
        ("attention.self.sr.", "attention.sequence_reduction.sequence_reduction."),
        ("attention.self.layer_norm.", "attention.sequence_reduction.layer_norm."),
        ("mlp.dense1.", "mlp.fc1."),
        ("mlp.dense2.", "mlp.fc2."),
    )
    for old, new in replacements:
        key = key.replace(old, new)
    return key


def forward_logits(model, images: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    outputs = model(images)
    if hasattr(outputs, "class_queries_logits") and hasattr(outputs, "masks_queries_logits"):
        return mask2former_semantic_logits(outputs, target_hw)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    if logits.shape[-2:] != target_hw:
        logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
    return logits


def mask2former_semantic_logits(outputs: Any, target_hw: tuple[int, int]) -> torch.Tensor:
    class_logits = outputs.class_queries_logits
    mask_logits = outputs.masks_queries_logits
    if mask_logits.shape[-2:] != target_hw:
        mask_logits = F.interpolate(mask_logits, size=target_hw, mode="bilinear", align_corners=False)

    class_probs = class_logits.softmax(dim=-1)[..., :2]
    mask_probs = mask_logits.sigmoid()
    semantic_probs = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
    semantic_probs = semantic_probs / semantic_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return semantic_probs.clamp_min(1e-6).log()


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
