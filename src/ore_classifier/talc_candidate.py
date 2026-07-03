from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TalcCandidateConfig:
    """Conservative optical talc candidate heuristic.

    This is a candidate mask, not expert talc ground truth. It exists so the
    end-to-end ore pipeline can exercise the talc-fraction rule before a neural
    talc model or accepted expert masks are available.
    """

    min_area_px: int = 320
    morphology_open_radius: int = 2
    morphology_close_radius: int = 5
    analyzed_min_value: int = 18
    hue_min: int = 35
    hue_max: int = 95
    saturation_min: int = 12
    saturation_max: int = 145
    value_min: int = 55
    value_max: int = 238
    green_bias_min: int = 10
    blue_bias_max: int = 35


def estimate_talc_candidate_mask(
    rgb: np.ndarray,
    sulfide_mask: np.ndarray | None = None,
    config: TalcCandidateConfig | None = None,
) -> np.ndarray:
    """Estimate a conservative talc candidate mask from RGB optical microscopy."""

    cfg = config or TalcCandidateConfig()
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected RGB image with shape HxWx3, got {rgb.shape}")
    rgb_u8 = rgb.astype(np.uint8, copy=False)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    rgb_i16 = rgb_u8.astype(np.int16)
    green_bias = rgb_i16[..., 1] - np.maximum(rgb_i16[..., 0], rgb_i16[..., 2])
    blue_bias = rgb_i16[..., 2] - np.maximum(rgb_i16[..., 0], rgb_i16[..., 1])
    analyzed = _build_analyzed_mask(rgb_u8, cfg)
    sulfide = np.zeros(hue.shape, dtype=bool) if sulfide_mask is None else sulfide_mask.astype(bool)

    green_gray = (
        (hue >= cfg.hue_min)
        & (hue <= cfg.hue_max)
        & (saturation >= cfg.saturation_min)
        & (saturation <= cfg.saturation_max)
        & (value >= cfg.value_min)
        & (value <= cfg.value_max)
        & (green_bias >= cfg.green_bias_min)
        & (blue_bias < cfg.blue_bias_max)
    )
    mask = (
        green_gray
        & analyzed.astype(bool)
        & ~sulfide
        & ~_blue_annotation_like(rgb_u8, saturation)
    ).astype(np.uint8)

    if cfg.morphology_open_radius > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse_kernel(cfg.morphology_open_radius))
    if cfg.morphology_close_radius > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse_kernel(cfg.morphology_close_radius))
    mask = _remove_small_components(mask, cfg.min_area_px)
    return (mask > 0).astype(np.uint8) * 255


def save_talc_candidate_outputs(
    *,
    out_dir: Path,
    rgb: np.ndarray,
    talc_mask: np.ndarray,
    sulfide_mask: np.ndarray | None = None,
    config: TalcCandidateConfig | None = None,
    preview_max_side: int = 1800,
) -> dict[str, str]:
    cfg = config or TalcCandidateConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / "talc_candidate_mask.png"
    overlay_path = out_dir / "talc_candidate_overlay_preview.jpg"
    summary_path = out_dir / "talc_candidate_summary.json"

    Image.fromarray((talc_mask > 0).astype(np.uint8) * 255, mode="L").save(mask_path)
    overlay = talc_candidate_overlay(rgb=rgb, talc_mask=talc_mask, sulfide_mask=sulfide_mask, max_side=preview_max_side)
    Image.fromarray(overlay, mode="RGB").save(overlay_path, quality=92, optimize=True)

    image_area = int(talc_mask.size)
    talc_area = int((talc_mask > 0).sum())
    summary: dict[str, Any] = {
        "schema_version": "talc-candidate-v0.1",
        "source": "automatic_color_heuristic",
        "note": "Automatic talc candidate, not expert geological ground truth.",
        "width": int(talc_mask.shape[1]),
        "height": int(talc_mask.shape[0]),
        "image_area_px": image_area,
        "talc_candidate_area_px": talc_area,
        "talc_candidate_fraction": talc_area / max(image_area, 1),
        "config": asdict(cfg),
        "paths": {
            "talc_candidate_mask": str(mask_path),
            "talc_candidate_overlay_preview": str(overlay_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "talc_candidate_mask": str(mask_path),
        "talc_candidate_overlay_preview": str(overlay_path),
        "talc_candidate_summary": str(summary_path),
    }


def talc_candidate_overlay(
    *,
    rgb: np.ndarray,
    talc_mask: np.ndarray,
    sulfide_mask: np.ndarray | None = None,
    max_side: int = 1800,
) -> np.ndarray:
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    talc_img = Image.fromarray((talc_mask > 0).astype(np.uint8) * 255, mode="L")
    sulfide_img = None if sulfide_mask is None else Image.fromarray((sulfide_mask > 0).astype(np.uint8) * 255, mode="L")
    if max_side and max(image.size) > max_side:
        scale = max_side / float(max(image.size))
        size = (max(1, int(image.size[0] * scale)), max(1, int(image.size[1] * scale)))
        image = image.resize(size, Image.Resampling.BILINEAR)
        talc_img = talc_img.resize(size, Image.Resampling.NEAREST)
        if sulfide_img is not None:
            sulfide_img = sulfide_img.resize(size, Image.Resampling.NEAREST)

    base = np.asarray(image).astype(np.float32)
    talc = np.asarray(talc_img) > 0
    color = np.zeros_like(base)
    color[talc] = (45, 130, 255)
    alpha = (talc.astype(np.float32) * 0.55)[..., None]
    if sulfide_img is not None:
        sulfide = (np.asarray(sulfide_img) > 0) & ~talc
        color[sulfide] = (255, 216, 0)
        alpha[sulfide] = 0.22
    return np.clip(base * (1.0 - alpha) + color * alpha, 0, 255).astype(np.uint8)


def _build_analyzed_mask(rgb: np.ndarray, config: TalcCandidateConfig) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    value = hsv[..., 2]
    saturation = hsv[..., 1]
    not_black = value >= config.analyzed_min_value
    not_blue_annotation = ~_blue_annotation_like(rgb, saturation)
    mask = np.logical_and(not_black, not_blue_annotation).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse_kernel(1)).astype(np.uint8)


def _blue_annotation_like(rgb: np.ndarray, saturation: np.ndarray) -> np.ndarray:
    rgb_i16 = rgb.astype(np.int16)
    blue_bias = rgb_i16[..., 2] - np.maximum(rgb_i16[..., 0], rgb_i16[..., 1])
    return (blue_bias > 45) & (saturation > 80)


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(num_labels, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels].astype(np.uint8)


def _ellipse_kernel(radius: int) -> np.ndarray:
    size = max(1, radius * 2 + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
