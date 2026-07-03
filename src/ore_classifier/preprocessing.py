"""Shared optical-microscopy preprocessing logic.

This module is the single source of truth for the deterministic preprocessing
transform used by the ore pipeline. It was extracted from
``apps/ore_pipeline_web.py`` so that both the browser UI and the offline
evaluation/robustness harness (``scripts/evaluate_official_pipeline.py``) apply
byte-identical preprocessing.

The pixel transform (:func:`apply_preprocessing`) is a verbatim move of the web
app implementation; :func:`normalize_preprocess_settings` reproduces the web
app's JSON-key normalization without the HTTP-specific error type so it can be
reused outside the web server. The web app keeps a thin wrapper that raises its
own ``ApiError`` for non-object payloads and delegates the rest here.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance

# Panorama scaling constants (shared with the web UI settings schema).
PANORAMA_SCALING_MODE_MAX_SIDE = "max_side"
PANORAMA_SCALING_MODE_SCALE_FACTOR = "scale_factor"
PANORAMA_SCALING_MODES = {PANORAMA_SCALING_MODE_MAX_SIDE, PANORAMA_SCALING_MODE_SCALE_FACTOR}
DEFAULT_PANORAMA_MAX_SIDE_PX = 1800
DEFAULT_PANORAMA_SCALE_FACTOR = 0.5
MIN_PANORAMA_MAX_SIDE_PX = 64
MAX_PANORAMA_MAX_SIDE_PX = 12000
MIN_PANORAMA_SCALE_FACTOR = 0.05
MAX_PANORAMA_SCALE_FACTOR = 1.0

# Canonical default preprocessing preset. Matches DEFAULT_APP_SETTINGS["preprocess"].
DEFAULT_PREPROCESS_SETTINGS: dict[str, Any] = {
    "preprocessing_enabled": True,
    "illumination_normalization": True,
    "denoise": True,
    "contrast_correction": True,
    "panorama_scaling": True,
    "panorama_scaling_mode": PANORAMA_SCALING_MODE_MAX_SIDE,
    "panorama_max_side_px": DEFAULT_PANORAMA_MAX_SIDE_PX,
    "panorama_scale_factor": DEFAULT_PANORAMA_SCALE_FACTOR,
}


def default_preprocess_settings() -> dict[str, Any]:
    return dict(DEFAULT_PREPROCESS_SETTINGS)


def _clamp_int(value: Any, fallback: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        parsed = int(fallback)
    return max(int(minimum), min(int(maximum), parsed))


def _clamp_float(value: Any, fallback: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(fallback)
    if not math.isfinite(parsed):
        parsed = float(fallback)
    return max(float(minimum), min(float(maximum), parsed))


def _scaling_mode(value: Any, fallback: str = PANORAMA_SCALING_MODE_MAX_SIDE) -> str:
    mode = str(value or fallback)
    return mode if mode in PANORAMA_SCALING_MODES else fallback


def _setting_bool(payload: dict[str, Any], key: str, fallback: bool, aliases: tuple[str, ...] = ()) -> bool:
    for candidate in (key, *aliases):
        if candidate in payload:
            return bool(payload[candidate])
    return bool(fallback)


def _setting_value(payload: dict[str, Any], key: str, fallback: Any, aliases: tuple[str, ...] = ()) -> Any:
    for candidate in (key, *aliases):
        if candidate in payload:
            return payload[candidate]
    return fallback


def normalize_preprocess_settings(payload: Any, base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize a preprocessing settings payload to a fully-populated preset.

    Accepts the same aliases as the web UI (e.g. ``enabled``, ``illumination``,
    ``noise_reduction``, ``contrast``, camelCase panorama keys). Raises
    ``ValueError`` when ``payload`` is neither ``None`` nor a mapping.
    """
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("preprocess settings must be an object")
    fallback = base if isinstance(base, dict) else DEFAULT_PREPROCESS_SETTINGS
    return {
        "preprocessing_enabled": _setting_bool(
            payload, "preprocessing_enabled", bool(fallback["preprocessing_enabled"]), ("enabled",)
        ),
        "illumination_normalization": _setting_bool(
            payload, "illumination_normalization", bool(fallback["illumination_normalization"]), ("illumination",)
        ),
        "denoise": _setting_bool(payload, "denoise", bool(fallback["denoise"]), ("noise_reduction",)),
        "contrast_correction": _setting_bool(
            payload, "contrast_correction", bool(fallback["contrast_correction"]), ("contrast",)
        ),
        "panorama_scaling": _setting_bool(
            payload, "panorama_scaling", bool(fallback["panorama_scaling"]), ("panoramaScaling",)
        ),
        "panorama_scaling_mode": _scaling_mode(
            _setting_value(
                payload,
                "panorama_scaling_mode",
                fallback.get("panorama_scaling_mode", PANORAMA_SCALING_MODE_MAX_SIDE),
                ("panoramaScalingMode",),
            ),
            PANORAMA_SCALING_MODE_MAX_SIDE,
        ),
        "panorama_max_side_px": _clamp_int(
            _setting_value(
                payload,
                "panorama_max_side_px",
                fallback.get("panorama_max_side_px", DEFAULT_PANORAMA_MAX_SIDE_PX),
                ("panoramaMaxSidePx", "panorama_max_side", "panoramaMaxSide"),
            ),
            DEFAULT_PANORAMA_MAX_SIDE_PX,
            MIN_PANORAMA_MAX_SIDE_PX,
            MAX_PANORAMA_MAX_SIDE_PX,
        ),
        "panorama_scale_factor": _clamp_float(
            _setting_value(
                payload,
                "panorama_scale_factor",
                fallback.get("panorama_scale_factor", DEFAULT_PANORAMA_SCALE_FACTOR),
                ("panoramaScaleFactor", "panorama_scaling_factor"),
            ),
            DEFAULT_PANORAMA_SCALE_FACTOR,
            MIN_PANORAMA_SCALE_FACTOR,
            MAX_PANORAMA_SCALE_FACTOR,
        ),
    }


def preprocessing_enabled(preset: dict[str, Any] | None) -> bool:
    if not isinstance(preset, dict):
        return False
    return bool(preset.get("preprocessing_enabled", preset.get("enabled", True)))


def apply_preprocessing(image: Image.Image, preset: dict[str, Any]) -> Image.Image:
    # Keep a single numpy RGB buffer across steps. The previous version round-tripped
    # PIL<->numpy once per enabled step, allocating several full-size copies of large
    # images (costly for panorama-scale inputs). Output is pixel-identical; only the
    # redundant intermediate PIL images and array copies are removed.
    arr = np.asarray(image.convert("RGB"))
    if preset.get("illumination_normalization"):
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        value = hsv[..., 2]
        sigma = max(9.0, min(value.shape) / 32.0)
        background = cv2.GaussianBlur(value, (0, 0), sigmaX=sigma)
        corrected = value.astype(np.float32) - background.astype(np.float32) + float(np.median(background))
        hsv[..., 2] = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        arr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if preset.get("denoise"):
        arr = cv2.fastNlMeansDenoisingColored(arr, None, 4, 4, 7, 21)
    if preset.get("contrast_correction"):
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        corrected_l = clahe.apply(l_channel)
        corrected = cv2.merge((corrected_l, a_channel, b_channel))
        arr = cv2.cvtColor(corrected, cv2.COLOR_LAB2RGB)
    result = Image.fromarray(arr, mode="RGB")
    if preset.get("contrast_correction"):
        result = ImageEnhance.Contrast(result).enhance(1.05)
    return result


def preprocess_image(image: Image.Image, preset: dict[str, Any]) -> Image.Image:
    """Convenience wrapper: apply preprocessing only when the preset enables it.

    Panorama scaling is intentionally not applied here. The offline batch path
    (``scripts/run_ore_pipeline.py``) tiles images at native resolution and does
    not resize inputs, so preprocessing for the harness is the pixel transform
    only, matching what reaches the model per tile in the UI.
    """
    if not preprocessing_enabled(preset):
        return image
    return apply_preprocessing(image, preset)
