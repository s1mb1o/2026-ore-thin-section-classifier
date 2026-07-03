from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

AUGMENTATION_SCHEMA_VERSION = "ore-pipeline-augmentation-v0.1"

DEFAULT_AUGMENTATION_SETTINGS: dict[str, Any] = {
    "schema_version": AUGMENTATION_SCHEMA_VERSION,
    "enabled": False,
    "color": {
        "brightness_pct": 4.0,
        "contrast_pct": 6.0,
        "saturation_pct": 3.0,
        "hue_degrees": 0.0,
        "gamma": 1.0,
    },
    "acquisition": {
        "blur_radius": 0.0,
        "gaussian_noise_std": 0.0,
    },
    "surface_artifacts": {
        "scratch_count": 6,
        "scratch_intensity_pct": 14.0,
        "polishing_haze_pct": 7.0,
        "pit_count": 18,
        "pit_intensity_pct": 12.0,
    },
    "runtime": {
        "geometry_preserving": True,
        "coordinate_mode": "original",
        "random_seed": 0,
    },
}


def default_augmentation_settings() -> dict[str, Any]:
    return deepcopy(DEFAULT_AUGMENTATION_SETTINGS)


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float_value(payload: dict[str, Any], key: str, fallback: float, *, min_value: float, max_value: float) -> float:
    try:
        value = float(payload.get(key, fallback))
    except (TypeError, ValueError):
        value = float(fallback)
    return max(min_value, min(max_value, value))


def _int_value(payload: dict[str, Any], key: str, fallback: int, *, min_value: int, max_value: int) -> int:
    try:
        value = int(payload.get(key, fallback))
    except (TypeError, ValueError):
        value = int(fallback)
    return max(min_value, min(max_value, value))


def normalize_augmentation_settings(payload: Any, base: dict[str, Any] | None = None) -> dict[str, Any]:
    values = _as_mapping(payload)
    fallback = default_augmentation_settings()
    if isinstance(base, dict):
        fallback = normalize_augmentation_settings(base)

    color_values = _as_mapping(values.get("color"))
    acquisition_values = _as_mapping(values.get("acquisition"))
    artifact_values = _as_mapping(values.get("surface_artifacts"))
    runtime_values = _as_mapping(values.get("runtime"))
    fallback_color = fallback["color"]
    fallback_acquisition = fallback["acquisition"]
    fallback_artifacts = fallback["surface_artifacts"]
    fallback_runtime = fallback["runtime"]

    return {
        "schema_version": AUGMENTATION_SCHEMA_VERSION,
        "enabled": bool(values.get("enabled", fallback["enabled"])),
        "color": {
            "brightness_pct": _float_value(color_values, "brightness_pct", fallback_color["brightness_pct"], min_value=-50.0, max_value=50.0),
            "contrast_pct": _float_value(color_values, "contrast_pct", fallback_color["contrast_pct"], min_value=-50.0, max_value=80.0),
            "saturation_pct": _float_value(color_values, "saturation_pct", fallback_color["saturation_pct"], min_value=-60.0, max_value=80.0),
            "hue_degrees": _float_value(color_values, "hue_degrees", fallback_color["hue_degrees"], min_value=-30.0, max_value=30.0),
            "gamma": _float_value(color_values, "gamma", fallback_color["gamma"], min_value=0.5, max_value=2.0),
        },
        "acquisition": {
            "blur_radius": _float_value(acquisition_values, "blur_radius", fallback_acquisition["blur_radius"], min_value=0.0, max_value=3.0),
            "gaussian_noise_std": _float_value(acquisition_values, "gaussian_noise_std", fallback_acquisition["gaussian_noise_std"], min_value=0.0, max_value=20.0),
        },
        "surface_artifacts": {
            "scratch_count": _int_value(artifact_values, "scratch_count", fallback_artifacts["scratch_count"], min_value=0, max_value=120),
            "scratch_intensity_pct": _float_value(artifact_values, "scratch_intensity_pct", fallback_artifacts["scratch_intensity_pct"], min_value=0.0, max_value=70.0),
            "polishing_haze_pct": _float_value(artifact_values, "polishing_haze_pct", fallback_artifacts["polishing_haze_pct"], min_value=0.0, max_value=60.0),
            "pit_count": _int_value(artifact_values, "pit_count", fallback_artifacts["pit_count"], min_value=0, max_value=600),
            "pit_intensity_pct": _float_value(artifact_values, "pit_intensity_pct", fallback_artifacts["pit_intensity_pct"], min_value=0.0, max_value=70.0),
        },
        "runtime": {
            "geometry_preserving": True,
            "coordinate_mode": "original",
            "random_seed": _int_value(runtime_values, "random_seed", fallback_runtime["random_seed"], min_value=0, max_value=2**31 - 1),
        },
    }


def augmentation_enabled(settings: dict[str, Any] | None) -> bool:
    return bool((settings or {}).get("enabled"))


def _apply_polishing_haze(image: Image.Image, haze_pct: float, rng: np.random.Generator) -> Image.Image:
    if haze_pct <= 1e-6:
        return image
    width, height = image.size
    low_width = max(8, min(96, width // 16 or 8))
    low_height = max(8, min(96, height // 16 or 8))
    haze = rng.normal(0.5, 0.22, size=(low_height, low_width)).astype(np.float32)
    haze = np.clip(haze, 0.0, 1.0)
    haze_image = Image.fromarray((haze * 255).astype(np.uint8), mode="L").resize((width, height), Image.Resampling.BILINEAR)
    haze_arr = np.asarray(haze_image, dtype=np.float32) / 255.0
    alpha = (haze_pct / 100.0) * (0.35 + 0.65 * haze_arr)
    arr = np.asarray(image, dtype=np.float32)
    arr = arr * (1.0 - alpha[..., None]) + 245.0 * alpha[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def _apply_scratches(image: Image.Image, count: int, intensity_pct: float, rng: np.random.Generator) -> Image.Image:
    if count <= 0 or intensity_pct <= 1e-6:
        return image
    width, height = image.size
    diagonal = float(np.hypot(width, height))
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    base_alpha = int(255 * intensity_pct / 100.0)
    max_line_width = max(1, min(5, int(round(min(width, height) / 220.0))))
    for _ in range(count):
        angle = float(rng.uniform(0, np.pi))
        length = float(rng.uniform(0.08, 0.45) * diagonal)
        cx = float(rng.uniform(0, width))
        cy = float(rng.uniform(0, height))
        dx = np.cos(angle) * length / 2.0
        dy = np.sin(angle) * length / 2.0
        alpha = max(0, min(255, int(base_alpha * rng.uniform(0.35, 1.0))))
        if rng.random() < 0.7:
            color = (255, 255, 255, alpha)
        else:
            dark_alpha = max(0, min(255, int(alpha * 0.65)))
            color = (20, 20, 20, dark_alpha)
        line_width = int(rng.integers(1, max_line_width + 1))
        draw.line((cx - dx, cy - dy, cx + dx, cy + dy), fill=color, width=line_width)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=0.35))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _apply_pits_and_dust(image: Image.Image, count: int, intensity_pct: float, rng: np.random.Generator) -> Image.Image:
    if count <= 0 or intensity_pct <= 1e-6:
        return image
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    base_alpha = int(255 * intensity_pct / 100.0)
    max_radius = max(1.2, min(5.0, min(width, height) / 120.0))
    for _ in range(count):
        radius = float(rng.uniform(0.7, max_radius))
        x = float(rng.uniform(radius, max(radius, width - radius)))
        y = float(rng.uniform(radius, max(radius, height - radius)))
        alpha = max(0, min(255, int(base_alpha * rng.uniform(0.45, 1.0))))
        if rng.random() < 0.62:
            color = (10, 10, 10, alpha)
        else:
            color = (255, 255, 255, int(alpha * 0.75))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=0.2))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def apply_augmentation(image: Image.Image, settings: dict[str, Any]) -> Image.Image:
    normalized = normalize_augmentation_settings(settings)
    result = image.convert("RGB")
    color = normalized["color"]
    acquisition = normalized["acquisition"]
    artifacts = normalized["surface_artifacts"]
    rng = np.random.default_rng(int(normalized["runtime"]["random_seed"]))

    brightness_factor = 1.0 + float(color["brightness_pct"]) / 100.0
    contrast_factor = 1.0 + float(color["contrast_pct"]) / 100.0
    saturation_factor = 1.0 + float(color["saturation_pct"]) / 100.0
    if abs(brightness_factor - 1.0) > 1e-6:
        result = ImageEnhance.Brightness(result).enhance(brightness_factor)
    if abs(contrast_factor - 1.0) > 1e-6:
        result = ImageEnhance.Contrast(result).enhance(contrast_factor)
    if abs(saturation_factor - 1.0) > 1e-6:
        result = ImageEnhance.Color(result).enhance(saturation_factor)

    hue_degrees = float(color["hue_degrees"])
    if abs(hue_degrees) > 1e-6:
        hsv = np.asarray(result.convert("HSV"), dtype=np.uint8).copy()
        shift = int(round(hue_degrees / 360.0 * 256.0))
        hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + shift) % 256).astype(np.uint8)
        result = Image.fromarray(hsv, mode="HSV").convert("RGB")

    gamma = float(color["gamma"])
    if abs(gamma - 1.0) > 1e-6:
        lut = np.clip(((np.arange(256, dtype=np.float32) / 255.0) ** (1.0 / gamma)) * 255.0, 0, 255).astype(np.uint8)
        result = Image.fromarray(lut[np.asarray(result, dtype=np.uint8)], mode="RGB")

    blur_radius = float(acquisition["blur_radius"])
    if blur_radius > 1e-6:
        result = result.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    noise_std = float(acquisition["gaussian_noise_std"])
    if noise_std > 1e-6:
        arr = np.asarray(result, dtype=np.float32)
        arr = np.clip(arr + rng.normal(0.0, noise_std, size=arr.shape), 0, 255).astype(np.uint8)
        result = Image.fromarray(arr, mode="RGB")

    result = _apply_polishing_haze(result, float(artifacts["polishing_haze_pct"]), rng)
    result = _apply_scratches(result, int(artifacts["scratch_count"]), float(artifacts["scratch_intensity_pct"]), rng)
    result = _apply_pits_and_dust(result, int(artifacts["pit_count"]), float(artifacts["pit_intensity_pct"]), rng)

    return result.convert("RGB")
