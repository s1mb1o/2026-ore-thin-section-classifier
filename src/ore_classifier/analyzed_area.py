from __future__ import annotations

import cv2
import numpy as np


def build_analyzed_mask(
    rgb: np.ndarray,
    *,
    min_value: int = 18,
    blue_bias_min: int = 45,
    blue_saturation_min: int = 80,
) -> np.ndarray:
    """Return pixels that should count in image-fraction denominators.

    The official photos may include black borders and blue markup strokes. Those
    pixels should not dilute talc/sulfide fractions used by ore-class rules.
    """

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected RGB image with shape HxWx3, got {rgb.shape}")
    rgb_u8 = rgb.astype(np.uint8, copy=False)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    value = hsv[..., 2]
    saturation = hsv[..., 1]
    mask = (value >= int(min_value)) & ~blue_annotation_like(
        rgb_u8,
        saturation=saturation,
        blue_bias_min=blue_bias_min,
        saturation_min=blue_saturation_min,
    )
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, ellipse_kernel(1)).astype(np.uint8)


def blue_annotation_like(
    rgb: np.ndarray,
    *,
    saturation: np.ndarray | None = None,
    blue_bias_min: int = 45,
    saturation_min: int = 80,
) -> np.ndarray:
    rgb_u8 = rgb.astype(np.uint8, copy=False)
    if saturation is None:
        saturation = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)[..., 1]
    rgb_i16 = rgb_u8.astype(np.int16)
    blue_bias = rgb_i16[..., 2] - np.maximum(rgb_i16[..., 0], rgb_i16[..., 1])
    return (blue_bias > int(blue_bias_min)) & (saturation > int(saturation_min))


def ellipse_kernel(radius: int) -> np.ndarray:
    size = max(1, int(radius) * 2 + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
