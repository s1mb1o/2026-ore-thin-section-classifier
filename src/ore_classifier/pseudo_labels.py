from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


# LumenStone / Petroscope class ids for sulfide ore minerals.
# Excluded by default: 0 background, 3 magnetite, 10 hematite, 14 native gold.
DEFAULT_SULFIDE_CLASS_IDS: tuple[int, ...] = (
    1,  # chalcopyrite
    2,  # galena
    4,  # bornite
    5,  # pyrrhotite
    6,  # pyrite
    7,  # pentlandite
    8,  # sphalerite
    9,  # arsenopyrite
    11,  # tenantite
    12,  # covellite
    13,  # marcasite
)


@dataclass(frozen=True)
class PseudoMask:
    mask: np.ndarray
    ignore: np.ndarray
    confidence: np.ndarray
    threshold: float | None = None


def parse_class_ids(raw: str | Iterable[int] | None) -> tuple[int, ...]:
    if raw is None:
        return DEFAULT_SULFIDE_CLASS_IDS
    if isinstance(raw, str):
        if not raw.strip():
            return DEFAULT_SULFIDE_CLASS_IDS
        return tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    return tuple(int(v) for v in raw)


def lumenstone_binary_mask(
    mask_array: np.ndarray,
    sulfide_class_ids: Iterable[int] = DEFAULT_SULFIDE_CLASS_IDS,
) -> PseudoMask:
    if mask_array.ndim == 3:
        class_ids = mask_array[..., 0]
    else:
        class_ids = mask_array
    sulfide_ids = np.asarray(tuple(sulfide_class_ids), dtype=class_ids.dtype)
    mask = np.isin(class_ids, sulfide_ids).astype(np.uint8)
    ignore = np.zeros_like(mask, dtype=np.uint8)
    confidence = np.where(mask > 0, 255, 230).astype(np.uint8)
    return PseudoMask(mask=mask, ignore=ignore, confidence=confidence)


def brightness_sulfide_pseudo_mask(
    rgb: np.ndarray,
    min_area: int = 48,
    uncertainty_margin: int = 18,
) -> PseudoMask:
    """Heuristic sulfide mask for reflected-light ore micrographs.

    This is intentionally conservative: bright metallic phases are positive,
    the matrix is negative, and near-threshold/boundary pixels are ignored.
    """

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected RGB image, got shape {rgb.shape}")

    rgb_u8 = rgb.astype(np.uint8, copy=False)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    value = hsv[..., 2]
    value_blur = cv2.GaussianBlur(value, (0, 0), sigmaX=1.2)

    threshold, _ = cv2.threshold(
        value_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = float(np.clip(threshold, 55, 235))

    raw_mask = (value_blur >= threshold).astype(np.uint8)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5)
    mask = _remove_small_components(mask, min_area=min_area)

    near_threshold = (
        np.abs(value_blur.astype(np.int16) - int(round(threshold)))
        <= uncertainty_margin
    )
    dilated = cv2.dilate(mask, kernel5, iterations=1)
    eroded = cv2.erode(mask, kernel5, iterations=1)
    boundary = dilated != eroded
    ignore = np.logical_or(near_threshold, boundary).astype(np.uint8)

    distance = np.abs(value_blur.astype(np.float32) - threshold)
    confidence = np.clip((distance / max(uncertainty_margin, 1)) * 255, 0, 255)
    confidence = confidence.astype(np.uint8)
    return PseudoMask(mask=mask.astype(np.uint8), ignore=ignore, confidence=confidence, threshold=threshold)


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros(num_labels, dtype=bool)
    keep[0] = False
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels].astype(np.uint8)
