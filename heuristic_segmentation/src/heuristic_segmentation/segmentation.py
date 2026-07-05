from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


CLASS_BACKGROUND = 0
CLASS_ORDINARY_INTERGROWTH = 1
CLASS_FINE_INTERGROWTH = 2
CLASS_TALC_CANDIDATE = 3

CLASS_COLORS = {
    CLASS_ORDINARY_INTERGROWTH: np.array([40, 190, 85], dtype=np.float32),
    CLASS_FINE_INTERGROWTH: np.array([230, 70, 65], dtype=np.float32),
    CLASS_TALC_CANDIDATE: np.array([65, 130, 245], dtype=np.float32),
}


@dataclass(frozen=True)
class HeuristicConfig:
    min_component_area: int = 64
    morphology_open_radius: int = 2
    morphology_close_radius: int = 4
    threshold_offset: float = 0.0
    fine_max_area_px: int = 450
    fine_min_replacement_ratio: float = 0.22
    fine_max_solidity: float = 0.78
    fine_max_compactness: float = 0.24
    footprint_close_radius: int = 9
    enable_talc_candidate: bool = True
    talc_min_area: int = 320
    talc_fraction_threshold: float = 0.10
    analyzed_min_value: int = 18


@dataclass(frozen=True)
class SegmentationResult:
    class_mask: np.ndarray
    sulfide_mask: np.ndarray
    talc_candidate_mask: np.ndarray
    analyzed_mask: np.ndarray
    components: list[dict[str, Any]]
    metrics: dict[str, Any]
    config: dict[str, Any]


def segment_image(rgb: np.ndarray, config: HeuristicConfig | None = None) -> SegmentationResult:
    if config is None:
        config = HeuristicConfig()
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected RGB image with shape HxWx3, got {rgb.shape}")

    rgb_u8 = rgb.astype(np.uint8, copy=False)
    analyzed_mask = _build_analyzed_mask(rgb_u8, config)
    sulfide_mask, sulfide_threshold = _segment_sulfides(rgb_u8, analyzed_mask, config)
    class_mask, components = _classify_sulfide_components(
        sulfide_mask=sulfide_mask,
        analyzed_mask=analyzed_mask,
        config=config,
    )
    talc_candidate_mask = (
        _segment_talc_candidates(rgb_u8, analyzed_mask, sulfide_mask, config)
        if config.enable_talc_candidate
        else np.zeros(sulfide_mask.shape, dtype=np.uint8)
    )
    class_mask[(talc_candidate_mask > 0) & (class_mask == CLASS_BACKGROUND)] = CLASS_TALC_CANDIDATE

    metrics = _build_metrics(
        class_mask=class_mask,
        sulfide_mask=sulfide_mask,
        talc_candidate_mask=talc_candidate_mask,
        analyzed_mask=analyzed_mask,
        components=components,
        sulfide_threshold=sulfide_threshold,
        talc_fraction_threshold=config.talc_fraction_threshold,
    )
    return SegmentationResult(
        class_mask=class_mask.astype(np.uint8),
        sulfide_mask=sulfide_mask.astype(np.uint8),
        talc_candidate_mask=talc_candidate_mask.astype(np.uint8),
        analyzed_mask=analyzed_mask.astype(np.uint8),
        components=components,
        metrics=metrics,
        config=asdict(config),
    )


def make_overlay(rgb: np.ndarray, class_mask: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    rgb_u8 = rgb.astype(np.uint8, copy=False)
    overlay = rgb_u8.astype(np.float32).copy()
    for class_id, color in CLASS_COLORS.items():
        pixels = class_mask == class_id
        if not np.any(pixels):
            continue
        overlay[pixels] = (1.0 - alpha) * overlay[pixels] + alpha * color
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _build_analyzed_mask(rgb: np.ndarray, config: HeuristicConfig) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    value = hsv[..., 2]
    saturation = hsv[..., 1]
    not_black = value >= config.analyzed_min_value
    not_blue_annotation = ~_blue_annotation_like(rgb, saturation)
    mask = np.logical_and(not_black, not_blue_annotation).astype(np.uint8)
    kernel = _ellipse_kernel(3)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel).astype(np.uint8)


def _segment_sulfides(
    rgb: np.ndarray,
    analyzed_mask: np.ndarray,
    config: HeuristicConfig,
) -> tuple[np.ndarray, float]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    value = hsv[..., 2]
    saturation = hsv[..., 1]
    value_blur = cv2.GaussianBlur(value, (0, 0), sigmaX=1.1)
    normalized = _normalize_illumination(value_blur)
    valid_values = normalized[analyzed_mask > 0]
    if valid_values.size == 0:
        return np.zeros(value.shape, dtype=np.uint8), 0.0

    threshold = _otsu_threshold(valid_values) + config.threshold_offset
    threshold = float(np.clip(threshold, 55, 235))
    rgb_i16 = rgb.astype(np.int16)
    green_bias = rgb_i16[..., 1] - np.maximum(rgb_i16[..., 0], rgb_i16[..., 2])

    bright = normalized >= threshold
    very_bright = value_blur >= max(210, threshold)
    non_green_matrix = green_bias < 18
    not_blue_annotation = ~_blue_annotation_like(rgb, saturation)
    raw = (
        analyzed_mask.astype(bool)
        & not_blue_annotation
        & (bright | very_bright)
        & (non_green_matrix | very_bright)
    )

    mask = raw.astype(np.uint8)
    if config.morphology_open_radius > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse_kernel(config.morphology_open_radius))
    if config.morphology_close_radius > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse_kernel(config.morphology_close_radius))
    mask = _remove_small_components(mask, config.min_component_area)
    return mask.astype(np.uint8), threshold


def _classify_sulfide_components(
    *,
    sulfide_mask: np.ndarray,
    analyzed_mask: np.ndarray,
    config: HeuristicConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    class_mask = np.zeros(sulfide_mask.shape, dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sulfide_mask, 8)
    components: list[dict[str, Any]] = []
    footprint_kernel = _ellipse_kernel(config.footprint_close_radius)
    analyzed_bool = analyzed_mask.astype(bool)
    height, width = sulfide_mask.shape
    roi_pad = max(1, 2 * int(config.footprint_close_radius) + 2)

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        x0 = max(0, x - roi_pad)
        y0 = max(0, y - roi_pad)
        x1 = min(width, x + w + roi_pad)
        y1 = min(height, y + h + roi_pad)

        labels_roi = labels[y0:y1, x0:x1]
        component = labels_roi == label_id
        contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
        hull_area = _convex_hull_area(contours)
        solidity = float(area / hull_area) if hull_area > 0 else 0.0
        compactness = float(4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0

        footprint = cv2.morphologyEx(component.astype(np.uint8), cv2.MORPH_CLOSE, footprint_kernel).astype(bool)
        analyzed_roi = analyzed_bool[y0:y1, x0:x1]
        footprint &= analyzed_roi
        footprint_area = int(footprint.sum())
        internal_dark_area = int(np.logical_and(footprint, ~component).sum())
        replacement_ratio = float(internal_dark_area / footprint_area) if footprint_area > 0 else 0.0

        is_fine = (
            area <= config.fine_max_area_px
            or replacement_ratio >= config.fine_min_replacement_ratio
            or solidity <= config.fine_max_solidity
            or compactness <= config.fine_max_compactness
        )
        class_id = CLASS_FINE_INTERGROWTH if is_fine else CLASS_ORDINARY_INTERGROWTH
        class_roi = class_mask[y0:y1, x0:x1]
        class_roi[component] = class_id
        components.append(
            {
                "component_id": label_id,
                "class_id": class_id,
                "class_label": "fine_intergrowth" if is_fine else "ordinary_intergrowth",
                "area_px": area,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "centroid_x": round(float(centroids[label_id][0]), 3),
                "centroid_y": round(float(centroids[label_id][1]), 3),
                "perimeter_px": round(perimeter, 3),
                "solidity": round(solidity, 6),
                "compactness": round(compactness, 6),
                "footprint_area_px": footprint_area,
                "internal_dark_area_px": internal_dark_area,
                "replacement_ratio": round(replacement_ratio, 6),
            }
        )
    return class_mask, components


def _segment_talc_candidates(
    rgb: np.ndarray,
    analyzed_mask: np.ndarray,
    sulfide_mask: np.ndarray,
    config: HeuristicConfig,
) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    rgb_i16 = rgb.astype(np.int16)
    green_bias = rgb_i16[..., 1] - np.maximum(rgb_i16[..., 0], rgb_i16[..., 2])
    blue_bias = rgb_i16[..., 2] - np.maximum(rgb_i16[..., 0], rgb_i16[..., 1])

    green_gray = (
        (hue >= 35)
        & (hue <= 95)
        & (saturation >= 12)
        & (saturation <= 145)
        & (value >= 55)
        & (value <= 238)
        & (green_bias >= 10)
        & (blue_bias < 35)
    )
    mask = (
        green_gray
        & analyzed_mask.astype(bool)
        & ~sulfide_mask.astype(bool)
        & ~_blue_annotation_like(rgb, saturation)
    ).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse_kernel(2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse_kernel(5))
    return _remove_small_components(mask, config.talc_min_area).astype(np.uint8)


def _build_metrics(
    *,
    class_mask: np.ndarray,
    sulfide_mask: np.ndarray,
    talc_candidate_mask: np.ndarray,
    analyzed_mask: np.ndarray,
    components: list[dict[str, Any]],
    sulfide_threshold: float,
    talc_fraction_threshold: float,
) -> dict[str, Any]:
    analyzed_area = int(analyzed_mask.sum())
    ordinary_area = int((class_mask == CLASS_ORDINARY_INTERGROWTH).sum())
    fine_area = int((class_mask == CLASS_FINE_INTERGROWTH).sum())
    talc_area = int(talc_candidate_mask.sum())
    sulfide_area = int(sulfide_mask.sum())
    denom = max(analyzed_area, 1)
    talc_fraction = talc_area / denom
    if talc_fraction > talc_fraction_threshold:
        ore_class = "talcose_candidate"
    elif fine_area > ordinary_area:
        ore_class = "fine_intergrowth_candidate"
    else:
        ore_class = "ordinary_intergrowth_candidate"
    return {
        "image_height": int(class_mask.shape[0]),
        "image_width": int(class_mask.shape[1]),
        "analyzed_area_px": analyzed_area,
        "sulfide_area_px": sulfide_area,
        "ordinary_area_px": ordinary_area,
        "fine_area_px": fine_area,
        "talc_candidate_area_px": talc_area,
        "sulfide_fraction": round(sulfide_area / denom, 6),
        "ordinary_fraction": round(ordinary_area / denom, 6),
        "fine_fraction": round(fine_area / denom, 6),
        "talc_candidate_fraction": round(talc_fraction, 6),
        "component_count": len(components),
        "ordinary_component_count": sum(1 for c in components if c["class_id"] == CLASS_ORDINARY_INTERGROWTH),
        "fine_component_count": sum(1 for c in components if c["class_id"] == CLASS_FINE_INTERGROWTH),
        "sulfide_threshold": round(float(sulfide_threshold), 3),
        "ore_class_candidate": ore_class,
        "rule_note": "Heuristic candidate, not expert ground truth.",
    }


def _normalize_illumination(value: np.ndarray) -> np.ndarray:
    h, w = value.shape
    sigma = max(9.0, min(h, w) / 32.0)
    background = cv2.GaussianBlur(value, (0, 0), sigmaX=sigma)
    corrected = value.astype(np.float32) - background.astype(np.float32) + float(np.median(background))
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    return corrected.astype(np.uint8)


def _otsu_threshold(values: np.ndarray) -> float:
    values_u8 = values.astype(np.uint8, copy=False).reshape(-1, 1)
    threshold, _ = cv2.threshold(values_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(threshold)


def _convex_hull_area(contours: list[np.ndarray]) -> float:
    if not contours:
        return 0.0
    points = np.vstack(contours)
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points)
    return float(cv2.contourArea(hull))


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(num_labels, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels].astype(np.uint8)


def _blue_annotation_like(rgb: np.ndarray, saturation: np.ndarray) -> np.ndarray:
    rgb_i16 = rgb.astype(np.int16)
    blue_bias = rgb_i16[..., 2] - np.maximum(rgb_i16[..., 0], rgb_i16[..., 1])
    return (blue_bias > 45) & (saturation > 80)


def _ellipse_kernel(radius: int) -> np.ndarray:
    size = max(1, radius * 2 + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
