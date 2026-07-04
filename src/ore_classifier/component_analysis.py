from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image


@dataclass
class ComponentFeatures:
    component_id: int
    label: str
    area_px: int
    footprint_area_px: int
    dark_inside_area_px: int
    dark_inside_ratio: float
    solidity: float
    compactness: float
    boundary_complexity: float
    perimeter_px: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    centroid_x: float
    centroid_y: float


@dataclass
class OreSummary:
    ore_class: str
    ore_class_ru: str
    sulfide_fraction: float
    sulfide_fraction_image: float
    ordinary_sulfide_fraction: float
    fine_sulfide_fraction: float
    talc_fraction: float
    talc_fraction_image: float
    sulfide_area_px: int
    ordinary_sulfide_area_px: int
    fine_sulfide_area_px: int
    talc_area_px: int
    image_area_px: int
    analysis_area_px: int
    analyzed_fraction: float
    component_count: int
    ordinary_component_count: int
    fine_component_count: int
    talc_margin: float
    intergrowth_margin: float
    needs_expert_review: bool
    warnings: list[str]
    rule_text_ru: str


@dataclass
class ComponentRuleConfig:
    min_component_area_px: int = 64
    close_kernel_px: int = 15
    fine_dark_inside_ratio: float = 0.18
    fine_solidity_max: float = 0.62
    fine_compactness_max: float = 0.12
    talc_fraction_threshold: float = 0.10
    # Variant B (default 0 = off): morphological OPEN radius applied to the grain
    # mask before measuring solidity/compactness/boundary_complexity, so fine-scale
    # serration (grinding/pluck-out) is ignored and only deep embayments (real
    # intergrowth) drive the boundary "fine" signal. Does NOT touch area/footprint/
    # dark_inside_ratio (the replacement signal).
    boundary_smooth_px: int = 0


def analyze_components(
    sulfide_mask: np.ndarray,
    talc_mask: np.ndarray | None = None,
    analyzed_mask: np.ndarray | None = None,
    config: ComponentRuleConfig | None = None,
    component_classifier: "Callable[[list[ComponentFeatures]], list[str]] | None" = None,
) -> tuple[OreSummary, list[ComponentFeatures], np.ndarray]:
    cfg = config or ComponentRuleConfig()
    sulfide_raw = (sulfide_mask > 0).astype(np.uint8)
    if analyzed_mask is None:
        analyzed = np.ones_like(sulfide_raw, dtype=np.uint8)
    else:
        if analyzed_mask.shape[:2] != sulfide_raw.shape[:2]:
            raise ValueError("analyzed_mask dimensions must match sulfide_mask")
        analyzed = (analyzed_mask > 0).astype(np.uint8)
    talc_raw = np.zeros_like(sulfide_raw, dtype=np.uint8) if talc_mask is None else (talc_mask > 0).astype(np.uint8)
    if talc_raw.shape[:2] != sulfide_raw.shape[:2]:
        raise ValueError("talc_mask dimensions must match sulfide_mask")
    sulfide = (sulfide_raw & analyzed).astype(np.uint8)
    talc = (talc_raw & analyzed).astype(np.uint8)
    image_area = int(sulfide.size)
    analysis_area = int(analyzed.sum())
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(sulfide, connectivity=8)
    classified = np.zeros_like(sulfide, dtype=np.uint8)
    components: list[ComponentFeatures] = []
    ordinary_area = 0
    fine_area = 0
    ordinary_count = 0
    fine_count = 0

    for component_id in range(1, labels_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < cfg.min_component_area_px:
            continue
        component_crop, sulfide_crop = crop_component(labels, sulfide, component_id, stats[component_id], cfg.close_kernel_px)
        features = component_features(component_id, component_crop, sulfide_crop, stats[component_id], centroids[component_id], cfg)
        components.append(features)

    if component_classifier is not None and components:
        # Learned relabel (e.g. component_grade_model); aggregation below is unchanged.
        for features, label in zip(components, component_classifier(components)):
            features.label = label

    for features in components:
        component_id = features.component_id
        area = features.area_px
        x, y, w, h = features.bbox_x, features.bbox_y, features.bbox_w, features.bbox_h
        component_pixels = labels[y : y + h, x : x + w] == component_id
        classified_crop = classified[y : y + h, x : x + w]
        if features.label == "fine_intergrowth":
            fine_area += area
            fine_count += 1
            classified_crop[component_pixels] = 2
        else:
            ordinary_area += area
            ordinary_count += 1
            classified_crop[component_pixels] = 1

    sulfide_area = int(sulfide.sum())
    talc_area = int(talc.sum())
    talc_fraction = talc_area / max(analysis_area, 1)
    ordinary_fraction = ordinary_area / max(sulfide_area, 1)
    fine_fraction = fine_area / max(sulfide_area, 1)
    if talc_fraction > cfg.talc_fraction_threshold:
        ore_class = "talcose_ore"
        ore_class_ru = "оталькованная руда"
    elif ordinary_area >= fine_area:
        ore_class = "row_ore"
        ore_class_ru = "рядовая руда"
    else:
        ore_class = "hard_to_process_ore"
        ore_class_ru = "труднообогатимая руда"
    talc_margin = talc_fraction - cfg.talc_fraction_threshold
    intergrowth_margin = ordinary_fraction - fine_fraction
    warnings = summary_warnings(
        sulfide_area=sulfide_area,
        analyzed_fraction=analysis_area / max(image_area, 1),
        talc_margin=talc_margin,
        intergrowth_margin=intergrowth_margin,
    )

    summary = OreSummary(
        ore_class=ore_class,
        ore_class_ru=ore_class_ru,
        sulfide_fraction=sulfide_area / max(analysis_area, 1),
        sulfide_fraction_image=sulfide_area / max(image_area, 1),
        ordinary_sulfide_fraction=ordinary_fraction,
        fine_sulfide_fraction=fine_fraction,
        talc_fraction=talc_fraction,
        talc_fraction_image=talc_area / max(image_area, 1),
        sulfide_area_px=sulfide_area,
        ordinary_sulfide_area_px=ordinary_area,
        fine_sulfide_area_px=fine_area,
        talc_area_px=talc_area,
        image_area_px=image_area,
        analysis_area_px=analysis_area,
        analyzed_fraction=analysis_area / max(image_area, 1),
        component_count=ordinary_count + fine_count,
        ordinary_component_count=ordinary_count,
        fine_component_count=fine_count,
        talc_margin=talc_margin,
        intergrowth_margin=intergrowth_margin,
        needs_expert_review=bool(warnings),
        warnings=warnings,
        rule_text_ru=rule_text_ru(ore_class_ru, talc_fraction, ordinary_area, fine_area),
    )
    return summary, components, classified


def crop_component(
    labels: np.ndarray,
    sulfide: np.ndarray,
    component_id: int,
    stat: np.ndarray,
    close_kernel_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    pad = max(2, int(close_kernel_px))
    x = int(stat[cv2.CC_STAT_LEFT])
    y = int(stat[cv2.CC_STAT_TOP])
    w = int(stat[cv2.CC_STAT_WIDTH])
    h = int(stat[cv2.CC_STAT_HEIGHT])
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(labels.shape[1], x + w + pad)
    y1 = min(labels.shape[0], y + h + pad)
    component = (labels[y0:y1, x0:x1] == component_id).astype(np.uint8)
    sulfide_crop = sulfide[y0:y1, x0:x1].astype(np.uint8)
    return component, sulfide_crop


def component_features(
    component_id: int,
    component: np.ndarray,
    sulfide: np.ndarray,
    stat: np.ndarray,
    centroid: np.ndarray,
    cfg: ComponentRuleConfig,
) -> ComponentFeatures:
    area = int(component.sum())
    footprint = reconstructed_footprint(component, cfg.close_kernel_px)
    footprint_area = int(footprint.sum())
    dark_inside = ((footprint > 0) & (sulfide == 0)).astype(np.uint8)
    dark_inside_area = int(dark_inside.sum())
    dark_inside_ratio = dark_inside_area / max(footprint_area, 1)
    # Boundary metrics measured on an optionally smoothed grain (variant B): the
    # OPEN removes shallow serration; deep embayments survive. area/footprint/
    # dark_inside above are left on the raw grain on purpose.
    shape_component = smoothed_grain(component, cfg.boundary_smooth_px)
    shape_area = int(shape_component.sum())
    solidity = component_solidity(shape_component)
    perimeter = component_perimeter(shape_component)
    compactness = 4.0 * math.pi * shape_area / max(perimeter * perimeter, 1e-6)
    boundary_complexity = perimeter / max(math.sqrt(max(shape_area, 1)), 1e-6)
    is_fine = (
        dark_inside_ratio >= cfg.fine_dark_inside_ratio
        or solidity <= cfg.fine_solidity_max
        or compactness <= cfg.fine_compactness_max
    )
    return ComponentFeatures(
        component_id=component_id,
        label="fine_intergrowth" if is_fine else "ordinary_intergrowth",
        area_px=area,
        footprint_area_px=footprint_area,
        dark_inside_area_px=dark_inside_area,
        dark_inside_ratio=dark_inside_ratio,
        solidity=solidity,
        compactness=compactness,
        boundary_complexity=boundary_complexity,
        perimeter_px=perimeter,
        bbox_x=int(stat[cv2.CC_STAT_LEFT]),
        bbox_y=int(stat[cv2.CC_STAT_TOP]),
        bbox_w=int(stat[cv2.CC_STAT_WIDTH]),
        bbox_h=int(stat[cv2.CC_STAT_HEIGHT]),
        centroid_x=float(centroid[0]),
        centroid_y=float(centroid[1]),
    )


def smoothed_grain(component: np.ndarray, smooth_px: int) -> np.ndarray:
    """Morphological OPEN of the grain mask by ``smooth_px`` (variant B), keeping
    the largest resulting blob so shape metrics stay single-component. Returns the
    input unchanged when ``smooth_px <= 0`` (preserves the default rule exactly)."""
    if smooth_px <= 0:
        return component
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(smooth_px) * 2 + 1,) * 2)
    opened = cv2.morphologyEx(component.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    if int(opened.sum()) == 0:
        return component  # opening erased a thin grain -> fall back to raw
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if count > 2:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (labels == largest).astype(np.uint8)
    return opened


def reconstructed_footprint(component: np.ndarray, close_kernel_px: int) -> np.ndarray:
    kernel_size = max(3, int(close_kernel_px) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(component.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return fill_holes(closed)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    padded = np.pad(mask_u8, 1, mode="constant", constant_values=0)
    background = (padded == 0).astype(np.uint8)
    h, w = background.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(background, flood_mask, (0, 0), 2)
    exterior = background[1:-1, 1:-1] == 2
    holes = (mask_u8 == 0) & ~exterior
    return np.where(holes, 1, mask_u8).astype(np.uint8)


def component_solidity(component: np.ndarray) -> float:
    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    return float(component.sum()) / max(hull_area, 1.0)


def component_perimeter(component: np.ndarray) -> float:
    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return float(sum(cv2.arcLength(contour, True) for contour in contours))


def rule_text_ru(ore_class_ru: str, talc_fraction: float, ordinary_area: int, fine_area: int) -> str:
    talc_pct = talc_fraction * 100.0
    sulfide_total = max(ordinary_area + fine_area, 1)
    ordinary_pct = ordinary_area / sulfide_total * 100.0
    fine_pct = fine_area / sulfide_total * 100.0
    return (
        f"Руда классифицирована как {ore_class_ru}: тальк {talc_pct:.1f}%, "
        f"обычные срастания {ordinary_pct:.1f}% сульфидной площади, "
        f"тонкие срастания {fine_pct:.1f}% сульфидной площади."
    )


def summary_warnings(
    *,
    sulfide_area: int,
    analyzed_fraction: float,
    talc_margin: float,
    intergrowth_margin: float,
    talc_margin_review: float = 0.02,
    intergrowth_margin_review: float = 0.10,
    analyzed_fraction_min: float = 0.50,
) -> list[str]:
    warnings: list[str] = []
    if sulfide_area == 0:
        warnings.append("zero_sulfide_area")
    if analyzed_fraction < analyzed_fraction_min:
        warnings.append("low_analyzed_fraction")
    if abs(talc_margin) <= talc_margin_review:
        warnings.append("talc_fraction_near_threshold")
    if abs(intergrowth_margin) <= intergrowth_margin_review:
        warnings.append("ordinary_fine_margin_near_threshold")
    return warnings


def save_component_outputs(
    out_dir: Path,
    summary: OreSummary,
    components: list[ComponentFeatures],
    classified_mask: np.ndarray,
    original_image: np.ndarray | None = None,
    talc_mask: np.ndarray | None = None,
    analyzed_mask: np.ndarray | None = None,
    preview_max_side: int = 1800,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "ore_summary.json"
    features_path = out_dir / "component_features.csv"
    mask_path = out_dir / "intergrowth_mask.png"
    analyzed_path = out_dir / "analyzed_mask.png"
    overlay_path = out_dir / "intergrowth_overlay_preview.jpg"
    summary_path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_component_csv(features_path, components)
    Image.fromarray(classified_mask.astype(np.uint8), mode="L").save(mask_path)
    paths = {
        "summary": str(summary_path),
        "component_features": str(features_path),
        "intergrowth_mask": str(mask_path),
    }
    if analyzed_mask is not None:
        Image.fromarray((analyzed_mask > 0).astype(np.uint8) * 255, mode="L").save(analyzed_path)
        paths["analyzed_mask"] = str(analyzed_path)
    if original_image is not None:
        overlay = intergrowth_overlay(original_image, classified_mask, talc_mask=talc_mask, max_side=preview_max_side)
        Image.fromarray(overlay, mode="RGB").save(overlay_path, quality=92, optimize=True)
        paths["intergrowth_overlay_preview"] = str(overlay_path)
    return paths


def write_component_csv(path: Path, components: list[ComponentFeatures]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        if not components:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=list(asdict(components[0]).keys()))
        writer.writeheader()
        for component in components:
            writer.writerow(asdict(component))


def intergrowth_overlay(
    rgb: np.ndarray,
    classified_mask: np.ndarray,
    talc_mask: np.ndarray | None = None,
    max_side: int = 1800,
) -> np.ndarray:
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    mask_img = Image.fromarray(classified_mask.astype(np.uint8), mode="L")
    talc_img = None if talc_mask is None else Image.fromarray((talc_mask > 0).astype(np.uint8) * 255, mode="L")
    if max_side and max(image.size) > max_side:
        scale = max_side / float(max(image.size))
        size = (max(1, int(image.size[0] * scale)), max(1, int(image.size[1] * scale)))
        image = image.resize(size, Image.Resampling.BILINEAR)
        mask_img = mask_img.resize(size, Image.Resampling.NEAREST)
        if talc_img is not None:
            talc_img = talc_img.resize(size, Image.Resampling.NEAREST)

    base = np.asarray(image).astype(np.float32)
    mask = np.asarray(mask_img)
    color = np.zeros_like(base)
    color[mask == 1] = (0, 220, 70)
    color[mask == 2] = (255, 40, 40)
    alpha = ((mask > 0).astype(np.float32) * 0.55)[..., None]
    if talc_img is not None:
        talc = np.asarray(talc_img) > 0
        color[talc] = (40, 120, 255)
        alpha[talc] = 0.55
    return np.clip(base * (1.0 - alpha) + color * alpha, 0, 255).astype(np.uint8)
