from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

GIS_SCHEMA_VERSION = "ore-pipeline-gis-export-v0.1"
COORDINATE_SPACE = "local_image_pixel_top_left"


@dataclass(frozen=True)
class GisClassSpec:
    class_id: int
    class_key: str
    class_label: str


def build_geojson_feature_collection(
    mask: np.ndarray,
    *,
    class_specs: list[GisClassSpec],
    run_id: str,
    source_mask: str,
    scale: dict[str, Any] | None = None,
    min_polygon_area_px: int = 4,
    simplify_tolerance_px: float = 1.0,
) -> dict[str, Any]:
    final_mask = _mask_array(mask)
    features: list[dict[str, Any]] = []
    for spec in class_specs:
        features.extend(
            _class_features(
                final_mask,
                spec=spec,
                run_id=run_id,
                source_mask=source_mask,
                scale=scale,
                min_polygon_area_px=min_polygon_area_px,
                simplify_tolerance_px=simplify_tolerance_px,
                feature_offset=len(features),
            )
        )
    return {
        "type": "FeatureCollection",
        "metadata": {
            "schema_version": GIS_SCHEMA_VERSION,
            "coordinate_space": COORDINATE_SPACE,
            "run_id": str(run_id),
            "source_mask": source_mask,
            "image_width": int(final_mask.shape[1]),
            "image_height": int(final_mask.shape[0]),
            "class_count": len(class_specs),
            "feature_count": len(features),
            "min_polygon_area_px": int(min_polygon_area_px),
            "simplify_tolerance_px": float(simplify_tolerance_px),
            "scale": _scale_metadata(scale),
        },
        "features": features,
    }


def write_geojson_export(
    mask_path: Path,
    output_path: Path,
    *,
    class_specs: list[GisClassSpec],
    run_id: str,
    source_mask: str,
    scale: dict[str, Any] | None = None,
    min_polygon_area_px: int = 4,
    simplify_tolerance_px: float = 1.0,
) -> dict[str, Any]:
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert("L"))
    collection = build_geojson_feature_collection(
        mask,
        class_specs=class_specs,
        run_id=run_id,
        source_mask=source_mask,
        scale=scale,
        min_polygon_area_px=min_polygon_area_px,
        simplify_tolerance_px=simplify_tolerance_px,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(collection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return collection


def _class_features(
    mask: np.ndarray,
    *,
    spec: GisClassSpec,
    run_id: str,
    source_mask: str,
    scale: dict[str, Any] | None,
    min_polygon_area_px: int,
    simplify_tolerance_px: float,
    feature_offset: int,
) -> list[dict[str, Any]]:
    class_mask = (mask == int(spec.class_id)).astype(np.uint8)
    if not np.any(class_mask):
        return []
    contours, hierarchy = cv2.findContours(class_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy_rows = hierarchy[0]
    features: list[dict[str, Any]] = []
    for contour_index, hierarchy_row in enumerate(hierarchy_rows):
        parent_index = int(hierarchy_row[3])
        if parent_index != -1:
            continue
        child_indices = _child_contour_indices(hierarchy_rows, contour_index)
        exterior = _contour_to_ring(contours[contour_index], simplify_tolerance_px)
        if exterior is None:
            continue
        holes = [
            ring
            for child_index in child_indices
            if (ring := _contour_to_ring(contours[child_index], simplify_tolerance_px)) is not None
        ]
        feature_mask = np.zeros(class_mask.shape, dtype=np.uint8)
        cv2.drawContours(feature_mask, [contours[contour_index]], -1, 1, thickness=cv2.FILLED)
        active = (feature_mask > 0) & (class_mask > 0)
        area_px = int(np.count_nonzero(active))
        if area_px < int(min_polygon_area_px):
            continue
        bbox_px = _bbox_from_mask(active)
        feature_id = feature_offset + len(features) + 1
        properties: dict[str, Any] = {
            "feature_id": feature_id,
            "run_id": str(run_id),
            "class_id": int(spec.class_id),
            "class_key": str(spec.class_key),
            "class_label": str(spec.class_label),
            "source_mask": source_mask,
            "area_px": area_px,
            "bbox_px": bbox_px,
            "coordinate_space": COORDINATE_SPACE,
        }
        properties.update(_physical_area_properties(area_px, scale))
        x, y, w, h = bbox_px
        features.append(
            {
                "type": "Feature",
                "id": feature_id,
                "bbox": [float(x), float(y), float(x + w), float(y + h)],
                "properties": properties,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [exterior, *holes],
                },
            }
        )
    return features


def _mask_array(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"expected 2D final mask, got shape {array.shape}")
    return array.astype(np.uint8, copy=False)


def _child_contour_indices(hierarchy_rows: np.ndarray, contour_index: int) -> list[int]:
    child = int(hierarchy_rows[contour_index][2])
    children: list[int] = []
    while child != -1:
        children.append(child)
        child = int(hierarchy_rows[child][0])
    return children


def _contour_to_ring(contour: np.ndarray, simplify_tolerance_px: float) -> list[list[float]] | None:
    epsilon = max(0.0, float(simplify_tolerance_px))
    approx = cv2.approxPolyDP(contour, epsilon, closed=True) if epsilon else contour
    points = approx.reshape(-1, 2)
    if len(points) < 3:
        return None
    ring = [[float(point[0]), float(point[1])] for point in points]
    unique = {tuple(point) for point in ring}
    if len(unique) < 3:
        return None
    if ring[0] != ring[-1]:
        ring.append([ring[0][0], ring[0][1]])
    if len(ring) < 4:
        return None
    return ring


def _bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max())
    y1 = int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def _physical_area_properties(area_px: int, scale: dict[str, Any] | None) -> dict[str, float]:
    if not scale:
        return {}
    try:
        area_um2 = float(area_px) * float(scale["area_um2_per_analysis_pixel"])
    except (KeyError, TypeError, ValueError):
        return {}
    return {
        "area_um2": area_um2,
        "area_mm2": area_um2 / 1_000_000.0,
    }


def _scale_metadata(scale: dict[str, Any] | None) -> dict[str, Any]:
    if not scale:
        return {"available": False}
    keys = [
        "schema_version",
        "available",
        "source_field",
        "microns_per_source_pixel",
        "microns_per_analysis_pixel_x",
        "microns_per_analysis_pixel_y",
        "effective_microns_per_analysis_pixel",
        "area_um2_per_analysis_pixel",
        "scale_source",
        "scale_confidence",
        "source_width",
        "source_height",
        "analysis_width",
        "analysis_height",
    ]
    return {key: scale[key] for key in keys if key in scale}
