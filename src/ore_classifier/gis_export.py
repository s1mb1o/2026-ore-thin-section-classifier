from __future__ import annotations

import io
import json
import struct
import zipfile
from dataclasses import dataclass
from datetime import date
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


def write_shapefile_zip_export(
    collection: dict[str, Any],
    output_path: Path,
    *,
    layer_name: str = "final_classes",
) -> dict[str, Any]:
    records = _shape_records_from_geojson(collection)
    shp_bytes, shx_bytes = _shp_shx_bytes(records)
    dbf_bytes = _dbf_bytes([record["properties"] for record in records])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    members = {
        f"{layer_name}.shp": shp_bytes,
        f"{layer_name}.shx": shx_bytes,
        f"{layer_name}.dbf": dbf_bytes,
        f"{layer_name}.cpg": b"UTF-8\n",
    }
    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return {
        "path": str(output_path),
        "feature_count": len(records),
        "members": sorted(members),
    }


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


def _shape_records_from_geojson(collection: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for feature in collection.get("features") or []:
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry") if isinstance(feature.get("geometry"), dict) else {}
        if geometry.get("type") != "Polygon":
            continue
        rings = _shapefile_rings(geometry.get("coordinates") or [])
        if not rings:
            continue
        records.append(
            {
                "rings": rings,
                "properties": feature.get("properties") if isinstance(feature.get("properties"), dict) else {},
            }
        )
    return records


def _shapefile_rings(raw_rings: Any) -> list[list[tuple[float, float]]]:
    rings: list[list[tuple[float, float]]] = []
    if not isinstance(raw_rings, list):
        return rings
    for index, raw_ring in enumerate(raw_rings):
        points = _ring_points(raw_ring)
        if len(points) < 4:
            continue
        rings.append(_orient_ring(points, clockwise=index == 0))
    return rings


def _ring_points(raw_ring: Any) -> list[tuple[float, float]]:
    if not isinstance(raw_ring, list):
        return []
    points: list[tuple[float, float]] = []
    for raw_point in raw_ring:
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 2:
            continue
        try:
            points.append((float(raw_point[0]), float(raw_point[1])))
        except (TypeError, ValueError):
            continue
    if len(points) >= 3 and points[0] != points[-1]:
        points.append(points[0])
    return points


def _orient_ring(points: list[tuple[float, float]], *, clockwise: bool) -> list[tuple[float, float]]:
    if len(points) < 4:
        return points
    open_points = points[:-1] if points[0] == points[-1] else points[:]
    signed_area = _signed_ring_area(open_points)
    should_reverse = (clockwise and signed_area > 0) or (not clockwise and signed_area < 0)
    if should_reverse:
        open_points = list(reversed(open_points))
    return [*open_points, open_points[0]]


def _signed_ring_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for index, (x0, y0) in enumerate(points):
        x1, y1 = points[(index + 1) % len(points)]
        area += x0 * y1 - x1 * y0
    return area / 2.0


def _shp_shx_bytes(records: list[dict[str, Any]]) -> tuple[bytes, bytes]:
    record_payloads: list[bytes] = []
    record_bboxes: list[tuple[float, float, float, float]] = []
    for record in records:
        payload, bbox = _shp_record_payload(record["rings"])
        record_payloads.append(payload)
        record_bboxes.append(bbox)
    bbox = _combined_bbox(record_bboxes)
    shp_length_bytes = 100 + sum(8 + len(payload) for payload in record_payloads)
    shx_length_bytes = 100 + 8 * len(record_payloads)
    shp = io.BytesIO()
    shx = io.BytesIO()
    shp.write(_shapefile_header(shp_length_bytes, bbox))
    shx.write(_shapefile_header(shx_length_bytes, bbox))
    offset_words = 50
    for record_number, payload in enumerate(record_payloads, start=1):
        content_length_words = len(payload) // 2
        shp.write(struct.pack(">2i", record_number, content_length_words))
        shp.write(payload)
        shx.write(struct.pack(">2i", offset_words, content_length_words))
        offset_words += 4 + content_length_words
    return shp.getvalue(), shx.getvalue()


def _shp_record_payload(rings: list[list[tuple[float, float]]]) -> tuple[bytes, tuple[float, float, float, float]]:
    parts: list[int] = []
    points: list[tuple[float, float]] = []
    for ring in rings:
        parts.append(len(points))
        points.extend(ring)
    bbox = _bbox_from_points(points)
    payload = io.BytesIO()
    payload.write(struct.pack("<i4d2i", 5, *bbox, len(parts), len(points)))
    payload.write(struct.pack(f"<{len(parts)}i", *parts) if parts else b"")
    for x, y in points:
        payload.write(struct.pack("<2d", x, y))
    return payload.getvalue(), bbox


def _shapefile_header(length_bytes: int, bbox: tuple[float, float, float, float]) -> bytes:
    header = io.BytesIO()
    header.write(struct.pack(">7i", 9994, 0, 0, 0, 0, 0, length_bytes // 2))
    header.write(struct.pack("<2i4d4d", 1000, 5, *bbox, 0.0, 0.0, 0.0, 0.0))
    return header.getvalue()


def _combined_bbox(bboxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    if not bboxes:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def _bbox_from_points(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _dbf_bytes(records: list[dict[str, Any]]) -> bytes:
    fields = [
        ("FID", "N", 10, 0, "feature_id"),
        ("RUN_ID", "C", 48, 0, "run_id"),
        ("CLASS_ID", "N", 10, 0, "class_id"),
        ("CLASS_KEY", "C", 16, 0, "class_key"),
        ("LABEL", "C", 64, 0, "class_label"),
        ("AREA_PX", "N", 18, 0, "area_px"),
        ("AREA_UM2", "N", 20, 6, "area_um2"),
        ("AREA_MM2", "N", 20, 9, "area_mm2"),
    ]
    today = date.today()
    header_length = 32 + 32 * len(fields) + 1
    record_length = 1 + sum(field[2] for field in fields)
    payload = io.BytesIO()
    payload.write(
        struct.pack(
            "<BBBBLHH20x",
            3,
            today.year - 1900,
            today.month,
            today.day,
            len(records),
            header_length,
            record_length,
        )
    )
    for name, field_type, width, decimals, _ in fields:
        name_bytes = name.encode("ascii")[:10]
        payload.write(name_bytes + b"\x00" * (11 - len(name_bytes)))
        payload.write(field_type.encode("ascii"))
        payload.write(b"\x00" * 4)
        payload.write(bytes([width, decimals]))
        payload.write(b"\x00" * 14)
    payload.write(b"\r")
    for record in records:
        payload.write(b" ")
        for _, field_type, width, decimals, key in fields:
            payload.write(_dbf_field_bytes(record.get(key), field_type=field_type, width=width, decimals=decimals))
    payload.write(b"\x1a")
    return payload.getvalue()


def _dbf_field_bytes(value: Any, *, field_type: str, width: int, decimals: int) -> bytes:
    if value is None:
        return b" " * width
    if field_type == "C":
        return _truncate_utf8(str(value), width).ljust(width, b" ")
    try:
        if decimals:
            text = f"{float(value):.{decimals}f}"
        else:
            text = str(int(round(float(value))))
    except (TypeError, ValueError):
        return b" " * width
    if len(text) > width:
        return b" " * width
    return text.rjust(width).encode("ascii")


def _truncate_utf8(value: str, width: int) -> bytes:
    text = value
    encoded = text.encode("utf-8")
    while len(encoded) > width and text:
        text = text[:-1]
        encoded = text.encode("utf-8")
    return encoded[:width]


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
