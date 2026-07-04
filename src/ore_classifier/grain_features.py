"""Shared feature engineering and model factory for the grain-level classifier.

Both `scripts/train_grain_classifier.py` (exports the model + grain-level CV) and
`scripts/aggregate_grade_from_grains.py` (leak-free image-grade CV) build the same
per-grain feature vector and use the same estimators via this module, so the two
never drift.

A "grain" is one connected sulfide component with the numeric fields of
`ore_classifier.component_analysis.ComponentFeatures`. We use morphology + a few
engineered ratios and deliberately DROP absolute position (bbox_x/y, centroid)
so the classifier keys on grain shape/texture, not where it sits in the frame.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# The two grade-relevant sulfide grain classes the human labels / heuristic emits.
GRAIN_CLASS_ORDER = ["ordinary_intergrowth", "fine_intergrowth"]

# Raw component_features columns consumed as-is (position columns excluded on purpose).
_RAW_FEATURES = [
    "area_px",
    "footprint_area_px",
    "dark_inside_area_px",
    "dark_inside_ratio",
    "solidity",
    "compactness",
    "boundary_complexity",
    "bbox_w",
    "bbox_h",
]

# Final feature vector order (raw + engineered).
FEATURE_NAMES = [
    *_RAW_FEATURES,
    "log_area",
    "aspect_ratio",
    "extent",
    "footprint_fill",
    "dark_inside_area_frac",
]


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def grain_feature_vector(row: dict[str, Any]) -> list[float]:
    """Build one grain's feature vector from a manifest / component_features row."""
    raw = {name: _to_float(row.get(name)) for name in _RAW_FEATURES}
    area = raw["area_px"]
    footprint = raw["footprint_area_px"]
    bbox_w = raw["bbox_w"]
    bbox_h = raw["bbox_h"]
    bbox_area = max(bbox_w * bbox_h, 1.0)
    engineered = {
        "log_area": math.log1p(max(area, 0.0)),
        "aspect_ratio": bbox_w / max(bbox_h, 1.0),
        "extent": area / bbox_area,
        "footprint_fill": area / max(footprint, 1.0),
        "dark_inside_area_frac": raw["dark_inside_area_px"] / max(footprint, 1.0),
    }
    values = [raw[name] for name in _RAW_FEATURES] + [engineered[name] for name in FEATURE_NAMES[len(_RAW_FEATURES):]]
    return [float(v) for v in values]


def build_grain_feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    matrix = np.array([grain_feature_vector(row) for row in rows], dtype=np.float64)
    if matrix.size == 0:
        return matrix.reshape(0, len(FEATURE_NAMES))
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def make_grain_model(name: str) -> Any:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=400, random_state=42, class_weight="balanced", min_samples_leaf=2, n_jobs=-1)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced", min_samples_leaf=2, n_jobs=-1)
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    raise ValueError(f"unknown grain model: {name}")


def resolve_grain_label(
    row: dict[str, Any],
    annotations: dict[str, dict[str, Any]] | None,
    *,
    require_human: bool,
) -> str | None:
    """Return the training label for a grain, or None to skip it.

    Human annotation (if present and not 'uncertain') wins; otherwise the
    heuristic pre-label is used as a weak-supervision bootstrap unless
    ``require_human`` is set.
    """
    grain_uid = str(row.get("grain_uid", ""))
    if annotations and grain_uid in annotations:
        label = str(annotations[grain_uid].get("label", ""))
        if label in GRAIN_CLASS_ORDER:
            return label
        return None  # 'uncertain' or unknown -> excluded from training
    if require_human:
        return None
    heuristic = str(row.get("heuristic_label", ""))
    return heuristic if heuristic in GRAIN_CLASS_ORDER else None
