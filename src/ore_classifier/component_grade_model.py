"""Learned per-component grade classifier (ordinary vs fine intergrowth).

Provides an optional replacement for the hand-tuned OR-rule in
``component_analysis``: a HistGradientBoosting model trained on per-component
shape features with weak folder-level labels. The shipped default is the
2026-07-05 no-magnification artifact, which avoids a scanner-frame shortcut and
uses a calibrated fine threshold from its ``meta.json``.

The model consumes the same ``ComponentFeatures`` the pipeline already
computes, so integration only relabels components before the area-weighted
aggregation — percentages and the ore-class verdict keep working unchanged.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ore_classifier.component_analysis import ComponentFeatures

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPONENT_MODEL_PATH = ROOT / "models/component_grade/hgb_weak100_nomag_20260705/model.joblib"

FEATURE_FIELDS = (
    "area_px",
    "footprint_area_px",
    "dark_inside_ratio",
    "solidity",
    "compactness",
    "boundary_complexity",
    "perimeter_px",
    "bbox_w",
    "bbox_h",
)
MAGNIFICATIONS = ("cam", "5x", "10x", "20x")
_MAG_RE = re.compile(r"(\d+)\s*[xхX]")

FINE_LABEL = "fine_intergrowth"
ORDINARY_LABEL = "ordinary_intergrowth"

ComponentClassifier = Callable[[Sequence["ComponentFeatures"]], list[str]]


def parse_magnification(image_name: str | None) -> str:
    """Magnification tag from an image filename ('10x', '5x', ...); 'cam' if absent."""
    if not image_name:
        return "cam"
    match = _MAG_RE.search(image_name)
    mag = f"{match.group(1)}x" if match else "cam"
    return mag if mag in MAGNIFICATIONS else "cam"


def feature_vector(component: "ComponentFeatures", magnification: str = "cam") -> list[float]:
    # NOTE: magnification one-hots were removed 2026-07-05 — in the weak-label
    # training set the 10x/20x scanner frames were 91-94% fine, so the model
    # learned "scanner frame => fine" (sampling bias, not geology) and stamped
    # every component of 10x frames fine regardless of shape (2550382-1 case).
    bw = float(component.bbox_w)
    bh = float(component.bbox_h)
    area = float(component.area_px)
    row = [float(getattr(component, f)) for f in FEATURE_FIELDS]
    row.append(float(np.log1p(area)))
    row.append(area / max(bw * bh, 1.0))
    row.append(bw / max(bh, 1.0))
    return row


@dataclass
class ComponentGradeModel:
    """Loaded model + metadata; produces a classifier callable per image."""

    model: object
    meta: dict
    path: Path

    @classmethod
    def load(cls, path: Path | str) -> "ComponentGradeModel":
        import joblib  # deferred: rule-only deployments don't need sklearn

        path = Path(path)
        model = joblib.load(path)
        meta_path = path.with_name("meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        expected = meta.get("feature_fields")
        if expected and tuple(expected) != FEATURE_FIELDS:
            raise ValueError(
                f"component grade model {path} was trained with features {expected}, "
                f"code expects {list(FEATURE_FIELDS)}"
            )
        return cls(model=model, meta=meta, path=path)

    def labeler(self, image_name: str | None = None) -> ComponentClassifier:
        """Classifier callable for one image."""
        magnification = parse_magnification(image_name)
        # Calibrated component threshold: weak labels are ~3:1 fine-heavy, which
        # inflates P(fine) globally; the artifact carries its own cutoff.
        threshold = float(self.meta.get("component_threshold", 0.5))

        def classify(components: Sequence["ComponentFeatures"]) -> list[str]:
            if not components:
                return []
            X = np.array([feature_vector(c, magnification) for c in components])
            proba_fine = self.model.predict_proba(X)[:, 1]
            return [FINE_LABEL if p >= threshold else ORDINARY_LABEL for p in proba_fine]

        return classify


def resolve_component_model(path: Path | str | None) -> ComponentGradeModel | None:
    """Load a model if a usable path is given; None means 'use the rule'."""
    if path is None:
        return None
    if str(path).lower() in {"", "none", "rule"}:
        return None
    return ComponentGradeModel.load(path)
