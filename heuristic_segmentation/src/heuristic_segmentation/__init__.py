"""Standalone heuristic segmentation baseline for the ore-classifier task."""

from .segmentation import (
    CLASS_BACKGROUND,
    CLASS_FINE_INTERGROWTH,
    CLASS_ORDINARY_INTERGROWTH,
    CLASS_TALC_CANDIDATE,
    HeuristicConfig,
    SegmentationResult,
    make_overlay,
    segment_image,
)

__all__ = [
    "CLASS_BACKGROUND",
    "CLASS_FINE_INTERGROWTH",
    "CLASS_ORDINARY_INTERGROWTH",
    "CLASS_TALC_CANDIDATE",
    "HeuristicConfig",
    "SegmentationResult",
    "make_overlay",
    "segment_image",
]
