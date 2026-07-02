from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heuristic_segmentation import (  # noqa: E402
    CLASS_FINE_INTERGROWTH,
    CLASS_ORDINARY_INTERGROWTH,
    CLASS_TALC_CANDIDATE,
    HeuristicConfig,
    make_overlay,
    segment_image,
)


class HeuristicSegmentationTest(unittest.TestCase):
    def test_segments_compact_fragmented_and_talc_candidate_regions(self) -> None:
        image = np.full((128, 160, 3), (42, 54, 38), dtype=np.uint8)
        image[18:46, 96:140] = (105, 158, 112)
        image[54:92, 20:60] = (235, 228, 162)
        image[54:60, 34:43] = (45, 50, 42)
        image[98:104, 84:91] = (232, 230, 180)
        image[110:116, 105:112] = (232, 230, 180)

        result = segment_image(
            image,
            HeuristicConfig(
                min_component_area=20,
                fine_max_area_px=70,
                fine_min_replacement_ratio=0.12,
                talc_min_area=120,
            ),
        )

        self.assertGreater(int((result.class_mask == CLASS_ORDINARY_INTERGROWTH).sum()), 0)
        self.assertGreater(int((result.class_mask == CLASS_FINE_INTERGROWTH).sum()), 0)
        self.assertGreater(int((result.class_mask == CLASS_TALC_CANDIDATE).sum()), 0)
        self.assertEqual(result.class_mask.shape, image.shape[:2])
        self.assertGreaterEqual(result.metrics["component_count"], 2)

    def test_overlay_preserves_image_shape(self) -> None:
        image = np.full((32, 40, 3), 50, dtype=np.uint8)
        result = segment_image(image, HeuristicConfig(min_component_area=5))
        overlay = make_overlay(image, result.class_mask)
        self.assertEqual(overlay.shape, image.shape)
        self.assertEqual(overlay.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
