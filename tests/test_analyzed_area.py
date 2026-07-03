from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.analyzed_area import (  # noqa: E402
    blue_annotation_like,
    build_analyzed_mask,
    ellipse_kernel,
)


class AnalyzedAreaTest(unittest.TestCase):
    def test_build_analyzed_mask_rejects_non_rgb(self) -> None:
        with self.assertRaises(ValueError):
            build_analyzed_mask(np.zeros((8, 8), dtype=np.uint8))
        with self.assertRaises(ValueError):
            build_analyzed_mask(np.zeros((8, 8, 4), dtype=np.uint8))

    def test_dark_border_pixels_are_excluded(self) -> None:
        rgb = np.full((16, 16, 3), 120, dtype=np.uint8)
        rgb[:4, :] = 0  # dark border below min_value
        mask = build_analyzed_mask(rgb, min_value=18)
        self.assertEqual(int(mask[0, 0]), 0)
        self.assertEqual(int(mask[8, 8]), 1)

    def test_blue_annotation_pixels_are_excluded(self) -> None:
        rgb = np.full((16, 16, 3), 120, dtype=np.uint8)
        rgb[6:10, 6:10] = (20, 40, 220)  # saturated blue markup
        mask = build_analyzed_mask(rgb)
        self.assertEqual(int(mask[8, 8]), 0)

    def test_blue_annotation_like_computes_saturation_when_omitted(self) -> None:
        # Covers the branch where saturation is derived internally.
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[..., 2] = 230  # strong, saturated blue
        without = blue_annotation_like(rgb)
        precomputed = blue_annotation_like(rgb, saturation=np.full((4, 4), 255, dtype=np.uint8))
        self.assertTrue(without.all())
        np.testing.assert_array_equal(without, precomputed)

    def test_ellipse_kernel_is_odd_and_minimum_one(self) -> None:
        self.assertEqual(ellipse_kernel(0).shape, (1, 1))
        self.assertEqual(ellipse_kernel(2).shape, (5, 5))


if __name__ == "__main__":
    unittest.main()
