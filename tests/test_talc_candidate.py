from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.talc_candidate import TalcCandidateConfig, estimate_talc_candidate_mask  # noqa: E402


class TalcCandidateTest(unittest.TestCase):
    def test_green_gray_region_is_talc_candidate(self) -> None:
        rgb = np.zeros((20, 20, 3), dtype=np.uint8)
        rgb[4:14, 4:14] = (90, 125, 85)
        mask = estimate_talc_candidate_mask(
            rgb,
            config=TalcCandidateConfig(min_area_px=1, morphology_open_radius=0, morphology_close_radius=0),
        )
        self.assertGreater(int(mask.sum()), 0)
        self.assertEqual(int(mask[8, 8]), 255)

    def test_sulfide_mask_excludes_candidate_pixels(self) -> None:
        rgb = np.zeros((20, 20, 3), dtype=np.uint8)
        rgb[4:14, 4:14] = (90, 125, 85)
        sulfide = np.zeros((20, 20), dtype=np.uint8)
        sulfide[4:14, 4:14] = 255
        mask = estimate_talc_candidate_mask(
            rgb,
            sulfide_mask=sulfide,
            config=TalcCandidateConfig(min_area_px=1, morphology_open_radius=0, morphology_close_radius=0),
        )
        self.assertEqual(int(mask.sum()), 0)

    def test_blue_annotation_like_pixels_are_excluded(self) -> None:
        rgb = np.zeros((20, 20, 3), dtype=np.uint8)
        rgb[4:14, 4:14] = (20, 70, 190)
        mask = estimate_talc_candidate_mask(
            rgb,
            config=TalcCandidateConfig(
                min_area_px=1,
                morphology_open_radius=0,
                morphology_close_radius=0,
                hue_min=0,
                hue_max=179,
                saturation_min=1,
                saturation_max=255,
                value_min=1,
                value_max=255,
                green_bias_min=-255,
                blue_bias_max=255,
            ),
        )
        self.assertEqual(int(mask.sum()), 0)


if __name__ == "__main__":
    unittest.main()
