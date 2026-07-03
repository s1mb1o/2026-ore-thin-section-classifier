from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.talc_candidate import (  # noqa: E402
    TalcCandidateConfig,
    estimate_talc_candidate_mask,
    save_talc_candidate_outputs,
    talc_candidate_overlay,
)


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


class TalcCandidateOverlayTest(unittest.TestCase):
    def test_overlay_tints_talc_pixels_blue(self) -> None:
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        talc = np.zeros((8, 8), dtype=np.uint8)
        talc[2:5, 2:5] = 255
        overlay = talc_candidate_overlay(rgb=rgb, talc_mask=talc, max_side=0)
        self.assertEqual(overlay.shape, (8, 8, 3))
        # talc pixels gain blue channel weight; untouched pixels stay black.
        self.assertGreater(int(overlay[3, 3, 2]), int(overlay[0, 0, 2]))
        np.testing.assert_array_equal(overlay[0, 0], np.zeros(3, dtype=np.uint8))

    def test_overlay_downscales_when_over_max_side(self) -> None:
        rgb = np.zeros((40, 20, 3), dtype=np.uint8)
        talc = np.zeros((40, 20), dtype=np.uint8)
        overlay = talc_candidate_overlay(rgb=rgb, talc_mask=talc, max_side=20)
        # longest side is clamped to max_side, aspect ratio preserved.
        self.assertEqual(overlay.shape[0], 20)
        self.assertEqual(overlay.shape[1], 10)

    def test_overlay_marks_sulfide_without_overlapping_talc(self) -> None:
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        talc = np.zeros((8, 8), dtype=np.uint8)
        talc[0:2, 0:2] = 255
        sulfide = np.zeros((8, 8), dtype=np.uint8)
        sulfide[0:4, 0:4] = 255
        overlay = talc_candidate_overlay(rgb=rgb, talc_mask=talc, sulfide_mask=sulfide, max_side=0)
        # a sulfide-only pixel is tinted yellow (high red+green, low blue).
        self.assertGreater(int(overlay[3, 3, 0]), int(overlay[3, 3, 2]))


class TalcCandidateSaveOutputsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self._tmp.name) / "talc"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_save_writes_all_artifacts_and_summary_math(self) -> None:
        rgb = np.full((10, 10, 3), 120, dtype=np.uint8)
        talc = np.zeros((10, 10), dtype=np.uint8)
        talc[0:5, 0:10] = 255  # 50 of 100 pixels
        paths = save_talc_candidate_outputs(out_dir=self.out_dir, rgb=rgb, talc_mask=talc)

        for key in ("talc_candidate_mask", "talc_candidate_overlay_preview", "talc_candidate_summary"):
            self.assertTrue(Path(paths[key]).exists(), key)

        summary = json.loads(Path(paths["talc_candidate_summary"]).read_text(encoding="utf-8"))
        self.assertEqual(summary["schema_version"], "talc-candidate-v0.1")
        self.assertEqual(summary["width"], 10)
        self.assertEqual(summary["height"], 10)
        self.assertEqual(summary["image_area_px"], 100)
        self.assertEqual(summary["talc_candidate_area_px"], 50)
        self.assertAlmostEqual(summary["talc_candidate_fraction_image"], 0.5)
        # fraction-of-analyzed uses the analyzed-area denominator, not image area.
        self.assertEqual(
            summary["talc_candidate_fraction"],
            50 / max(summary["analyzed_area_px"], 1),
        )
        self.assertIn("config", summary)


if __name__ == "__main__":
    unittest.main()
