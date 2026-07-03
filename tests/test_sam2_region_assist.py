from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.sam2_region_assist import Sam2AssistFailure, _postprocess_sam2_mask  # noqa: E402


class Sam2RegionAssistTest(unittest.TestCase):
    def test_rectangle_prompt_clips_mask_to_box(self) -> None:
        mask = np.full((10, 12), 255, dtype=np.uint8)

        clipped = _postprocess_sam2_mask(
            mask,
            {"type": "rectangle_xyxy", "x1": 2, "y1": 3, "x2": 7, "y2": 8},
            mask.shape,
            max_fraction=1.0,
        )

        self.assertEqual(int(np.count_nonzero(clipped)), 25)
        self.assertEqual(int(np.count_nonzero(clipped[:3, :])), 0)
        self.assertEqual(int(np.count_nonzero(clipped[:, :2])), 0)
        self.assertEqual(int(np.count_nonzero(clipped[8:, :])), 0)
        self.assertEqual(int(np.count_nonzero(clipped[:, 7:])), 0)

    def test_rejects_full_image_sam2_mask(self) -> None:
        mask = np.full((10, 12), 255, dtype=np.uint8)

        with self.assertRaises(Sam2AssistFailure) as raised:
            _postprocess_sam2_mask(mask, {"type": "point_xy", "x": 4, "y": 5}, mask.shape, max_fraction=0.5)

        self.assertIn("SAM2 mask covers", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
