from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.talc_blue_line_converter import (  # noqa: E402
    TalcConversionConfig,
    apply_edit_mask,
    convert_talc_annotation_image,
    polygon_mask,
    read_mask,
    rectangle_mask,
)


class TalcBlueLineConverterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.out_root = ROOT / "outputs/test_talc_blue_line_converter"
        shutil.rmtree(self.out_root, ignore_errors=True)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.image_path = self.out_root / "synthetic_blue_line.jpg"
        image = np.full((120, 160, 3), (55, 70, 45), dtype=np.uint8)
        cv2.rectangle(image, (30, 25), (130, 95), (0, 0, 255), thickness=4)
        cv2.rectangle(image, (70, 48), (92, 72), (245, 240, 185), thickness=-1)
        Image.fromarray(image, mode="RGB").save(self.image_path)

    def test_converter_fills_blue_region_and_subtracts_sulfide(self) -> None:
        summary = convert_talc_annotation_image(
            self.image_path,
            self.out_root / "converted",
            TalcConversionConfig(
                gap_close_px=8,
                line_dilate_px=2,
                markup_ignore_dilate_px=1,
                sulfide_bright_percentile=96.0,
                sulfide_min_area_px=20,
                fallback_hull=False,
            ),
        )
        self.assertEqual(summary["schema_version"], "talc-blue-line-conversion-v0.1")
        self.assertGreater(summary["raw_blue_stroke_pixels"], 0)
        self.assertGreater(summary["candidate_talc_pixels"], 0)
        self.assertGreater(summary["overlap_pixels"], 0)

        final_mask = read_mask(Path(summary["paths"]["final_talc_mask"]))
        overlap_mask = read_mask(Path(summary["paths"]["sulfide_overlap_mask"]))
        self.assertGreater(final_mask[45, 45], 0)
        self.assertEqual(final_mask[60, 80], 0)
        self.assertGreater(overlap_mask[60, 80], 0)

    def test_apply_rectangle_review_edit(self) -> None:
        talc = np.zeros((20, 30), dtype=np.uint8)
        ignore = np.zeros_like(talc)
        rect = rectangle_mask(talc.shape, 5, 6, 12, 14)
        talc, ignore = apply_edit_mask(talc, ignore, rect, "add_talc")
        self.assertGreater(talc[7, 6], 0)
        talc, ignore = apply_edit_mask(talc, ignore, rect, "uncertain")
        self.assertEqual(talc[7, 6], 0)
        self.assertGreater(ignore[7, 6], 0)

    def test_polygon_review_edit_fills_area(self) -> None:
        mask = polygon_mask((20, 30), [[5, 5], [20, 5], [12, 15]])
        self.assertGreater(mask[8, 12], 0)
        self.assertEqual(mask[2, 2], 0)


if __name__ == "__main__":
    unittest.main()
