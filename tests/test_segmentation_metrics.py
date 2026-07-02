from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.segmentation_metrics import (  # noqa: E402
    BinarySegmentationAccumulator,
    auc_from_hist,
    binary_confusion_counts,
    binary_scores_from_counts,
    hausdorff_and_hd95_px,
)


class SegmentationMetricsTest(unittest.TestCase):
    def test_confusion_scores(self) -> None:
        target = np.array([[1, 1, 0], [0, 1, 0]], dtype=np.uint8)
        pred = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.uint8)
        counts = binary_confusion_counts(target, pred)
        self.assertEqual(counts, {"tp": 2, "fp": 1, "fn": 1, "tn": 2})

        scores = binary_scores_from_counts(**counts)
        self.assertAlmostEqual(scores["iou_sulfide"], 0.5)
        self.assertAlmostEqual(scores["f1_sulfide"], 2 * 2 / (2 * 2 + 1 + 1))
        self.assertAlmostEqual(scores["pixel_acc"], 4 / 6)

    def test_auc_from_hist_prefers_positive_scores(self) -> None:
        pos_hist = np.array([0, 0, 3], dtype=np.int64)
        neg_hist = np.array([3, 0, 0], dtype=np.int64)
        self.assertEqual(auc_from_hist(pos_hist, neg_hist), 1.0)

    def test_hausdorff_zero_for_identical_masks(self) -> None:
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[4:12, 5:13] = 1
        hausdorff_px, hd95_px = hausdorff_and_hd95_px(mask, mask)
        self.assertEqual(hausdorff_px, 0.0)
        self.assertEqual(hd95_px, 0.0)

    def test_accumulator_summary_includes_auc_and_hd95(self) -> None:
        target = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        pred = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        prob = np.array([[0.9, 0.2], [0.4, 0.1]], dtype=np.float32)
        acc = BinarySegmentationAccumulator(auc_bins=8)
        acc.update_confusion(target, pred, prob_sulfide=prob)
        acc.update_hausdorff(target, pred)
        summary = acc.summary().to_dict()
        self.assertEqual(summary["tp"], 1)
        self.assertGreater(summary["auc_sulfide"], 0.5)
        self.assertEqual(summary["hausdorff_items"], 1)
        self.assertIsNotNone(summary["hd95_px_mean"])


if __name__ == "__main__":
    unittest.main()
