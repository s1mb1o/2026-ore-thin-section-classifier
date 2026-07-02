from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.curation import (  # noqa: E402
    hardness_from_probabilities,
    image_feature_vector,
    near_duplicate_pairs,
    segmentation_label_issue_summary,
    uniqueness_scores,
)


class CurationTest(unittest.TestCase):
    def test_uniqueness_and_near_duplicate_pairs(self) -> None:
        dark = np.zeros((8, 8, 3), dtype=np.uint8)
        dark_copy = dark.copy()
        bright = np.full((8, 8, 3), 255, dtype=np.uint8)
        features = np.stack([image_feature_vector(item) for item in [dark, dark_copy, bright]], axis=0)
        unique = uniqueness_scores(features)
        pairs = near_duplicate_pairs(["dark", "dark_copy", "bright"], features, distance_threshold=0.0)
        self.assertEqual(pairs[0].left_id, "dark")
        self.assertEqual(pairs[0].right_id, "dark_copy")
        self.assertGreater(float(unique[2]), float(unique[0]))

    def test_hardness_from_probabilities(self) -> None:
        probs = np.array(
            [
                [[0.9, 0.5], [0.51, 0.1]],
                [[0.1, 0.5], [0.49, 0.9]],
            ],
            dtype=np.float32,
        )
        hardness = hardness_from_probabilities(probs)
        self.assertAlmostEqual(float(hardness[0, 0]), 0.1, places=5)
        self.assertAlmostEqual(float(hardness[0, 1]), 0.5, places=5)

    def test_segmentation_label_issue_summary(self) -> None:
        labels = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        probs = np.array(
            [
                [[0.9, 0.85], [0.1, 0.9]],
                [[0.1, 0.15], [0.9, 0.1]],
            ],
            dtype=np.float32,
        )
        summary = segmentation_label_issue_summary(labels, probs, confidence_threshold=0.8)
        self.assertAlmostEqual(summary.issue_fraction, 0.25)
        self.assertEqual(int(summary.issue_mask[0, 1]), 1)
        self.assertEqual(int(summary.issue_mask[0, 0]), 0)


if __name__ == "__main__":
    unittest.main()
