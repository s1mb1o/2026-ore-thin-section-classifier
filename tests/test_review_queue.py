from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.review_queue import (  # noqa: E402
    build_review_queue,
    decision_impact_from_threshold_margin,
    expert_questions_from_candidates,
)


class ReviewQueueTest(unittest.TestCase):
    def test_review_queue_prioritizes_decision_impact(self) -> None:
        uncertainty = np.zeros((12, 12), dtype=np.float32)
        uncertainty[1:4, 1:4] = 0.8
        uncertainty[7:10, 7:10] = 0.8
        impact = np.ones_like(uncertainty) * 0.2
        impact[7:10, 7:10] = 1.0
        candidates = build_review_queue(
            uncertainty,
            decision_impact_map=impact,
            threshold=0.5,
            min_area_px=4,
            padding_px=1,
        )
        self.assertEqual(len(candidates), 2)
        self.assertGreater(candidates[0].x, candidates[1].x)
        self.assertIn("near decision threshold", candidates[0].reason)

    def test_threshold_margin_map_peaks_near_threshold(self) -> None:
        values = np.array([[0.49, 0.2], [0.6, 0.9]], dtype=np.float32)
        impact = decision_impact_from_threshold_margin(values, threshold=0.5, max_margin=0.2)
        self.assertGreater(float(impact[0, 0]), float(impact[1, 1]))
        self.assertEqual(float(impact[1, 1]), 0.0)

    def test_expert_questions_include_bbox(self) -> None:
        uncertainty = np.zeros((8, 8), dtype=np.float32)
        uncertainty[2:5, 3:6] = 0.9
        candidates = build_review_queue(uncertainty, threshold=0.5, min_area_px=3)
        questions = expert_questions_from_candidates(candidates, image_id="sample")
        self.assertEqual(questions[0]["image_id"], "sample")
        self.assertEqual(questions[0]["bbox_xywh"], [3, 2, 3, 3])
        self.assertIn("Проверьте", questions[0]["question_ru"])


if __name__ == "__main__":
    unittest.main()
