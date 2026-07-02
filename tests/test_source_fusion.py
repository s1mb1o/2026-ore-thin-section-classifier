from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.source_fusion import (  # noqa: E402
    MaskSource,
    fuse_source_masks,
    source_agreement_summary,
)


class SourceFusionTest(unittest.TestCase):
    def test_weighted_fusion_and_disagreement(self) -> None:
        a = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        b = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        fused = fuse_source_masks([MaskSource("a", a), MaskSource("b", b)])
        self.assertEqual(float(fused.probability[0, 0]), 1.0)
        self.assertEqual(float(fused.probability[0, 1]), 0.5)
        self.assertEqual(int(fused.mask[0, 1]), 1)
        self.assertEqual(float(fused.disagreement[0, 1]), 1.0)
        self.assertEqual(float(fused.disagreement[0, 0]), 0.0)

    def test_agreement_summary_reports_conflicts(self) -> None:
        a = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        b = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        summary = source_agreement_summary([MaskSource("a", a), MaskSource("b", b)])
        self.assertEqual(summary["source_count"], 2)
        self.assertAlmostEqual(summary["conflict_fraction"], 0.5)
        self.assertIn("a__b", summary["pairwise_iou"])


if __name__ == "__main__":
    unittest.main()
