from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_talc_segformer_folds import (  # noqa: E402
    make_stratified_folds,
    parse_folds_to_run,
    parse_thresholds,
    threshold_row,
)


class TalcSegformerFoldsTest(unittest.TestCase):
    def test_stratified_folds_keep_sample_ids_unique(self) -> None:
        samples = [
            {"sample_id": f"DSCN{i}", "group": "dscn_na"} for i in range(6)
        ] + [
            {"sample_id": f"scan{i}", "group": "scan_10x"} for i in range(4)
        ]

        folds = make_stratified_folds(samples, k=3, seed=11)

        all_ids = [sample_id for fold_ids in folds.values() for sample_id in fold_ids]
        self.assertEqual(sorted(all_ids), sorted(sample["sample_id"] for sample in samples))
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(folds.keys()), {"0", "1", "2"})
        self.assertTrue(all(folds[str(i)] for i in range(3)))

    def test_parse_fold_selection_and_thresholds(self) -> None:
        self.assertEqual(parse_folds_to_run("all", 3), [0, 1, 2])
        self.assertEqual(parse_folds_to_run("0,2", 3), [0, 2])
        self.assertEqual(parse_thresholds("0.5,0.25,0.5"), [0.25, 0.5])
        with self.assertRaises(ValueError):
            parse_folds_to_run("3", 3)
        with self.assertRaises(ValueError):
            parse_thresholds("0,0.5")

    def test_threshold_row_reports_talc_metrics(self) -> None:
        row = threshold_row(
            0.5,
            {
                "tp": 8,
                "fp": 2,
                "fn": 4,
                "tn": 6,
                "valid": 20,
            },
        )

        self.assertEqual(row["threshold"], 0.5)
        self.assertAlmostEqual(row["iou_talc"], 8 / 14)
        self.assertAlmostEqual(row["precision_talc"], 8 / 10)
        self.assertAlmostEqual(row["recall_talc"], 8 / 12)
        self.assertAlmostEqual(row["pixel_acc"], 14 / 20)


if __name__ == "__main__":
    unittest.main()
