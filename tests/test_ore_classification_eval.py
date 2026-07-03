from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_ore_classification import binary_auc, evaluate_rows  # noqa: E402


class OreClassificationEvalTest(unittest.TestCase):
    def test_binary_auc_handles_ties(self) -> None:
        self.assertAlmostEqual(binary_auc([1, 0, 1, 0], [0.8, 0.2, 0.5, 0.5]), 0.875)

    def test_evaluate_rows_reports_macro_metrics(self) -> None:
        rows = [
            {
                "run_id": "a",
                "source_label": "ordinary_intergrowth",
                "predicted_ore_class": "row_ore",
                "sulfide_fraction": "0.2",
                "ordinary_sulfide_fraction": "0.9",
                "fine_sulfide_fraction": "0.1",
                "talc_fraction": "0.01",
            },
            {
                "run_id": "b",
                "source_label": "fine_intergrowth",
                "predicted_ore_class": "hard_to_process_ore",
                "sulfide_fraction": "0.2",
                "ordinary_sulfide_fraction": "0.1",
                "fine_sulfide_fraction": "0.9",
                "talc_fraction": "0.01",
            },
            {
                "run_id": "c",
                "source_label": "talcose",
                "predicted_ore_class": "talcose_ore",
                "sulfide_fraction": "0.1",
                "ordinary_sulfide_fraction": "0.4",
                "fine_sulfide_fraction": "0.6",
                "talc_fraction": "0.2",
            },
        ]
        metrics = evaluate_rows(rows)
        self.assertEqual(metrics["rows_used"], 3)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["confusion_matrix"]["talcose_ore"]["talcose_ore"], 1)
        self.assertIsNotNone(metrics["macro_auc_ovr"])


if __name__ == "__main__":
    unittest.main()
