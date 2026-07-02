from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.report_cards import (  # noqa: E402
    render_dataset_card,
    render_model_card,
    render_run_fact_sheet,
)


class ReportCardsTest(unittest.TestCase):
    def test_model_card_contains_provenance_metrics_and_limits(self) -> None:
        card = render_model_card(
            model_name="segformer-b0-sulfide",
            intended_use="Binary sulfide segmentation for official OM images.",
            provenance={"checkpoint": "best.pt", "labels": "weak"},
            metrics={"iou": 0.95},
            limitations=["Not a talc classifier."],
        )
        self.assertIn("# Model Card", card)
        self.assertIn("`checkpoint`: best.pt", card)
        self.assertIn("Not a talc classifier.", card)

    def test_dataset_card_and_run_fact_sheet_have_expected_sections(self) -> None:
        dataset = render_dataset_card(
            dataset_name="official-om-v2",
            composition={"images": 1236},
            labels={"class_folders": "image-level"},
            recommended_use="Training with weak supervision.",
        )
        fact_sheet = render_run_fact_sheet(
            run_id="smoke",
            inputs={"image": "sample.jpg"},
            outputs={"mask": "mask.png"},
            parameters={"threshold": 0.5},
        )
        self.assertIn("## Composition", dataset)
        self.assertIn("Pseudo-labels", dataset)
        self.assertIn("## Parameters", fact_sheet)


if __name__ == "__main__":
    unittest.main()
