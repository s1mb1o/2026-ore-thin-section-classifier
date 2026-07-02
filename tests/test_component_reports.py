from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.component_analysis import ComponentRuleConfig, analyze_components  # noqa: E402
from ore_classifier.component_reports import (  # noqa: E402
    association_contacts,
    component_liberation_proxies,
    ore_decision_margins,
)


class ComponentReportsTest(unittest.TestCase):
    def test_association_contacts_count_adjacent_labels(self) -> None:
        class_mask = np.array([[0, 1, 1], [0, 2, 2], [3, 3, 2]], dtype=np.uint8)
        contacts = association_contacts(class_mask, {0: "matrix", 1: "ordinary", 2: "fine", 3: "talc"})
        records = {(item.name_a, item.name_b): item.contact_px for item in contacts}
        self.assertGreater(records[("matrix", "ordinary")], 0)
        self.assertGreater(records[("fine", "talc")], 0)

    def test_component_liberation_proxy_tracks_talc_touch(self) -> None:
        sulfide = np.zeros((12, 12), dtype=np.uint8)
        cv2.rectangle(sulfide, (3, 3), (5, 5), 1, thickness=-1)
        talc = np.zeros_like(sulfide)
        talc[3:6, 6] = 1
        rows = component_liberation_proxies(sulfide, talc_mask=talc, min_area_px=1)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].touches_talc)
        self.assertGreater(rows[0].matrix_contact_px, 0)

    def test_ore_decision_margins_flags_near_talc_threshold(self) -> None:
        sulfide = np.zeros((100, 100), dtype=np.uint8)
        sulfide[10:30, 10:30] = 1
        talc = np.zeros_like(sulfide)
        talc[:10, :10] = 1
        summary, _, _ = analyze_components(
            sulfide,
            talc_mask=talc,
            config=ComponentRuleConfig(min_component_area_px=1, talc_fraction_threshold=0.01),
        )
        margins = ore_decision_margins(summary, ComponentRuleConfig(talc_fraction_threshold=0.01), talc_review_margin=0.001)
        self.assertTrue(margins["needs_expert_review"])
        self.assertIn("talc fraction near threshold", margins["review_reasons"])


if __name__ == "__main__":
    unittest.main()
