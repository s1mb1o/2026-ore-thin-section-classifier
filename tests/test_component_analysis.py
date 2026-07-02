from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.component_analysis import ComponentRuleConfig, analyze_components  # noqa: E402
from ore_classifier.component_analysis import fill_holes  # noqa: E402


class ComponentAnalysisTest(unittest.TestCase):
    def test_solid_component_is_ordinary_and_row_ore(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (20, 20), (70, 70), 255, thickness=-1)
        summary, components, classified = analyze_components(mask, config=ComponentRuleConfig(min_component_area_px=10))
        self.assertEqual(summary.ore_class, "row_ore")
        self.assertEqual(len(components), 1)
        self.assertEqual(components[0].label, "ordinary_intergrowth")
        self.assertGreater(int((classified == 1).sum()), 0)

    def test_holey_component_is_fine(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (20, 20), (80, 80), 255, thickness=-1)
        cv2.rectangle(mask, (38, 38), (62, 62), 0, thickness=-1)
        summary, components, classified = analyze_components(
            mask,
            config=ComponentRuleConfig(
                min_component_area_px=10,
                close_kernel_px=31,
                fine_dark_inside_ratio=0.05,
            ),
        )
        self.assertEqual(components[0].label, "fine_intergrowth")
        self.assertEqual(summary.ore_class, "hard_to_process_ore")
        self.assertGreater(int((classified == 2).sum()), 0)

    def test_talc_fraction_overrides_ore_class(self) -> None:
        sulfide = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(sulfide, (20, 20), (40, 40), 255, thickness=-1)
        talc = np.zeros_like(sulfide)
        talc[:40, :40] = 255
        summary, _, _ = analyze_components(sulfide, talc_mask=talc, config=ComponentRuleConfig(min_component_area_px=10))
        self.assertEqual(summary.ore_class, "talcose_ore")

    def test_fill_holes_does_not_fill_exterior_when_component_touches_border(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[0:10, 0:10] = 1
        filled = fill_holes(mask)
        self.assertEqual(int(filled[15, 15]), 0)
        self.assertEqual(int(filled[5, 5]), 1)


if __name__ == "__main__":
    unittest.main()
