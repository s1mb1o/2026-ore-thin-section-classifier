from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.component_analysis import ComponentRuleConfig, analyze_components  # noqa: E402
from ore_classifier.component_analysis import crop_component  # noqa: E402
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

    def test_component_classifier_relabels_components_before_aggregation(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (30, 30), 255, thickness=-1)
        cv2.rectangle(mask, (60, 60), (82, 82), 255, thickness=-1)

        summary, components, classified = analyze_components(
            mask,
            config=ComponentRuleConfig(min_component_area_px=10),
            component_classifier=lambda found: ["fine_intergrowth" for _ in found],
        )

        self.assertEqual(summary.ore_class, "hard_to_process_ore")
        self.assertEqual({component.label for component in components}, {"fine_intergrowth"})
        self.assertEqual(int((classified == 1).sum()), 0)
        self.assertEqual(int((classified == 2).sum()), int((mask > 0).sum()))

    def test_component_classifier_must_return_one_label_per_component(self) -> None:
        mask = np.zeros((50, 50), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (30, 30), 255, thickness=-1)

        with self.assertRaises(ValueError):
            analyze_components(
                mask,
                config=ComponentRuleConfig(min_component_area_px=10),
                component_classifier=lambda found: [],
            )

    def test_talc_fraction_overrides_ore_class(self) -> None:
        sulfide = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(sulfide, (20, 20), (40, 40), 255, thickness=-1)
        talc = np.zeros_like(sulfide)
        talc[:40, :40] = 255
        summary, _, _ = analyze_components(sulfide, talc_mask=talc, config=ComponentRuleConfig(min_component_area_px=10))
        self.assertEqual(summary.ore_class, "talcose_ore")

    def test_analyzed_mask_controls_fraction_denominator(self) -> None:
        sulfide = np.zeros((100, 100), dtype=np.uint8)
        sulfide[10:20, 10:20] = 255
        analyzed = np.zeros_like(sulfide)
        analyzed[:50, :] = 255

        summary, _, _ = analyze_components(
            sulfide,
            analyzed_mask=analyzed,
            config=ComponentRuleConfig(min_component_area_px=1),
        )

        self.assertEqual(summary.image_area_px, 10000)
        self.assertEqual(summary.analysis_area_px, 5000)
        self.assertAlmostEqual(summary.sulfide_fraction, 100 / 5000)
        self.assertAlmostEqual(summary.sulfide_fraction_image, 100 / 10000)

    def test_fill_holes_does_not_fill_exterior_when_component_touches_border(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[0:10, 0:10] = 1
        filled = fill_holes(mask)
        self.assertEqual(int(filled[15, 15]), 0)
        self.assertEqual(int(filled[5, 5]), 1)

    def test_crop_component_limits_feature_work_to_padded_bbox(self) -> None:
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[90:100, 95:105] = 1
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        self.assertEqual(labels_count, 2)

        component, sulfide_crop = crop_component(labels, mask, 1, stats[1], close_kernel_px=7)

        self.assertEqual(component.shape, sulfide_crop.shape)
        self.assertLess(component.shape[0], mask.shape[0])
        self.assertLess(component.shape[1], mask.shape[1])
        self.assertEqual(int(component.sum()), 100)


if __name__ == "__main__":
    unittest.main()
