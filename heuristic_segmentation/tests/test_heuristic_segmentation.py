from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heuristic_segmentation import (  # noqa: E402
    CLASS_FINE_INTERGROWTH,
    CLASS_ORDINARY_INTERGROWTH,
    CLASS_TALC_CANDIDATE,
    HeuristicConfig,
    make_overlay,
    segment_image,
)
from heuristic_segmentation.segmentation import (  # noqa: E402
    _classify_sulfide_components,
    _convex_hull_area,
    _ellipse_kernel,
)


class HeuristicSegmentationTest(unittest.TestCase):
    def test_segments_compact_fragmented_and_talc_candidate_regions(self) -> None:
        image = np.full((128, 160, 3), (42, 54, 38), dtype=np.uint8)
        image[18:46, 96:140] = (105, 158, 112)
        image[54:92, 20:60] = (235, 228, 162)
        image[54:60, 34:43] = (45, 50, 42)
        image[98:104, 84:91] = (232, 230, 180)
        image[110:116, 105:112] = (232, 230, 180)

        result = segment_image(
            image,
            HeuristicConfig(
                min_component_area=20,
                fine_max_area_px=70,
                fine_min_replacement_ratio=0.12,
                talc_min_area=120,
            ),
        )

        self.assertGreater(int((result.class_mask == CLASS_ORDINARY_INTERGROWTH).sum()), 0)
        self.assertGreater(int((result.class_mask == CLASS_FINE_INTERGROWTH).sum()), 0)
        self.assertGreater(int((result.class_mask == CLASS_TALC_CANDIDATE).sum()), 0)
        self.assertEqual(result.class_mask.shape, image.shape[:2])
        self.assertGreaterEqual(result.metrics["component_count"], 2)

    def test_overlay_preserves_image_shape(self) -> None:
        image = np.full((32, 40, 3), 50, dtype=np.uint8)
        result = segment_image(image, HeuristicConfig(min_component_area=5))
        overlay = make_overlay(image, result.class_mask)
        self.assertEqual(overlay.shape, image.shape)
        self.assertEqual(overlay.dtype, np.uint8)

    def test_component_classification_roi_matches_full_frame_reference(self) -> None:
        sulfide_mask = np.zeros((180, 220), dtype=np.uint8)
        analyzed_mask = np.ones_like(sulfide_mask, dtype=np.uint8)
        sulfide_mask[4:34, 3:45] = 1
        sulfide_mask[14:24, 17:31] = 0
        sulfide_mask[92:122, 114:152] = 1
        sulfide_mask[126:132, 165:172] = 1
        sulfide_mask[150:158, 12:21] = 1

        config = HeuristicConfig(
            fine_max_area_px=80,
            fine_min_replacement_ratio=0.12,
            fine_max_solidity=0.68,
            fine_max_compactness=0.18,
            footprint_close_radius=7,
        )
        expected_mask, expected_components = _classify_sulfide_components_reference(
            sulfide_mask=sulfide_mask,
            analyzed_mask=analyzed_mask,
            config=config,
        )
        actual_mask, actual_components = _classify_sulfide_components(
            sulfide_mask=sulfide_mask,
            analyzed_mask=analyzed_mask,
            config=config,
        )

        np.testing.assert_array_equal(actual_mask, expected_mask)
        self.assertEqual(actual_components, expected_components)


def _classify_sulfide_components_reference(
    *,
    sulfide_mask: np.ndarray,
    analyzed_mask: np.ndarray,
    config: HeuristicConfig,
) -> tuple[np.ndarray, list[dict]]:
    class_mask = np.zeros(sulfide_mask.shape, dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sulfide_mask, 8)
    components: list[dict] = []
    footprint_kernel = _ellipse_kernel(config.footprint_close_radius)

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        component = labels == label_id
        contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
        hull_area = _convex_hull_area(contours)
        solidity = float(area / hull_area) if hull_area > 0 else 0.0
        compactness = float(4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0

        footprint = cv2.morphologyEx(component.astype(np.uint8), cv2.MORPH_CLOSE, footprint_kernel).astype(bool)
        footprint &= analyzed_mask.astype(bool)
        footprint_area = int(footprint.sum())
        internal_dark_area = int(np.logical_and(footprint, ~component).sum())
        replacement_ratio = float(internal_dark_area / footprint_area) if footprint_area > 0 else 0.0

        is_fine = (
            area <= config.fine_max_area_px
            or replacement_ratio >= config.fine_min_replacement_ratio
            or solidity <= config.fine_max_solidity
            or compactness <= config.fine_max_compactness
        )
        class_id = CLASS_FINE_INTERGROWTH if is_fine else CLASS_ORDINARY_INTERGROWTH
        class_mask[component] = class_id
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        components.append(
            {
                "component_id": label_id,
                "class_id": class_id,
                "class_label": "fine_intergrowth" if is_fine else "ordinary_intergrowth",
                "area_px": area,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "centroid_x": round(float(centroids[label_id][0]), 3),
                "centroid_y": round(float(centroids[label_id][1]), 3),
                "perimeter_px": round(perimeter, 3),
                "solidity": round(solidity, 6),
                "compactness": round(compactness, 6),
                "footprint_area_px": footprint_area,
                "internal_dark_area_px": internal_dark_area,
                "replacement_ratio": round(replacement_ratio, 6),
            }
        )
    return class_mask, components


if __name__ == "__main__":
    unittest.main()
