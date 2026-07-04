from __future__ import annotations

import csv
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from apps.ore_pipeline_web import OrePipelineStore, render_html_page  # noqa: E402
from ore_classifier.component_analysis import ComponentRuleConfig, analyze_components, write_component_csv  # noqa: E402


class OrePipelineGrainReportingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="test_ore_pipeline_grain_reporting_"))
        self.store = OrePipelineStore(
            workspace_dir=self.root / "workspace",
            backend="heuristic",
            checkpoint=None,
            processing_max_side=128,
            panorama_max_side=128,
            preview_max_sides=(128,),
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_component_csv_includes_perimeter_px_for_new_runs(self) -> None:
        mask = np.zeros((32, 32), dtype=np.uint8)
        cv2.rectangle(mask, (8, 8), (19, 19), 255, thickness=-1)

        _, components, _ = analyze_components(mask, config=ComponentRuleConfig(min_component_area_px=1))

        csv_path = self.root / "component_features.csv"
        write_component_csv(csv_path, components)
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertIn("perimeter_px", rows[0])
        self.assertGreater(float(rows[0]["perimeter_px"]), 0.0)

    def test_sulfide_grain_payload_returns_om_proxy_fields(self) -> None:
        run_id = "grain_report"
        run_dir = self.store.runs_dir / run_id
        reports_dir = run_dir / "reports"
        masks_dir = run_dir / "masks"
        reports_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        sulfide = np.zeros((12, 12), dtype=np.uint8)
        sulfide[3:6, 3:6] = 255
        talc = np.zeros_like(sulfide)
        talc[3:6, 6] = 255
        sulfide_path = masks_dir / "sulfide_mask.png"
        talc_path = masks_dir / "talc_mask.png"
        Image.fromarray(sulfide, mode="L").save(sulfide_path)
        Image.fromarray(talc, mode="L").save(talc_path)

        csv_path = reports_dir / "component_features.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "component_id",
                    "label",
                    "area_px",
                    "boundary_complexity",
                    "bbox_x",
                    "bbox_y",
                    "bbox_w",
                    "bbox_h",
                    "centroid_x",
                    "centroid_y",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "component_id": 1,
                    "label": "ordinary_intergrowth",
                    "area_px": 9,
                    "boundary_complexity": 6.0,
                    "bbox_x": 3,
                    "bbox_y": 3,
                    "bbox_w": 3,
                    "bbox_h": 3,
                    "centroid_x": 4,
                    "centroid_y": 4,
                }
            )

        payload = self.store._sulfide_grains_payload(
            run_id,
            {
                "reports": {"component_features_csv": str(csv_path)},
                "masks": {"sulfide": str(sulfide_path), "talc": str(talc_path)},
            },
            {"sulfide_area_px": 9},
        )

        self.assertEqual(payload["schema_version"], "ore-pipeline-sulfide-grains-v0.1")
        self.assertEqual(payload["share_denominator_px"], 9)
        self.assertTrue((masks_dir / "sulfide_component_labels_rgb.png").exists())
        self.assertEqual(len(payload["items"]), 1)
        grain = payload["items"][0]
        self.assertAlmostEqual(grain["equivalent_diameter_px"], math.sqrt(4 * 9 / math.pi))
        self.assertAlmostEqual(grain["perimeter_px"], 18.0)
        self.assertAlmostEqual(grain["sulfide_area_share"], 1.0)
        self.assertAlmostEqual(grain["share_percent"], 100.0)
        self.assertGreater(grain["contacts"]["matrix_px"], 0)
        self.assertGreater(grain["contacts"]["talc_px"], 0)
        self.assertEqual(grain["contacts"]["other_contact_px"], 0)
        self.assertAlmostEqual(
            grain["liberation_proxy"],
            grain["contacts"]["matrix_px"] / grain["contacts"]["total_px"],
        )
        self.assertTrue(grain["locked_composite_proxy"])
        self.assertIn("other_contact", grain["association_percentages"])

    def test_static_ui_exposes_enriched_grain_report_columns(self) -> None:
        html = render_html_page()

        self.assertIn("sulfideGrainsHeaderDiameter", html)
        self.assertIn("sulfideGrainsHeaderPerimeter", html)
        self.assertIn("sulfideGrainsHeaderLiberation", html)
        self.assertIn("sulfideGrainsHeaderContacts", html)
        self.assertIn("sulfideGrainsHeaderLocked", html)
        self.assertIn("function formatGrainContacts", html)
        self.assertIn("liberation_proxy", html)
        self.assertIn("locked_composite_proxy", html)
        self.assertIn("OM-mask proxies", html)


if __name__ == "__main__":
    unittest.main()
