from __future__ import annotations

import csv
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_ore_rules import RuleConfig, calibrate_rules, load_component_tables  # noqa: E402


class OreRuleCalibrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / "outputs/test_ore_rule_calibration"
        shutil.rmtree(self.root, ignore_errors=True)
        self.batch_dir = self.root / "batch"
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.summary_csv = self.batch_dir / "summary.csv"
        rows = [
            self._write_run(
                run_id="ordinary_a",
                source_label="ordinary_intergrowth",
                expected="row_ore",
                talc_fraction=0.0,
                components=[{"area_px": 1000, "dark_inside_ratio": 0.04, "solidity": 0.91, "compactness": 0.42}],
            ),
            self._write_run(
                run_id="fine_a",
                source_label="fine_intergrowth",
                expected="hard_to_process_ore",
                talc_fraction=0.0,
                components=[{"area_px": 1000, "dark_inside_ratio": 0.24, "solidity": 0.88, "compactness": 0.39}],
            ),
            self._write_run(
                run_id="talc_a",
                source_label="talcose",
                expected="talcose_ore",
                talc_fraction=0.12,
                components=[{"area_px": 1000, "dark_inside_ratio": 0.03, "solidity": 0.92, "compactness": 0.41}],
            ),
        ]
        with self.summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_run(
        self,
        *,
        run_id: str,
        source_label: str,
        expected: str,
        talc_fraction: float,
        components: list[dict[str, float]],
    ) -> dict[str, str]:
        run_dir = self.batch_dir / "runs" / source_label / run_id
        feature_path = run_dir / "ore_analysis/component_features.csv"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["area_px", "dark_inside_ratio", "solidity", "compactness"]
        with feature_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(components)
        return {
            "run_id": run_id,
            "source_label": source_label,
            "expected_ore_class": expected,
            "sulfide_fraction": "0.25",
            "ordinary_sulfide_fraction": "0",
            "fine_sulfide_fraction": "0",
            "talc_fraction": str(talc_fraction),
            "run_dir": str(run_dir),
        }

    def test_calibration_finds_perfect_synthetic_rule(self) -> None:
        with self.summary_csv.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        components = load_component_tables(rows, summary_csv=self.summary_csv)

        result = calibrate_rules(
            rows,
            components,
            configs=[
                RuleConfig(
                    fine_dark_inside_ratio=0.18,
                    fine_solidity_max=0.62,
                    fine_compactness_max=0.12,
                    talc_fraction_threshold=0.10,
                )
            ],
            top_k=3,
        )

        self.assertEqual(result["rows_used"], 3)
        self.assertEqual(result["configurations_tested"], 1)
        self.assertAlmostEqual(result["best_metrics"]["macro_f1"], 1.0)
        self.assertEqual(result["best_metrics"]["confusion_matrix"]["talcose_ore"]["talcose_ore"], 1)


if __name__ == "__main__":
    unittest.main()
