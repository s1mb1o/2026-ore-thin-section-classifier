from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_official_batch import build_summary_row  # noqa: E402


class RunOfficialBatchTest(unittest.TestCase):
    def test_build_summary_row_includes_rule_config_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs/ordinary_intergrowth/example"
            binary_dir = run_dir / "binary_sulfide"
            ore_dir = run_dir / "ore_analysis"
            talc_dir = run_dir / "talc_candidate"
            for path in [binary_dir, ore_dir, talc_dir]:
                path.mkdir(parents=True, exist_ok=True)
            write_json(binary_dir / "summary.json", {"sulfide_fraction": 0.2, "seconds": 1.0})
            write_json(
                ore_dir / "ore_summary.json",
                {
                    "ore_class": "row_ore",
                    "ore_class_ru": "рядовая руда",
                    "sulfide_fraction": 0.2,
                    "ordinary_sulfide_fraction": 0.9,
                    "fine_sulfide_fraction": 0.1,
                    "talc_fraction": 0.01,
                    "component_count": 2,
                    "ordinary_component_count": 1,
                    "fine_component_count": 1,
                },
            )
            write_json(talc_dir / "talc_candidate_summary.json", {"talc_candidate_fraction": 0.01})
            write_json(
                run_dir / "pipeline_summary.json",
                {
                    "talc_source": "auto_candidate",
                    "rule_config": {
                        "fine_dark_inside_ratio": 0.22,
                        "fine_solidity_max": 0.70,
                        "fine_compactness_max": 0.16,
                        "talc_fraction_threshold": 0.05,
                    },
                    "paths": {
                        "binary_sulfide_summary": str(binary_dir / "summary.json"),
                        "ore_summary": str(ore_dir / "ore_summary.json"),
                        "talc_candidate_summary": str(talc_dir / "talc_candidate_summary.json"),
                    },
                },
            )

            row = build_summary_row(
                item={"label": "ordinary_intergrowth", "path": "sample.jpg", "width": 10, "height": 20},
                image_path=Path("dataset/sample.jpg"),
                run_dir=run_dir,
            )

        self.assertEqual(row["expected_ore_class"], "row_ore")
        self.assertEqual(row["rule_fine_dark_inside_ratio"], 0.22)
        self.assertEqual(row["rule_fine_solidity_max"], 0.70)
        self.assertEqual(row["rule_fine_compactness_max"], 0.16)
        self.assertEqual(row["rule_talc_fraction_threshold"], 0.05)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
