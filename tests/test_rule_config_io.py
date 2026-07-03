from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.rule_config_io import (  # noqa: E402
    load_rule_config,
    resolve_rule_config_from_args,
    rule_config_cli_args,
)


class RuleConfigIoTest(unittest.TestCase):
    def test_loads_best_config_from_calibration_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ore_rule_calibration.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "ore-rule-calibration-v0.1",
                        "best_config": {
                            "fine_dark_inside_ratio": 0.22,
                            "fine_solidity_max": 0.70,
                            "fine_compactness_max": 0.16,
                            "talc_fraction_threshold": 0.05,
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_rule_config(path)

        self.assertEqual(config["fine_dark_inside_ratio"], 0.22)
        self.assertEqual(config["fine_solidity_max"], 0.70)
        self.assertEqual(config["fine_compactness_max"], 0.16)
        self.assertEqual(config["talc_fraction_threshold"], 0.05)

    def test_cli_values_override_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "direct_rule_config.json"
            path.write_text(
                json.dumps(
                    {
                        "fine_dark_inside_ratio": 0.10,
                        "fine_solidity_max": 0.45,
                        "fine_compactness_max": 0.06,
                        "talc_fraction_threshold": 0.01,
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                rule_config_json=path,
                fine_dark_inside_ratio=None,
                fine_solidity_max=0.80,
                fine_compactness_max=None,
                talc_fraction_threshold=None,
            )

            config = resolve_rule_config_from_args(args)

        self.assertEqual(config["fine_dark_inside_ratio"], 0.10)
        self.assertEqual(config["fine_solidity_max"], 0.80)
        self.assertEqual(config["fine_compactness_max"], 0.06)
        self.assertEqual(config["talc_fraction_threshold"], 0.01)

    def test_rule_config_cli_args_use_public_flag_names(self) -> None:
        cli_args = rule_config_cli_args(
            {
                "fine_dark_inside_ratio": 0.1,
                "fine_solidity_max": 0.2,
                "fine_compactness_max": 0.3,
                "talc_fraction_threshold": 0.4,
            }
        )

        self.assertEqual(
            cli_args,
            [
                "--fine-dark-inside-ratio",
                "0.1",
                "--fine-solidity-max",
                "0.2",
                "--fine-compactness-max",
                "0.3",
                "--talc-fraction-threshold",
                "0.4",
            ],
        )

    def test_analyze_ore_from_masks_applies_rule_config_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.full((40, 40, 3), 180, dtype=np.uint8)
            sulfide = np.zeros((40, 40), dtype=np.uint8)
            sulfide[10:30, 10:30] = 255
            talc = np.zeros((40, 40), dtype=np.uint8)
            talc[:16, :16] = 255

            image_path = root / "image.png"
            sulfide_path = root / "sulfide.png"
            talc_path = root / "talc.png"
            config_path = root / "rule_config.json"
            out_dir = root / "out"
            Image.fromarray(image, mode="RGB").save(image_path)
            Image.fromarray(sulfide, mode="L").save(sulfide_path)
            Image.fromarray(talc, mode="L").save(talc_path)
            config_path.write_text(
                json.dumps(
                    {
                        "fine_dark_inside_ratio": 0.18,
                        "fine_solidity_max": 0.62,
                        "fine_compactness_max": 0.12,
                        "talc_fraction_threshold": 0.50,
                    }
                ),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/analyze_ore_from_masks.py"),
                    "--image",
                    str(image_path),
                    "--sulfide-mask",
                    str(sulfide_path),
                    "--talc-mask",
                    str(talc_path),
                    "--rule-config-json",
                    str(config_path),
                    "--min-component-area-px",
                    "1",
                    "--out-dir",
                    str(out_dir),
                ],
                check=True,
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
            )

            summary = json.loads((out_dir / "ore_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["ore_class"], "row_ore")
        self.assertAlmostEqual(summary["talc_fraction"], 256 / 1600)


if __name__ == "__main__":
    unittest.main()
