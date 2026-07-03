from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_ore_feature_classifier import build_feature_table, evaluate_models  # noqa: E402


class OreFeatureClassifierTest(unittest.TestCase):
    def test_build_feature_table_reads_component_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs/ordinary_intergrowth/example"
            component_path = run_dir / "ore_analysis/component_features.csv"
            component_path.parent.mkdir(parents=True, exist_ok=True)
            write_csv(
                component_path,
                [
                    {
                        "label": "ordinary_intergrowth",
                        "area_px": "100",
                        "footprint_area_px": "120",
                        "dark_inside_area_px": "5",
                        "dark_inside_ratio": "0.04",
                        "solidity": "0.9",
                        "compactness": "0.8",
                        "boundary_complexity": "4.0",
                        "bbox_w": "10",
                        "bbox_h": "12",
                    }
                ],
            )
            summary_csv = root / "summary.csv"
            row = summary_row(run_dir=run_dir, label="ordinary_intergrowth", expected="row_ore", run_id="example")
            write_csv(summary_csv, [row])

            matrix, labels, feature_names = build_feature_table([row], summary_csv=summary_csv)

        self.assertEqual(labels, ["row_ore"])
        self.assertEqual(matrix.shape[0], 1)
        self.assertIn("component_area_px_mean", feature_names)
        self.assertAlmostEqual(matrix[0, feature_names.index("component_area_px_mean")], 100.0)

    def test_evaluate_models_reports_cross_validated_metrics(self) -> None:
        features = []
        labels = []
        for idx, label in enumerate(["row_ore", "hard_to_process_ore", "talcose_ore"]):
            for offset in range(4):
                features.append([float(idx), float(offset), float(idx * 10 + offset)])
                labels.append(label)

        result = evaluate_models(
            features=np.asarray(features, dtype=float),
            labels=labels,
            feature_names=["a", "b", "c"],
            model_names=["extra_trees"],
            folds=2,
        )

        self.assertEqual(result["best_model"], "extra_trees")
        self.assertIn("macro_f1", result["best_metrics"])


def summary_row(*, run_dir: Path, label: str, expected: str, run_id: str) -> dict[str, str]:
    return {
        "run_id": run_id,
        "source_label": label,
        "expected_ore_class": expected,
        "run_dir": str(run_dir),
        "width": "100",
        "height": "80",
        "sulfide_fraction": "0.2",
        "ordinary_sulfide_fraction": "0.8",
        "fine_sulfide_fraction": "0.2",
        "talc_fraction": "0.01",
        "talc_candidate_fraction": "0.01",
        "component_count": "1",
        "ordinary_component_count": "1",
        "fine_component_count": "0",
        "binary_sulfide_fraction": "0.2",
        "binary_inference_seconds": "1.0",
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
