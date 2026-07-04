"""Tests for the official pipeline evaluation harness (scripts/evaluate_official_pipeline.py).

The harness is a thin orchestrator over tested step scripts; these tests cover
its own logic: settings loading, split subset selection, the perturbed-dataset
builder, and the combined metrics summary/markdown rendering.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_official_pipeline as eop  # noqa: E402


class LoadSettingsTest(unittest.TestCase):
    def test_none_returns_none(self) -> None:
        self.assertIsNone(eop.load_settings(None, dict))

    def test_inline_json_is_normalized(self) -> None:
        marker = []

        def normalizer(payload):
            marker.append(payload)
            return {"normalized": True}

        result = eop.load_settings('{"enabled": true}', normalizer)
        self.assertEqual(result, {"normalized": True})
        self.assertEqual(marker, [{"enabled": True}])

    def test_file_path_is_read_and_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text('{"enabled": false}', encoding="utf-8")
            result = eop.load_settings(str(path), lambda payload: payload)
        self.assertEqual(result, {"enabled": False})

    def test_invalid_value_exits(self) -> None:
        with self.assertRaises(SystemExit):
            eop.load_settings("definitely not json and not a file", dict)


class SelectItemsTest(unittest.TestCase):
    ITEMS = [
        {"path": "a1.jpg", "label": "row_ore"},
        {"path": "a2.jpg", "label": "row_ore"},
        {"path": "b1.jpg", "label": "talcose_ore"},
        {"path": "b2.jpg", "label": "talcose_ore"},
        {"path": "c1.jpg", "label": "hard_to_process_ore"},
    ]

    def test_no_filters_returns_all(self) -> None:
        self.assertEqual(eop.select_items(self.ITEMS, labels=None, per_label=None, max_total=None), self.ITEMS)

    def test_labels_filter(self) -> None:
        selected = eop.select_items(self.ITEMS, labels={"talcose_ore"}, per_label=None, max_total=None)
        self.assertEqual([item["path"] for item in selected], ["b1.jpg", "b2.jpg"])

    def test_per_label_cap(self) -> None:
        selected = eop.select_items(self.ITEMS, labels=None, per_label=1, max_total=None)
        self.assertEqual([item["path"] for item in selected], ["a1.jpg", "b1.jpg", "c1.jpg"])

    def test_max_total_stops_early(self) -> None:
        selected = eop.select_items(self.ITEMS, labels=None, per_label=None, max_total=2)
        self.assertEqual([item["path"] for item in selected], ["a1.jpg", "a2.jpg"])


class BuildTransformedDatasetTest(unittest.TestCase):
    def test_perturbs_split_images_preserving_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_root = tmp_path / "dataset"
            dest_root = tmp_path / "transformed"
            rel = Path("grade_a/img_1.JPG")
            (source_root / rel).parent.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(5)
            rgb = rng.integers(30, 220, size=(40, 50, 3), dtype=np.uint8)
            Image.fromarray(rgb, mode="RGB").save(source_root / rel)
            split_json = tmp_path / "split.json"
            split_json.write_text(
                json.dumps({"items": [{"path": str(rel), "label": "row_ore"}]}),
                encoding="utf-8",
            )
            from ore_classifier.augmentation import normalize_augmentation_settings

            augmentation = normalize_augmentation_settings(
                {"enabled": True, "color": {"brightness_pct": 30}, "runtime": {"random_seed": 3}}
            )
            report = eop.build_transformed_dataset(
                split_json=split_json,
                source_root=source_root,
                dest_root=dest_root,
                augmentation=augmentation,
                preprocess=None,
                labels=None,
                per_label=None,
                max_total=None,
            )
            self.assertEqual(report["transformed_images"], 1)
            out_path = dest_root / rel
            self.assertTrue(out_path.exists())  # original suffix preserved
            with Image.open(out_path) as image:
                self.assertEqual(image.format, "PNG")  # lossless content despite .JPG name
                transformed = np.asarray(image.convert("RGB")).astype(float)
            self.assertGreater(transformed.mean(), rgb.astype(float).mean())  # brightness applied


class SummaryRenderingTest(unittest.TestCase):
    def make_summary(self, *, perturbed: bool) -> dict:
        args = argparse.Namespace(checkpoint=Path("models/best.pt"), split_json=Path("split.json"))
        rule_metrics = {
            "rows_used": 6,
            "accuracy": 0.5,
            "macro_f1": 0.4,
            "weighted_f1": 0.45,
            "macro_auc_ovr": None,
            "per_class": {
                "row_ore": {"f1": 0.6, "precision": 0.7, "recall": 0.5},
                "hard_to_process_ore": {"f1": 0.3, "precision": 0.4, "recall": 0.25},
                # talcose_ore intentionally missing -> None fields
            },
            "confusion_matrix": [[1, 0], [0, 1]],
        }
        feature_metrics = {
            "best_model": "extra_trees",
            "best_metrics": {
                "accuracy": 0.7,
                "macro_f1": 0.65,
                "weighted_f1": 0.66,
                "macro_auc_ovr": 0.8,
                "per_class": {"row_ore": {"f1": 0.71}},
            },
        }
        return eop.build_combined_summary(
            args=args,
            dataset_root=Path("dataset"),
            augmentation={"enabled": True} if perturbed else None,
            preprocess=None,
            transform_report={"augmentation_applied": perturbed, "preprocessing_applied": False},
            rule_metrics=rule_metrics,
            feature_metrics=feature_metrics,
        )

    def test_build_combined_summary_shapes_metrics(self) -> None:
        summary = self.make_summary(perturbed=False)
        self.assertEqual(summary["schema_version"], "official-pipeline-eval-v0.1")
        self.assertEqual(summary["rows_used"], 6)
        rule = summary["deterministic_rule_metrics"]
        self.assertEqual(rule["accuracy"], 0.5)
        self.assertEqual(rule["per_class"]["row_ore"]["f1"], 0.6)
        self.assertIsNone(rule["per_class"]["talcose_ore"]["f1"])
        feat = summary["feature_classifier_cv_metrics"]
        self.assertEqual(feat["best_model"], "extra_trees")
        self.assertEqual(feat["per_class"]["row_ore"]["f1"], 0.71)
        self.assertIsNone(feat["per_class"]["talcose_ore"]["f1"])

    def test_render_summary_md_baseline_and_none_values(self) -> None:
        markdown = eop.render_summary_md(self.make_summary(perturbed=False))
        self.assertIn("# Official Pipeline Evaluation", markdown)
        self.assertIn("baseline (no perturbation)", markdown)
        self.assertIn("Accuracy: 0.5000", markdown)
        self.assertIn("Macro AUC OVR: n/a", markdown)  # None renders as n/a
        for name in eop.CLASS_ORDER:
            self.assertIn(f"| {name} |", markdown)

    def test_render_summary_md_labels_perturbed_condition(self) -> None:
        markdown = eop.render_summary_md(self.make_summary(perturbed=True))
        self.assertIn("perturbed: augmentation", markdown)

    def test_fmt(self) -> None:
        self.assertEqual(eop.fmt(None), "n/a")
        self.assertEqual(eop.fmt(0.12344), "0.1234")
        self.assertEqual(eop.fmt(1), "1.0000")


if __name__ == "__main__":
    unittest.main()
