"""Tests for optional MLflow tracking (src/ore_classifier/tracking.py).

The contract under test: training scripts behave identically when tracking is
off or mlflow is missing — every tracker call must be a harmless no-op.
"""
from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.tracking import (  # noqa: E402
    _MlflowTracker,
    _NullTracker,
    _scalarize,
    add_mlflow_args,
    mlflow_run,
)


class ScalarizeTest(unittest.TestCase):
    def test_scalars_pass_through(self) -> None:
        for value in (None, "text", True, 3, 2.5):
            self.assertEqual(_scalarize(value), value)

    def test_rich_values_become_json_strings(self) -> None:
        self.assertEqual(_scalarize({"a": 1}), '{"a": 1}')
        self.assertEqual(_scalarize([1, 2]), "[1, 2]")

    def test_long_values_are_truncated_to_250(self) -> None:
        text = _scalarize(list(range(200)))
        self.assertEqual(len(text), 250)
        self.assertTrue(text.endswith("..."))


class NullTrackerTest(unittest.TestCase):
    def test_disabled_and_all_methods_are_noops(self) -> None:
        tracker = _NullTracker()
        self.assertFalse(tracker.enabled)
        self.assertIsNone(tracker.log_params({"a": 1}))
        self.assertIsNone(tracker.log_metrics({"loss": 1.0}, step=3))
        self.assertIsNone(tracker.log_artifact("missing.txt"))
        self.assertIsNone(tracker.log_artifacts("missing_dir"))
        self.assertIsNone(tracker.log_dict({"a": 1}, "a.json"))


class AddMlflowArgsTest(unittest.TestCase):
    def test_registers_flags_with_defaults(self) -> None:
        parser = argparse.ArgumentParser()
        add_mlflow_args(parser, default_experiment="unit-test")
        args = parser.parse_args([])
        self.assertFalse(args.mlflow)
        self.assertEqual(args.mlflow_experiment, "unit-test")
        self.assertIsNone(args.mlflow_tracking_uri)
        self.assertIsNone(args.mlflow_run_name)
        enabled = parser.parse_args(["--mlflow", "--mlflow-run-name", "r1"])
        self.assertTrue(enabled.mlflow)
        self.assertEqual(enabled.mlflow_run_name, "r1")


class MlflowRunTest(unittest.TestCase):
    def test_flag_off_yields_null_tracker(self) -> None:
        args = argparse.Namespace(mlflow=False)
        with mlflow_run(args, params={"lr": 0.1}) as tracker:
            self.assertIsInstance(tracker, _NullTracker)

    def test_missing_args_attribute_yields_null_tracker(self) -> None:
        with mlflow_run(argparse.Namespace()) as tracker:
            self.assertIsInstance(tracker, _NullTracker)

    def test_mlflow_import_failure_falls_back_to_null_tracker(self) -> None:
        args = argparse.Namespace(mlflow=True)
        # None in sys.modules makes `import mlflow` raise ImportError even if installed
        with mock.patch.dict(sys.modules, {"mlflow": None}):
            with mlflow_run(args, params={"lr": 0.1}) as tracker:
                self.assertIsInstance(tracker, _NullTracker)


class MlflowTrackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mlflow = mock.Mock()
        self.tracker = _MlflowTracker(self.mlflow)

    def test_enabled_flag(self) -> None:
        self.assertTrue(self.tracker.enabled)

    def test_log_params_scalarizes_values(self) -> None:
        self.tracker.log_params({"lr": 0.1, "sizes": [1, 2]})
        self.mlflow.log_params.assert_called_once_with({"lr": 0.1, "sizes": "[1, 2]"})

    def test_log_metrics_keeps_only_numeric_non_bool(self) -> None:
        self.tracker.log_metrics({"loss": 1.5, "epoch": 3, "name": "x", "flag": True}, step=7)
        self.mlflow.log_metrics.assert_called_once_with({"loss": 1.5, "epoch": 3.0}, step=7)

    def test_log_metrics_skips_call_when_nothing_numeric(self) -> None:
        self.tracker.log_metrics({"name": "x"})
        self.mlflow.log_metrics.assert_not_called()

    def test_log_artifact_only_when_path_exists(self) -> None:
        self.tracker.log_artifact("definitely_missing_file.bin")
        self.mlflow.log_artifact.assert_not_called()
        self.tracker.log_artifact(__file__)
        self.mlflow.log_artifact.assert_called_once_with(__file__)

    def test_log_artifacts_only_for_directories(self) -> None:
        self.tracker.log_artifacts(__file__)  # a file, not a dir
        self.mlflow.log_artifacts.assert_not_called()
        directory = str(Path(__file__).parent)
        self.tracker.log_artifacts(directory)
        self.mlflow.log_artifacts.assert_called_once_with(directory)


if __name__ == "__main__":
    unittest.main()
