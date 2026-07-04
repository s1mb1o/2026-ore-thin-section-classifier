"""Optional MLflow experiment tracking for the training scripts (debug only).

Off by default. Enable a run with ``--mlflow``. When the flag is absent, or when
mlflow is not installed, every tracker call is a no-op and the scripts behave
exactly as before — their JSON/CSV/checkpoint outputs under ``--out-dir`` are
untouched. MLflow is a *dev* dependency (``requirements-dev.txt``); it is never
required to run training.

The default store is a local file store at ``<repo>/mlruns``. Browse it with
``mlflow ui`` from the repo root, then open http://127.0.0.1:5000.

Usage in a train script::

    from ore_classifier.tracking import add_mlflow_args, mlflow_run

    add_mlflow_args(parser, default_experiment="grade-classifier")
    ...
    with mlflow_run(args, params={"lr": args.lr, "epochs": args.epochs}) as run:
        for epoch in ...:
            run.log_metrics({"train_loss": loss, "val_macro_f1": f1}, step=epoch)
        run.log_artifacts(args.out_dir)
"""
from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_DIR = ROOT / "mlruns"


def add_mlflow_args(parser: argparse.ArgumentParser, *, default_experiment: str) -> None:
    """Register the shared, optional --mlflow* flags on a parser."""
    group = parser.add_argument_group("mlflow (debug tracking, off by default)")
    group.add_argument("--mlflow", action="store_true", help="Log this run to MLflow (debug only).")
    group.add_argument("--mlflow-experiment", default=default_experiment, help="MLflow experiment name.")
    group.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="MLflow tracking URI (default: local ./mlruns file store).",
    )
    group.add_argument("--mlflow-run-name", default=None, help="Optional MLflow run name.")


def _scalarize(value: Any) -> Any:
    """MLflow params must be scalar-ish; stringify+truncate anything richer."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= 250 else text[:247] + "..."


class _NullTracker:
    """No-op tracker used when tracking is disabled or unavailable."""

    enabled = False

    def log_params(self, params: dict[str, Any]) -> None:  # noqa: D102
        return None

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:  # noqa: D102
        return None

    def log_artifact(self, path: Any) -> None:  # noqa: D102
        return None

    def log_artifacts(self, directory: Any) -> None:  # noqa: D102
        return None

    def log_dict(self, obj: Any, artifact_file: str) -> None:  # noqa: D102
        return None


class _MlflowTracker:
    """Thin wrapper that forwards to mlflow while sanitising inputs."""

    enabled = True

    def __init__(self, mlflow: Any) -> None:
        self._mlflow = mlflow

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params({k: _scalarize(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        numeric = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if numeric:
            self._mlflow.log_metrics(numeric, step=step)

    def log_artifact(self, path: Any) -> None:
        p = Path(path)
        if p.exists():
            self._mlflow.log_artifact(str(p))

    def log_artifacts(self, directory: Any) -> None:
        d = Path(directory)
        if d.is_dir():
            self._mlflow.log_artifacts(str(d))

    def log_dict(self, obj: Any, artifact_file: str) -> None:
        self._mlflow.log_dict(obj, artifact_file)


@contextmanager
def mlflow_run(args: argparse.Namespace, *, params: dict[str, Any] | None = None) -> Iterator[Any]:
    """Context manager yielding a tracker.

    Yields a no-op tracker (and leaves outputs untouched) when ``--mlflow`` is
    not set or mlflow is missing; otherwise opens an mlflow run, logs ``params``,
    and closes the run on exit.
    """
    if not getattr(args, "mlflow", False):
        yield _NullTracker()
        return
    # The local file store is in maintenance mode on mlflow>=3; opt back in so the
    # default ./mlruns store works for debug without a database backend.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        import mlflow
    except ImportError:
        print(
            "[mlflow] --mlflow set but mlflow is not installed; skipping tracking "
            "(pip install -r requirements-dev.txt). Training continues normally.",
            flush=True,
        )
        yield _NullTracker()
        return

    uri = getattr(args, "mlflow_tracking_uri", None) or DEFAULT_TRACKING_DIR.as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(getattr(args, "mlflow_experiment", None) or "ore-classifier")
    tracker = _MlflowTracker(mlflow)
    with mlflow.start_run(run_name=getattr(args, "mlflow_run_name", None)):
        print(
            f"[mlflow] logging run to {uri} "
            f"(experiment={getattr(args, 'mlflow_experiment', 'ore-classifier')})",
            flush=True,
        )
        if params:
            tracker.log_params(params)
        yield tracker
