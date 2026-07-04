#!/usr/bin/env python3
"""Train the grain-level (ordinary vs fine intergrowth) classifier — path B, stage 3.

Reads the grain manifest from `build_grain_dataset.py` and, if present, human
grain annotations from `grain_review_web.py`. Labels come from human annotations
where available, otherwise the heuristic pre-label as a weak-supervision
bootstrap (unless --require-human). Evaluation is GroupKFold by specimen so all
photos of one аншлиф stay on one side of a fold. Exports the fitted model
(joblib) + metadata + metrics.

This produces the standalone grain model and its grain-level CV number for
inspection. The leak-free IMAGE-GRADE metric is produced separately by
`aggregate_grade_from_grains.py` (which retrains per fold to avoid same-image
leakage).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.grain_features import (  # noqa: E402
    FEATURE_NAMES,
    GRAIN_CLASS_ORDER,
    build_grain_feature_matrix,
    make_grain_model,
    resolve_grain_label,
)
from ore_classifier.tracking import add_mlflow_args, mlflow_run  # noqa: E402

try:
    import joblib
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError as exc:  # pragma: no cover - optional deps
    raise SystemExit("scikit-learn + joblib are required for train_grain_classifier.py") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="grains_manifest.csv from build_grain_dataset.py")
    parser.add_argument("--annotations", type=Path, default=None, help="annotations.json from grain_review_web.py (optional).")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--require-human", action="store_true", help="Train only on human-labelled grains (no heuristic bootstrap).")
    parser.add_argument("--models", nargs="+", default=["extra_trees", "random_forest", "logistic"])
    parser.add_argument("--folds", type=int, default=5)
    add_mlflow_args(parser, default_experiment="grain-classifier")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    annotations = load_annotations(args.annotations)
    labelled: list[dict[str, Any]] = []
    labels: list[str] = []
    label_source = Counter()
    for row in rows:
        label = resolve_grain_label(row, annotations, require_human=args.require_human)
        if label is None:
            continue
        labelled.append(row)
        labels.append(label)
        is_human = bool(annotations and str(row.get("grain_uid", "")) in annotations)
        label_source["human" if is_human else "heuristic_bootstrap"] += 1

    if len(labelled) < 10:
        raise SystemExit(f"too few labelled grains to train: {len(labelled)}")

    features = build_grain_feature_matrix(labelled)
    y = np.array([GRAIN_CLASS_ORDER.index(label) for label in labels], dtype=np.int64)
    groups = np.array([str(row.get("specimen_group", row.get("grain_uid", i))) for i, row in enumerate(labelled)])

    result = evaluate_models(features, y, groups, model_names=args.models, folds=args.folds)
    best_name = result["best_model"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    final_model = make_grain_model(best_name)
    final_model.fit(features, y)
    joblib.dump(final_model, args.out_dir / "model.joblib")

    metadata = {
        "schema_version": "grain-classifier-v0.1",
        "manifest": str(args.manifest),
        "annotations": str(args.annotations) if args.annotations else None,
        "class_order": GRAIN_CLASS_ORDER,
        "feature_names": FEATURE_NAMES,
        "sklearn_version": sklearn_version(),
        "label_source_counts": dict(label_source),
        "training_is_bootstrap": label_source.get("human", 0) == 0,
        "labelled_grains": len(labelled),
        "class_counts": {GRAIN_CLASS_ORDER[i]: int((y == i).sum()) for i in range(len(GRAIN_CLASS_ORDER))},
        "n_groups": int(len(set(groups.tolist()))),
        "best_model": best_name,
        "cv": result,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "metrics.md").write_text(render_md(metadata), encoding="utf-8")

    best_metrics = result["best_metrics"]
    with mlflow_run(
        args,
        params={
            "models": args.models,
            "folds": args.folds,
            "require_human": args.require_human,
            "best_model": best_name,
            "labelled_grains": len(labelled),
            "n_groups": metadata["n_groups"],
        },
    ) as run:
        run.log_metrics(
            {
                "cv_macro_f1": best_metrics.get("macro_f1", 0.0),
                "cv_accuracy": best_metrics.get("accuracy", 0.0),
            }
        )
        run.log_artifact(args.out_dir / "metadata.json")
        run.log_artifact(args.out_dir / "model.joblib")

    print(render_md(metadata))
    return 0


def evaluate_models(
    features: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    model_names: list[str],
    folds: int,
) -> dict[str, Any]:
    n_splits = grouped_n_splits(y, groups, folds)
    model_results = []
    for model_name in model_names:
        predicted = np.full_like(y, fill_value=-1)
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_idx, test_idx in splitter.split(features, y, groups):
            model = make_grain_model(model_name)
            model.fit(features[train_idx], y[train_idx])
            predicted[test_idx] = model.predict(features[test_idx])
        mask = predicted >= 0
        model_results.append({"model": model_name, "folds": n_splits, "metrics": binary_metrics(y[mask], predicted[mask])})
    model_results.sort(key=lambda item: float(item["metrics"]["macro_f1"]), reverse=True)
    return {"best_model": model_results[0]["model"], "best_metrics": model_results[0]["metrics"], "model_results": model_results}


def grouped_n_splits(y: np.ndarray, groups: np.ndarray, folds: int) -> int:
    # StratifiedGroupKFold needs >= n_splits distinct groups in EVERY class. Do NOT
    # floor at 2: with only one group in a class that silently produced an
    # empty-train fold (crash) or a single-class train fold (garbage macro-F1).
    # Fail loudly instead so the number is never misleading.
    min_groups_per_class = min(len(set(groups[y == cls].tolist())) for cls in np.unique(y))
    n_splits = min(folds, min_groups_per_class)
    if n_splits < 2:
        raise SystemExit(
            f"grouped CV needs >=2 specimen groups in the smallest class; got {min_groups_per_class}. "
            "Label grains spanning more аншлифы."
        )
    return n_splits


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(GRAIN_CLASS_ORDER))), zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(GRAIN_CLASS_ORDER))))
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            GRAIN_CLASS_ORDER[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(GRAIN_CLASS_ORDER))
        },
        "confusion_matrix": {
            GRAIN_CLASS_ORDER[i]: {GRAIN_CLASS_ORDER[j]: int(matrix[i, j]) for j in range(len(GRAIN_CLASS_ORDER))}
            for i in range(len(GRAIN_CLASS_ORDER))
        },
    }


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"manifest missing or empty: {path}")
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_annotations(path: Path | None) -> dict[str, dict[str, Any]] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("labels", {}) if isinstance(payload, dict) else None


def sklearn_version() -> str:
    try:
        import sklearn

        return sklearn.__version__
    except Exception:  # pragma: no cover
        return "unknown"


def render_md(metadata: dict[str, Any]) -> str:
    best = metadata["cv"]["best_metrics"]
    lines = [
        "# Grain Classifier (ordinary vs fine)",
        "",
        f"- Labelled grains: {metadata['labelled_grains']} ({metadata['label_source_counts']})",
        f"- Training is bootstrap (heuristic labels only): {metadata['training_is_bootstrap']}",
        f"- Specimen groups: {metadata['n_groups']}",
        f"- Best model: `{metadata['best_model']}`  |  grouped-CV macro-F1: {best['macro_f1']:.4f}, acc {best['accuracy']:.4f}",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in GRAIN_CLASS_ORDER:
        pc = best["per_class"][name]
        lines.append(f"| {name} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} | {pc['support']} |")
    lines += ["", "| Model | Folds | Macro F1 | Accuracy |", "| --- | ---: | ---: | ---: |"]
    for item in metadata["cv"]["model_results"]:
        m = item["metrics"]
        lines.append(f"| {item['model']} | {item['folds']} | {m['macro_f1']:.4f} | {m['accuracy']:.4f} |")
    if metadata["training_is_bootstrap"]:
        lines += ["", "> Bootstrap run on heuristic pre-labels — replace with human grain labels (grain_review_web.py) for the real gain."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
