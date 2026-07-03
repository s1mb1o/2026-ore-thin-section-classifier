#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover - exercised only in missing optional envs.
    raise SystemExit("scikit-learn is required for evaluate_ore_feature_classifier.py") from exc


CLASS_ORDER = ["row_ore", "hard_to_process_ore", "talcose_ore"]
CLASS_RU = {
    "row_ore": "рядовая руда",
    "hard_to_process_ore": "труднообогатимая руда",
    "talcose_ore": "оталькованная руда",
}
BASE_FEATURES = [
    "width",
    "height",
    "sulfide_fraction",
    "ordinary_sulfide_fraction",
    "fine_sulfide_fraction",
    "talc_fraction",
    "talc_candidate_fraction",
    "component_count",
    "ordinary_component_count",
    "fine_component_count",
    "binary_sulfide_fraction",
    "binary_inference_seconds",
]
COMPONENT_FEATURES = [
    "area_px",
    "footprint_area_px",
    "dark_inside_area_px",
    "dark_inside_ratio",
    "solidity",
    "compactness",
    "boundary_complexity",
    "bbox_w",
    "bbox_h",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-validate a tabular ore classifier from pipeline summary and component aggregate features."
    )
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["extra_trees", "random_forest", "logistic"],
        choices=["extra_trees", "random_forest", "logistic"],
    )
    args = parser.parse_args()

    rows = list(csv.DictReader(args.summary_csv.open(encoding="utf-8", newline="")))
    feature_rows, labels, feature_names = build_feature_table(rows, summary_csv=args.summary_csv)
    result = evaluate_models(feature_rows, labels, feature_names, model_names=args.models, folds=args.folds)
    result["schema_version"] = "ore-feature-classifier-cv-v0.1"
    result["summary_csv"] = str(args.summary_csv)
    result["rows_total"] = len(rows)
    result["rows_used"] = len(labels)
    result["class_counts"] = dict(Counter(labels))
    result["note"] = (
        "This is an image-level cross-validation benchmark over features extracted from the segmentation pipeline. "
        "It is not a pixel-level geological ground truth score and should be reported separately from deterministic-rule metrics."
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_feature_table(
    rows: list[dict[str, Any]],
    *,
    summary_csv: Path,
) -> tuple[np.ndarray, list[str], list[str]]:
    feature_dicts: list[dict[str, float]] = []
    labels: list[str] = []
    for row in rows:
        label = str(row.get("expected_ore_class", ""))
        if label not in CLASS_ORDER:
            continue
        features = base_features(row)
        component_rows = read_component_rows(resolve_component_path(row, summary_csv=summary_csv))
        features.update(component_aggregate_features(component_rows))
        feature_dicts.append(features)
        labels.append(label)
    feature_names = sorted({key for features in feature_dicts for key in features})
    matrix = np.array([[features.get(key, 0.0) for key in feature_names] for features in feature_dicts], dtype=np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix, labels, feature_names


def base_features(row: dict[str, Any]) -> dict[str, float]:
    width = to_float(row.get("width"))
    height = to_float(row.get("height"))
    area = max(width * height, 1.0)
    features = {key: to_float(row.get(key)) for key in BASE_FEATURES}
    features.update(
        {
            "image_area_log": math.log1p(area),
            "aspect_ratio": width / max(height, 1.0),
            "ordinary_minus_fine_fraction": to_float(row.get("ordinary_sulfide_fraction"))
            - to_float(row.get("fine_sulfide_fraction")),
            "fine_to_ordinary_component_ratio": to_float(row.get("fine_component_count"))
            / max(to_float(row.get("ordinary_component_count")), 1.0),
        }
    )
    return features


def resolve_component_path(row: dict[str, Any], *, summary_csv: Path) -> Path:
    run_dir = Path(str(row.get("run_dir", "")))
    candidates: list[Path] = []
    if run_dir:
        candidates.append(run_dir / "ore_analysis/component_features.csv")
    source_label = str(row.get("source_label", ""))
    run_id = str(row.get("run_id", ""))
    if source_label and run_id:
        shard_dir = Path(str(row.get("shard_dir", "")))
        if str(shard_dir):
            candidates.append(shard_dir / "runs" / source_label / run_id / "ore_analysis/component_features.csv")
        candidates.append(summary_csv.parent / "runs" / source_label / run_id / "ore_analysis/component_features.csv")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else Path("__missing_component_features.csv")


def read_component_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({key: value if key == "label" else to_float(value) for key, value in row.items()})
    return rows


def component_aggregate_features(rows: list[dict[str, Any]]) -> dict[str, float]:
    features: dict[str, float] = {"component_table_count": float(len(rows))}
    if not rows:
        for name in COMPONENT_FEATURES:
            for suffix in ["mean", "std", "min", "max", "p25", "p50", "p75", "weighted_mean"]:
                features[f"component_{name}_{suffix}"] = 0.0
        return features

    weights = np.array([max(row.get("area_px", 0.0), 0.0) for row in rows], dtype=np.float64)
    weight_sum = float(weights.sum())
    for name in COMPONENT_FEATURES:
        values = np.array([row.get(name, 0.0) for row in rows], dtype=np.float64)
        features[f"component_{name}_mean"] = float(values.mean())
        features[f"component_{name}_std"] = float(values.std())
        features[f"component_{name}_min"] = float(values.min())
        features[f"component_{name}_max"] = float(values.max())
        features[f"component_{name}_p25"] = float(np.percentile(values, 25))
        features[f"component_{name}_p50"] = float(np.percentile(values, 50))
        features[f"component_{name}_p75"] = float(np.percentile(values, 75))
        features[f"component_{name}_weighted_mean"] = float(np.dot(values, weights) / weight_sum) if weight_sum > 0 else 0.0
    fine_count = sum(1 for row in rows if str(row.get("label", "")) == "fine_intergrowth")
    ordinary_count = sum(1 for row in rows if str(row.get("label", "")) == "ordinary_intergrowth")
    features["component_fine_label_fraction"] = fine_count / max(len(rows), 1)
    features["component_ordinary_label_fraction"] = ordinary_count / max(len(rows), 1)
    return features


def evaluate_models(
    features: np.ndarray,
    labels: list[str],
    feature_names: list[str],
    *,
    model_names: list[str],
    folds: int,
) -> dict[str, Any]:
    y = np.array([CLASS_ORDER.index(label) for label in labels], dtype=np.int64)
    min_class_count = min(Counter(y).values())
    n_splits = max(2, min(folds, min_class_count))
    model_results = []
    for model_name in model_names:
        predicted = np.zeros_like(y)
        probabilities = np.zeros((len(y), len(CLASS_ORDER)), dtype=np.float64)
        split = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_idx, test_idx in split.split(features, y):
            model = make_model(model_name)
            model.fit(features[train_idx], y[train_idx])
            predicted[test_idx] = model.predict(features[test_idx])
            probabilities[test_idx] = model.predict_proba(features[test_idx])
        metrics = classification_metrics(y, predicted, probabilities)
        final_model = make_model(model_name)
        final_model.fit(features, y)
        model_results.append(
            {
                "model": model_name,
                "folds": n_splits,
                "metrics": metrics,
                "top_features": top_features(final_model, feature_names),
            }
        )
    model_results.sort(
        key=lambda item: (
            float(item["metrics"]["macro_f1"]),
            float(item["metrics"]["macro_auc_ovr"] if item["metrics"]["macro_auc_ovr"] is not None else -1.0),
            float(item["metrics"]["accuracy"]),
        ),
        reverse=True,
    )
    return {"best_model": model_results[0]["model"], "best_metrics": model_results[0]["metrics"], "model_results": model_results}


def make_model(name: str) -> Any:
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=300, random_state=42, class_weight="balanced", min_samples_leaf=2, n_jobs=-1)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", min_samples_leaf=2, n_jobs=-1)
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        )
    raise ValueError(f"unknown model: {name}")


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_ORDER))),
        zero_division=0,
    )
    try:
        macro_auc = float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro", labels=list(range(len(CLASS_ORDER)))))
    except ValueError:
        macro_auc = None
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_ORDER))))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)),
        "macro_auc_ovr": macro_auc,
        "confusion_matrix": {
            CLASS_ORDER[i]: {CLASS_ORDER[j]: int(matrix[i, j]) for j in range(len(CLASS_ORDER))}
            for i in range(len(CLASS_ORDER))
        },
        "per_class": {
            CLASS_ORDER[i]: {
                "label_ru": CLASS_RU[CLASS_ORDER[i]],
                "support": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
            for i in range(len(CLASS_ORDER))
        },
    }


def top_features(model: Any, feature_names: list[str], limit: int = 20) -> list[dict[str, Any]]:
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("logisticregression", model)
    if hasattr(estimator, "feature_importances_"):
        scores = np.asarray(estimator.feature_importances_, dtype=np.float64)
    elif hasattr(estimator, "coef_"):
        scores = np.abs(np.asarray(estimator.coef_, dtype=np.float64)).mean(axis=0)
    else:
        return []
    order = np.argsort(scores)[::-1][:limit]
    return [{"feature": feature_names[int(i)], "importance": float(scores[int(i)])} for i in order]


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Ore Feature Classifier CV",
        "",
        f"- Rows used: {result.get('rows_used', 0)} / {result.get('rows_total', 0)}",
        f"- Best model: `{result.get('best_model', '')}`",
        f"- Best macro F1: {result.get('best_metrics', {}).get('macro_f1', 0.0):.4f}",
        f"- Best macro AUC OVR: {format_optional(result.get('best_metrics', {}).get('macro_auc_ovr'))}",
        f"- Best accuracy: {result.get('best_metrics', {}).get('accuracy', 0.0):.4f}",
        "",
        "## Model Comparison",
        "",
        "| Model | Folds | Accuracy | Macro F1 | Weighted F1 | Macro AUC OVR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in result.get("model_results", []):
        metrics = item["metrics"]
        lines.append(
            f"| {item['model']} | {item['folds']} | {metrics['accuracy']:.4f} | "
            f"{metrics['macro_f1']:.4f} | {metrics['weighted_f1']:.4f} | {format_optional(metrics['macro_auc_ovr'])} |"
        )
    lines.extend(["", "## Best Per-Class Metrics", "", "| Class | Support | Precision | Recall | F1 |", "| --- | ---: | ---: | ---: | ---: |"])
    for class_name in CLASS_ORDER:
        item = result.get("best_metrics", {}).get("per_class", {}).get(class_name, {})
        lines.append(
            f"| {class_name} | {item.get('support', 0)} | {item.get('precision', 0.0):.4f} | "
            f"{item.get('recall', 0.0):.4f} | {item.get('f1', 0.0):.4f} |"
        )
    lines.extend(["", "## Top Features", "", "| Rank | Feature | Importance |", "| ---: | --- | ---: |"])
    best = result.get("model_results", [{}])[0]
    for index, item in enumerate(best.get("top_features", []), start=1):
        lines.append(f"| {index} | `{item['feature']}` | {item['importance']:.6f} |")
    lines.extend(["", f"Note: {result.get('note', '')}", ""])
    return "\n".join(lines)


def to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def format_optional(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
