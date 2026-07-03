#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CLASS_ORDER = ["row_ore", "hard_to_process_ore", "talcose_ore"]
CLASS_RU = {
    "row_ore": "рядовая руда",
    "hard_to_process_ore": "труднообогатимая руда",
    "talcose_ore": "оталькованная руда",
}
LABEL_TO_ORE_CLASS = {
    "ordinary_intergrowth": "row_ore",
    "fine_intergrowth": "hard_to_process_ore",
    "talcose": "talcose_ore",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate image-level ore classification F1/AUC from a batch summary CSV.")
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.summary_csv.open(encoding="utf-8")))
    metrics = evaluate_rows(rows)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(metrics), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        true_class = row.get("expected_ore_class") or LABEL_TO_ORE_CLASS.get(str(row.get("source_label", "")), "")
        pred_class = str(row.get("predicted_ore_class", ""))
        if true_class not in CLASS_ORDER:
            skipped.append({"run_id": row.get("run_id", ""), "reason": "unknown_true_class"})
            continue
        usable.append({**row, "_true_class": true_class, "_pred_class": pred_class})

    matrix_columns = CLASS_ORDER + ["unknown"]
    confusion = {true_class: {pred_class: 0 for pred_class in matrix_columns} for true_class in CLASS_ORDER}
    for row in usable:
        true_class = row["_true_class"]
        pred_class = row["_pred_class"] if row["_pred_class"] in CLASS_ORDER else "unknown"
        confusion[true_class][pred_class] += 1

    per_class = {}
    total = len(usable)
    correct = 0
    for class_name in CLASS_ORDER:
        tp = confusion[class_name][class_name]
        fp = sum(confusion[other][class_name] for other in CLASS_ORDER if other != class_name)
        fn = sum(confusion[class_name][pred] for pred in matrix_columns if pred != class_name)
        support = sum(confusion[class_name].values())
        correct += tp
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        auc = binary_auc(
            labels=[1 if row["_true_class"] == class_name else 0 for row in usable],
            scores=[score_for_class(row, class_name) for row in usable],
        )
        per_class[class_name] = {
            "label_ru": CLASS_RU[class_name],
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc_ovr": auc,
        }

    supports = [per_class[class_name]["support"] for class_name in CLASS_ORDER]
    f1_values = [per_class[class_name]["f1"] for class_name in CLASS_ORDER]
    auc_values = [per_class[class_name]["auc_ovr"] for class_name in CLASS_ORDER if per_class[class_name]["auc_ovr"] is not None]
    weighted_f1 = safe_div(
        sum(per_class[class_name]["f1"] * per_class[class_name]["support"] for class_name in CLASS_ORDER),
        sum(supports),
    )
    return {
        "schema_version": "ore-classification-eval-v0.1",
        "class_order": CLASS_ORDER,
        "rows_total": len(rows),
        "rows_used": total,
        "rows_skipped": len(skipped),
        "skipped": skipped,
        "accuracy": safe_div(correct, total),
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "weighted_f1": weighted_f1,
        "macro_auc_ovr": sum(auc_values) / len(auc_values) if auc_values else None,
        "confusion_matrix": confusion,
        "per_class": per_class,
        "score_note": (
            "AUC uses deterministic rule scores from ore summary: talc_fraction for talcose, "
            "ordinary_sulfide_fraction for row ore, fine_sulfide_fraction for hard-to-process ore."
        ),
    }


def score_for_class(row: dict[str, Any], class_name: str) -> float:
    talc_fraction = to_float(row.get("talc_fraction"))
    ordinary_fraction = to_float(row.get("ordinary_sulfide_fraction"))
    fine_fraction = to_float(row.get("fine_sulfide_fraction"))
    sulfide_fraction = to_float(row.get("sulfide_fraction"))
    non_talc_weight = max(0.0, 1.0 - talc_fraction)
    if class_name == "talcose_ore":
        return talc_fraction
    if class_name == "row_ore":
        return non_talc_weight * ordinary_fraction * max(sulfide_fraction, 1e-6)
    if class_name == "hard_to_process_ore":
        return non_talc_weight * fine_fraction * max(sulfide_fraction, 1e-6)
    raise ValueError(f"unsupported class: {class_name}")


def binary_auc(labels: list[int], scores: list[float]) -> float | None:
    pos_count = sum(1 for label in labels if label == 1)
    neg_count = sum(1 for label in labels if label == 0)
    if pos_count == 0 or neg_count == 0:
        return None

    pairs = sorted(zip(scores, labels, strict=True), key=lambda item: item[0])
    rank_sum_pos = 0.0
    rank = 1
    index = 0
    while index < len(pairs):
        next_index = index + 1
        while next_index < len(pairs) and pairs[next_index][0] == pairs[index][0]:
            next_index += 1
        average_rank = (rank + rank + (next_index - index) - 1) / 2.0
        rank_sum_pos += average_rank * sum(1 for _, label in pairs[index:next_index] if label == 1)
        rank += next_index - index
        index = next_index
    return (rank_sum_pos - pos_count * (pos_count + 1) / 2.0) / float(pos_count * neg_count)


def render_markdown(metrics: dict[str, Any]) -> str:
    lines = [
        "# Ore Classification Evaluation",
        "",
        f"- Rows used: {metrics['rows_used']} / {metrics['rows_total']}",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Macro F1: {metrics['macro_f1']:.4f}",
        f"- Weighted F1: {metrics['weighted_f1']:.4f}",
        f"- Macro AUC OVR: {format_optional(metrics['macro_auc_ovr'])}",
        "",
        "## Per-Class Metrics",
        "",
        "| Class | Support | Precision | Recall | F1 | AUC OVR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for class_name in CLASS_ORDER:
        item = metrics["per_class"][class_name]
        lines.append(
            f"| {class_name} | {item['support']} | {item['precision']:.4f} | "
            f"{item['recall']:.4f} | {item['f1']:.4f} | {format_optional(item['auc_ovr'])} |"
        )
    lines.extend(["", "## Confusion Matrix", "", "Rows are true classes; columns are predicted classes.", ""])
    columns = CLASS_ORDER + ["unknown"]
    lines.append("| True \\ Pred | " + " | ".join(columns) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in columns) + " |")
    for true_class in CLASS_ORDER:
        values = [str(metrics["confusion_matrix"][true_class][pred_class]) for pred_class in columns]
        lines.append("| " + true_class + " | " + " | ".join(values) + " |")
    lines.extend(["", f"Note: {metrics['score_note']}", ""])
    return "\n".join(lines)


def to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
