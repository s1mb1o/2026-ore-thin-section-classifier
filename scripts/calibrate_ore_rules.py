#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_ore_classification import CLASS_ORDER, evaluate_rows  # noqa: E402


DEFAULT_DARK_GRID = [0.10, 0.14, 0.18, 0.22, 0.28, 0.35]
DEFAULT_SOLIDITY_GRID = [0.45, 0.55, 0.62, 0.70, 0.80]
DEFAULT_COMPACTNESS_GRID = [0.06, 0.09, 0.12, 0.16, 0.22]
DEFAULT_TALC_GRID = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15]


@dataclass(frozen=True)
class RuleConfig:
    fine_dark_inside_ratio: float
    fine_solidity_max: float
    fine_compactness_max: float
    talc_fraction_threshold: float


@dataclass(frozen=True)
class ComponentRow:
    area_px: int
    dark_inside_ratio: float
    solidity: float
    compactness: float


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Grid-search deterministic ore classification thresholds from a completed "
            "official batch summary and per-run component_features.csv files."
        )
    )
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--fine-dark-grid", default=",".join(map(str, DEFAULT_DARK_GRID)))
    parser.add_argument("--fine-solidity-grid", default=",".join(map(str, DEFAULT_SOLIDITY_GRID)))
    parser.add_argument("--fine-compactness-grid", default=",".join(map(str, DEFAULT_COMPACTNESS_GRID)))
    parser.add_argument("--talc-threshold-grid", default=",".join(map(str, DEFAULT_TALC_GRID)))
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.summary_csv.open(encoding="utf-8")))
    row_components = load_component_tables(rows, summary_csv=args.summary_csv)
    configs = [
        RuleConfig(dark, solidity, compactness, talc)
        for dark, solidity, compactness, talc in product(
            parse_float_grid(args.fine_dark_grid),
            parse_float_grid(args.fine_solidity_grid),
            parse_float_grid(args.fine_compactness_grid),
            parse_float_grid(args.talc_threshold_grid),
        )
    ]
    result = calibrate_rules(rows, row_components, configs=configs, top_k=args.top_k)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def parse_float_grid(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("grid must contain at least one float value")
    return values


def load_component_tables(rows: list[dict[str, Any]], *, summary_csv: Path) -> dict[str, list[ComponentRow]]:
    by_run_id: dict[str, list[ComponentRow]] = {}
    for row in rows:
        run_id = str(row.get("run_id", ""))
        component_path = resolve_component_path(row, summary_csv=summary_csv)
        by_run_id[run_id] = read_component_rows(component_path)
    return by_run_id


def resolve_component_path(row: dict[str, Any], *, summary_csv: Path) -> Path:
    run_dir = Path(str(row.get("run_dir", "")))
    candidates: list[Path] = []
    if run_dir:
        candidates.append(run_dir / "ore_analysis/component_features.csv")
    source_label = str(row.get("source_label", ""))
    run_id = str(row.get("run_id", ""))
    if source_label and run_id:
        candidates.append(summary_csv.parent / "runs" / source_label / run_id / "ore_analysis/component_features.csv")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else Path("__missing_component_features.csv")


def read_component_rows(path: Path) -> list[ComponentRow]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    components: list[ComponentRow] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            components.append(
                ComponentRow(
                    area_px=int(float(row.get("area_px") or 0)),
                    dark_inside_ratio=float(row.get("dark_inside_ratio") or 0.0),
                    solidity=float(row.get("solidity") or 0.0),
                    compactness=float(row.get("compactness") or 0.0),
                )
            )
    return components


def calibrate_rules(
    rows: list[dict[str, Any]],
    row_components: dict[str, list[ComponentRow]],
    *,
    configs: list[RuleConfig],
    top_k: int,
) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for config in configs:
        predicted_rows = apply_config(rows, row_components, config)
        metrics = evaluate_rows(predicted_rows)
        scored.append(
            {
                "config": asdict(config),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "macro_auc_ovr": metrics["macro_auc_ovr"],
                "metrics": metrics,
            }
        )
    scored.sort(key=score_sort_key, reverse=True)
    best = scored[0] if scored else {}
    return {
        "schema_version": "ore-rule-calibration-v0.1",
        "rows_total": len(rows),
        "rows_used": best.get("metrics", {}).get("rows_used", 0),
        "configurations_tested": len(configs),
        "selection_order": ["macro_f1", "macro_auc_ovr", "accuracy", "weighted_f1"],
        "best_config": best.get("config", {}),
        "best_metrics": best.get("metrics", {}),
        "top_results": [{k: v for k, v in item.items() if k != "metrics"} for item in scored[: max(1, top_k)]],
        "note": (
            "Calibration uses image-level official folder labels and component CSVs from a completed batch. "
            "It does not create pixel-level geological ground truth and can overfit; keep the output as an explicit demo/report artifact."
        ),
    }


def apply_config(
    rows: list[dict[str, Any]],
    row_components: dict[str, list[ComponentRow]],
    config: RuleConfig,
) -> list[dict[str, Any]]:
    predicted: list[dict[str, Any]] = []
    for row in rows:
        components = row_components.get(str(row.get("run_id", "")), [])
        ordinary_area = 0
        fine_area = 0
        ordinary_count = 0
        fine_count = 0
        for component in components:
            is_fine = (
                component.dark_inside_ratio >= config.fine_dark_inside_ratio
                or component.solidity <= config.fine_solidity_max
                or component.compactness <= config.fine_compactness_max
            )
            if is_fine:
                fine_area += component.area_px
                fine_count += 1
            else:
                ordinary_area += component.area_px
                ordinary_count += 1
        classified_area = max(ordinary_area + fine_area, 1)
        talc_fraction = to_float(row.get("talc_fraction"))
        if talc_fraction > config.talc_fraction_threshold:
            ore_class = "talcose_ore"
        elif ordinary_area >= fine_area:
            ore_class = "row_ore"
        else:
            ore_class = "hard_to_process_ore"

        predicted.append(
            {
                **row,
                "predicted_ore_class": ore_class,
                "ordinary_sulfide_fraction": ordinary_area / classified_area,
                "fine_sulfide_fraction": fine_area / classified_area,
                "ordinary_component_count": ordinary_count,
                "fine_component_count": fine_count,
            }
        )
    return predicted


def score_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    macro_auc = item["macro_auc_ovr"]
    return (
        float(item["macro_f1"]),
        float(macro_auc if macro_auc is not None else -1.0),
        float(item["accuracy"]),
        float(item["weighted_f1"]),
    )


def render_markdown(result: dict[str, Any]) -> str:
    best = result.get("best_config", {})
    metrics = result.get("best_metrics", {})
    lines = [
        "# Ore Rule Calibration",
        "",
        f"- Rows used: {result.get('rows_used', 0)} / {result.get('rows_total', 0)}",
        f"- Configurations tested: {result.get('configurations_tested', 0)}",
        f"- Best macro F1: {metrics.get('macro_f1', 0.0):.4f}",
        f"- Best macro AUC OVR: {format_optional(metrics.get('macro_auc_ovr'))}",
        f"- Best accuracy: {metrics.get('accuracy', 0.0):.4f}",
        "",
        "## Best Config",
        "",
        "| Parameter | Value |",
        "| --- | ---: |",
    ]
    for key in [
        "fine_dark_inside_ratio",
        "fine_solidity_max",
        "fine_compactness_max",
        "talc_fraction_threshold",
    ]:
        lines.append(f"| `{key}` | {float(best.get(key, 0.0)):.4f} |")
    lines.extend(
        [
            "",
            "## Per-Class Metrics",
            "",
            "| Class | Support | Precision | Recall | F1 | AUC OVR |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for class_name in CLASS_ORDER:
        item = metrics.get("per_class", {}).get(class_name, {})
        lines.append(
            f"| {class_name} | {item.get('support', 0)} | {item.get('precision', 0.0):.4f} | "
            f"{item.get('recall', 0.0):.4f} | {item.get('f1', 0.0):.4f} | {format_optional(item.get('auc_ovr'))} |"
        )
    lines.extend(
        [
            "",
            "## Top Results",
            "",
            "| Rank | Macro F1 | Macro AUC | Accuracy | Talc thr | Dark ratio | Solidity | Compactness |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for index, item in enumerate(result.get("top_results", []), start=1):
        cfg = item["config"]
        lines.append(
            f"| {index} | {item['macro_f1']:.4f} | {format_optional(item['macro_auc_ovr'])} | "
            f"{item['accuracy']:.4f} | {cfg['talc_fraction_threshold']:.4f} | "
            f"{cfg['fine_dark_inside_ratio']:.4f} | {cfg['fine_solidity_max']:.4f} | "
            f"{cfg['fine_compactness_max']:.4f} |"
        )
    lines.extend(["", f"Note: {result.get('note', '')}", ""])
    return "\n".join(lines)


def to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def format_optional(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
