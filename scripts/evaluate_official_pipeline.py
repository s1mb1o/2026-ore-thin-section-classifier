#!/usr/bin/env python3
"""End-to-end official ore-pipeline evaluation harness.

One command that runs OUR pipeline over the official dataset and prints OUR
metrics on a leak-free, deconflicted balanced split:

    dataset -> manifest -> label audit -> deconflicted balanced split
            -> (optional augmentation + preprocessing perturbation)
            -> full sulfide/talc/ore pipeline (run_official_batch.py)
            -> deterministic-rule metrics + feature-classifier CV metrics
            -> combined metrics_summary.json / .md

"Images present in multiple variants" (identical content filed under conflicting
grade folders, plus exact duplicate content) are excluded up front by building
the split with --exclude-conflicts --dedupe-sha256, the same deconflicting the
project uses for its preferred 345-image evaluation set.

Robustness testing (requirement #5): pass --augmentation-json and/or
--preprocess-json (a file path or an inline JSON string) to perturb every split
image before inference with the exact same augmentation
(ore_classifier.augmentation) and preprocessing (ore_classifier.preprocessing)
used by the browser UI, then re-measure. Omit both for the clean baseline.

This is a thin orchestrator: it shells out to the existing, tested step scripts
rather than reimplementing them, so results match the manual command sequence in
COMMANDS.md.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.augmentation import (  # noqa: E402
    apply_augmentation,
    augmentation_enabled,
    normalize_augmentation_settings,
)
from ore_classifier.preprocessing import (  # noqa: E402
    normalize_preprocess_settings,
    preprocess_image,
    preprocessing_enabled,
)

Image.MAX_IMAGE_PIXELS = None

CLASS_ORDER = ["row_ore", "hard_to_process_ore", "talcose_ore"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Binary sulfide segmentation checkpoint.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Destination for batch runs and metrics.")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--official-manifest", type=Path, default=ROOT / "outputs/official_manifest.json")
    parser.add_argument("--label-audit-dir", type=Path, default=ROOT / "outputs/official_label_audit")
    parser.add_argument("--split-json", type=Path, default=ROOT / "outputs/official_balanced_eval_split_deconflicted.json")
    parser.add_argument("--split-csv", type=Path, default=ROOT / "outputs/official_balanced_eval_split_deconflicted.csv")
    # Perturbation settings for robustness testing (requirement #5).
    parser.add_argument("--augmentation-json", default=None, help="Path to JSON file or inline JSON string of augmentation settings.")
    parser.add_argument("--preprocess-json", default=None, help="Path to JSON file or inline JSON string of preprocessing settings.")
    # Rebuild controls (default: reuse prerequisite artifacts if present).
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--rebuild-audit", action="store_true")
    parser.add_argument("--rebuild-split", action="store_true")
    parser.add_argument("--overwrite-batch", action="store_true", help="Re-run inference even if per-image outputs exist.")
    parser.add_argument("--skip-inference", action="store_true", help="Reuse an existing summary.csv in --out-dir; only re-evaluate.")
    # Subset controls (smoke).
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--per-label", type=int, default=None)
    parser.add_argument("--max-total", type=int, default=None)
    # Pipeline knobs (passed through to run_official_batch.py).
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-component-area-px", type=int, default=128)
    parser.add_argument("--close-kernel-px", type=int, default=21)
    parser.add_argument("--talc-min-area-px", type=int, default=320)
    parser.add_argument("--preview-max-side", type=int, default=1800)
    parser.add_argument("--no-auto-talc-candidate", action="store_true")
    parser.add_argument("--talc-checkpoint", type=Path, default=None, help="Trained talc segmentation checkpoint (e.g. SegFormer-B0) used instead of the color auto-candidate.")
    parser.add_argument("--talc-threshold", type=float, default=0.5)
    parser.add_argument("--rule-config-json", type=Path, default=None)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--resident",
        action="store_true",
        help="Use the single-load resident batch (scripts/run_resident_batch.py) instead of the per-image subprocess batch.",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    augmentation = load_settings(args.augmentation_json, normalize_augmentation_settings)
    preprocess = load_settings(args.preprocess_json, normalize_preprocess_settings)
    aug_on = bool(augmentation and augmentation_enabled(augmentation))
    pre_on = bool(preprocess and preprocessing_enabled(preprocess))
    perturbed = aug_on or pre_on

    # 1-3. Prerequisite artifacts: manifest -> audit -> deconflicted split.
    ensure_manifest(args)
    ensure_audit(args)
    ensure_split(args)

    # 4. Optional perturbation -> a transformed dataset root mirroring split paths.
    dataset_root = args.dataset_root
    transform_report: dict[str, Any] = {"augmentation_applied": aug_on, "preprocessing_applied": pre_on}
    if perturbed and not args.skip_inference:
        dataset_root = out_dir / "transformed_dataset"
        transform_report.update(
            build_transformed_dataset(
                split_json=args.split_json,
                source_root=args.dataset_root,
                dest_root=dataset_root,
                augmentation=augmentation if aug_on else None,
                preprocess=preprocess if pre_on else None,
                labels=set(args.labels) if args.labels else None,
                per_label=args.per_label,
                max_total=args.max_total,
            )
        )

    # 5. Run the full pipeline over the split (unless reusing an existing batch).
    summary_csv = out_dir / "summary.csv"
    if args.skip_inference:
        if not summary_csv.exists():
            raise SystemExit(f"--skip-inference set but {summary_csv} is missing")
        print(f"[skip-inference] reusing {summary_csv}", flush=True)
    else:
        run_batch(args, dataset_root=dataset_root, out_dir=out_dir)

    # 6. Evaluate: deterministic-rule metrics + feature-classifier cross-validation.
    rule_json = out_dir / "ore_classification_metrics.json"
    rule_md = out_dir / "ore_classification_metrics.md"
    run([
        sys.executable, "scripts/evaluate_ore_classification.py",
        "--summary-csv", str(summary_csv),
        "--out-json", str(rule_json),
        "--out-md", str(rule_md),
    ])
    feat_json = out_dir / "ore_feature_classifier_cv.json"
    feat_md = out_dir / "ore_feature_classifier_cv.md"
    run([
        sys.executable, "scripts/evaluate_ore_feature_classifier.py",
        "--summary-csv", str(summary_csv),
        "--out-json", str(feat_json),
        "--out-md", str(feat_md),
        "--folds", str(args.cv_folds),
    ])

    # 7. Combined metrics summary.
    summary = build_combined_summary(
        args=args,
        dataset_root=dataset_root,
        augmentation=augmentation,
        preprocess=preprocess,
        transform_report=transform_report,
        rule_metrics=read_json(rule_json),
        feature_metrics=read_json(feat_json),
    )
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "metrics_summary.md").write_text(render_summary_md(summary), encoding="utf-8")
    print("\n" + render_summary_md(summary))
    return 0


def load_settings(value: str | None, normalizer: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    candidate = Path(value)
    if candidate.exists():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"could not read settings (not a file, not valid JSON): {value}") from exc
    return normalizer(payload)


def ensure_manifest(args: argparse.Namespace) -> None:
    if args.official_manifest.exists() and not args.rebuild_manifest:
        return
    run([
        sys.executable, "scripts/build_official_manifest.py",
        "--dataset-root", str(args.dataset_root),
        "--out", str(args.official_manifest),
    ])


def ensure_audit(args: argparse.Namespace) -> None:
    audit_summary = args.label_audit_dir / "summary.json"
    if audit_summary.exists() and not args.rebuild_audit:
        return
    run([
        sys.executable, "scripts/audit_official_labels.py",
        "--official-manifest", str(args.official_manifest),
        "--dataset-root", str(args.dataset_root),
        "--out-dir", str(args.label_audit_dir),
    ])


def ensure_split(args: argparse.Namespace) -> None:
    if args.split_json.exists() and not args.rebuild_split:
        return
    run([
        sys.executable, "scripts/build_official_balanced_eval_split.py",
        "--official-manifest", str(args.official_manifest),
        "--label-audit-json", str(args.label_audit_dir / "summary.json"),
        "--exclude-conflicts",
        "--dedupe-sha256",
        "--out-json", str(args.split_json),
        "--out-csv", str(args.split_csv),
    ])


def build_transformed_dataset(
    *,
    split_json: Path,
    source_root: Path,
    dest_root: Path,
    augmentation: dict[str, Any] | None,
    preprocess: dict[str, Any] | None,
    labels: set[str] | None,
    per_label: int | None,
    max_total: int | None,
) -> dict[str, Any]:
    split = json.loads(split_json.read_text(encoding="utf-8"))
    items = select_items(split.get("items", []), labels=labels, per_label=per_label, max_total=max_total)
    written = 0
    for index, item in enumerate(items, start=1):
        rel_path = Path(item["path"])
        src = source_root / rel_path
        dst = dest_root / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        image = Image.open(src).convert("RGB")
        if augmentation is not None:
            image = apply_augmentation(image, augmentation)
        if preprocess is not None:
            image = preprocess_image(image, preprocess)
        # Lossless write; PIL detects format by content, so the original suffix
        # (e.g. .JPG) is preserved for run_official_batch path reconstruction
        # while avoiding a JPEG re-compression confound in the perturbation test.
        image.save(dst, format="PNG", compress_level=1)
        written += 1
        if index % 25 == 0 or index == len(items):
            print(f"[transform] {index}/{len(items)} images perturbed", flush=True)
    return {"transformed_images": written, "transformed_dataset": str(dest_root)}


def select_items(
    items: list[dict[str, Any]],
    labels: set[str] | None,
    per_label: int | None,
    max_total: int | None,
) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    selected: list[dict[str, Any]] = []
    for item in items:
        label = item.get("label", "")
        if labels is not None and label not in labels:
            continue
        if per_label is not None and counts.get(label, 0) >= per_label:
            continue
        selected.append(item)
        counts[label] = counts.get(label, 0) + 1
        if max_total is not None and len(selected) >= max_total:
            break
    return selected


def run_batch(args: argparse.Namespace, *, dataset_root: Path, out_dir: Path) -> None:
    batch_script = "scripts/run_resident_batch.py" if args.resident else "scripts/run_official_batch.py"
    cmd = [
        sys.executable, batch_script,
        "--split-json", str(args.split_json),
        "--dataset-root", str(dataset_root),
        "--checkpoint", str(args.checkpoint),
        "--out-dir", str(out_dir),
        "--tile-size", str(args.tile_size),
        "--stride", str(args.stride),
        "--batch-size", str(args.batch_size),
        "--device", args.device,
        "--threshold", str(args.threshold),
        "--min-component-area-px", str(args.min_component_area_px),
        "--close-kernel-px", str(args.close_kernel_px),
        "--talc-min-area-px", str(args.talc_min_area_px),
        "--preview-max-side", str(args.preview_max_side),
        "--keep-going",
    ]
    if args.labels:
        cmd.extend(["--labels", *args.labels])
    if args.per_label is not None:
        cmd.extend(["--per-label", str(args.per_label)])
    if args.max_total is not None:
        cmd.extend(["--max-total", str(args.max_total)])
    if args.talc_checkpoint is not None:
        cmd.extend(["--talc-checkpoint", str(args.talc_checkpoint), "--talc-threshold", str(args.talc_threshold), "--no-auto-talc-candidate"])
    elif args.no_auto_talc_candidate:
        cmd.append("--no-auto-talc-candidate")
    if args.overwrite_batch:
        cmd.append("--overwrite")
    if args.rule_config_json is not None:
        cmd.extend(["--rule-config-json", str(args.rule_config_json)])
    run(cmd)


def build_combined_summary(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    augmentation: dict[str, Any] | None,
    preprocess: dict[str, Any] | None,
    transform_report: dict[str, Any],
    rule_metrics: dict[str, Any],
    feature_metrics: dict[str, Any],
) -> dict[str, Any]:
    best = feature_metrics.get("best_metrics", {})
    return {
        "schema_version": "official-pipeline-eval-v0.1",
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(dataset_root),
        "split_json": str(args.split_json),
        "rows_used": rule_metrics.get("rows_used"),
        "perturbation": {
            **transform_report,
            "augmentation": augmentation,
            "preprocess": preprocess,
        },
        "deterministic_rule_metrics": {
            "accuracy": rule_metrics.get("accuracy"),
            "macro_f1": rule_metrics.get("macro_f1"),
            "weighted_f1": rule_metrics.get("weighted_f1"),
            "macro_auc_ovr": rule_metrics.get("macro_auc_ovr"),
            "per_class": {
                name: {
                    "f1": rule_metrics.get("per_class", {}).get(name, {}).get("f1"),
                    "precision": rule_metrics.get("per_class", {}).get(name, {}).get("precision"),
                    "recall": rule_metrics.get("per_class", {}).get(name, {}).get("recall"),
                }
                for name in CLASS_ORDER
            },
            "confusion_matrix": rule_metrics.get("confusion_matrix"),
        },
        "feature_classifier_cv_metrics": {
            "best_model": feature_metrics.get("best_model"),
            "accuracy": best.get("accuracy"),
            "macro_f1": best.get("macro_f1"),
            "weighted_f1": best.get("weighted_f1"),
            "macro_auc_ovr": best.get("macro_auc_ovr"),
            "per_class": {
                name: {"f1": best.get("per_class", {}).get(name, {}).get("f1")}
                for name in CLASS_ORDER
            },
        },
    }


def render_summary_md(summary: dict[str, Any]) -> str:
    rule = summary["deterministic_rule_metrics"]
    feat = summary["feature_classifier_cv_metrics"]
    pert = summary["perturbation"]
    pert_label = "baseline (no perturbation)"
    if pert.get("augmentation_applied") or pert.get("preprocessing_applied"):
        parts = []
        if pert.get("augmentation_applied"):
            parts.append("augmentation")
        if pert.get("preprocessing_applied"):
            parts.append("preprocessing")
        pert_label = "perturbed: " + " + ".join(parts)
    lines = [
        "# Official Pipeline Evaluation",
        "",
        f"- Images used: {summary.get('rows_used')}",
        f"- Checkpoint: `{Path(summary['checkpoint']).name}`",
        f"- Condition: {pert_label}",
        "",
        "## Deterministic rule pipeline (image-level, folder-label GT)",
        "",
        f"- Accuracy: {fmt(rule['accuracy'])}",
        f"- Macro F1: {fmt(rule['macro_f1'])}",
        f"- Weighted F1: {fmt(rule['weighted_f1'])}",
        f"- Macro AUC OVR: {fmt(rule['macro_auc_ovr'])}",
        "",
        "| Class | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in CLASS_ORDER:
        pc = rule["per_class"][name]
        lines.append(f"| {name} | {fmt(pc['precision'])} | {fmt(pc['recall'])} | {fmt(pc['f1'])} |")
    lines += [
        "",
        "## Feature classifier (5-fold CV over pipeline features, folder-label GT)",
        "",
        f"- Best model: `{feat.get('best_model')}`",
        f"- Accuracy: {fmt(feat['accuracy'])}",
        f"- Macro F1: {fmt(feat['macro_f1'])}",
        f"- Weighted F1: {fmt(feat['weighted_f1'])}",
        f"- Macro AUC OVR: {fmt(feat['macro_auc_ovr'])}",
        "",
        "| Class | F1 |",
        "| --- | ---: |",
    ]
    for name in CLASS_ORDER:
        lines.append(f"| {name} | {fmt(feat['per_class'][name]['f1'])} |")
    lines += [
        "",
        "> GT = official grade-folder label propagated to every photo of the аншлиф; "
        "conflicting/duplicate content excluded. Report both metrics: the rule pipeline is "
        "the pure deterministic result; the feature-CV is the learnable ceiling from the same "
        "pipeline features.",
        "",
    ]
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
