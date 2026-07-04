#!/usr/bin/env python3
"""Aggregate grain predictions into an image grade and evaluate it — path B, stage 4.

Leak-free, grouped-by-specimen nested CV over the images of a completed batch:

  for each specimen-grouped fold:
    1. train the grain classifier on TRAIN-image grains only (labels: human
       annotations if present, else heuristic bootstrap);
    2. predict P(fine) for every grain of every image in the fold;
    3. per image, area-weighted fine_fraction = Σ area·P(fine) / Σ area;
    4. talc_fraction: the trained talc segmentation model when --talc-checkpoint
       is given (recommended — the colour auto-candidate is ≈0 for talcose), else
       the batch's ore_summary.json auto-candidate;
    5. calibrate (τ_fine, τ_talc) on TRAIN images to maximise grade macro-F1;
    6. predict TEST-image grades with:
         talc_fraction ≥ τ_talc → talcose_ore
         elif fine_fraction ≥ τ_fine → hard_to_process_ore
         else → row_ore

Because BOTH the grain model and the thresholds are fit on train images only, the
resulting image-level grade macro-F1 is leak-free and directly comparable to the
harness (rule 0.185 / feature-CV 0.747) and competitor A (0.88).

GT grade is the folder label (`expected_ore_class`) from the batch summary.csv.
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
    GRAIN_CLASS_ORDER,
    build_grain_feature_matrix,
    grain_feature_vector,
    make_grain_model,
    resolve_grain_label,
)
from ore_classifier.specimen import specimen_group  # noqa: E402

try:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scikit-learn is required for aggregate_grade_from_grains.py") from exc

GRADE_ORDER = ["row_ore", "hard_to_process_ore", "talcose_ore"]
GRADE_RU = {"row_ore": "рядовая", "hard_to_process_ore": "труднообогатимая", "talcose_ore": "оталькованная"}
FINE_INDEX = GRAIN_CLASS_ORDER.index("fine_intergrowth")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-dir", type=Path, required=True, help="Completed official batch dir (summary.csv + runs/).")
    parser.add_argument("--manifest", type=Path, required=True, help="grains_manifest.csv (source of grain training labels).")
    parser.add_argument("--annotations", type=Path, default=None, help="annotations.json from grain_review_web.py (optional).")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--require-human", action="store_true")
    parser.add_argument("--grain-model", default="extra_trees")
    parser.add_argument("--folds", type=int, default=5)
    # Talcose branch (v0.2): score talc with the trained talc segmentation model
    # instead of the batch's colour auto-candidate (which is ≈0 for talcose images).
    parser.add_argument("--talc-checkpoint", type=Path, default=None, help="Trained talc segmentation checkpoint; overrides the auto-candidate talc_fraction.")
    parser.add_argument("--talc-threshold", type=float, default=0.5)
    parser.add_argument("--talc-device", default="auto")
    parser.add_argument("--talc-tile-size", type=int, default=1024)
    parser.add_argument("--talc-stride", type=int, default=768)
    parser.add_argument("--talc-batch-size", type=int, default=4)
    args = parser.parse_args()

    talc_scorer = None
    if args.talc_checkpoint is not None:
        if not args.talc_checkpoint.exists():
            raise SystemExit(f"talc checkpoint not found: {args.talc_checkpoint}")
        talc_scorer = TalcModelScorer(
            checkpoint=args.talc_checkpoint,
            device=args.talc_device,
            threshold=args.talc_threshold,
            tile=args.talc_tile_size,
            stride=args.talc_stride,
            batch=args.talc_batch_size,
        )
        print(f"[talc] scoring talcose with trained model {talc_scorer.meta.get('model')} on {talc_scorer.device}", flush=True)

    images = load_images(args.batch_dir, talc_scorer=talc_scorer)
    if not images:
        raise SystemExit(f"no usable images with grains under {args.batch_dir}")
    train_labels_by_run = build_grain_label_index(args.manifest, args.annotations, require_human=args.require_human)

    fine_grid = [round(x, 3) for x in np.linspace(0.15, 0.85, 15)]
    talc_grid = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]

    y = np.array([GRADE_ORDER.index(img["grade"]) for img in images], dtype=np.int64)
    groups = np.array([img["specimen_group"] for img in images])
    n_splits = grouped_n_splits(y, groups, args.folds)

    oof_pred: list[str | None] = [None] * len(images)
    fold_thresholds = []
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold_i, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(images)), y, groups)):
        train_runs = {images[i]["run_id"] for i in train_idx}
        model, bootstrap = fit_fold_grain_model(train_runs, train_labels_by_run, args.grain_model)
        # P(fine) per image (train+test needed: train for calibration, test for OOF).
        fine_fraction = {i: image_fine_fraction(images[i], model) for i in list(train_idx) + list(test_idx)}
        tau_fine, tau_talc = calibrate_thresholds(
            [images[i] for i in train_idx],
            {i: fine_fraction[i] for i in train_idx},
            fine_grid,
            talc_grid,
        )
        fold_thresholds.append({"fold": fold_i, "tau_fine": tau_fine, "tau_talc": tau_talc, "bootstrap": bootstrap})
        for i in test_idx:
            oof_pred[i] = predict_grade(fine_fraction[i], images[i]["talc_fraction"], tau_fine, tau_talc)

    metrics = grade_metrics(
        [img["grade"] for img in images],
        oof_pred,
        count_fine=[img.get("count_fine_fraction", 0.0) for img in images],
    )
    any_human = any(v["source"] == "human" for run in train_labels_by_run.values() for v in run)
    talc_sources = Counter(img.get("talc_source", "unknown") for img in images)
    summary = {
        "schema_version": "grain-grade-aggregation-v0.2",
        "batch_dir": str(args.batch_dir),
        "images_used": len(images),
        "specimen_groups": int(len(set(groups.tolist()))),
        "folds": n_splits,
        "grain_model": args.grain_model,
        "training_is_bootstrap": not any_human,
        "talc_checkpoint": str(args.talc_checkpoint) if args.talc_checkpoint else None,
        "talc_source_counts": dict(talc_sources),
        "fold_thresholds": fold_thresholds,
        "grade_metrics": metrics,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "grade_from_grains.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "grade_from_grains.md").write_text(render_md(summary), encoding="utf-8")
    print(render_md(summary))
    return 0


def load_images(batch_dir: Path, *, talc_scorer: "TalcModelScorer | None" = None) -> list[dict[str, Any]]:
    summary_csv = batch_dir / "summary.csv"
    if not summary_csv.exists():
        raise SystemExit(f"batch summary not found: {summary_csv}")
    images: list[dict[str, Any]] = []
    for row in csv.DictReader(summary_csv.open(encoding="utf-8")):
        grade = str(row.get("expected_ore_class", ""))
        if grade not in GRADE_ORDER:
            continue
        run_dir = resolve_run_dir(Path(str(row.get("run_dir", ""))))
        grains = read_component_grains(run_dir / "ore_analysis" / "component_features.csv")
        if not grains:
            continue
        features = build_grain_feature_matrix(grains)
        areas = np.array([float(g.get("area_px", 0) or 0) for g in grains], dtype=np.float64)
        talc_fraction, talc_source = resolve_talc_fraction(row, run_dir, talc_scorer)
        heuristic_fine = np.array([1.0 if str(g.get("label")) == "fine_intergrowth" else 0.0 for g in grains])
        count_fine_fraction = float(heuristic_fine.mean()) if len(heuristic_fine) else 0.0
        images.append(
            {
                "run_id": run_dir.name,
                "grade": grade,
                "specimen_group": specimen_group(str(row.get("source_rel_path", run_dir.name))),
                "features": features,
                "areas": areas,
                "talc_fraction": talc_fraction,
                "talc_source": talc_source,
                "count_fine_fraction": count_fine_fraction,
            }
        )
    return images


def resolve_talc_fraction(row: dict[str, Any], run_dir: Path, talc_scorer: "TalcModelScorer | None") -> tuple[float, str]:
    """Talcose signal: the trained talc model if provided (and its inputs exist),
    else the batch's auto-candidate talc_fraction from ore_summary.json."""
    if talc_scorer is not None:
        from PIL import Image

        source_path = Path(str(row.get("source_dataset_path", "")))
        sulfide_path = run_dir / "binary_sulfide" / "sulfide_mask.png"
        analyzed_path = run_dir / "binary_sulfide" / "analyzed_mask.png"
        if source_path.exists() and sulfide_path.exists() and analyzed_path.exists():
            sulfide = np.asarray(Image.open(sulfide_path).convert("L"))
            analyzed = np.asarray(Image.open(analyzed_path).convert("L"))
            return talc_scorer.talc_fraction(source_path, sulfide, analyzed), "trained_model"
        return read_talc_fraction(run_dir / "ore_analysis" / "ore_summary.json"), "ore_summary_fallback"
    return read_talc_fraction(run_dir / "ore_analysis" / "ore_summary.json"), "ore_summary_auto_candidate"


class TalcModelScorer:
    """Loads the trained talc segmentation model once and computes a per-image
    talc_fraction = talc∩non-sulfide pixels / analyzed pixels, matching the ore
    rule's fraction (identical tiling to the resident pipeline). torch is imported
    lazily so the default (auto-candidate) path needs no torch."""

    def __init__(self, checkpoint: Path, *, device: str, threshold: float, tile: int, stride: int, batch: int) -> None:
        import torch

        from ore_classifier.model_io import load_binary_segmentation_checkpoint, resolve_device
        from ore_classifier.resident_pipeline import _tile_weight

        self._torch = torch
        self.device = resolve_device(device)
        self.model, self.meta = load_binary_segmentation_checkpoint(Path(checkpoint), self.device)
        self.model.eval()
        self.threshold = threshold
        self.tile = tile
        self.stride = stride
        self.batch = batch
        self.weight = _tile_weight(tile)

    def talc_fraction(self, image_path: Path, sulfide_mask: np.ndarray, analyzed_mask: np.ndarray) -> float:
        from PIL import Image

        from ore_classifier.model_io import forward_logits
        from ore_classifier.resident_pipeline import _batched, _preprocess_tile
        from ore_classifier.tiling import iter_tiles

        torch = self._torch
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        tiles = iter_tiles(width=w, height=h, tile_size=self.tile, stride=self.stride)
        prob_sum = np.zeros((h, w), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)
        with torch.no_grad():
            for batch_tiles in _batched(tiles, self.batch):
                tensor = torch.stack([_preprocess_tile(image, t) for t in batch_tiles]).to(self.device)
                logits = forward_logits(self.model, tensor, (self.tile, self.tile))
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().astype(np.float32)
                for tile, prob in zip(batch_tiles, probs, strict=True):
                    vh = min(tile.height, h - tile.y)
                    vw = min(tile.width, w - tile.x)
                    tw = self.weight[:vh, :vw]
                    prob_sum[tile.y:tile.y + vh, tile.x:tile.x + vw] += prob[:vh, :vw] * tw
                    weight_sum[tile.y:tile.y + vh, tile.x:tile.x + vw] += tw
        prob = prob_sum / np.maximum(weight_sum, 1e-6)
        analyzed = analyzed_mask > 0
        non_sulfide = analyzed & ~(sulfide_mask > 0)
        talc = (prob >= self.threshold) & non_sulfide
        return int(talc.sum()) / max(int(analyzed.sum()), 1)


def build_grain_label_index(
    manifest: Path,
    annotations_path: Path | None,
    *,
    require_human: bool,
) -> dict[str, list[dict[str, Any]]]:
    annotations = None
    if annotations_path and annotations_path.exists():
        payload = json.loads(annotations_path.read_text(encoding="utf-8"))
        annotations = payload.get("labels", {}) if isinstance(payload, dict) else None
    index: dict[str, list[dict[str, Any]]] = {}
    if not manifest.exists() or manifest.stat().st_size == 0:
        return index
    for row in csv.DictReader(manifest.open(encoding="utf-8")):
        label = resolve_grain_label(row, annotations, require_human=require_human)
        if label is None:
            continue
        is_human = bool(annotations and str(row.get("grain_uid", "")) in annotations)
        index.setdefault(str(row.get("run_id", "")), []).append(
            {"features": grain_feature_vector(row), "label": label, "source": "human" if is_human else "heuristic"}
        )
    return index


def fit_fold_grain_model(train_runs: set[str], labels_by_run: dict[str, list[dict[str, Any]]], model_name: str):
    feats: list[list[float]] = []
    ys: list[int] = []
    bootstrap = True
    for run_id in train_runs:
        for grain in labels_by_run.get(run_id, []):
            feats.append(grain["features"])
            ys.append(GRAIN_CLASS_ORDER.index(grain["label"]))
            if grain["source"] == "human":
                bootstrap = False
    if len(set(ys)) < 2:
        # Degenerate fold (only one grain class labelled) — fall back to a constant.
        return _ConstantFineModel(np.mean(ys) if ys else 0.0), bootstrap
    model = make_grain_model(model_name)
    model.fit(np.array(feats, dtype=np.float64), np.array(ys, dtype=np.int64))
    return model, bootstrap


class _ConstantFineModel:
    """Fallback used when a fold has only one grain class; returns a constant P(fine)."""

    def __init__(self, p_fine: float) -> None:
        self.p_fine = float(p_fine)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        n = len(features)
        col = np.full((n, 1), self.p_fine)
        return np.hstack([1.0 - col, col])


def image_fine_fraction(image: dict[str, Any], model: Any) -> float:
    features = image["features"]
    areas = image["areas"]
    if len(features) == 0 or areas.sum() <= 0:
        return 0.0
    proba = model.predict_proba(features)
    fine_col = fine_probability_column(model, proba)
    return float(np.dot(areas, fine_col) / areas.sum())


def fine_probability_column(model: Any, proba: np.ndarray) -> np.ndarray:
    classes = list(getattr(model, "classes_", [0, 1]))
    if FINE_INDEX in classes:
        return proba[:, classes.index(FINE_INDEX)]
    # Model never saw the fine class in training -> P(fine)=0.
    return np.zeros(proba.shape[0])


def calibrate_thresholds(
    train_images: list[dict[str, Any]],
    fine_fraction: dict[int, float],
    fine_grid: list[float],
    talc_grid: list[float],
) -> tuple[float, float]:
    # fine_fraction is keyed by the global image index in the same order as
    # train_images (both follow train_idx), so they zip position-for-position.
    fracs = list(fine_fraction.values())
    talc = [img["talc_fraction"] for img in train_images]
    truth = [img["grade"] for img in train_images]
    best = (-1.0, fine_grid[len(fine_grid) // 2], 0.10)
    for tau_talc in talc_grid:
        for tau_fine in fine_grid:
            preds = [predict_grade(fracs[k], talc[k], tau_fine, tau_talc) for k in range(len(fracs))]
            score = macro_f1(truth, preds)
            if score > best[0]:
                best = (score, tau_fine, tau_talc)
    return best[1], best[2]


def predict_grade(fine_fraction: float, talc_fraction: float, tau_fine: float, tau_talc: float) -> str:
    if talc_fraction >= tau_talc:
        return "talcose_ore"
    if fine_fraction >= tau_fine:
        return "hard_to_process_ore"
    return "row_ore"


def macro_f1(truth: list[str], preds: list[str]) -> float:
    y_true = np.array([GRADE_ORDER.index(t) for t in truth])
    y_pred = np.array([GRADE_ORDER.index(p) for p in preds])
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(GRADE_ORDER))), zero_division=0)
    return float(np.mean(f1))


def grade_metrics(truth: list[str], preds: list[str | None], count_fine: list[float]) -> dict[str, Any]:
    pairs = [(t, p) for t, p in zip(truth, preds) if p is not None]
    y_true = np.array([GRADE_ORDER.index(t) for t, _ in pairs])
    y_pred = np.array([GRADE_ORDER.index(p) for _, p in pairs])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(GRADE_ORDER))), zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(GRADE_ORDER))))
    return {
        "images_scored": len(pairs),
        "accuracy": float((y_true == y_pred).mean()) if len(pairs) else 0.0,
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)) if support.sum() else 0.0,
        "per_class": {
            GRADE_ORDER[i]: {
                "label_ru": GRADE_RU[GRADE_ORDER[i]],
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(GRADE_ORDER))
        },
        "confusion_matrix": {
            GRADE_ORDER[i]: {GRADE_ORDER[j]: int(matrix[i, j]) for j in range(len(GRADE_ORDER))}
            for i in range(len(GRADE_ORDER))
        },
        "predicted_distribution": dict(Counter(p for _, p in pairs)),
    }


def grouped_n_splits(y: np.ndarray, groups: np.ndarray, folds: int) -> int:
    # See train_grain_classifier.grouped_n_splits: never floor at 2, or a grade
    # class confined to one specimen group yields an empty-train / single-class
    # fold and a misleading number. Fail loudly instead.
    min_groups_per_class = min(len(set(groups[y == cls].tolist())) for cls in np.unique(y))
    n_splits = min(folds, min_groups_per_class)
    if n_splits < 2:
        raise SystemExit(
            f"grouped grade CV needs >=2 specimen groups in the smallest grade; got {min_groups_per_class}."
        )
    return n_splits


def resolve_run_dir(run_dir: Path) -> Path:
    return run_dir if run_dir.is_absolute() else (ROOT / run_dir)


def read_component_grains(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_talc_fraction(path: Path) -> float:
    if not path.exists():
        return 0.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data.get("talc_fraction", 0.0) or 0.0)
    except (ValueError, OSError):
        return 0.0


def render_md(summary: dict[str, Any]) -> str:
    m = summary["grade_metrics"]
    lines = [
        "# Grade from grains (path B) — leak-free grouped CV",
        "",
        f"- Images scored: {m['images_scored']} | specimen groups: {summary['specimen_groups']} | folds: {summary['folds']}",
        f"- Grain model: `{summary['grain_model']}` | training is bootstrap (heuristic labels): {summary['training_is_bootstrap']}",
        f"- Talc signal: {summary.get('talc_source_counts')}" + (f" | checkpoint `{Path(summary['talc_checkpoint']).name}`" if summary.get("talc_checkpoint") else ""),
        f"- **Grade macro-F1: {m['macro_f1']:.4f}** | weighted-F1: {m['weighted_f1']:.4f} | accuracy: {m['accuracy']:.4f}",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in GRADE_ORDER:
        pc = m["per_class"][name]
        lines.append(f"| {name} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} | {pc['support']} |")
    lines += ["", "## Confusion (rows=true, cols=pred)", "", "| True \\ Pred | " + " | ".join(GRADE_ORDER) + " |", "| --- | " + " | ".join("---:" for _ in GRADE_ORDER) + " |"]
    for t in GRADE_ORDER:
        lines.append("| " + t + " | " + " | ".join(str(m["confusion_matrix"][t][p]) for p in GRADE_ORDER) + " |")
    lines += [
        "",
        f"Fold thresholds: {[{k: v for k, v in ft.items() if k in ('fold','tau_fine','tau_talc')} for ft in summary['fold_thresholds']]}",
        "",
        "> Comparison: harness deterministic rule 0.185, feature-CV 0.747; competitor A (trained CNN) 0.880. "
        "This number is leak-free (grain model + thresholds fit per train fold).",
    ]
    if summary["training_is_bootstrap"]:
        lines.append("> Bootstrap run on heuristic grain labels — human labels via grain_review_web.py are the path to the real gain.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
