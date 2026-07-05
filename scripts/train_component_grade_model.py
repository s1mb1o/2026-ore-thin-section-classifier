#!/usr/bin/env python3
"""Train the per-component grade (ordinary/fine) classifier from pipeline runs.

Consumes ore-pipeline batch output (runs with component_features.csv), uses weak
folder labels (Рядовые -> ordinary, Труднообогатимые/тонкие -> fine), reports
honest image-level GroupKFold CV with area-weighted aggregation, then fits the
final model on ALL data and saves model.joblib + meta.json for
``ore_classifier.component_grade_model``.

    python3 scripts/train_component_grade_model.py \
        --runs-dir outputs/evaluations/ch1_dark_green_notalc_20260704/run/runs \
        --out-dir models/component_grade/hgb_weak100_nomag_20260705
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.component_grade_model import (  # noqa: E402
    FEATURE_FIELDS,
    parse_magnification,
)


def weak_label(image_path: str) -> int | None:
    if "Труднообогат" in image_path or "/тонкие/" in image_path.lower():
        return 1
    if "Рядовые" in image_path or "/рядовые/" in image_path.lower():
        return 0
    return None


def row_vector(row: dict, magnification: str = "cam") -> list[float]:
    # magnification one-hots removed 2026-07-05: sampling-bias shortcut (see component_grade_model)
    bw, bh = float(row["bbox_w"]), float(row["bbox_h"])
    area = float(row["area_px"])
    vec = [float(row[f]) for f in FEATURE_FIELDS]
    vec.append(float(np.log1p(area)))
    vec.append(area / max(bw * bh, 1.0))
    vec.append(bw / max(bh, 1.0))
    return vec


def make_model():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.08, max_iter=300,
        l2_regularization=1.0, min_samples_leaf=40,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    X, y, groups, areas = [], [], [], []
    frame_gt: dict[str, int] = {}
    for ps_path in sorted(glob.glob(str(args.runs_dir / "**/pipeline_summary.json"), recursive=True)):
        summary = json.loads(Path(ps_path).read_text(encoding="utf-8"))
        gt = weak_label(summary["image"])
        csv_path = Path(ps_path).parent / "ore_analysis" / "component_features.csv"
        if gt is None or not csv_path.exists():
            continue
        frame_id = Path(ps_path).parent.name
        frame_gt[frame_id] = gt
        mag = parse_magnification(Path(summary["image"]).name)
        for row in csv.DictReader(csv_path.open(encoding="utf-8")):
            X.append(row_vector(row, mag))
            y.append(gt)
            groups.append(frame_id)
            areas.append(float(row["area_px"]))

    X = np.array(X); y = np.array(y); groups = np.array(groups); areas = np.array(areas)
    n_frames = len(frame_gt)
    print(f"components: {len(X)}  frames: {n_frames}  fine frames: {sum(frame_gt.values())}")

    # honest CV: image-level folds, area-weighted aggregation to a frame verdict
    from sklearn.model_selection import GroupKFold

    frame_score: dict[str, list[float]] = {}
    for train_idx, test_idx in GroupKFold(n_splits=args.folds).split(X, y, groups=groups):
        model = make_model()
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        for j, idx in enumerate(test_idx):
            acc = frame_score.setdefault(groups[idx], [0.0, 0.0])
            acc[0] += areas[idx] * proba[j]
            acc[1] += areas[idx]

    ok = ord_ok = fine_ok = n_ord = n_fine = 0
    for fid, gt in frame_gt.items():
        s, a = frame_score[fid]
        pred = 1 if s / max(a, 1) >= 0.5 else 0
        ok += pred == gt
        if gt == 0:
            n_ord += 1; ord_ok += pred == gt
        else:
            n_fine += 1; fine_ok += pred == gt
    cv = {
        "frame_accuracy": ok / max(n_frames, 1),
        "ordinary_recall": ord_ok / max(n_ord, 1),
        "fine_recall": fine_ok / max(n_fine, 1),
        "frames": n_frames, "folds": args.folds,
    }
    print(f"CV frame accuracy: {cv['frame_accuracy']:.1%}  "
          f"(ordinary {cv['ordinary_recall']:.1%} / fine {cv['fine_recall']:.1%})")

    # final model on ALL data
    final = make_model()
    final.fit(X, y)

    import joblib

    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, args.out_dir / "model.joblib")
    meta = {
        "kind": "component_grade_hgb",
        "trained_on": str(args.runs_dir),
        "components": int(len(X)),
        "frames": n_frames,
        "labels": {"0": "ordinary_intergrowth", "1": "fine_intergrowth"},
        "feature_fields": list(FEATURE_FIELDS),
        "derived_features": ["log1p_area", "bbox_fill", "aspect"],
        "magnifications": [],
        "weak_labels": "class folder (Рядовые/Труднообогатимые)",
        "cv": cv,
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {args.out_dir}/model.joblib (+meta.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
