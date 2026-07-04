#!/usr/bin/env python3
"""Seed grade-review masks with the trained tabular model (approach A) instead of
the rule. Uses out-of-fold predictions (each frame classified by a model that did
NOT see it), so the seed reflects the honest ~73% model, not memorised folder
labels. The pipeline's ore footprint is untouched: only the kept ore components
(those present in component_features.csv) are recoloured; background stays 0.

    python3 scripts/seed_masks_with_model.py \
        --runs-dir outputs/evaluations/ch1_dark_green_notalc_20260704/run/runs \
        --out-dir  outputs/grade_seed_modelA_v0
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

Image.MAX_IMAGE_PIXELS = None

FEATS = ["area_px", "footprint_area_px", "dark_inside_ratio", "solidity", "compactness",
         "boundary_complexity", "perimeter_px", "bbox_w", "bbox_h"]
MAGS = ["cam", "5x", "10x", "20x"]
MAG_RE = re.compile(r"(\d+)\s*[xхX]")


def parse_mag(n):
    m = MAG_RE.search(n)
    return (m.group(1) + "x") if m else "cam"


def feat_vec(r, mag):
    bw, bh = float(r["bbox_w"]), float(r["bbox_h"])
    v = [float(r[f]) for f in FEATS]
    v += [np.log1p(float(r["area_px"])), float(r["area_px"]) / max(bw * bh, 1), bw / max(bh, 1)]
    v += [1.0 if mag == m else 0.0 for m in MAGS]
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- gather training rows (weak folder labels) ----
    frames = []  # (frame_id, run_dir, mag, gt, rows)
    X, y, grp, keyrows = [], [], [], []
    for ps in sorted(glob.glob(str(args.runs_dir / "**/pipeline_summary.json"), recursive=True)):
        d = json.loads(Path(ps).read_text(encoding="utf-8"))
        run_dir = Path(ps).parent
        p = d["image"]
        gt = 1 if "Труднообогат" in p else (0 if "Рядовые" in p else None)
        csvp = run_dir / "ore_analysis" / "component_features.csv"
        if gt is None or not csvp.exists():
            continue
        mag = parse_mag(Path(p).name)
        rows = list(csv.DictReader(csvp.open(encoding="utf-8")))
        fid = run_dir.name
        frames.append((fid, run_dir, mag, gt, rows))
        for r in rows:
            X.append(feat_vec(r, mag)); y.append(gt); grp.append(fid)
            keyrows.append((fid, int(r["component_id"])))

    X = np.array(X); y = np.array(y); grp = np.array(grp)
    print(f"components: {len(X)}  frames: {len(frames)}")

    # ---- out-of-fold predicted class per component ----
    pred = {}  # (fid, cid) -> 1 ordinary / 2 fine
    gkf = GroupKFold(n_splits=args.folds)
    for tr, te in gkf.split(X, y, groups=grp):
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=300,
                                             l2_regularization=1.0, min_samples_leaf=40)
        clf.fit(X[tr], y[tr])
        pf = clf.predict_proba(X[te])[:, 1]
        for j, idx in enumerate(te):
            fid, cid = keyrows[idx]
            pred[(fid, cid)] = 2 if pf[j] >= 0.5 else 1

    # ---- repaint per frame (only kept ore components; footprint untouched) ----
    manifest = []
    for fid, run_dir, mag, gt, rows in frames:
        smask = np.asarray(Image.open(run_dir / "binary_sulfide" / "sulfide_mask.png").convert("L"))
        analyzed = np.asarray(Image.open(run_dir / "ore_analysis" / "analyzed_mask.png").convert("L"))
        sul = ((smask > 0) & (analyzed > 0)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(sul, connectivity=8)
        lut = np.zeros(n, dtype=np.uint8)
        ncomp = 0
        for r in rows:
            cid = int(r["component_id"])
            if 1 <= cid < n and (fid, cid) in pred:
                lut[cid] = pred[(fid, cid)]; ncomp += 1
        seed = lut[labels].astype(np.uint8)
        Image.fromarray(seed).save(args.out_dir / f"{fid}.png")
        ord_px = int((seed == 1).sum()); fine_px = int((seed == 2).sum())
        klass = "row_ore" if ord_px >= fine_px else "hard_to_process_ore"
        manifest.append({"frame_id": fid, "gt": "hard_to_process_ore" if gt else "row_ore",
                         "seed_class": klass, "ordinary_px": ord_px, "fine_px": fine_px,
                         "components": ncomp})

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # seed frame-accuracy vs folder GT (sanity)
    ok = sum(1 for m in manifest if m["seed_class"] == m["gt"])
    print(f"seed masks: {len(manifest)} -> {args.out_dir}")
    print(f"seed frame-accuracy vs folder GT: {ok}/{len(manifest)} = {ok/max(len(manifest),1):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
