#!/usr/bin/env python3
"""Fused 3-class ore verdict = talc branch (talcose) ⊕ Grade-CNN (ordinary↔fine).

Primary-verdict fusion:
  - talcose  <- the deterministic talc branch (B0 talc-fraction rule) from an
    existing B2+B0 batch summary (`predicted_ore_class == talcose_ore`);
  - ordinary vs fine  <- the Grade-CNN (efficientnet_b3, pp-aware), replacing the
    weak morphology rule on the non-talcose images.

Leak-free: the Grade-CNN was trained excluding the entire 345 eval split. Reports
3-class accuracy / per-class P-R-F1 / macro-F1 / confusion, and the deltas vs the
pure rule pipeline. Meant to run where the batch summary + dataset + grade
checkpoint live (gx10).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
Image.MAX_IMAGE_PIXELS = None

from ore_classifier.grade_classifier import load_grade_model, predict_grade  # noqa: E402

CLASS_ORDER = ["row_ore", "hard_to_process_ore", "talcose_ore"]
CLASS_RU = {"row_ore": "рядовая", "hard_to_process_ore": "труднообогатимая", "talcose_ore": "оталькованная"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-csv", type=Path, required=True, help="B2+B0 batch summary.csv (talc branch + true labels).")
    ap.add_argument("--grade-checkpoint", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.summary_csv.open(encoding="utf-8", newline="")))
    grade = load_grade_model(args.grade_checkpoint, device=args.device)
    print(f"loaded grade model: classes={grade.classes} device={grade.device}", flush=True)

    fused_conf = {t: {p: 0 for p in CLASS_ORDER} for t in CLASS_ORDER}
    rule_conf = {t: {p: 0 for p in CLASS_ORDER} for t in CLASS_ORDER}
    used_cnn = 0
    used_talc = 0
    per_image = []
    for row in rows:
        true = row.get("expected_ore_class", "")
        if true not in CLASS_ORDER:
            continue
        rule_pred = row.get("predicted_ore_class", "")
        img_path = row.get("source_dataset_path") or str(args.dataset_root / row.get("source_rel_path", ""))
        if rule_pred == "talcose_ore":
            fused = "talcose_ore"
            used_talc += 1
        else:
            pred = predict_grade(grade, Image.open(img_path))
            fused = pred["predicted_ore_class"]  # row_ore or hard_to_process_ore
            used_cnn += 1
        if fused not in CLASS_ORDER:
            fused = "row_ore"
        fused_conf[true][fused] += 1
        rp = rule_pred if rule_pred in CLASS_ORDER else "row_ore"
        rule_conf[true][rp] += 1
        per_image.append({"path": row.get("source_rel_path", ""), "true": true, "rule": rule_pred, "fused": fused})

    result = {
        "schema_version": "grade-fusion-eval-v0.1",
        "fusion": "talcose<-B0 talc branch; ordinary/fine<-Grade-CNN(pp-aware); morphology rule = fallback",
        "grade_checkpoint": str(args.grade_checkpoint),
        "n_images": sum(sum(v.values()) for v in fused_conf.values()),
        "decided_by_talc_branch": used_talc,
        "decided_by_cnn": used_cnn,
        "fused": metrics_from_confusion(fused_conf),
        "rule_baseline": metrics_from_confusion(rule_conf),
        "per_image": per_image,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.write_text(render_md(result), encoding="utf-8")
    print(render_md(result))
    return 0


def metrics_from_confusion(conf: dict) -> dict:
    per_class = {}
    f1s = []
    total = sum(sum(conf[t].values()) for t in CLASS_ORDER)
    correct = sum(conf[t][t] for t in CLASS_ORDER)
    for c in CLASS_ORDER:
        tp = conf[c][c]
        fp = sum(conf[o][c] for o in CLASS_ORDER if o != c)
        fn = sum(conf[c][o] for o in CLASS_ORDER if o != c)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_class[c] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "support": sum(conf[c].values())}
        f1s.append(f1)
    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "macro_f1": round(float(np.mean(f1s)), 4),
        "per_class": per_class,
        "confusion_matrix": conf,
    }


def render_md(res: dict) -> str:
    f = res["fused"]; b = res["rule_baseline"]
    lines = [
        "# Fused 3-class ore verdict (talc branch ⊕ Grade-CNN)",
        "",
        f"- Images: {res['n_images']} | decided by talc branch: {res['decided_by_talc_branch']} | by Grade-CNN: {res['decided_by_cnn']}",
        f"- Fusion: {res['fusion']}",
        "",
        f"**Fused macro-F1: {f['macro_f1']:.4f}** (acc {f['accuracy']:.4f}) — vs pure rule {b['macro_f1']:.4f} (acc {b['accuracy']:.4f})",
        "",
        "| Class | fused P | fused R | fused F1 | rule F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for c in CLASS_ORDER:
        fc = f["per_class"][c]; bc = b["per_class"][c]
        lines.append(f"| {c} ({CLASS_RU[c]}) | {fc['precision']:.3f} | {fc['recall']:.3f} | **{fc['f1']:.3f}** | {bc['f1']:.3f} |")
    cm = f["confusion_matrix"]
    lines += ["", "Fused confusion (rows=true, cols=pred): " + "; ".join(
        f"{c}=[{', '.join(str(cm[c][p]) for p in CLASS_ORDER)}]" for c in CLASS_ORDER), ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
