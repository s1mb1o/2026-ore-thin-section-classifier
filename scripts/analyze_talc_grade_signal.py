#!/usr/bin/env python3
"""Quick correlation analysis: does the TRAINED talc segmentation give a signal
for the talcose GRADE? (Decision gate before investing in a talcose branch.)

The color-heuristic talc candidate is ~0 on talcose-grade images (talcose-vs-rest
AUC 0.38). This checks whether the *trained* talc SegFormer-B0 does better: it
runs the model over a balanced subset of the deconflicted 345 eval split, reuses
the sulfide masks already produced by the baseline batch, computes per-image
`talc_fraction_analyzed`, and reports per-grade distributions plus the
talcose-vs-rest ROC-AUC of that fraction. Talcose images that belong to the 42
blue-contour training set are flagged (leak) and the AUC is reported both with
and without them.

Decision: AUC ≳ 0.65-0.70 → exploitable grade signal, worth investing;
AUC ≈ 0.5 → the trained segmenter does not separate the grade either → prefer the
3-class CNN route.

Reuses helpers from scripts/infer_talc_segmentation.py; runs the model once.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ore_classifier.analyzed_area import build_analyzed_mask  # noqa: E402
from ore_classifier.model_io import (  # noqa: E402
    forward_logits,
    load_binary_segmentation_checkpoint,
    resolve_device,
)
from ore_classifier.tiling import iter_tiles  # noqa: E402
import infer_talc_segmentation as tinf  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
GRADES = ["row_ore", "hard_to_process_ore", "talcose_ore"]
LABEL_TO_GRADE = {
    "ordinary_intergrowth": "row_ore",
    "fine_intergrowth": "hard_to_process_ore",
    "talcose": "talcose_ore",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-json", type=Path, default=ROOT / "outputs/official_balanced_eval_split_deconflicted.json")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--baseline-summary", type=Path, default=ROOT / "outputs/evaluations/harness_baseline_20260704/summary.csv")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/talc_segformer_folds/segformer_b0_full_20260703/fold_00/segformer_b0/best.pt")
    parser.add_argument("--annotated-dir", type=Path, default=ROOT / "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования")
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--tile-size", type=int, default=384)
    parser.add_argument("--stride", type=int, default=288)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--analyzed-min-value", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-json", type=Path, default=ROOT / "outputs/evaluations/talc_grade_signal_20260704.json")
    parser.add_argument("--out-md", type=Path, default=ROOT / "outputs/evaluations/talc_grade_signal_20260704.md")
    args = parser.parse_args()

    sulfide_by_path = load_sulfide_map(args.baseline_summary)
    leak_names = {p.name for p in args.annotated_dir.glob("*") if p.is_file()}
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    subset = select_subset(split.get("items", []), per_class=args.per_class, leak_names=leak_names)
    print(f"subset: {sum(len(v) for v in subset.values())} images ({ {k: len(v) for k, v in subset.items()} })", flush=True)

    device = resolve_device(args.device)
    model, _ = load_binary_segmentation_checkpoint(args.checkpoint, device)
    model.eval()

    records: list[dict[str, Any]] = []
    started = time.time()
    total = sum(len(v) for v in subset.values())
    done = 0
    for label, items in subset.items():
        grade = LABEL_TO_GRADE[label]
        for item in items:
            rel = item["path"]
            frac = talc_fraction_for_image(
                image_path=args.dataset_root / rel,
                sulfide_mask_path=sulfide_by_path.get(rel),
                model=model,
                device=device,
                tile_size=args.tile_size,
                stride=args.stride,
                batch_size=args.batch_size,
                threshold=args.threshold,
                analyzed_min_value=args.analyzed_min_value,
            )
            leak = Path(rel).name in leak_names
            records.append({"path": rel, "grade": grade, "talc_fraction": frac, "leak": leak})
            done += 1
            if done % 10 == 0 or done == total:
                print(f"[{done}/{total}] {grade} frac={frac:.5f}{' LEAK' if leak else ''}", flush=True)

    result = summarize(records)
    result.update({
        "schema_version": "talc-grade-signal-v0.1",
        "checkpoint": str(args.checkpoint),
        "per_class": args.per_class,
        "tile_size": args.tile_size,
        "stride": args.stride,
        "threshold": args.threshold,
        "seconds": round(time.time() - started, 1),
    })
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = render_md(result)
    args.out_md.write_text(md, encoding="utf-8")
    print("\n" + md)
    return 0


def load_sulfide_map(summary_csv: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with summary_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            path = row.get("source_rel_path")
            mask = row.get("sulfide_mask")
            if path and mask:
                mapping[path] = mask
    return mapping


def select_subset(items: list[dict[str, Any]], *, per_class: int, leak_names: set[str]) -> dict[str, list[dict[str, Any]]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sorted(items, key=lambda x: x["path"]):
        # For talcose, drop images whose clean original is in the talc training set
        # (same basename as a 42-annotated аншлиф) so the AUC measures generalization.
        if item["label"] == "talcose" and Path(item["path"]).name in leak_names:
            continue
        by_label[item["label"]].append(item)
    return {label: by_label[label][:per_class] for label in ("ordinary_intergrowth", "fine_intergrowth", "talcose") if by_label[label]}


def talc_fraction_for_image(
    *,
    image_path: Path,
    sulfide_mask_path: str | None,
    model,
    device,
    tile_size: int,
    stride: int,
    batch_size: int,
    threshold: float,
    analyzed_min_value: int,
) -> float:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    tiles = iter_tiles(width=width, height=height, tile_size=tile_size, stride=stride)
    weight = tinf.tile_weight(tile_size)
    prob_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    with torch.no_grad():
        for batch_tiles in tinf.batched(tiles, batch_size):
            tensor = torch.stack([tinf.preprocess_tile(image, t) for t in batch_tiles]).to(device)
            logits = forward_logits(model, tensor, (tile_size, tile_size))
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().astype(np.float32)
            for tile, prob in zip(batch_tiles, probs, strict=True):
                vh = min(tile.height, height - tile.y)
                vw = min(tile.width, width - tile.x)
                w = weight[:vh, :vw]
                prob_sum[tile.y:tile.y + vh, tile.x:tile.x + vw] += prob[:vh, :vw] * w
                weight_sum[tile.y:tile.y + vh, tile.x:tile.x + vw] += w
    prob = prob_sum / np.maximum(weight_sum, 1e-6)
    rgb = np.asarray(image, dtype=np.uint8)
    analyzed = build_analyzed_mask(rgb, min_value=analyzed_min_value).astype(bool)
    sulfide = tinf.load_optional_mask(Path(sulfide_mask_path) if sulfide_mask_path else None, (height, width))
    non_sulfide = analyzed & ~sulfide
    talc = (prob >= threshold) & non_sulfide
    analyzed_area = int(analyzed.sum())
    return float(int(talc.sum()) / max(analyzed_area, 1))


def roc_auc(labels: list[int], scores: list[float]) -> float | None:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    pairs = sorted(zip(scores, labels, strict=True), key=lambda p: p[0])
    rank_sum = 0.0
    i = 0
    rank = 1
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (rank + rank + (j - i) - 1) / 2.0
        rank_sum += avg * sum(1 for _, l in pairs[i:j] if l == 1)
        rank += j - i
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / float(pos * neg)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_grade: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_grade[r["grade"]].append(r["talc_fraction"])
    per_grade = {}
    for g in GRADES:
        v = sorted(by_grade.get(g, []))
        if v:
            per_grade[g] = {
                "n": len(v),
                "mean": round(float(np.mean(v)), 6),
                "median": round(float(np.median(v)), 6),
                "p90": round(float(np.percentile(v, 90)), 6),
                "max": round(float(np.max(v)), 6),
            }
    # talcose-vs-rest AUC, with and without leak-set talcose
    all_labels = [1 if r["grade"] == "talcose_ore" else 0 for r in records]
    all_scores = [r["talc_fraction"] for r in records]
    noleak = [r for r in records if not (r["grade"] == "talcose_ore" and r["leak"])]
    nl_labels = [1 if r["grade"] == "talcose_ore" else 0 for r in noleak]
    nl_scores = [r["talc_fraction"] for r in noleak]
    leak_count = sum(1 for r in records if r["grade"] == "talcose_ore" and r["leak"])
    return {
        "per_grade": per_grade,
        "talcose_vs_rest_auc_all": roc_auc(all_labels, all_scores),
        "talcose_vs_rest_auc_no_leak": roc_auc(nl_labels, nl_scores),
        "talcose_leak_count": leak_count,
        "talcose_total": sum(1 for r in records if r["grade"] == "talcose_ore"),
        "records": records,
    }


def render_md(r: dict[str, Any]) -> str:
    auc_nl = r["talcose_vs_rest_auc_no_leak"]
    auc_all = r["talcose_vs_rest_auc_all"]
    verdict = "n/a"
    if auc_nl is not None:
        if auc_nl >= 0.65:
            verdict = "SIGNAL — trained talc segmentation separates talcose; worth investing in a talcose branch."
        elif auc_nl >= 0.55:
            verdict = "WEAK signal — marginal; likely not enough alone."
        else:
            verdict = "NO signal — trained segmenter does not separate the grade either; prefer the 3-class CNN route."
    lines = [
        "# Talc-segmentation signal for the talcose grade",
        "",
        f"- Checkpoint: `{Path(r['checkpoint']).parent.parent.name}` (trained talc SegFormer-B0)",
        f"- Subset: {r.get('per_class')} images/class; tile {r.get('tile_size')}/{r.get('stride')}; thr {r.get('threshold')}",
        f"- Talcose leak-set (42-annotated) in subset: {r['talcose_leak_count']}/{r['talcose_total']}",
        "",
        "## Predicted talc_fraction_analyzed by grade",
        "",
        "| Grade | n | mean | median | p90 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for g in GRADES:
        pg = r["per_grade"].get(g)
        if pg:
            lines.append(f"| {g} | {pg['n']} | {pg['mean']:.5f} | {pg['median']:.5f} | {pg['p90']:.5f} | {pg['max']:.5f} |")
    lines += [
        "",
        f"- **talcose-vs-rest ROC-AUC (excl. leak): {fmt(auc_nl)}**",
        f"- talcose-vs-rest ROC-AUC (incl. leak): {fmt(auc_all)}",
        "",
        f"**Verdict:** {verdict}",
        "",
    ]
    return "\n".join(lines)


def fmt(v: Any) -> str:
    return "n/a" if v is None else f"{float(v):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
