#!/usr/bin/env python3
"""CHECK: is the trained talc segmentation model discriminative across grades?

The talc model was trained only on the 42 talcose (оталькованные) blue-contour
masks, so before wiring it into the talcose grade decision we must verify it
predicts high talc_fraction on talcose images but LOW on ordinary/fine images
(else it just adds false positives that hurt the other two classes).

Reuses each sampled baseline run's sulfide_mask + analyzed_mask so the fraction
is computed exactly as the ore rule would (talc_area / analyzed_area, non-sulfide
clipped), and compares against the baseline colour auto-candidate talc_fraction.
"""
from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.model_io import forward_logits, load_binary_segmentation_checkpoint, resolve_device  # noqa: E402
from ore_classifier.resident_pipeline import _batched, _preprocess_tile, _tile_weight  # noqa: E402
from ore_classifier.tiling import iter_tiles  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

TALC_CKPT = ROOT / "models/talc_segmentation/resunet_non_sulfide_20260703_local/best.pt"
BASELINE = ROOT / "outputs/evaluations/harness_baseline_20260704"
PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
THRESHOLD = 0.5
TILE, STRIDE, BS = 1024, 768, 4
TALC_RULE_THRESHOLD = 0.10


def talc_fraction_for(model, device, weight, image_path, sulfide_mask, analyzed_mask) -> float:
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    tiles = iter_tiles(width=w, height=h, tile_size=TILE, stride=STRIDE)
    prob_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    with torch.no_grad():
        for bt in _batched(tiles, BS):
            tensor = torch.stack([_preprocess_tile(image, t) for t in bt]).to(device)
            logits = forward_logits(model, tensor, (TILE, TILE))
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().astype(np.float32)
            for t, prob in zip(bt, probs, strict=True):
                vh = min(t.height, h - t.y); vw = min(t.width, w - t.x)
                tw = weight[:vh, :vw]
                prob_sum[t.y:t.y + vh, t.x:t.x + vw] += prob[:vh, :vw] * tw
                weight_sum[t.y:t.y + vh, t.x:t.x + vw] += tw
    prob = prob_sum / np.maximum(weight_sum, 1e-6)
    analyzed = analyzed_mask > 0
    non_sulfide = analyzed & ~(sulfide_mask > 0)
    talc = (prob >= THRESHOLD) & non_sulfide
    return int(talc.sum()) / max(int(analyzed.sum()), 1)


def main() -> int:
    rows = list(csv.DictReader((BASELINE / "summary.csv").open(encoding="utf-8")))
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[r["source_label"]].append(r)

    device = resolve_device("auto")
    model, meta = load_binary_segmentation_checkpoint(TALC_CKPT, device)
    model.eval()
    weight = _tile_weight(TILE)
    print(f"talc model={meta.get('model')} device={device} threshold={THRESHOLD} rule_thr={TALC_RULE_THRESHOLD}\n")

    per_class_model: dict[str, list[float]] = defaultdict(list)
    per_class_auto: dict[str, list[float]] = defaultdict(list)
    for label in ["ordinary_intergrowth", "fine_intergrowth", "talcose"]:
        sample = by_label[label][:PER_CLASS]
        for r in sample:
            run_dir = Path(r["run_dir"])
            img = Path(r["source_dataset_path"])
            sm = np.asarray(Image.open(run_dir / "binary_sulfide/sulfide_mask.png").convert("L"))
            am = np.asarray(Image.open(run_dir / "binary_sulfide/analyzed_mask.png").convert("L"))
            frac = talc_fraction_for(model, device, weight, img, sm, am)
            per_class_model[label].append(frac)
            per_class_auto[label].append(float(r.get("talc_fraction") or 0.0))
        m = per_class_model[label]; a = per_class_auto[label]
        crossed = sum(1 for x in m if x > TALC_RULE_THRESHOLD)
        print(f"{label:22s} n={len(m)}")
        print(f"  MODEL  talc_fraction: mean={statistics.mean(m):.4f} med={statistics.median(m):.4f} max={max(m):.4f} | > {TALC_RULE_THRESHOLD} in {crossed}/{len(m)}")
        print(f"  AUTO   talc_fraction: mean={statistics.mean(a):.4f} med={statistics.median(a):.4f} max={max(a):.4f}")
    print("\nDISCRIMINATIVE if talcose MODEL fraction >> ordinary/fine MODEL fraction and crosses the 0.10 rule mostly on talcose only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
