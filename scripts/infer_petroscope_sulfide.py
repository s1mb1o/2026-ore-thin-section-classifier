#!/usr/bin/env python3
"""Generate sulfide masks for a talc annotation workspace with Petroscope ResUNet.

Uses the LumenStone-pretrained Petroscope checkpoint (e.g. resunet_s2_x05_*.pth)
from the teammate bundle to predict mineral classes on each workspace sample,
then binarizes sulfide codes into `sulfide_mask.png` next to the sample and
registers the path in conversion_summary.json / manifest.json so the review
app can show the sulfide layer and protect sulfides while drawing.

The x05 checkpoints are trained at half resolution: images are downscaled by
--scale before prediction and the mask is upscaled back with nearest.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Sulfide class codes shared with src/ore_classifier/pseudo_labels.py
DEFAULT_SULFIDE_CODES = (1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--petroscope-repo", type=Path, required=True, help="Path to experiments/petroscope checkout.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--classes", default="S2")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(args.petroscope_repo))
    from petroscope.segmentation.classes import LumenStoneClasses
    from petroscope.segmentation.models.resunet.model import ResUNet

    class_set = LumenStoneClasses.from_name(args.classes)
    idx_to_code = np.array([c.code for c in class_set.classes], dtype=np.int32)
    sulfide_codes = set(DEFAULT_SULFIDE_CODES)

    model = ResUNet.from_pretrained(str(args.weights), device=args.device)

    sample_dirs = sorted(d for d in (args.workspace / "samples").iterdir() if d.is_dir())
    if args.limit:
        sample_dirs = sample_dirs[: args.limit]

    rows = []
    for i, sample_dir in enumerate(sample_dirs, 1):
        summary_path = sample_dir / "conversion_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        mask_path = sample_dir / "sulfide_mask.png"
        if mask_path.exists() and not args.overwrite:
            mask = np.asarray(Image.open(mask_path).convert("L")) > 0
            rows.append({"sample_id": sample_dir.name, "sulfide_fraction": round(float(mask.mean()), 6), "cached": True})
            continue
        started = time.time()
        image = Image.open(summary["image_path"]).convert("RGB")
        full_w, full_h = image.size
        small = image.resize((max(1, int(full_w * args.scale)), max(1, int(full_h * args.scale))), Image.Resampling.BILINEAR)
        pred_idx = model.predict_image(np.asarray(small, dtype=np.uint8), return_logits=False)
        codes = idx_to_code[pred_idx]
        sulfide_small = np.isin(codes, list(sulfide_codes)).astype(np.uint8) * 255
        sulfide = Image.fromarray(sulfide_small, mode="L").resize((full_w, full_h), Image.Resampling.NEAREST)
        sulfide.save(mask_path)

        summary.setdefault("paths", {})["sulfide_mask"] = str(mask_path)
        summary["sulfide_source"] = f"petroscope_resunet_{args.classes.lower()}:{args.weights.name}"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        fraction = float(np.asarray(sulfide).mean() / 255.0)
        rows.append({"sample_id": sample_dir.name, "sulfide_fraction": round(fraction, 6), "seconds": round(time.time() - started, 2)})
        print(f"[{i}/{len(sample_dirs)}] {sample_dir.name}: sulfide {fraction*100:.1f}% ({rows[-1].get('seconds', 0)}s)", flush=True)

    # sync manifest with per-sample summaries
    manifest_path = args.workspace / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        by_id = {}
        for sample_dir in sample_dirs:
            sp = sample_dir / "conversion_summary.json"
            if sp.exists():
                by_id[sample_dir.name] = json.loads(sp.read_text(encoding="utf-8"))
        manifest["samples"] = [by_id.get(str(s.get("image_id")), s) for s in manifest.get("samples", [])]
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    out_csv = args.workspace / "sulfide_masks_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "sulfide_fraction", "seconds", "cached"], extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"done: {len(rows)} samples -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
