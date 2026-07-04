#!/usr/bin/env python3
"""Build a per-segment dataset for grade (ordinary/fine) classification.

Reads ore-pipeline run outputs (original image + sulfide_mask.png) and emits, per
connected sulfide component:

  - context crop (expanded bbox at NATIVE scale, no per-segment resize)     -> crops/<seg_id>.jpg
  - the exact component mask aligned to that crop (CNN guidance channel)     -> masks/<seg_id>.png
  - a preview with a thin contour of the segment + scale bar (for annotator) -> previews/<seg_id>.jpg
  - a manifest row with geometry/features + parsed magnification            -> segments.csv

Both approach A (tabular on features) and B (CNN on crops) consume this manifest.
Magnification is preserved (parsed from filename); NOTHING is scale-normalised here
so the downstream model can decide how to handle it.
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

Image.MAX_IMAGE_PIXELS = None

MAG_RE = re.compile(r"(\d+)\s*[xхX]")  # matches 10x / 5х (latin+cyrillic)


def parse_mag(name: str) -> str:
    m = MAG_RE.search(name)
    return f"{m.group(1)}x" if m else "cam"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, required=True,
                    help="Dir containing */pipeline_summary.json (ore pipeline batch output).")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-area-px", type=int, default=500,
                    help="Skip components smaller than this (still counted in %, just not labeled).")
    ap.add_argument("--pad-frac", type=float, default=0.6,
                    help="Context padding as fraction of the larger bbox side.")
    ap.add_argument("--pad-min-px", type=int, default=80)
    ap.add_argument("--max-per-image", type=int, default=0,
                    help="Cap labeled segments per image (0 = all). Largest-area first.")
    args = ap.parse_args()

    crops = args.out_dir / "crops"
    masks = args.out_dir / "masks"
    previews = args.out_dir / "previews"
    for d in (crops, masks, previews):
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    summaries = sorted(glob.glob(str(args.runs_dir / "**/pipeline_summary.json"), recursive=True))
    for ps_path in summaries:
        ps = json.loads(Path(ps_path).read_text(encoding="utf-8"))
        run_dir = Path(ps_path).parent
        img_path = Path(ps["image"])
        smask_path = run_dir / "binary_sulfide" / "sulfide_mask.png"
        if not img_path.exists() or not smask_path.exists():
            continue
        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        H, W = rgb.shape[:2]
        smask = np.asarray(Image.open(smask_path).convert("L"))
        if smask.shape[:2] != (H, W):
            smask = np.asarray(Image.fromarray(smask).resize((W, H), Image.NEAREST))
        sul = (smask > 127).astype(np.uint8)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(sul, connectivity=8)

        name = img_path.name
        mag = parse_mag(name)
        # weak label from the GT folder (ordinary/fine) for approach A bootstrap
        p = str(img_path)
        weak = "row_ore" if "Рядовые" in p else ("hard_to_process_ore" if "Труднообогат" in p else "")

        comps = []
        for cid in range(1, n):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area < args.min_area_px:
                continue
            comps.append((area, cid))
        comps.sort(reverse=True)
        if args.max_per_image > 0:
            comps = comps[: args.max_per_image]

        for area, cid in comps:
            x = int(stats[cid, cv2.CC_STAT_LEFT]); y = int(stats[cid, cv2.CC_STAT_TOP])
            bw = int(stats[cid, cv2.CC_STAT_WIDTH]); bh = int(stats[cid, cv2.CC_STAT_HEIGHT])
            pad = max(args.pad_min_px, int(args.pad_frac * max(bw, bh)))
            x0 = max(0, x - pad); y0 = max(0, y - pad)
            x1 = min(W, x + bw + pad); y1 = min(H, y + bh + pad)
            crop = rgb[y0:y1, x0:x1].copy()
            comp_mask = (labels[y0:y1, x0:x1] == cid).astype(np.uint8) * 255

            seg_id = f"{run_dir.name}__c{cid}"
            Image.fromarray(crop).save(crops / f"{seg_id}.jpg", quality=90)
            Image.fromarray(comp_mask).save(masks / f"{seg_id}.png")

            # preview: thin contour of the exact segment + a scale bar
            prev = crop.copy()
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(prev, contours, -1, (255, 40, 40), 2)
            bar = min(200, max(40, (x1 - x0) // 4))  # scale bar length in native px
            ch, cw = prev.shape[:2]
            cv2.rectangle(prev, (10, ch - 22), (10 + bar, ch - 16), (255, 255, 255), -1)
            cv2.putText(prev, f"{bar}px  {mag}", (10, ch - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            Image.fromarray(prev).save(previews / f"{seg_id}.jpg", quality=88)

            rows.append({
                "seg_id": seg_id, "image": name, "image_path": p, "component_id": cid,
                "mag": mag, "weak_label": weak, "area_px": area,
                "bbox_x": x, "bbox_y": y, "bbox_w": bw, "bbox_h": bh,
                "crop_x0": x0, "crop_y0": y0, "crop_x1": x1, "crop_y1": y1,
                "crop": f"crops/{seg_id}.jpg", "mask": f"masks/{seg_id}.png",
                "preview": f"previews/{seg_id}.jpg",
            })

    rows.sort(key=lambda r: -r["area_px"])
    fields = list(rows[0].keys()) if rows else []
    with (args.out_dir / "segments.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"segments: {len(rows)} from {len(summaries)} images -> {args.out_dir/'segments.csv'}")
    by_mag = {}
    for r in rows:
        by_mag[r["mag"]] = by_mag.get(r["mag"], 0) + 1
    print("by magnification:", by_mag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
