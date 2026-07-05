#!/usr/bin/env python3
"""Out-of-fold talc-fraction error on the 42 expert-annotated talc images.

Best-available numeric proxy to the organizers' "talc-fraction error ≤ ±3% vs
expert annotation" criterion. GT = the 42 reviewed blue-contour talc masks (the
only talc annotation that exists; itself weak/auto-converted). Leak-free: each
image is predicted by the SegFormer-B0 fold in which it was in validation, at that
fold's calibrated threshold. Reports |predicted% − reviewed%| in percentage points
on image / analyzed / non-sulfide denominators, and the share within ±3pp.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
Image.MAX_IMAGE_PIXELS = None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds-dir", type=Path, default=ROOT / "outputs/talc_segformer_folds/segformer_b0_full_20260703")
    ap.add_argument("--conversion-dir", type=Path, default=ROOT / "outputs/talc_blue_line_conversion")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tile-size", type=int, default=384)
    ap.add_argument("--stride", type=int, default=288)
    args = ap.parse_args()

    folds = json.loads((args.folds_dir / "folds.json").read_text(encoding="utf-8"))["folds"]
    rows = []
    work = Path(tempfile.mkdtemp(prefix="talcfrac_"))
    for fold_id, stems in folds.items():
        ckpt = args.folds_dir / f"fold_{int(fold_id):02d}/segformer_b0/best.pt"
        thr = float(json.loads((args.folds_dir / f"fold_{int(fold_id):02d}/summary.json").read_text())["best_threshold_metrics"]["threshold"])
        for stem in stems:
            sample = args.conversion_dir / "samples" / stem
            image = next((p for p in sample.glob(f"{stem}.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}), None)
            sulfide = sample / "sulfide_mask.png"
            reviewed = sample / "reviewed" / "reviewed_talc_mask.png"
            if not (image and sulfide.exists() and reviewed.exists()):
                print(f"skip {stem}: missing inputs", flush=True)
                continue
            out = work / stem
            subprocess.run([
                sys.executable, "scripts/infer_talc_segmentation.py",
                "--image", str(image), "--sulfide-mask", str(sulfide),
                "--checkpoint", str(ckpt), "--out-dir", str(out),
                "--tile-size", str(args.tile_size), "--stride", str(args.stride),
                "--threshold", str(thr), "--device", args.device,
            ], check=True, cwd=str(ROOT), stdout=subprocess.DEVNULL)
            s = json.loads((out / "summary.json").read_text())
            rev_px = int((np.asarray(Image.open(reviewed).convert("L")) > 0).sum())
            image_area = float(s["image_area_px"]); analyzed = float(s["analyzed_area_px"]); nonsulf = float(s["non_sulfide_area_px"])
            rec = {
                "stem": stem, "fold": int(fold_id), "threshold": thr,
                "pred_talc_px": int(s["talc_area_px"]), "reviewed_talc_px": rev_px,
                "err_pp_image": 100.0 * (s["talc_area_px"] - rev_px) / max(image_area, 1),
                "err_pp_analyzed": 100.0 * (s["talc_area_px"] - rev_px) / max(analyzed, 1),
                "err_pp_non_sulfide": 100.0 * (s["talc_area_px"] - rev_px) / max(nonsulf, 1),
                "pred_frac_image_pct": 100.0 * s["talc_area_px"] / max(image_area, 1),
                "gt_frac_image_pct": 100.0 * rev_px / max(image_area, 1),
            }
            rows.append(rec)
            print(f"[fold {fold_id}] {stem}: pred {rec['pred_frac_image_pct']:.1f}% gt {rec['gt_frac_image_pct']:.1f}% err(img) {rec['err_pp_image']:+.1f}pp", flush=True)

    result = {"schema_version": "talc-fraction-error-oof-v0.1", "n_images": len(rows),
              "note": "OOF (leak-free) on the 42 expert blue-contour talc images; GT is the reviewed masks (weak/auto-converted), not a true expert fraction."}
    for key in ("err_pp_image", "err_pp_analyzed", "err_pp_non_sulfide"):
        errs = np.array([abs(r[key]) for r in rows], dtype=float)
        signed = np.array([r[key] for r in rows], dtype=float)
        result[key] = {
            "mean_abs_pp": float(errs.mean()), "median_abs_pp": float(np.median(errs)),
            "p90_abs_pp": float(np.percentile(errs, 90)), "max_abs_pp": float(errs.max()),
            "mean_signed_pp": float(signed.mean()),
            "within_3pp_pct": float(100.0 * (errs <= 3.0).mean()),
        }
    result["per_image"] = rows
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out_md:
        lines = ["# Talc-fraction error (OOF, 42 annotated images)", "",
                 f"- Images: {result['n_images']} (leak-free out-of-fold)",
                 "- GT = reviewed blue-contour masks (weak, not true expert fractions)", "",
                 "| denominator | mean abs | median abs | p90 abs | max abs | mean signed | within ±3pp |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for key, lab in (("err_pp_image", "image"), ("err_pp_analyzed", "analyzed"), ("err_pp_non_sulfide", "non-sulfide")):
            m = result[key]
            lines.append(f"| {lab} | {m['mean_abs_pp']:.2f}pp | {m['median_abs_pp']:.2f}pp | {m['p90_abs_pp']:.2f}pp | {m['max_abs_pp']:.2f}pp | {m['mean_signed_pp']:+.2f}pp | {m['within_3pp_pct']:.0f}% |")
        lines += ["", f"Note: {result['note']}", ""]
        args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("n_images", "err_pp_image", "err_pp_analyzed")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
