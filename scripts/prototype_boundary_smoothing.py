#!/usr/bin/env python3
"""Prototype of variant B — boundary smoothing before shape metrics (path B).

The heuristic calls a massive homogeneous grain with a *strongly ragged contour*
"fine" purely from low solidity/compactness, at zero internal replacement (see
docs/notes/2026-07-04-ore-vs-gangue-feature-extraction.md §4c). Variant B tests
whether a morphological OPEN of radius r **before** measuring solidity/compactness
removes fine-scale serration (grinding/pluck-out) while keeping deep embayments
(real intergrowth) — i.e. selectively kills the "boundary-only fine" false
positives without touching genuinely replaced grains.

This is a **standalone prototype**: it re-labels grains from the existing sulfide
masks of a completed batch (no re-inference, NO change to core
`component_analysis.py`), reusing that module's own helpers so raw metrics match
production. It reports, per grade, how many boundary-only-fine grains smoothing
converts to ordinary and confirms replacement-fine grains are preserved.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.component_analysis import (  # noqa: E402
    component_perimeter,
    component_solidity,
    crop_component,
    reconstructed_footprint,
)

Image.MAX_IMAGE_PIXELS = None

FINE_DARK_INSIDE_RATIO = 0.18
FINE_SOLIDITY_MAX = 0.62
FINE_COMPACTNESS_MAX = 0.12
GRADES = ["ordinary_intergrowth", "fine_intergrowth", "talcose"]


def compactness(area: float, perimeter: float) -> float:
    return 4.0 * math.pi * area / max(perimeter * perimeter, 1e-6)


def is_fine(dark_inside_ratio: float, solidity: float, compact: float) -> bool:
    return (
        dark_inside_ratio >= FINE_DARK_INSIDE_RATIO
        or solidity <= FINE_SOLIDITY_MAX
        or compact <= FINE_COMPACTNESS_MAX
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-dir", type=Path, default=ROOT / "outputs/evaluations/harness_baseline_20260704")
    parser.add_argument("--smooth-px", type=int, default=3, help="Morphological OPEN radius applied before shape metrics.")
    parser.add_argument("--per-class", type=int, default=12, help="Images sampled per grade (0 = all).")
    parser.add_argument("--min-area", type=int, default=128, help="Min component area (match the batch's min_component_area_px).")
    parser.add_argument("--close-kernel-px", type=int, default=21)
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()

    rows = list(csv.DictReader((args.batch_dir / "summary.csv").open(encoding="utf-8")))
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row.get("source_label", "")].append(row)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, args.smooth_px) * 2 + 1,) * 2)
    stats: dict[str, dict[str, int]] = {}
    examples: list[str] = []

    for grade in GRADES:
        sample = by_label.get(grade, [])
        if args.per_class > 0:
            sample = sample[: args.per_class]
        s = {"grains": 0, "raw_fine": 0, "smooth_fine": 0, "boundary_only_raw": 0,
             "boundary_only_flipped_ordinary": 0, "replacement_fine": 0, "replacement_fine_kept": 0}
        for row in sample:
            run_dir = resolve(Path(row.get("run_dir", "")))
            sm_path = run_dir / "binary_sulfide" / "sulfide_mask.png"
            am_path = run_dir / "binary_sulfide" / "analyzed_mask.png"
            if not sm_path.exists() or not am_path.exists():
                continue
            sulfide_mask = np.asarray(Image.open(sm_path).convert("L"))
            analyzed_mask = np.asarray(Image.open(am_path).convert("L"))
            sulfide = ((sulfide_mask > 0) & (analyzed_mask > 0)).astype(np.uint8)
            count, labels, cc_stats, _ = cv2.connectedComponentsWithStats(sulfide, connectivity=8)
            for cid in range(1, count):
                area = int(cc_stats[cid, cv2.CC_STAT_AREA])
                if area < args.min_area:
                    continue
                comp, sulf_crop = crop_component(labels, sulfide, cid, cc_stats[cid], args.close_kernel_px)
                footprint = reconstructed_footprint(comp, args.close_kernel_px)
                footprint_area = int(footprint.sum())
                dark_inside_ratio = int(((footprint > 0) & (sulf_crop == 0)).sum()) / max(footprint_area, 1)

                sol_raw = component_solidity(comp)
                cmp_raw = compactness(int(comp.sum()), component_perimeter(comp))
                comp_s = cv2.morphologyEx(comp, cv2.MORPH_OPEN, kernel)
                if int(comp_s.sum()) == 0:  # opening erased a thin grain -> keep raw
                    comp_s = comp
                else:
                    # OPEN can split a ragged grain; keep the largest blob so
                    # solidity/compactness stay single-component (solidity ≤ 1).
                    n_s, lab_s, st_s, _ = cv2.connectedComponentsWithStats(comp_s, connectivity=8)
                    if n_s > 2:
                        largest = 1 + int(np.argmax(st_s[1:, cv2.CC_STAT_AREA]))
                        comp_s = (lab_s == largest).astype(np.uint8)
                sol_s = component_solidity(comp_s)
                cmp_s = compactness(int(comp_s.sum()), component_perimeter(comp_s))

                fine_raw = is_fine(dark_inside_ratio, sol_raw, cmp_raw)
                fine_smooth = is_fine(dark_inside_ratio, sol_s, cmp_s)
                boundary_only = (dark_inside_ratio < FINE_DARK_INSIDE_RATIO) and (
                    sol_raw <= FINE_SOLIDITY_MAX or cmp_raw <= FINE_COMPACTNESS_MAX
                )
                replacement_fine = dark_inside_ratio >= FINE_DARK_INSIDE_RATIO

                s["grains"] += 1
                s["raw_fine"] += int(fine_raw)
                s["smooth_fine"] += int(fine_smooth)
                s["boundary_only_raw"] += int(boundary_only)
                if boundary_only and not fine_smooth:
                    s["boundary_only_flipped_ordinary"] += 1
                    if len(examples) < args.examples:
                        examples.append(
                            f"  {grade[:12]:12s} dir={dark_inside_ratio:.2f} "
                            f"solidity {sol_raw:.2f}→{sol_s:.2f}  compact {cmp_raw:.3f}→{cmp_s:.3f}  fine→ordinary"
                        )
                if replacement_fine:
                    s["replacement_fine"] += 1
                    s["replacement_fine_kept"] += int(fine_smooth)
        stats[grade] = s

    print(f"# Variant B prototype — boundary OPEN radius {args.smooth_px}px (batch {args.batch_dir.name})\n")
    print(f"{'grade':22s} {'grains':>7} {'raw_fine':>9} {'smth_fine':>10} {'bnd_only':>9} {'flipped→ord':>12} {'repl_fine_kept':>15}")
    for grade in GRADES:
        s = stats.get(grade, {})
        if not s.get("grains"):
            continue
        print(
            f"{grade:22s} {s['grains']:>7} {s['raw_fine']:>9} {s['smooth_fine']:>10} "
            f"{s['boundary_only_raw']:>9} {s['boundary_only_flipped_ordinary']:>12} "
            f"{s['replacement_fine_kept']:>7}/{s['replacement_fine']:<7}"
        )
    print("\nExamples of boundary-only 'fine' that smoothing corrects to 'ordinary':")
    print("\n".join(examples) if examples else "  (none in this sample)")
    print(
        "\nRead: high 'flipped→ord' vs 'bnd_only' = smoothing removes ragged-edge false positives; "
        "'repl_fine_kept' ≈ 'replacement_fine' = genuinely replaced grains are preserved."
    )
    return 0


def resolve(run_dir: Path) -> Path:
    return run_dir if run_dir.is_absolute() else (ROOT / run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
