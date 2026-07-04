#!/usr/bin/env python3
"""Regenerate per-grain component_features from a completed batch's EXISTING masks.

Re-runs `analyze_components` over each run's already-computed
`binary_sulfide/{sulfide_mask,analyzed_mask}.png` with a chosen
`ComponentRuleConfig` (e.g. `boundary_smooth_px` for variant B) and writes a new
batch dir with `summary.csv` + `runs/<label>/<run_id>/ore_analysis/component_features.csv`.
No segmentation re-inference (cv2 only). Downstream `build_grain_dataset.py` /
`aggregate_grade_from_grains.py` can then run on the new batch to measure the
effect on the image grade.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.component_analysis import (  # noqa: E402
    ComponentRuleConfig,
    analyze_components,
    write_component_csv,
)

Image.MAX_IMAGE_PIXELS = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--boundary-smooth-px", type=int, default=0)
    parser.add_argument("--min-component-area-px", type=int, default=128)
    parser.add_argument("--close-kernel-px", type=int, default=21)
    args = parser.parse_args()

    rows = list(csv.DictReader((args.batch_dir / "summary.csv").open(encoding="utf-8")))
    cfg = ComponentRuleConfig(
        min_component_area_px=args.min_component_area_px,
        close_kernel_px=args.close_kernel_px,
        boundary_smooth_px=args.boundary_smooth_px,
    )
    out_rows: list[dict] = []
    done = 0
    for i, row in enumerate(rows, start=1):
        orig_run = resolve(Path(row.get("run_dir", "")))
        run_id = orig_run.name
        source_label = row.get("source_label", "")
        sm = orig_run / "binary_sulfide" / "sulfide_mask.png"
        am = orig_run / "binary_sulfide" / "analyzed_mask.png"
        if not sm.exists() or not am.exists():
            continue
        sulfide = np.asarray(Image.open(sm).convert("L"))
        analyzed = np.asarray(Image.open(am).convert("L"))
        _, components, _ = analyze_components(sulfide_mask=sulfide, talc_mask=None, analyzed_mask=analyzed, config=cfg)
        new_run = args.out_dir / "runs" / source_label / run_id
        (new_run / "ore_analysis").mkdir(parents=True, exist_ok=True)
        write_component_csv(new_run / "ore_analysis" / "component_features.csv", components)
        new_row = dict(row)
        new_row["run_dir"] = str(new_run)
        out_rows.append(new_row)
        done += 1
        if i % 50 == 0:
            print(f"[{i}/{len(rows)}] regenerated", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if out_rows:
        with (args.out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
    print(f"regenerated {done} runs (boundary_smooth_px={args.boundary_smooth_px}) -> {args.out_dir}", flush=True)
    return 0


def resolve(run_dir: Path) -> Path:
    return run_dir if run_dir.is_absolute() else (ROOT / run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
