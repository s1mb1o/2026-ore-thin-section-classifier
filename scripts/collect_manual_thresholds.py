#!/usr/bin/env python3
"""Collect per-sample manual talc thresholds saved from the review app.

Each sample dir may contain a `manual_threshold.json` written by the
`Save threshold` button (POST /api/samples/<id>/save_threshold). This tool
gathers them into one CSV and prints a quick fit of

    threshold = k * matrix_mean          (and vs matrix_mode)

so the relationship between the user-chosen threshold and the matrix
brightness statistics can be inspected before committing to a formula.

    python3 scripts/collect_manual_thresholds.py \
      --workspace outputs/talc_annotation_v1 \
      --out-csv outputs/talc_annotation_v1/manual_thresholds.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path


FIELDS = [
    "sample_id",
    "label",
    "threshold_luma",
    "matrix_mean",
    "matrix_median",
    "matrix_mode",
    "matrix_std",
    "k_vs_mean",
    "k_vs_mode",
    "ore_mask_source",
    "matrix_pixel_share",
    "p10",
    "p50",
    "p90",
    "saved_at",
]


def load_records(workspace: Path) -> list[dict]:
    records = []
    for path in sorted(glob.glob(str(workspace / "samples" / "*" / "manual_threshold.json"))):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        pct = data.get("matrix_percentiles") or {}
        records.append(
            {
                "sample_id": data.get("sample_id"),
                "label": data.get("label"),
                "threshold_luma": data.get("threshold_luma"),
                "matrix_mean": data.get("matrix_mean"),
                "matrix_median": data.get("matrix_median"),
                "matrix_mode": data.get("matrix_mode"),
                "matrix_std": data.get("matrix_std"),
                "k_vs_mean": data.get("k_vs_mean"),
                "k_vs_mode": data.get("k_vs_mode"),
                "ore_mask_source": data.get("ore_mask_source"),
                "matrix_pixel_share": data.get("matrix_pixel_share"),
                "p10": pct.get("p10"),
                "p50": pct.get("p50"),
                "p90": pct.get("p90"),
                "saved_at": data.get("saved_at"),
            }
        )
    return records


def summarize(records: list[dict]) -> None:
    if not records:
        print("No manual_threshold.json files found yet.")
        return
    import statistics as st

    ks_mean = [r["k_vs_mean"] for r in records if isinstance(r.get("k_vs_mean"), (int, float))]
    ks_mode = [r["k_vs_mode"] for r in records if isinstance(r.get("k_vs_mode"), (int, float))]
    print(f"collected: {len(records)} thresholds")
    if ks_mean:
        print(
            f"k vs matrix_mean: median {st.median(ks_mean):.3f}  "
            f"mean {st.mean(ks_mean):.3f}  min {min(ks_mean):.3f}  max {max(ks_mean):.3f}  "
            f"stdev {st.pstdev(ks_mean):.3f}"
        )
    if ks_mode:
        print(
            f"k vs matrix_mode: median {st.median(ks_mode):.3f}  "
            f"mean {st.mean(ks_mode):.3f}  min {min(ks_mode):.3f}  max {max(ks_mode):.3f}  "
            f"stdev {st.pstdev(ks_mode):.3f}"
        )

    # Least-squares slopes through the origin: threshold = k * feature.
    def slope_through_origin(xs, ys):
        num = sum(x * y for x, y in zip(xs, ys))
        den = sum(x * x for x in xs)
        return num / den if den else float("nan")

    for feat in ("matrix_mean", "matrix_mode", "matrix_median"):
        pairs = [
            (r[feat], r["threshold_luma"])
            for r in records
            if isinstance(r.get(feat), (int, float)) and isinstance(r.get("threshold_luma"), (int, float))
        ]
        if len(pairs) >= 3:
            xs, ys = zip(*pairs)
            k = slope_through_origin(xs, ys)
            resid = [y - k * x for x, y in pairs]
            mae = sum(abs(r) for r in resid) / len(resid)
            print(f"fit threshold = k*{feat}:  k={k:.3f}  MAE={mae:.2f} luma  (n={len(pairs)})")

    by_label: dict[str, list[float]] = {}
    for r in records:
        if isinstance(r.get("k_vs_mean"), (int, float)):
            by_label.setdefault(str(r.get("label")), []).append(r["k_vs_mean"])
    if len(by_label) > 1:
        print("k vs mean by group:")
        for lab, v in sorted(by_label.items()):
            v.sort()
            print(f"  {lab:14s} n={len(v):2d}  median {v[len(v)//2]:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    records = load_records(args.workspace)
    out_csv = args.out_csv or (args.workspace / "manual_thresholds.csv")
    if records:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        print(f"wrote {out_csv}")
    summarize(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
