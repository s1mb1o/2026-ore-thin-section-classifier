#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ore_classifier.talc_blue_line_converter import (  # noqa: E402
    TalcConversionConfig,
    convert_talc_annotation_folder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert blue-line talc annotations into QA-ready mask candidates.")
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory with blue-line talc annotations.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for candidate masks and manifest.")
    parser.add_argument("--sulfide-mask-dir", type=Path, default=None, help="Optional directory with precomputed binary sulfide masks named by image stem.")
    parser.add_argument(
        "--silicate-mask-dir",
        type=Path,
        default=None,
        help="Optional directory with binary silicon/silicate support masks named by image stem.",
    )
    parser.add_argument("--sulfide-mode", choices=["heuristic", "none"], default="heuristic")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--blue-hue-min", type=int, default=90)
    parser.add_argument("--blue-hue-max", type=int, default=135)
    parser.add_argument("--gap-close-px", type=int, default=25)
    parser.add_argument("--line-dilate-px", type=int, default=5)
    parser.add_argument("--markup-ignore-dilate-px", type=int, default=4)
    parser.add_argument("--min-region-area-px", type=int, default=600)
    parser.add_argument("--sulfide-bright-percentile", type=float, default=88.0)
    parser.add_argument("--talc-positive-core-erode-px", type=int, default=2)
    parser.add_argument("--silicate-hard-negative-margin-px", type=int, default=4)
    parser.add_argument("--fallback-hull", action="store_true", help="Enable aggressive convex-hull fallback for open blue strokes.")
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional path for a copy of the manifest JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TalcConversionConfig(
        blue_hue_min=args.blue_hue_min,
        blue_hue_max=args.blue_hue_max,
        line_dilate_px=args.line_dilate_px,
        gap_close_px=args.gap_close_px,
        markup_ignore_dilate_px=args.markup_ignore_dilate_px,
        min_region_area_px=args.min_region_area_px,
        fallback_hull=args.fallback_hull,
        sulfide_mode=args.sulfide_mode,
        sulfide_bright_percentile=args.sulfide_bright_percentile,
        talc_positive_core_erode_px=args.talc_positive_core_erode_px,
        silicate_hard_negative_margin_px=args.silicate_hard_negative_margin_px,
    )
    manifest = convert_talc_annotation_folder(
        args.input,
        args.output_dir,
        config,
        sulfide_mask_dir=args.sulfide_mask_dir,
        silicate_mask_dir=args.silicate_mask_dir,
        limit=args.limit,
    )
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
