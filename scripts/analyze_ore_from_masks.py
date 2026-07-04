#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.component_analysis import (  # noqa: E402
    ComponentRuleConfig,
    analyze_components,
    save_component_outputs,
)
from ore_classifier.analyzed_area import build_analyzed_mask  # noqa: E402
from ore_classifier.component_grade_model import resolve_component_model  # noqa: E402
from ore_classifier.rule_config_io import add_rule_config_arguments, resolve_rule_config_from_args  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify ore type from sulfide/talc masks.")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--sulfide-mask", type=Path, required=True)
    parser.add_argument("--talc-mask", type=Path, default=None)
    parser.add_argument("--analyzed-mask", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-component-area-px", type=int, default=64)
    parser.add_argument("--close-kernel-px", type=int, default=15)
    add_rule_config_arguments(parser)
    parser.add_argument("--preview-max-side", type=int, default=1800)
    parser.add_argument(
        "--component-model",
        type=Path,
        default=None,
        help="Learned per-component grade classifier (model.joblib); omit or 'none' to use the rule.",
    )
    args = parser.parse_args()

    image = None if args.image is None else np.asarray(Image.open(args.image).convert("RGB"))
    sulfide_mask = np.asarray(Image.open(args.sulfide_mask).convert("L"))
    talc_mask = None if args.talc_mask is None else np.asarray(Image.open(args.talc_mask).convert("L"))
    if args.analyzed_mask is not None:
        analyzed_mask = np.asarray(Image.open(args.analyzed_mask).convert("L"))
    elif image is not None:
        analyzed_mask = build_analyzed_mask(image)
    else:
        analyzed_mask = None
    rule_config = resolve_rule_config_from_args(args)
    cfg = ComponentRuleConfig(
        min_component_area_px=args.min_component_area_px,
        close_kernel_px=args.close_kernel_px,
        fine_dark_inside_ratio=rule_config["fine_dark_inside_ratio"],
        fine_solidity_max=rule_config["fine_solidity_max"],
        fine_compactness_max=rule_config["fine_compactness_max"],
        talc_fraction_threshold=rule_config["talc_fraction_threshold"],
    )
    grade_model = resolve_component_model(args.component_model)
    component_classifier = None
    if grade_model is not None:
        component_classifier = grade_model.labeler(args.image.name if args.image is not None else None)
    summary, components, classified = analyze_components(
        sulfide_mask=sulfide_mask,
        talc_mask=talc_mask,
        analyzed_mask=analyzed_mask,
        config=cfg,
        component_classifier=component_classifier,
    )
    paths = save_component_outputs(
        out_dir=args.out_dir,
        summary=summary,
        components=components,
        classified_mask=classified,
        original_image=image,
        talc_mask=talc_mask,
        analyzed_mask=analyzed_mask,
        preview_max_side=args.preview_max_side,
    )
    output = {"summary": summary.__dict__, "rule_config": rule_config, "paths": paths}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
