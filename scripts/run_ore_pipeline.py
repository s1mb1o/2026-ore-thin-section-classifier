#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.talc_candidate import (  # noqa: E402
    TalcCandidateConfig,
    estimate_talc_candidate_mask,
    save_talc_candidate_outputs,
)

Image.MAX_IMAGE_PIXELS = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run image -> sulfide mask -> ore summary pipeline.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-component-area-px", type=int, default=128)
    parser.add_argument("--close-kernel-px", type=int, default=21)
    parser.add_argument("--talc-mask", type=Path, default=None, help="Optional accepted/manual talc mask to pass into ore analysis.")
    parser.add_argument(
        "--auto-talc-candidate",
        action="store_true",
        help="Generate a conservative color-heuristic talc candidate mask and pass it into ore analysis.",
    )
    parser.add_argument("--talc-min-area-px", type=int, default=320)
    parser.add_argument("--preview-max-side", type=int, default=1800)
    args = parser.parse_args()

    if args.talc_mask is not None and args.auto_talc_candidate:
        raise ValueError("--talc-mask and --auto-talc-candidate are mutually exclusive")

    inference_dir = args.out_dir / "binary_sulfide"
    analysis_dir = args.out_dir / "ore_analysis"
    talc_dir = args.out_dir / "talc_candidate"
    run(
        [
            sys.executable,
            "scripts/infer_binary_sulfide.py",
            "--image",
            str(args.image),
            "--checkpoint",
            str(args.checkpoint),
            "--out-dir",
            str(inference_dir),
            "--tile-size",
            str(args.tile_size),
            "--stride",
            str(args.stride),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--threshold",
            str(args.threshold),
            "--preview-max-side",
            str(args.preview_max_side),
        ]
    )

    talc_mask_path: Path | None = None
    talc_paths: dict[str, str] = {}
    talc_source = "none"
    if args.talc_mask is not None:
        talc_mask_path = args.talc_mask
        talc_source = "provided_mask"
    elif args.auto_talc_candidate:
        image_arr = np.asarray(Image.open(args.image).convert("RGB"))
        sulfide_arr = np.asarray(Image.open(inference_dir / "sulfide_mask.png").convert("L"))
        cfg = TalcCandidateConfig(min_area_px=args.talc_min_area_px)
        talc_arr = estimate_talc_candidate_mask(image_arr, sulfide_mask=sulfide_arr, config=cfg)
        talc_paths = save_talc_candidate_outputs(
            out_dir=talc_dir,
            rgb=image_arr,
            talc_mask=talc_arr,
            sulfide_mask=sulfide_arr,
            config=cfg,
            preview_max_side=args.preview_max_side,
        )
        talc_mask_path = Path(talc_paths["talc_candidate_mask"])
        talc_source = "auto_candidate"

    analyze_cmd = [
        sys.executable,
        "scripts/analyze_ore_from_masks.py",
        "--image",
        str(args.image),
        "--sulfide-mask",
        str(inference_dir / "sulfide_mask.png"),
        "--out-dir",
        str(analysis_dir),
        "--min-component-area-px",
        str(args.min_component_area_px),
        "--close-kernel-px",
        str(args.close_kernel_px),
        "--preview-max-side",
        str(args.preview_max_side),
    ]
    if talc_mask_path is not None:
        analyze_cmd.extend(["--talc-mask", str(talc_mask_path)])
    run(analyze_cmd)

    summary = {
        "schema_version": "ore-pipeline-run-v0.2",
        "image": str(args.image),
        "checkpoint": str(args.checkpoint),
        "talc_source": talc_source,
        "paths": {
            "binary_sulfide_summary": str(inference_dir / "summary.json"),
            "sulfide_mask": str(inference_dir / "sulfide_mask.png"),
            "confidence": str(inference_dir / "confidence.png"),
            "analyzed_mask": str(inference_dir / "analyzed_mask.png"),
            "sulfide_overlay_preview": str(inference_dir / "overlay_preview.jpg"),
            "talc_mask": str(talc_mask_path) if talc_mask_path is not None else None,
            "talc_candidate_summary": talc_paths.get("talc_candidate_summary"),
            "talc_candidate_overlay_preview": talc_paths.get("talc_candidate_overlay_preview"),
            "ore_summary": str(analysis_dir / "ore_summary.json"),
            "component_features": str(analysis_dir / "component_features.csv"),
            "analysis_analyzed_mask": str(analysis_dir / "analyzed_mask.png"),
            "intergrowth_overlay_preview": str(analysis_dir / "intergrowth_overlay_preview.jpg"),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pipeline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
