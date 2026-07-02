#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


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
    parser.add_argument("--preview-max-side", type=int, default=1800)
    args = parser.parse_args()

    inference_dir = args.out_dir / "binary_sulfide"
    analysis_dir = args.out_dir / "ore_analysis"
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
    run(
        [
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
    )
    summary = {
        "schema_version": "ore-pipeline-run-v0.1",
        "image": str(args.image),
        "checkpoint": str(args.checkpoint),
        "paths": {
            "binary_sulfide_summary": str(inference_dir / "summary.json"),
            "sulfide_mask": str(inference_dir / "sulfide_mask.png"),
            "confidence": str(inference_dir / "confidence.png"),
            "sulfide_overlay_preview": str(inference_dir / "overlay_preview.jpg"),
            "ore_summary": str(analysis_dir / "ore_summary.json"),
            "component_features": str(analysis_dir / "component_features.csv"),
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
