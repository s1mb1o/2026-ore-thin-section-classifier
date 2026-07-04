#!/usr/bin/env python3
"""Resident (single-load) drop-in replacement for run_official_batch.py.

Loads the sulfide segmentation model once and runs the full ore pipeline over the
official split in-process, instead of spawning a Python process and reloading the
checkpoint per image. Produces a schema-identical summary.csv/json by reusing
run_official_batch's row builder, so downstream evaluators are unaffected.

Same CLI as run_official_batch.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_official_batch as rob  # noqa: E402  (reuse identical helpers)
from ore_classifier.resident_pipeline import ResidentSulfidePipeline  # noqa: E402
from ore_classifier.rule_config_io import (  # noqa: E402
    add_rule_config_arguments,
    resolve_rule_config_from_args,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Resident single-load ore pipeline over the official split.")
    parser.add_argument("--split-json", type=Path, default=Path("outputs/official_balanced_eval_split.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--per-label", type=int, default=None)
    parser.add_argument("--max-total", type=int, default=None)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-component-area-px", type=int, default=128)
    parser.add_argument("--close-kernel-px", type=int, default=21)
    add_rule_config_arguments(parser)
    parser.add_argument("--talc-checkpoint", type=Path, default=None, help="Optional trained talc segmentation checkpoint.")
    parser.add_argument("--talc-threshold", type=float, default=0.5)
    parser.add_argument("--talc-min-area-px", type=int, default=320)
    parser.add_argument("--preview-max-side", type=int, default=1800)
    parser.add_argument("--no-auto-talc-candidate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    rule_config = resolve_rule_config_from_args(args)

    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    selected = rob.select_items(
        split.get("items", []),
        labels=set(args.labels) if args.labels else None,
        per_label=args.per_label,
        max_total=args.max_total,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = ResidentSulfidePipeline(
        checkpoint=args.checkpoint,
        device=args.device,
        tile_size=args.tile_size,
        stride=args.stride,
        batch_size=args.batch_size,
        threshold=args.threshold,
        talc_checkpoint=args.talc_checkpoint,
        talc_threshold=args.talc_threshold,
        preview_max_side=args.preview_max_side,
    )
    print(
        f"[resident] model loaded once on {pipeline.device}: sulfide={pipeline.checkpoint_meta.get('model')} "
        f"talc={pipeline.talc_checkpoint_meta.get('model') if pipeline.talc_checkpoint_meta else 'none'}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, item in enumerate(selected, start=1):
        rel_path = Path(item["path"])
        image_path = args.dataset_root / rel_path
        source_label = item["label"]
        run_id = rob.safe_run_id(source_label, str(rel_path))
        run_dir = args.out_dir / "runs" / source_label / run_id
        pipeline_summary_path = run_dir / "pipeline_summary.json"
        print(f"[{index}/{len(selected)}] {source_label}: {rel_path}", flush=True)

        if args.overwrite or not pipeline_summary_path.exists():
            try:
                pipeline.run_image(
                    image_path=image_path,
                    out_dir=run_dir,
                    rule_config=rule_config,
                    min_component_area_px=args.min_component_area_px,
                    close_kernel_px=args.close_kernel_px,
                    talc_min_area_px=args.talc_min_area_px,
                    auto_talc_candidate=not args.no_auto_talc_candidate,
                )
            except Exception as exc:  # noqa: BLE001 - mirror run_official_batch failure handling
                failure = {"source_rel_path": str(rel_path), "source_label": source_label, "error": repr(exc)}
                failures.append(failure)
                if not args.keep_going:
                    rob.write_batch_outputs(args.out_dir, rows, failures)
                    raise
                continue

        rows.append(rob.build_summary_row(item=item, image_path=image_path, run_dir=run_dir))

    rob.write_batch_outputs(args.out_dir, rows, failures)
    print(json.dumps({"rows": len(rows), "failures": len(failures), "out_dir": str(args.out_dir), "resident": True}, ensure_ascii=False, indent=2))
    # Exit-code contract (see docs/plans/39, F6/§4): 0 = all images done; 3 = completed
    # with tolerated per-image failures under --keep-going (result set incomplete but
    # usable); 2 = fatal (every selected image failed -> no usable output). Callers such
    # as evaluate_official_pipeline.py must treat 3 as success-with-warnings, not abort.
    if not failures:
        return 0
    if not rows:
        print(f"[resident] FATAL: all {len(failures)} selected image(s) failed", file=sys.stderr, flush=True)
        return 2
    print(
        f"[resident] completed with {len(failures)} skipped/failed image(s) under --keep-going; "
        "result set is incomplete (see failures.json)",
        file=sys.stderr,
        flush=True,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
