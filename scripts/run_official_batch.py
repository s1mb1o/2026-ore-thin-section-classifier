#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


LABEL_TO_ORE_CLASS = {
    "ordinary_intergrowth": "row_ore",
    "fine_intergrowth": "hard_to_process_ore",
    "talcose": "talcose_ore",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full ore pipeline over the official balanced image split.")
    parser.add_argument("--split-json", type=Path, default=Path("outputs/official_balanced_eval_split.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--labels", nargs="*", default=None, help="Optional source labels to include.")
    parser.add_argument("--per-label", type=int, default=None)
    parser.add_argument("--max-total", type=int, default=None)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-component-area-px", type=int, default=128)
    parser.add_argument("--close-kernel-px", type=int, default=21)
    parser.add_argument("--talc-min-area-px", type=int, default=320)
    parser.add_argument("--preview-max-side", type=int, default=1800)
    parser.add_argument("--no-auto-talc-candidate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    selected = select_items(
        split.get("items", []),
        labels=set(args.labels) if args.labels else None,
        per_label=args.per_label,
        max_total=args.max_total,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for index, item in enumerate(selected, start=1):
        rel_path = Path(item["path"])
        image_path = args.dataset_root / rel_path
        source_label = item["label"]
        run_id = safe_run_id(source_label, str(rel_path))
        run_dir = args.out_dir / "runs" / source_label / run_id
        pipeline_summary_path = run_dir / "pipeline_summary.json"
        print(f"[{index}/{len(selected)}] {source_label}: {rel_path}", flush=True)

        if args.overwrite or not pipeline_summary_path.exists():
            cmd = [
                sys.executable,
                "scripts/run_ore_pipeline.py",
                "--image",
                str(image_path),
                "--checkpoint",
                str(args.checkpoint),
                "--out-dir",
                str(run_dir),
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
                "--min-component-area-px",
                str(args.min_component_area_px),
                "--close-kernel-px",
                str(args.close_kernel_px),
                "--talc-min-area-px",
                str(args.talc_min_area_px),
                "--preview-max-side",
                str(args.preview_max_side),
            ]
            if not args.no_auto_talc_candidate:
                cmd.append("--auto-talc-candidate")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                failure = {"source_rel_path": str(rel_path), "source_label": source_label, "error": str(exc)}
                failures.append(failure)
                if not args.keep_going:
                    write_batch_outputs(args.out_dir, rows, failures)
                    raise
                continue

        rows.append(build_summary_row(item=item, image_path=image_path, run_dir=run_dir))

    write_batch_outputs(args.out_dir, rows, failures)
    print(json.dumps({"rows": len(rows), "failures": len(failures), "out_dir": str(args.out_dir)}, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


def select_items(
    items: list[dict[str, Any]],
    labels: set[str] | None,
    per_label: int | None,
    max_total: int | None,
) -> list[dict[str, Any]]:
    counts: dict[str, int] = defaultdict(int)
    selected: list[dict[str, Any]] = []
    for item in items:
        label = item.get("label", "")
        if labels is not None and label not in labels:
            continue
        if per_label is not None and counts[label] >= per_label:
            continue
        selected.append(item)
        counts[label] += 1
        if max_total is not None and len(selected) >= max_total:
            break
    return selected


def build_summary_row(item: dict[str, Any], image_path: Path, run_dir: Path) -> dict[str, Any]:
    pipeline = read_json(run_dir / "pipeline_summary.json")
    binary = read_json(Path(pipeline["paths"]["binary_sulfide_summary"]))
    ore = read_json(Path(pipeline["paths"]["ore_summary"]))
    talc_summary_path = pipeline["paths"].get("talc_candidate_summary")
    talc_candidate = read_json(Path(talc_summary_path)) if talc_summary_path else {}
    paths = pipeline.get("paths", {})
    source_label = item["label"]
    return {
        "run_id": run_dir.name,
        "source_label": source_label,
        "expected_ore_class": LABEL_TO_ORE_CLASS.get(source_label, ""),
        "source_rel_path": item["path"],
        "source_dataset_path": str(image_path),
        "width": item.get("width", binary.get("width", "")),
        "height": item.get("height", binary.get("height", "")),
        "predicted_ore_class": ore.get("ore_class", ""),
        "predicted_ore_class_ru": ore.get("ore_class_ru", ""),
        "sulfide_fraction": ore.get("sulfide_fraction", ""),
        "ordinary_sulfide_fraction": ore.get("ordinary_sulfide_fraction", ""),
        "fine_sulfide_fraction": ore.get("fine_sulfide_fraction", ""),
        "talc_fraction": ore.get("talc_fraction", ""),
        "talc_source": pipeline.get("talc_source", ""),
        "talc_candidate_fraction": talc_candidate.get("talc_candidate_fraction", ""),
        "component_count": ore.get("component_count", ""),
        "ordinary_component_count": ore.get("ordinary_component_count", ""),
        "fine_component_count": ore.get("fine_component_count", ""),
        "binary_sulfide_fraction": binary.get("sulfide_fraction", ""),
        "binary_inference_seconds": binary.get("seconds", ""),
        "run_dir": str(run_dir),
        "sulfide_mask": paths.get("sulfide_mask", ""),
        "confidence": paths.get("confidence", ""),
        "talc_mask": paths.get("talc_mask", ""),
        "ore_summary": paths.get("ore_summary", ""),
        "intergrowth_overlay_preview": paths.get("intergrowth_overlay_preview", ""),
    }


def write_batch_outputs(out_dir: Path, rows: list[dict[str, Any]], failures: list[dict[str, str]]) -> None:
    summary_csv = out_dir / "summary.csv"
    summary_json = out_dir / "summary.json"
    failures_json = out_dir / "failures.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        summary_csv.write_text("", encoding="utf-8")
    summary_json.write_text(json.dumps({"rows": rows, "failures": failures}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failures_json.write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_run_id(label: str, rel_path: str) -> str:
    digest = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:12]
    stem = Path(rel_path).stem.lower()
    safe_stem = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")[:48]
    return f"{label}_{safe_stem}_{digest}"


if __name__ == "__main__":
    raise SystemExit(main())
