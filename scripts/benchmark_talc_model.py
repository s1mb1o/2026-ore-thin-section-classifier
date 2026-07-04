#!/usr/bin/env python3
"""Benchmark held-out talc segmentation fold checkpoints on full images."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_talc_dataset import (  # noqa: E402
    DEFAULT_CLEAN_IMAGE_DIR,
    DEFAULT_CONVERSION_DIR,
    list_reviewed_samples,
)
from infer_talc_segmentation import preprocess_tile, tile_weight  # noqa: E402
from ore_classifier.analyzed_area import build_analyzed_mask  # noqa: E402
from ore_classifier.model_io import (  # noqa: E402
    forward_logits,
    load_binary_segmentation_checkpoint,
    resolve_device,
)
from ore_classifier.tiling import iter_tiles  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds-dir", type=Path, default=Path("outputs/talc_segformer_folds/segformer_b0_full_20260703"))
    parser.add_argument("--conversion-dir", type=Path, default=DEFAULT_CONVERSION_DIR)
    parser.add_argument("--clean-image-dir", type=Path, default=DEFAULT_CLEAN_IMAGE_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/benchmarks/talc_model_full_image"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--analyzed-min-value", type=int, default=8)
    parser.add_argument("--skip-boundary", action="store_true", help="Skip Hausdorff/HD95 distance metrics.")
    parser.add_argument("--max-samples", type=int, default=0, help="Debug cap; 0 evaluates all held-out fold samples.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.stride > args.tile_size:
        raise ValueError("--stride must be <= --tile-size")
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.out_dir} is not empty; pass --overwrite")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    folds_dir = resolve_path(args.folds_dir)
    fold_summary_path = folds_dir / "summary.json"
    fold_summary = json.loads(fold_summary_path.read_text(encoding="utf-8"))
    stats: defaultdict[str, int] = defaultdict(int)
    samples = {
        sample["sample_id"]: sample
        for sample in list_reviewed_samples(
            resolve_path(args.conversion_dir),
            resolve_path(args.clean_image_dir),
            stats,
        )
    }
    if not samples:
        raise RuntimeError("no reviewed talc samples found")

    device = resolve_device(args.device)
    started = time.time()
    rows: list[dict] = []
    global_counts = empty_counts()
    baseline_counts = empty_counts()
    fold_rows = []
    samples_done = 0

    for fold in fold_summary["fold_summaries"]:
        checkpoint = resolve_path(Path(fold["checkpoint"]))
        threshold = float(fold["best_threshold_metrics"]["threshold"])
        model, checkpoint_meta = load_binary_segmentation_checkpoint(checkpoint, device)
        model.eval()
        fold_counts = empty_counts()
        fold_baseline_counts = empty_counts()
        fold_start = time.time()
        fold_sample_rows = []

        for sample_id in fold["val_samples"]:
            if args.max_samples and samples_done >= args.max_samples:
                break
            if sample_id not in samples:
                raise KeyError(f"fold sample is absent from reviewed sample list: {sample_id}")
            sample = samples[sample_id]
            sample_row, sample_counts, sample_baseline_counts = benchmark_sample(
                sample=sample,
                conversion_dir=resolve_path(args.conversion_dir),
                model=model,
                device=device,
                threshold=threshold,
                tile_size=args.tile_size,
                stride=args.stride,
                batch_size=args.batch_size,
                analyzed_min_value=args.analyzed_min_value,
                include_boundary=not args.skip_boundary,
            )
            sample_row.update(
                {
                    "fold_id": fold["fold_id"],
                    "threshold": threshold,
                    "checkpoint": str(checkpoint),
                }
            )
            rows.append(sample_row)
            fold_sample_rows.append(sample_row)
            add_counts(global_counts, sample_counts)
            add_counts(fold_counts, sample_counts)
            add_counts(baseline_counts, sample_baseline_counts)
            add_counts(fold_baseline_counts, sample_baseline_counts)
            samples_done += 1

        fold_metrics = metrics_from_counts(fold_counts)
        fold_baseline_metrics = metrics_from_counts(fold_baseline_counts)
        fold_rows.append(
            {
                "fold_id": fold["fold_id"],
                "threshold": threshold,
                "checkpoint": str(checkpoint),
                "sample_count": len(fold_sample_rows),
                "seconds": round(time.time() - fold_start, 3),
                "metrics": fold_metrics,
                "baseline_blue_line_metrics": fold_baseline_metrics,
                "fraction_mae_pp": mean_abs([r["fraction_error_pp"] for r in fold_sample_rows]),
                "fraction_bias_pp": mean([r["fraction_signed_error_pp"] for r in fold_sample_rows]),
                "fraction_within_3pp": fraction([abs(r["fraction_signed_error_pp"]) <= 3.0 for r in fold_sample_rows]),
                "checkpoint_meta": checkpoint_meta,
            }
        )
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        if args.max_samples and samples_done >= args.max_samples:
            break

    aggregate = metrics_from_counts(global_counts)
    baseline_aggregate = metrics_from_counts(baseline_counts)
    summary = {
        "schema_version": "talc-model-full-image-benchmark-v0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "folds_dir": str(folds_dir),
        "fold_summary": str(fold_summary_path),
        "model": fold_summary.get("model"),
        "device": str(device),
        "tile_size": args.tile_size,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "sample_count": len(rows),
        "source_reviewed_samples": len(samples),
        "seconds": round(time.time() - started, 3),
        "metrics": aggregate,
        "baseline_blue_line_metrics": baseline_aggregate,
        "fraction_metrics": fraction_summary(rows),
        "boundary_metrics": boundary_summary(rows) if not args.skip_boundary else None,
        "folds": fold_rows,
        "notes": [
            "Model predictions are clipped to analyzed non-sulfide pixels before pixel metrics.",
            "Fraction metrics use analyzed non-ignored pixels as denominator, matching ore-fraction reporting intent.",
            "Baseline is the original blue-line converter final_talc_mask.png, not the human-reviewed mask.",
            "Ground truth is non-expert reviewed masks from the talc review app, not independent geological ground truth.",
        ],
    }
    write_csv(args.out_dir / "per_sample_metrics.csv", rows)
    write_json(args.out_dir / "summary.json", summary)
    write_markdown(args.out_dir / "summary.md", summary, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


def benchmark_sample(
    *,
    sample: dict,
    conversion_dir: Path,
    model,
    device: torch.device,
    threshold: float,
    tile_size: int,
    stride: int,
    batch_size: int,
    analyzed_min_value: int,
    include_boundary: bool,
) -> tuple[dict, dict, dict]:
    sample_id = sample["sample_id"]
    sample_dir = conversion_dir / "samples" / sample_id
    sample_started = time.time()
    image = Image.open(sample["image_path"]).convert("RGB")
    width, height = image.size
    rgb = np.asarray(image, dtype=np.uint8)
    gt = load_mask(sample["talc_mask_path"], (height, width))
    reviewed_ignore = load_optional_mask(sample.get("ignore_mask_path"), (height, width))
    sulfide = load_optional_mask(sample.get("sulfide_mask_path"), (height, width))
    baseline_path = sample_dir / "final_talc_mask.png"
    baseline = load_optional_mask(baseline_path if baseline_path.exists() else None, (height, width))

    analyzed = build_analyzed_mask(rgb, min_value=analyzed_min_value).astype(bool)
    analyzed_eval = analyzed & ~reviewed_ignore
    valid = analyzed_eval & ~sulfide

    probability, tiles_processed = infer_probability(
        image=image,
        model=model,
        device=device,
        tile_size=tile_size,
        stride=stride,
        batch_size=batch_size,
    )
    pred = (probability >= threshold) & valid
    gt_valid = gt & valid
    baseline_pred = baseline & valid

    counts = counts_for(pred, gt_valid, valid)
    baseline_counts = counts_for(baseline_pred, gt_valid, valid)
    metrics = metrics_from_counts(counts)
    baseline_metrics = metrics_from_counts(baseline_counts)

    gt_fraction = safe_div(int((gt & analyzed_eval).sum()), int(analyzed_eval.sum()))
    pred_fraction = safe_div(int(pred.sum()), int(analyzed_eval.sum()))
    baseline_fraction = safe_div(int((baseline & analyzed_eval).sum()), int(analyzed_eval.sum()))
    row = {
        "sample_id": sample_id,
        "image_path": str(sample["image_path"]),
        "width": width,
        "height": height,
        "valid_px": int(valid.sum()),
        "analyzed_eval_px": int(analyzed_eval.sum()),
        "gt_talc_px": int(gt_valid.sum()),
        "pred_talc_px": int(pred.sum()),
        "baseline_talc_px": int(baseline_pred.sum()),
        "gt_fraction_analyzed": gt_fraction,
        "pred_fraction_analyzed": pred_fraction,
        "baseline_fraction_analyzed": baseline_fraction,
        "fraction_signed_error_pp": (pred_fraction - gt_fraction) * 100.0,
        "fraction_error_pp": abs(pred_fraction - gt_fraction) * 100.0,
        "baseline_fraction_error_pp": abs(baseline_fraction - gt_fraction) * 100.0,
        "iou_talc": metrics["iou_talc"],
        "f1_talc": metrics["f1_talc"],
        "precision_talc": metrics["precision_talc"],
        "recall_talc": metrics["recall_talc"],
        "pixel_acc": metrics["pixel_acc"],
        "baseline_iou_talc": baseline_metrics["iou_talc"],
        "baseline_f1_talc": baseline_metrics["f1_talc"],
        "tiles": tiles_processed,
        "seconds": round(time.time() - sample_started, 3),
    }
    if include_boundary:
        hd = boundary_distances(pred, gt_valid)
        row.update(hd)
    return row, counts, baseline_counts


@torch.no_grad()
def infer_probability(
    *,
    image: Image.Image,
    model,
    device: torch.device,
    tile_size: int,
    stride: int,
    batch_size: int,
) -> tuple[np.ndarray, int]:
    width, height = image.size
    tiles = iter_tiles(width=width, height=height, tile_size=tile_size, stride=stride)
    weight = tile_weight(tile_size)
    prob_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    processed = 0
    for index in range(0, len(tiles), batch_size):
        batch_tiles = tiles[index : index + batch_size]
        tensor = torch.stack([preprocess_tile(image, tile) for tile in batch_tiles]).to(device)
        logits = forward_logits(model, tensor, (tile_size, tile_size))
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().astype(np.float32)
        for tile, prob in zip(batch_tiles, probs, strict=True):
            valid_h = min(tile.height, height - tile.y)
            valid_w = min(tile.width, width - tile.x)
            y_slice = slice(tile.y, tile.y + valid_h)
            x_slice = slice(tile.x, tile.x + valid_w)
            tile_weight_valid = weight[:valid_h, :valid_w]
            prob_sum[y_slice, x_slice] += prob[:valid_h, :valid_w] * tile_weight_valid
            weight_sum[y_slice, x_slice] += tile_weight_valid
            processed += 1
    return np.asarray(prob_sum / np.maximum(weight_sum, 1e-6), dtype=np.float32), processed


def load_mask(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        expected = (target_hw[1], target_hw[0])
        if image.size != expected:
            image = image.resize(expected, Image.Resampling.NEAREST)
        return np.asarray(image, dtype=np.uint8) > 0


def load_optional_mask(path: Path | None, target_hw: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(target_hw, dtype=bool)
    return load_mask(path, target_hw)


def counts_for(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict:
    pred = pred & valid
    target = target & valid
    return {
        "tp": int((pred & target).sum()),
        "fp": int((pred & ~target & valid).sum()),
        "fn": int((~pred & target & valid).sum()),
        "tn": int((~pred & ~target & valid).sum()),
        "valid": int(valid.sum()),
    }


def empty_counts() -> dict:
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "valid": 0}


def add_counts(total: dict, part: dict) -> None:
    for key in total:
        total[key] += int(part.get(key, 0))


def metrics_from_counts(counts: dict) -> dict:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    valid = counts["valid"]
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "iou_talc": safe_div(tp, tp + fp + fn),
        "iou_not_talc": safe_div(tn, tn + fp + fn),
        "precision_talc": precision,
        "recall_talc": recall,
        "f1_talc": f1,
        "pixel_acc": safe_div(tp + tn, valid),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "valid": valid,
    }


def boundary_distances(pred: np.ndarray, target: np.ndarray) -> dict:
    pred_boundary = mask_boundary(pred)
    target_boundary = mask_boundary(target)
    height, width = pred.shape
    diagonal = float((height * height + width * width) ** 0.5)
    if not pred_boundary.any() and not target_boundary.any():
        return {"hausdorff_px": 0.0, "hd95_px": 0.0}
    if not pred_boundary.any() or not target_boundary.any():
        return {"hausdorff_px": diagonal, "hd95_px": diagonal}

    dist_to_target = ndimage.distance_transform_edt(~target_boundary)
    dist_to_pred = ndimage.distance_transform_edt(~pred_boundary)
    distances = np.concatenate([dist_to_target[pred_boundary], dist_to_pred[target_boundary]])
    return {
        "hausdorff_px": float(distances.max()) if distances.size else 0.0,
        "hd95_px": float(np.percentile(distances, 95)) if distances.size else 0.0,
    }


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return mask ^ eroded


def fraction_summary(rows: list[dict]) -> dict:
    errors = [r["fraction_error_pp"] for r in rows]
    signed = [r["fraction_signed_error_pp"] for r in rows]
    baseline_errors = [r["baseline_fraction_error_pp"] for r in rows]
    return {
        "mae_pp": mean_abs(errors),
        "median_abs_error_pp": percentile(errors, 50),
        "p90_abs_error_pp": percentile(errors, 90),
        "max_abs_error_pp": max(errors) if errors else None,
        "signed_bias_pp": mean(signed),
        "within_3pp_fraction": fraction([abs(v) <= 3.0 for v in signed]),
        "baseline_blue_line_mae_pp": mean_abs(baseline_errors),
        "baseline_blue_line_median_abs_error_pp": percentile(baseline_errors, 50),
    }


def boundary_summary(rows: list[dict]) -> dict:
    hd = [r["hausdorff_px"] for r in rows if "hausdorff_px" in r]
    hd95 = [r["hd95_px"] for r in rows if "hd95_px" in r]
    return {
        "hausdorff_mean_px": mean(hd),
        "hausdorff_median_px": percentile(hd, 50),
        "hd95_mean_px": mean(hd95),
        "hd95_median_px": percentile(hd95, 50),
    }


def write_markdown(path: Path, summary: dict, rows: list[dict]) -> None:
    metrics = summary["metrics"]
    baseline = summary["baseline_blue_line_metrics"]
    frac = summary["fraction_metrics"]
    boundary = summary["boundary_metrics"] or {}
    lines = [
        "# Talc Model Full-Image Benchmark",
        "",
        f"Created: `{summary['created_at']}`",
        "",
        "## Scope",
        "",
        f"- Fold run: `{summary['folds_dir']}`",
        f"- Model: `{summary.get('model')}`",
        f"- Samples: `{summary['sample_count']}` held-out reviewed talc images",
        f"- Device: `{summary['device']}`",
        f"- Tiling: `{summary['tile_size']}` / `{summary['stride']}`, batch `{summary['batch_size']}`",
        "- Evaluation: predictions clipped to analyzed non-sulfide pixels; fraction denominator is analyzed non-ignored pixels.",
        "- Ground truth caveat: reviewed masks are non-expert QA masks from the blue-line talc workflow, not independent geological ground truth.",
        "",
        "## Aggregate Metrics",
        "",
        "| Source | Talc IoU | Talc F1 | Precision | Recall | Pixel acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        format_metric_row("SegFormer-B0 held-out folds", metrics),
        format_metric_row("Blue-line converter baseline", baseline),
        "",
        "## Talc Fraction Error",
        "",
        f"- MAE: `{frac['mae_pp']:.3f}` percentage points",
        f"- Median absolute error: `{frac['median_abs_error_pp']:.3f}` pp",
        f"- P90 absolute error: `{frac['p90_abs_error_pp']:.3f}` pp",
        f"- Signed bias: `{frac['signed_bias_pp']:.3f}` pp",
        f"- Within ±3 pp: `{frac['within_3pp_fraction']:.3f}`",
        f"- Blue-line baseline MAE: `{frac['baseline_blue_line_mae_pp']:.3f}` pp",
        "",
        "## Boundary Metrics",
        "",
    ]
    if boundary:
        lines.extend(
            [
                f"- Hausdorff mean: `{boundary['hausdorff_mean_px']:.2f}` px",
                f"- Hausdorff median: `{boundary['hausdorff_median_px']:.2f}` px",
                f"- HD95 mean: `{boundary['hd95_mean_px']:.2f}` px",
                f"- HD95 median: `{boundary['hd95_median_px']:.2f}` px",
            ]
        )
    else:
        lines.append("- Boundary metrics skipped.")
    lines.extend(
        [
            "",
            "## Fold Summary",
            "",
            "| Fold | Samples | Threshold | Talc IoU | Talc F1 | Fraction MAE pp | Within ±3 pp | Seconds |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for fold in summary["folds"]:
        lines.append(
            "| {fold_id} | {sample_count} | {threshold:.2f} | {iou:.4f} | {f1:.4f} | {mae:.3f} | {within:.3f} | {seconds:.1f} |".format(
                fold_id=fold["fold_id"],
                sample_count=fold["sample_count"],
                threshold=fold["threshold"],
                iou=fold["metrics"]["iou_talc"],
                f1=fold["metrics"]["f1_talc"],
                mae=fold["fraction_mae_pp"],
                within=fold["fraction_within_3pp"],
                seconds=fold["seconds"],
            )
        )
    worst = sorted(rows, key=lambda row: row["fraction_error_pp"], reverse=True)[:8]
    lines.extend(
        [
            "",
            "## Worst Fraction Errors",
            "",
            "| Sample | GT frac | Pred frac | Error pp | IoU | F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in worst:
        lines.append(
            f"| `{row['sample_id']}` | {row['gt_fraction_analyzed']:.4f} | {row['pred_fraction_analyzed']:.4f} | {row['fraction_error_pp']:.3f} | {row['iou_talc']:.4f} | {row['f1_talc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `summary.json`",
            "- `per_sample_metrics.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_metric_row(label: str, metrics: dict) -> str:
    return (
        f"| {label} | {metrics['iou_talc']:.4f} | {metrics['f1_talc']:.4f} | "
        f"{metrics['precision_talc']:.4f} | {metrics['recall_talc']:.4f} | {metrics['pixel_acc']:.4f} |"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def mean_abs(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(abs(value) for value in values) / len(values))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def fraction(values: list[bool]) -> float | None:
    if not values:
        return None
    return float(sum(1 for value in values if value) / len(values))


if __name__ == "__main__":
    raise SystemExit(main())
