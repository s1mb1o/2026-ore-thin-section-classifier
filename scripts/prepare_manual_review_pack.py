#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.review_queue import (  # noqa: E402
    build_review_queue,
    candidates_to_records,
    expert_questions_from_candidates,
)

Image.MAX_IMAGE_PIXELS = None


DEFAULT_CHECKPOINT = (
    ROOT
    / "models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare B2 ore-pipeline images, panels, and CSV templates for manual review."
    )
    parser.add_argument("--split-json", type=Path, default=ROOT / "outputs/official_balanced_eval_split.json")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/manual_review/b2_balanced_review_pack")
    parser.add_argument("--per-label", type=int, default=3)
    parser.add_argument("--panorama-count", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--preview-max-side", type=int, default=1800)
    parser.add_argument("--candidate-top-k", type=int, default=3)
    parser.add_argument("--candidate-threshold", type=float, default=0.75)
    parser.add_argument("--candidate-crop-side", type=int, default=640)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.per_label < 0:
        raise ValueError("--per-label must be non-negative")
    if args.panorama_count < 0:
        raise ValueError("--panorama-count must be non-negative")
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    split = read_json(args.split_json)
    selected = select_review_items(split, args.per_label, args.panorama_count)
    runs_dir = args.out_dir / "runs"
    reviews_dir = args.out_dir / "reviews"
    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for index, item in enumerate(selected, start=1):
        review_id = review_id_for(index, item)
        run_dir = runs_dir / review_id
        image_path = args.dataset_root / item["path"]
        run_pipeline(args, image_path, run_dir)
        row = finalize_run(args, item, review_id, image_path, run_dir)
        rows.append(row)
        candidate_rows.extend(write_candidate_crops(args, review_id, image_path, run_dir))

    write_csv(args.out_dir / "review_manifest.csv", rows)
    write_feedback_template(args.out_dir / "feedback_template.csv", rows)
    write_csv(args.out_dir / "review_candidates.csv", candidate_rows, fields=candidate_fields())
    write_json(
        args.out_dir / "review_manifest.json",
        {
            "schema_version": "manual-review-pack-v0.1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "split_json": str(args.split_json),
            "dataset_root": str(args.dataset_root),
            "checkpoint": str(args.checkpoint),
            "out_dir": str(args.out_dir),
            "runs_dir": str(runs_dir),
            "reviews_dir": str(reviews_dir),
            "per_label": args.per_label,
            "panorama_count": args.panorama_count,
            "items": rows,
            "candidate_count": len(candidate_rows),
            "streamlit_command": (
                f"streamlit run apps/deprecated/streamlit/sulfide_qa_streamlit.py -- "
                f"--runs-dir {runs_dir} --review-dir {reviews_dir}"
            ),
        },
    )
    write_readme(args.out_dir, rows, len(candidate_rows), runs_dir, reviews_dir)
    print(json.dumps({"out_dir": str(args.out_dir), "items": len(rows), "candidates": len(candidate_rows)}, ensure_ascii=False))
    return 0


def select_review_items(split: dict[str, Any], per_label: int, panorama_count: int) -> list[dict[str, Any]]:
    labels = split.get("labels", [])
    items = split.get("items", [])
    selected: list[dict[str, Any]] = []
    for label in labels:
        labelled = sorted([item for item in items if item.get("label") == label], key=lambda item: item["path"])
        selected.extend(pick_evenly(labelled, per_label))
    panoramas = sorted(split.get("panorama_items_unlabelled", []), key=lambda item: item["path"])
    for item in pick_evenly(panoramas, panorama_count):
        with_label = dict(item)
        with_label["label"] = "panorama_unlabelled"
        selected.append(with_label)
    return selected


def pick_evenly(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return items
    if count == 1:
        return [items[0]]
    indexes = np.linspace(0, len(items) - 1, count)
    return [items[int(round(index))] for index in indexes]


def review_id_for(index: int, item: dict[str, Any]) -> str:
    stem = Path(item["path"]).stem
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
    return f"{index:02d}_{item.get('label', 'unknown')}_{safe_stem}"


def run_pipeline(args: argparse.Namespace, image_path: Path, run_dir: Path) -> None:
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
        "--preview-max-side",
        str(args.preview_max_side),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def finalize_run(
    args: argparse.Namespace,
    item: dict[str, Any],
    review_id: str,
    image_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    pipeline_summary_path = run_dir / "pipeline_summary.json"
    pipeline_summary = read_json(pipeline_summary_path)
    binary_summary = read_json(run_dir / "binary_sulfide/summary.json")
    ore_summary = read_json(run_dir / "ore_analysis/ore_summary.json")

    source_preview_path = run_dir / "source_preview.jpg"
    confidence_heatmap_path = run_dir / "binary_sulfide/confidence_heatmap.jpg"
    review_panel_path = run_dir / "review_panel.jpg"
    save_source_preview(image_path, source_preview_path, args.preview_max_side)
    save_confidence_heatmap(run_dir / "binary_sulfide/confidence.png", confidence_heatmap_path)
    save_review_panel(
        image_paths=[
            ("Source", source_preview_path),
            ("Sulfide overlay", run_dir / "binary_sulfide/overlay_preview.jpg"),
            ("Confidence heatmap", confidence_heatmap_path),
            ("Ordinary / fine overlay", run_dir / "ore_analysis/intergrowth_overlay_preview.jpg"),
        ],
        out_path=review_panel_path,
    )

    paths = pipeline_summary.setdefault("paths", {})
    paths["source_preview"] = str(source_preview_path)
    paths["confidence_heatmap"] = str(confidence_heatmap_path)
    paths["review_panel"] = str(review_panel_path)
    write_json(pipeline_summary_path, pipeline_summary)

    row = {
        "review_id": review_id,
        "source_label": item.get("label", ""),
        "image_path": str(image_path),
        "relative_path": item.get("path", ""),
        "width": item.get("width", ""),
        "height": item.get("height", ""),
        "run_dir": str(run_dir),
        "review_panel": str(review_panel_path),
        "source_preview": str(source_preview_path),
        "sulfide_overlay": str(run_dir / "binary_sulfide/overlay_preview.jpg"),
        "confidence_heatmap": str(confidence_heatmap_path),
        "intergrowth_overlay": str(run_dir / "ore_analysis/intergrowth_overlay_preview.jpg"),
        "sulfide_mask": str(run_dir / "binary_sulfide/sulfide_mask.png"),
        "confidence": str(run_dir / "binary_sulfide/confidence.png"),
        "predicted_ore_class": ore_summary.get("ore_class", ""),
        "predicted_ore_class_ru": ore_summary.get("ore_class_ru", ""),
        "sulfide_fraction": binary_summary.get("sulfide_fraction", ""),
        "ordinary_sulfide_fraction": ore_summary.get("ordinary_sulfide_fraction", ""),
        "fine_sulfide_fraction": ore_summary.get("fine_sulfide_fraction", ""),
        "talc_fraction": ore_summary.get("talc_fraction", ""),
        "component_count": ore_summary.get("component_count", ""),
        "rule_text_ru": ore_summary.get("rule_text_ru", ""),
    }
    return row


def write_candidate_crops(
    args: argparse.Namespace,
    review_id: str,
    image_path: Path,
    run_dir: Path,
) -> list[dict[str, Any]]:
    confidence = np.asarray(Image.open(run_dir / "binary_sulfide/confidence.png").convert("L"), dtype=np.float32) / 255.0
    uncertainty = 1.0 - np.clip(np.abs(confidence - 0.5) / 0.5, 0.0, 1.0)
    candidates = build_review_queue(
        uncertainty,
        threshold=args.candidate_threshold,
        min_area_px=256,
        padding_px=64,
        top_k=args.candidate_top_k,
    )
    questions = expert_questions_from_candidates(candidates, review_id)
    write_json(run_dir / "review_candidates.json", {"candidates": candidates_to_records(candidates), "questions": questions})
    if not candidates:
        return []

    image = Image.open(image_path).convert("RGB")
    mask = np.asarray(Image.open(run_dir / "binary_sulfide/sulfide_mask.png").convert("L"))
    conf_u8 = np.asarray(Image.open(run_dir / "binary_sulfide/confidence.png").convert("L"))
    crop_dir = run_dir / "candidate_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for crop_index, candidate in enumerate(candidates, start=1):
        x0, y0, x1, y1 = centered_crop_box(
            int(round(candidate.centroid_x)),
            int(round(candidate.centroid_y)),
            args.candidate_crop_side,
            image.size,
        )
        source_crop = image.crop((x0, y0, x1, y1))
        mask_crop = mask[y0:y1, x0:x1]
        conf_crop = conf_u8[y0:y1, x0:x1]
        crop_path = crop_dir / f"candidate_{crop_index:02d}.jpg"
        save_candidate_panel(source_crop, mask_crop, conf_crop, crop_path)
        rows.append(
            {
                "review_id": review_id,
                "candidate_index": crop_index,
                "crop_path": str(crop_path),
                "bbox_x": candidate.x,
                "bbox_y": candidate.y,
                "bbox_w": candidate.width,
                "bbox_h": candidate.height,
                "crop_x0": x0,
                "crop_y0": y0,
                "crop_x1": x1,
                "crop_y1": y1,
                "score": candidate.score,
                "uncertainty": candidate.uncertainty,
                "reason": candidate.reason,
                "question_ru": questions[crop_index - 1]["question_ru"] if crop_index - 1 < len(questions) else "",
            }
        )
    return rows


def centered_crop_box(cx: int, cy: int, side: int, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    side = min(side, width, height)
    x0 = max(0, min(width - side, cx - side // 2))
    y0 = max(0, min(height - side, cy - side // 2))
    return x0, y0, x0 + side, y0 + side


def save_source_preview(image_path: Path, out_path: Path, max_side: int) -> None:
    image = Image.open(image_path).convert("RGB")
    if max_side and max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=92, optimize=True)


def save_confidence_heatmap(confidence_path: Path, out_path: Path) -> None:
    confidence = np.asarray(Image.open(confidence_path).convert("L"))
    colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    heatmap_bgr = cv2.applyColorMap(confidence, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(heatmap_rgb, mode="RGB").save(out_path, quality=92, optimize=True)


def save_review_panel(image_paths: list[tuple[str, Path]], out_path: Path) -> None:
    cell_w, cell_h = 760, 540
    title_h = 34
    panel = Image.new("RGB", (cell_w * 2, (cell_h + title_h) * 2), (245, 245, 245))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    for idx, (title, path) in enumerate(image_paths):
        x = (idx % 2) * cell_w
        y = (idx // 2) * (cell_h + title_h)
        draw.rectangle([x, y, x + cell_w, y + title_h], fill=(24, 28, 34))
        draw.text((x + 12, y + 10), title, fill=(255, 255, 255), font=font)
        image = Image.open(path).convert("RGB")
        image.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        ox = x + (cell_w - image.size[0]) // 2
        oy = y + title_h + (cell_h - image.size[1]) // 2
        panel.paste(image, (ox, oy))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path, quality=92, optimize=True)


def save_candidate_panel(source_crop: Image.Image, mask: np.ndarray, confidence: np.ndarray, out_path: Path) -> None:
    source = source_crop.convert("RGB")
    overlay = sulfide_overlay(source, mask, confidence)
    heatmap = confidence_heatmap(confidence)
    save_review_panel_from_images(
        [("Source crop", source), ("Mask overlay", overlay), ("Confidence", heatmap)],
        out_path,
    )


def save_review_panel_from_images(items: list[tuple[str, Image.Image]], out_path: Path) -> None:
    cell_w, cell_h = 480, 420
    title_h = 34
    panel = Image.new("RGB", (cell_w * len(items), cell_h + title_h), (245, 245, 245))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    for idx, (title, image) in enumerate(items):
        x = idx * cell_w
        draw.rectangle([x, 0, x + cell_w, title_h], fill=(24, 28, 34))
        draw.text((x + 12, 10), title, fill=(255, 255, 255), font=font)
        preview = image.copy()
        preview.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        ox = x + (cell_w - preview.size[0]) // 2
        oy = title_h + (cell_h - preview.size[1]) // 2
        panel.paste(preview, (ox, oy))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path, quality=92, optimize=True)


def sulfide_overlay(source: Image.Image, mask: np.ndarray, confidence: np.ndarray) -> Image.Image:
    base = np.asarray(source).astype(np.float32)
    mask_bool = mask > 0
    conf = confidence.astype(np.float32) / 255.0
    color = np.zeros_like(base)
    color[..., 0] = 255.0
    color[..., 1] = 216.0
    alpha = np.where(mask_bool, 0.25 + 0.45 * conf, 0.0)[..., None]
    overlay = base * (1.0 - alpha) + color * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")


def confidence_heatmap(confidence: np.ndarray) -> Image.Image:
    colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    heatmap_bgr = cv2.applyColorMap(confidence.astype(np.uint8), colormap)
    return Image.fromarray(cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB), mode="RGB")


def write_feedback_template(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "review_id",
        "source_label",
        "predicted_ore_class",
        "status",
        "error_types",
        "mask_feedback",
        "ordinary_fine_feedback",
        "talc_feedback",
        "reviewer",
        "notes",
        "review_panel",
        "run_dir",
    ]
    feedback_rows = []
    for row in rows:
        feedback_rows.append(
            {
                "review_id": row["review_id"],
                "source_label": row["source_label"],
                "predicted_ore_class": row["predicted_ore_class"],
                "status": "",
                "error_types": "",
                "mask_feedback": "",
                "ordinary_fine_feedback": "",
                "talc_feedback": "",
                "reviewer": "",
                "notes": "",
                "review_panel": row["review_panel"],
                "run_dir": row["run_dir"],
            }
        )
    write_csv(path, feedback_rows, fields=fields)


def candidate_fields() -> list[str]:
    return [
        "review_id",
        "candidate_index",
        "crop_path",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "crop_x0",
        "crop_y0",
        "crop_x1",
        "crop_y1",
        "score",
        "uncertainty",
        "reason",
        "question_ru",
    ]


def write_readme(out_dir: Path, rows: list[dict[str, Any]], candidate_count: int, runs_dir: Path, reviews_dir: Path) -> None:
    lines = [
        "# Manual Review Pack",
        "",
        f"Created: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Review items: `{len(rows)}`",
        f"- Candidate uncertainty crops: `{candidate_count}`",
        "- Open `review_manifest.csv` for the full index.",
        "- Fill `feedback_template.csv` for spreadsheet-based feedback.",
        "- Open each `review_panel.jpg` for quick visual review.",
        "",
        "Streamlit QA:",
        "",
        "```bash",
        f"streamlit run apps/deprecated/streamlit/sulfide_qa_streamlit.py -- --runs-dir {runs_dir} --review-dir {reviews_dir}",
        "```",
        "",
        "Suggested statuses: `accepted`, `needs_mask_fix`, `uncertain`, `exclude_artifact`, `bad_input`.",
        "Suggested error types: `missed_sulfide`, `false_sulfide`, `bad_boundary`, `wrong_ordinary_fine`, `talc_issue`, `artifact`.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        field_set: list[str] = []
        for row in rows:
            for key in row:
                if key not in field_set:
                    field_set.append(key)
        fields = field_set
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


if __name__ == "__main__":
    raise SystemExit(main())
