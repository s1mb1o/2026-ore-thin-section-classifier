from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class TalcConversionConfig:
    blue_hue_min: int = 90
    blue_hue_max: int = 135
    blue_saturation_min: int = 70
    blue_value_min: int = 50
    blue_channel_margin: int = 35
    blue_channel_min: int = 90
    line_dilate_px: int = 5
    gap_close_px: int = 25
    markup_ignore_dilate_px: int = 4
    min_region_area_px: int = 600
    min_stroke_component_area_px: int = 200
    fallback_hull: bool = False
    max_hull_fraction: float = 0.75
    sulfide_mode: str = "heuristic"
    sulfide_bright_percentile: float = 88.0
    sulfide_min_area_px: int = 80
    sulfide_close_px: int = 5
    sulfide_dilate_px: int = 2
    talc_positive_core_erode_px: int = 2
    silicate_hard_negative_margin_px: int = 4
    overlay_alpha: float = 0.42


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_uint8_mask(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def read_image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def write_image_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image_rgb, 0, 255).astype(np.uint8), mode="RGB").save(path)


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), ensure_uint8_mask(mask))
    if not ok:
        raise RuntimeError(f"failed to write mask: {path}")


def read_mask(path: Path, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask: {path}")
    if shape_hw is not None and mask.shape[:2] != shape_hw:
        mask = cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return ensure_uint8_mask(mask)


def kernel_size(px: int) -> int:
    return max(1, int(px) * 2 + 1)


def ellipse_kernel(px: int) -> np.ndarray:
    size = kernel_size(px)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    binary = ensure_uint8_mask(mask)
    if min_area_px <= 1:
        return binary
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area >= min_area_px:
            out[labels == component_id] = 255
    return out


def erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    binary = ensure_uint8_mask(mask)
    if px <= 0:
        return binary
    return ensure_uint8_mask(cv2.erode(binary, ellipse_kernel(px), iterations=1))


def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    binary = ensure_uint8_mask(mask)
    if px <= 0:
        return binary
    return ensure_uint8_mask(cv2.dilate(binary, ellipse_kernel(px), iterations=1))


def detect_blue_stroke(image_rgb: np.ndarray, config: TalcConversionConfig) -> np.ndarray:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    hsv_mask = (
        (hue >= config.blue_hue_min)
        & (hue <= config.blue_hue_max)
        & (saturation >= config.blue_saturation_min)
        & (value >= config.blue_value_min)
    )
    red = image_rgb[:, :, 0].astype(np.int16)
    green = image_rgb[:, :, 1].astype(np.int16)
    blue = image_rgb[:, :, 2].astype(np.int16)
    channel_mask = (
        (blue >= config.blue_channel_min)
        & (blue - red >= config.blue_channel_margin)
        & (blue - green >= config.blue_channel_margin)
    )
    mask = ensure_uint8_mask(hsv_mask | channel_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse_kernel(1))
    return remove_small_components(mask, max(8, config.min_stroke_component_area_px // 8))


def close_stroke_boundaries(stroke_mask: np.ndarray, config: TalcConversionConfig) -> np.ndarray:
    closed = ensure_uint8_mask(stroke_mask)
    if config.line_dilate_px > 0:
        closed = cv2.dilate(closed, ellipse_kernel(config.line_dilate_px), iterations=1)
    if config.gap_close_px > 0:
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, ellipse_kernel(config.gap_close_px))
    return ensure_uint8_mask(closed)


def _border_seeds(free_mask: np.ndarray) -> list[tuple[int, int]]:
    height, width = free_mask.shape[:2]
    seeds: list[tuple[int, int]] = []
    for x in range(width):
        if free_mask[0, x] > 0:
            seeds.append((x, 0))
        if free_mask[height - 1, x] > 0:
            seeds.append((x, height - 1))
    for y in range(height):
        if free_mask[y, 0] > 0:
            seeds.append((0, y))
        if free_mask[y, width - 1] > 0:
            seeds.append((width - 1, y))
    return seeds


def fill_regions_from_barrier(barrier_mask: np.ndarray) -> np.ndarray:
    barrier = ensure_uint8_mask(barrier_mask)
    free = np.where(barrier > 0, 0, 255).astype(np.uint8)
    flood = free.copy()
    flood_mask = np.zeros((free.shape[0] + 2, free.shape[1] + 2), dtype=np.uint8)
    for seed in _border_seeds(free):
        if flood[seed[1], seed[0]] == 255:
            cv2.floodFill(flood, flood_mask, seed, 128)
    inside = (free == 255) & (flood != 128)
    return ensure_uint8_mask(inside)


def fill_component_hulls(stroke_mask: np.ndarray, config: TalcConversionConfig) -> np.ndarray:
    if not config.fallback_hull:
        return np.zeros_like(stroke_mask, dtype=np.uint8)
    stroke = ensure_uint8_mask(stroke_mask)
    height, width = stroke.shape[:2]
    max_area = float(height * width) * max(0.0, config.max_hull_fraction)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(stroke, connectivity=8)
    hull_mask = np.zeros_like(stroke)
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < config.min_stroke_component_area_px:
            continue
        component = labels == component_id
        ys, xs = np.where(component)
        if len(xs) < 3:
            continue
        points = np.column_stack([xs, ys]).astype(np.int32)
        hull = cv2.convexHull(points)
        hull_area = float(cv2.contourArea(hull))
        if hull_area < config.min_region_area_px or hull_area > max_area:
            continue
        cv2.fillPoly(hull_mask, [hull], 255)
    return ensure_uint8_mask(hull_mask)


def detect_bright_sulfide_mask(image_rgb: np.ndarray, config: TalcConversionConfig) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0]
    smooth = cv2.GaussianBlur(lightness, (0, 0), sigmaX=1.2, sigmaY=1.2)
    percentile_threshold = float(np.percentile(smooth, config.sulfide_bright_percentile))
    otsu_threshold, _ = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(percentile_threshold, float(otsu_threshold))
    mask = ensure_uint8_mask(smooth >= threshold)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse_kernel(1))
    if config.sulfide_close_px > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ellipse_kernel(config.sulfide_close_px))
    mask = remove_small_components(mask, config.sulfide_min_area_px)
    if config.sulfide_dilate_px > 0:
        mask = cv2.dilate(mask, ellipse_kernel(config.sulfide_dilate_px), iterations=1)
    return ensure_uint8_mask(mask)


def make_overlay(
    image_rgb: np.ndarray,
    *,
    talc_mask: np.ndarray | None = None,
    stroke_mask: np.ndarray | None = None,
    sulfide_mask: np.ndarray | None = None,
    overlap_mask: np.ndarray | None = None,
    ignore_mask: np.ndarray | None = None,
    alpha: float = 0.42,
) -> np.ndarray:
    overlay = image_rgb.astype(np.float32).copy()
    layers: list[tuple[np.ndarray | None, tuple[int, int, int], float]] = [
        (talc_mask, (0, 85, 255), alpha),
        (sulfide_mask, (255, 220, 55), 0.28),
        (ignore_mask, (180, 180, 180), 0.38),
        (overlap_mask, (255, 45, 45), 0.58),
        (stroke_mask, (0, 0, 255), 0.82),
    ]
    for mask, color, layer_alpha in layers:
        if mask is None:
            continue
        active = mask > 0
        if not np.any(active):
            continue
        color_array = np.array(color, dtype=np.float32)
        overlay[active] = overlay[active] * (1.0 - layer_alpha) + color_array * layer_alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def mask_pixels(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask > 0))


def confidence_label(summary: dict[str, Any]) -> str:
    if summary["final_talc_pixels"] == 0:
        return "needs_manual_review"
    if summary.get("silicate_source") not in {None, "none"} and float(summary.get("silicate_supported_fraction") or 0.0) < 0.20:
        return "silicate_support_review_required"
    overlap_fraction = summary["overlap_pixels"] / max(1, summary["candidate_talc_pixels"])
    if summary["fallback_hull_pixels"] > summary["closed_fill_pixels"] * 2:
        return "candidate_from_hull_review_required"
    if overlap_fraction > 0.15:
        return "sulfide_overlap_review_required"
    return "candidate_ok"


def find_matching_mask(mask_dir: Path, image_path: Path) -> Path | None:
    for suffix in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        candidate = mask_dir / f"{image_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def convert_talc_annotation_image(
    image_path: Path,
    out_dir: Path,
    config: TalcConversionConfig | None = None,
    *,
    sulfide_mask_path: Path | None = None,
    silicate_mask_path: Path | None = None,
) -> dict[str, Any]:
    config = config or TalcConversionConfig()
    image_path = image_path.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    image_rgb = read_image_rgb(image_path)
    shape_hw = image_rgb.shape[:2]

    raw_stroke = detect_blue_stroke(image_rgb, config)
    closed_stroke = close_stroke_boundaries(raw_stroke, config)
    closed_fill = fill_regions_from_barrier(closed_stroke)
    hull_fill = fill_component_hulls(closed_stroke, config)
    filled_region = remove_small_components(closed_fill | hull_fill, config.min_region_area_px)

    markup_ignore = raw_stroke.copy()
    if config.markup_ignore_dilate_px > 0:
        markup_ignore = cv2.dilate(markup_ignore, ellipse_kernel(config.markup_ignore_dilate_px), iterations=1)

    if config.sulfide_mode == "none":
        sulfide_mask = np.zeros(shape_hw, dtype=np.uint8)
        sulfide_source = "none"
    elif sulfide_mask_path is not None:
        sulfide_mask = read_mask(sulfide_mask_path, shape_hw)
        sulfide_source = str(sulfide_mask_path)
    else:
        sulfide_mask = detect_bright_sulfide_mask(image_rgb, config)
        sulfide_source = "heuristic_bright_phase"

    if silicate_mask_path is not None:
        silicate_support = read_mask(silicate_mask_path, shape_hw)
        silicate_source = str(silicate_mask_path)
    else:
        silicate_support = np.zeros(shape_hw, dtype=np.uint8)
        silicate_source = "none"

    candidate_talc = ensure_uint8_mask((filled_region > 0) & (markup_ignore == 0))
    overlap = ensure_uint8_mask((candidate_talc > 0) & (sulfide_mask > 0))
    talc_without_sulfide = ensure_uint8_mask((candidate_talc > 0) & (sulfide_mask == 0))
    if silicate_mask_path is not None:
        silicate_supported_talc = ensure_uint8_mask((talc_without_sulfide > 0) & (silicate_support > 0))
        silicate_unsupported_talc = ensure_uint8_mask((talc_without_sulfide > 0) & (silicate_support == 0))
        final_talc = silicate_supported_talc
    else:
        silicate_supported_talc = np.zeros(shape_hw, dtype=np.uint8)
        silicate_unsupported_talc = np.zeros(shape_hw, dtype=np.uint8)
        final_talc = talc_without_sulfide
    candidate_guard = dilate_mask(candidate_talc | markup_ignore, config.silicate_hard_negative_margin_px)
    silicate_hard_negative = ensure_uint8_mask(
        (silicate_support > 0) & (candidate_guard == 0) & (sulfide_mask == 0) & (markup_ignore == 0)
    )
    talc_positive_core = erode_mask(final_talc, config.talc_positive_core_erode_px)
    ignore = ensure_uint8_mask((markup_ignore > 0) | (overlap > 0) | (silicate_unsupported_talc > 0))

    copied_image = out_dir / image_path.name
    if copied_image.resolve() != image_path:
        shutil.copy2(image_path, copied_image)

    paths = {
        "source_image": copied_image,
        "raw_blue_stroke": out_dir / "raw_blue_stroke.png",
        "closed_blue_stroke": out_dir / "closed_blue_stroke.png",
        "filled_talc_region": out_dir / "filled_talc_region.png",
        "candidate_talc_mask": out_dir / "candidate_talc_mask.png",
        "sulfide_mask": out_dir / "sulfide_mask.png",
        "sulfide_overlap_mask": out_dir / "sulfide_overlap_mask.png",
        "silicate_support_mask": out_dir / "silicate_support_mask.png",
        "silicate_supported_talc_mask": out_dir / "silicate_supported_talc_mask.png",
        "silicate_unsupported_talc_mask": out_dir / "silicate_unsupported_talc_mask.png",
        "talc_positive_core_mask": out_dir / "talc_positive_core_mask.png",
        "silicate_hard_negative_mask": out_dir / "silicate_hard_negative_mask.png",
        "ignore_mask": out_dir / "ignore_mask.png",
        "final_talc_mask": out_dir / "final_talc_mask.png",
        "qa_overlay": out_dir / "qa_overlay.png",
        "summary_json": out_dir / "conversion_summary.json",
    }
    write_mask(paths["raw_blue_stroke"], raw_stroke)
    write_mask(paths["closed_blue_stroke"], closed_stroke)
    write_mask(paths["filled_talc_region"], filled_region)
    write_mask(paths["candidate_talc_mask"], candidate_talc)
    write_mask(paths["sulfide_mask"], sulfide_mask)
    write_mask(paths["sulfide_overlap_mask"], overlap)
    write_mask(paths["silicate_support_mask"], silicate_support)
    write_mask(paths["silicate_supported_talc_mask"], silicate_supported_talc)
    write_mask(paths["silicate_unsupported_talc_mask"], silicate_unsupported_talc)
    write_mask(paths["talc_positive_core_mask"], talc_positive_core)
    write_mask(paths["silicate_hard_negative_mask"], silicate_hard_negative)
    write_mask(paths["ignore_mask"], ignore)
    write_mask(paths["final_talc_mask"], final_talc)
    write_image_rgb(
        paths["qa_overlay"],
        make_overlay(
            image_rgb,
            talc_mask=final_talc,
            stroke_mask=raw_stroke,
            sulfide_mask=sulfide_mask,
            overlap_mask=overlap,
            ignore_mask=ignore,
            alpha=config.overlay_alpha,
        ),
    )

    summary = {
        "schema_version": "talc-blue-line-conversion-v0.2",
        "generated_at": utc_now_iso(),
        "image_id": image_path.stem,
        "image_path": str(image_path),
        "image_sha256": sha256_file(image_path),
        "width": int(shape_hw[1]),
        "height": int(shape_hw[0]),
        "sulfide_source": sulfide_source,
        "silicate_source": silicate_source,
        "config": asdict(config),
        "raw_blue_stroke_pixels": mask_pixels(raw_stroke),
        "closed_stroke_pixels": mask_pixels(closed_stroke),
        "closed_fill_pixels": mask_pixels(closed_fill),
        "fallback_hull_pixels": mask_pixels(hull_fill),
        "filled_region_pixels": mask_pixels(filled_region),
        "candidate_talc_pixels": mask_pixels(candidate_talc),
        "sulfide_pixels": mask_pixels(sulfide_mask),
        "overlap_pixels": mask_pixels(overlap),
        "talc_without_sulfide_pixels": mask_pixels(talc_without_sulfide),
        "silicate_support_pixels": mask_pixels(silicate_support),
        "silicate_supported_talc_pixels": mask_pixels(silicate_supported_talc),
        "silicate_unsupported_talc_pixels": mask_pixels(silicate_unsupported_talc),
        "silicate_supported_fraction": mask_pixels(silicate_supported_talc) / max(1, mask_pixels(talc_without_sulfide)),
        "talc_positive_core_pixels": mask_pixels(talc_positive_core),
        "silicate_hard_negative_pixels": mask_pixels(silicate_hard_negative),
        "ignore_pixels": mask_pixels(ignore),
        "final_talc_pixels": mask_pixels(final_talc),
        "paths": {key: str(value) for key, value in paths.items() if key != "summary_json"},
    }
    summary["status"] = confidence_label(summary)
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def iter_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    files = [path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=lambda path: path.name.lower())


def convert_talc_annotation_folder(
    input_path: Path,
    out_dir: Path,
    config: TalcConversionConfig | None = None,
    *,
    sulfide_mask_dir: Path | None = None,
    silicate_mask_dir: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    config = config or TalcConversionConfig()
    images = iter_images(input_path)
    if limit is not None:
        images = images[: max(0, int(limit))]
    samples: list[dict[str, Any]] = []
    samples_dir = out_dir / "samples"
    for image_path in images:
        sulfide_mask_path = find_matching_mask(sulfide_mask_dir, image_path) if sulfide_mask_dir else None
        silicate_mask_path = find_matching_mask(silicate_mask_dir, image_path) if silicate_mask_dir else None
        sample_out = samples_dir / image_path.stem
        samples.append(
            convert_talc_annotation_image(
                image_path,
                sample_out,
                config,
                sulfide_mask_path=sulfide_mask_path,
                silicate_mask_path=silicate_mask_path,
            )
        )
    manifest = {
        "schema_version": "talc-blue-line-conversion-manifest-v0.2",
        "generated_at": utc_now_iso(),
        "input_path": str(input_path),
        "output_dir": str(out_dir),
        "sample_count": len(samples),
        "config": asdict(config),
        "samples": samples,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def rectangle_mask(shape_hw: tuple[int, int], x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    height, width = shape_hw
    left, right = sorted((max(0, min(width, int(x1))), max(0, min(width, int(x2)))))
    top, bottom = sorted((max(0, min(height, int(y1))), max(0, min(height, int(y2)))))
    out = np.zeros(shape_hw, dtype=np.uint8)
    if left < right and top < bottom:
        out[top:bottom, left:right] = 255
    return out


def polygon_mask(shape_hw: tuple[int, int], points_xy: list[list[int]] | list[tuple[int, int]]) -> np.ndarray:
    height, width = shape_hw
    out = np.zeros(shape_hw, dtype=np.uint8)
    points: list[tuple[int, int]] = []
    for point in points_xy:
        if len(point) < 2:
            continue
        x = max(0, min(width - 1, int(round(float(point[0])))))
        y = max(0, min(height - 1, int(round(float(point[1])))))
        points.append((x, y))
    if len(points) < 3:
        return out
    cv2.fillPoly(out, [np.asarray(points, dtype=np.int32)], 255)
    return ensure_uint8_mask(out)


def apply_edit_mask(
    talc_mask: np.ndarray,
    ignore_mask: np.ndarray,
    edit_mask: np.ndarray,
    action: str,
) -> tuple[np.ndarray, np.ndarray]:
    talc = ensure_uint8_mask(talc_mask).copy()
    ignore = ensure_uint8_mask(ignore_mask).copy()
    active = edit_mask > 0
    normalized = action.strip().lower()
    if normalized in {"add_talc", "talc", "restore_talc"}:
        talc[active] = 255
        ignore[active] = 0
    elif normalized in {"erase_talc", "not_talc", "not_talc_or_matrix"}:
        talc[active] = 0
        ignore[active] = 0
    elif normalized in {"uncertain", "exclude_artifact", "ignore"}:
        talc[active] = 0
        ignore[active] = 255
    else:
        raise ValueError(f"unsupported review action: {action}")
    return ensure_uint8_mask(talc), ensure_uint8_mask(ignore)


def save_reviewed_masks(
    sample_dir: Path,
    talc_mask: np.ndarray,
    ignore_mask: np.ndarray,
    edits: list[dict[str, Any]],
) -> dict[str, Any]:
    summary_path = sample_dir / "conversion_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"missing conversion summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    image_path = Path(summary["paths"]["source_image"])
    image_rgb = read_image_rgb(image_path)
    reviewed_dir = sample_dir / "reviewed"
    reviewed_dir.mkdir(parents=True, exist_ok=True)
    reviewed_talc_path = reviewed_dir / "reviewed_talc_mask.png"
    reviewed_ignore_path = reviewed_dir / "reviewed_ignore_mask.png"
    reviewed_overlay_path = reviewed_dir / "reviewed_overlay.png"
    patch_path = reviewed_dir / "review_patch.json"
    review_summary_path = reviewed_dir / "review_summary.json"
    write_mask(reviewed_talc_path, talc_mask)
    write_mask(reviewed_ignore_path, ignore_mask)
    write_image_rgb(
        reviewed_overlay_path,
        make_overlay(image_rgb, talc_mask=talc_mask, ignore_mask=ignore_mask),
    )
    patch = {
        "schema_version": "talc-review-patch-v0.1",
        "image_id": summary["image_id"],
        "base_summary": str(summary_path),
        "edits": edits,
        "saved_at": utc_now_iso(),
    }
    patch_path.write_text(json.dumps(patch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    review_summary = {
        "schema_version": "talc-review-summary-v0.1",
        "image_id": summary["image_id"],
        "saved_at": utc_now_iso(),
        "reviewed_talc_pixels": mask_pixels(talc_mask),
        "reviewed_ignore_pixels": mask_pixels(ignore_mask),
        "paths": {
            "reviewed_talc_mask": str(reviewed_talc_path),
            "reviewed_ignore_mask": str(reviewed_ignore_path),
            "reviewed_overlay": str(reviewed_overlay_path),
            "review_patch": str(patch_path),
        },
    }
    review_summary_path.write_text(json.dumps(review_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return review_summary
