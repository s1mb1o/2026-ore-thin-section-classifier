#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from dataclasses import asdict
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default as email_default_policy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFile, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
HEURISTIC_SRC = ROOT / "heuristic_segmentation/src"
for source_root in (SRC, HEURISTIC_SRC):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from heuristic_segmentation.segmentation import segment_image  # noqa: E402
from ore_classifier.analyzed_area import build_analyzed_mask  # noqa: E402
from ore_classifier.component_analysis import (  # noqa: E402
    ComponentRuleConfig,
    OreSummary,
    analyze_components,
    summary_warnings,
    write_component_csv,
)
from ore_classifier.rule_config_io import default_rule_config  # noqa: E402
from ore_classifier.tiling import iter_tiles  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_WORKSPACE_DIR = ROOT / "outputs/ore_pipeline_ui"
DEFAULT_CHECKPOINT = ROOT / "models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
MAX_JSON_BYTES = 220 * 1024 * 1024
DISPLAY_TILE_SIZE = 1024
DISPLAY_TILE_STRIDE = 768
RAW_EXTENSIONS = {".raw", ".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".raf", ".pef", ".srw"}
IMAGE_EXTENSIONS = RAW_EXTENSIONS | {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
CLASS_COLORS = {
    1: (30, 185, 85, 150),
    2: (230, 65, 65, 160),
    3: (40, 120, 245, 165),
}
CLASS_LABELS_RU = {
    "sulfide_fraction": "Общая доля сульфидов",
    "ordinary_sulfide_fraction": "Доля обычных срастаний",
    "fine_sulfide_fraction": "Доля тонких срастаний",
    "talc_fraction": "Доля талька",
}
DEFAULT_RULE_CONFIG = default_rule_config()
ORE_CLASS_SHORT_RU = {
    "talcose_ore": "оталькованная",
    "row_ore": "рядовая",
    "hard_to_process_ore": "труднообогатимая",
}


class ApiError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class RunCancelled(RuntimeError):
    """Internal control-flow signal for cooperative run cancellation."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_response(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def safe_name(name: str) -> str:
    stem = Path(name).stem or "image"
    suffix = Path(name).suffix.lower()
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem).strip("_")
    if not cleaned:
        cleaned = "image"
    return f"{cleaned[:90]}{suffix}"


def resolve_path(path: Path | str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def file_sha1(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image_pil(path: Path) -> Image.Image:
    suffix = path.suffix.lower()
    if suffix in RAW_EXTENSIONS:
        try:
            import rawpy  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "RAW decoding requires optional dependency rawpy. Install rawpy or convert the file to TIFF/PNG/JPEG first.",
            ) from exc
        try:
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
            return Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        except Exception as exc:  # noqa: BLE001 - report decoder failure to the UI.
            raise ApiError(HTTPStatus.BAD_REQUEST, f"failed to decode RAW image: {exc}") from exc
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            return image.convert("RGB")
    except Exception as exc:  # noqa: BLE001 - report unsupported image to the UI.
        raise ApiError(HTTPStatus.BAD_REQUEST, f"failed to decode image: {exc}") from exc


def image_dimensions(path: Path) -> tuple[int, int]:
    suffix = path.suffix.lower()
    if suffix in RAW_EXTENSIONS:
        image = load_image_pil(path)
        return image.size
    with Image.open(path) as image:
        return image.size


def downscaled_image(path: Path, max_side: int | None = None, size: tuple[int, int] | None = None) -> Image.Image:
    image = load_image_pil(path)
    if size is not None:
        if image.size != size:
            image = image.resize(size, Image.Resampling.BILINEAR)
        return image.convert("RGB")
    if max_side and max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
    return image.convert("RGB")


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def decode_mask_data_url(data_url: str, expected_shape_hw: tuple[int, int], *, final_mask: bool = False) -> np.ndarray:
    if not isinstance(data_url, str) or "," not in data_url:
        raise ApiError(HTTPStatus.BAD_REQUEST, "mask_png must be a PNG data URL")
    header, encoded = data_url.split(",", 1)
    if "base64" not in header.lower():
        raise ApiError(HTTPStatus.BAD_REQUEST, "mask_png must be base64 encoded")
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            mask = np.asarray(image.convert("L"))
    except Exception as exc:  # noqa: BLE001 - malformed client payload.
        raise ApiError(HTTPStatus.BAD_REQUEST, f"failed to decode mask PNG: {exc}") from exc
    if mask.shape[:2] != expected_shape_hw:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            f"mask dimensions {mask.shape[1]}x{mask.shape[0]} do not match run "
            f"{expected_shape_hw[1]}x{expected_shape_hw[0]}",
        )
    if final_mask:
        return np.clip(np.rint(mask), 0, 3).astype(np.uint8)
    return (mask > 0).astype(np.uint8) * 255


def preset_from_payload(payload: dict[str, Any]) -> dict[str, bool]:
    return {
        "illumination_normalization": bool(payload.get("illumination_normalization") or payload.get("illumination")),
        "denoise": bool(payload.get("denoise") or payload.get("noise_reduction")),
        "contrast_correction": bool(payload.get("contrast_correction") or payload.get("contrast")),
        "panorama_scaling": bool(payload.get("panorama_scaling") or payload.get("panoramaScaling")),
    }


def apply_preprocessing(image: Image.Image, preset: dict[str, bool]) -> Image.Image:
    result = image.convert("RGB")
    if preset.get("illumination_normalization"):
        arr = np.asarray(result)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        value = hsv[..., 2]
        sigma = max(9.0, min(value.shape) / 32.0)
        background = cv2.GaussianBlur(value, (0, 0), sigmaX=sigma)
        corrected = value.astype(np.float32) - background.astype(np.float32) + float(np.median(background))
        hsv[..., 2] = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        result = Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), mode="RGB")
    if preset.get("denoise"):
        arr = np.asarray(result)
        denoised = cv2.fastNlMeansDenoisingColored(arr, None, 4, 4, 7, 21)
        result = Image.fromarray(denoised, mode="RGB")
    if preset.get("contrast_correction"):
        arr = np.asarray(result)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        corrected_l = clahe.apply(l_channel)
        corrected = cv2.merge((corrected_l, a_channel, b_channel))
        result = Image.fromarray(cv2.cvtColor(corrected, cv2.COLOR_LAB2RGB), mode="RGB")
        result = ImageEnhance.Contrast(result).enhance(1.05)
    return result


def save_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode == "RGBA":
        image.save(path, format="PNG", optimize=True)
    elif image.mode == "L":
        image.save(path, format="PNG", optimize=True)
    else:
        image.convert("RGB").save(path, format="PNG", optimize=True)


def save_preview_pyramid(
    image: Image.Image,
    out_dir: Path,
    stem: str,
    max_sides: tuple[int, ...],
    *,
    nearest: bool = False,
    prefer_png: bool = False,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    previews: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for max_side in sorted(max_sides):
        preview = image.copy()
        if max(preview.size) > max_side:
            resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
            preview.thumbnail((max_side, max_side), resample)
        if preview.size in seen:
            continue
        seen.add(preview.size)
        ext = ".png" if prefer_png or preview.mode in {"RGBA", "L"} else ".jpg"
        path = out_dir / f"{stem}_{max(preview.size)}{ext}"
        if ext == ".png":
            preview.save(path, format="PNG", optimize=True)
        else:
            preview.convert("RGB").save(path, format="JPEG", quality=90, optimize=True)
        previews.append(
            {
                "max_side": max(preview.size),
                "width": preview.size[0],
                "height": preview.size[1],
                "path": str(path),
            }
        )
    return previews


def build_tiling_manifest(
    *,
    source_width: int,
    source_height: int,
    analysis_width: int,
    analysis_height: int,
    source_scaled: bool,
    tile_size: int = DISPLAY_TILE_SIZE,
    stride: int = DISPLAY_TILE_STRIDE,
) -> dict[str, Any]:
    tiles = iter_tiles(width=analysis_width, height=analysis_height, tile_size=tile_size, stride=stride)
    return {
        "schema_version": "ore-pipeline-tiling-v0.1",
        "source_width": int(source_width),
        "source_height": int(source_height),
        "analysis_width": int(analysis_width),
        "analysis_height": int(analysis_height),
        "tile_size": int(tile_size),
        "stride": int(stride),
        "source_scaled_for_processing": bool(source_scaled),
        "enabled": bool(source_scaled or len(tiles) > 1),
        "tile_count": len(tiles),
        "tiles": [
            {
                "x": int(tile.x),
                "y": int(tile.y),
                "width": int(min(tile.width, analysis_width - tile.x)),
                "height": int(min(tile.height, analysis_height - tile.y)),
            }
            for tile in tiles
        ],
    }


def colored_overlay(mask: np.ndarray, class_id: int | None, rgba: tuple[int, int, int, int]) -> Image.Image:
    if class_id is None:
        active = mask > 0
    else:
        active = mask == class_id
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    overlay[active] = np.array(rgba, dtype=np.uint8)
    return Image.fromarray(overlay, mode="RGBA")


def final_mask_from_classified(classified: np.ndarray, talc_mask: np.ndarray | None) -> np.ndarray:
    final_mask = classified.astype(np.uint8).copy()
    if talc_mask is not None:
        final_mask[talc_mask > 0] = 3
    return final_mask


def text_output_for_summary(summary: dict[str, Any]) -> str:
    ore = ORE_CLASS_SHORT_RU.get(str(summary.get("ore_class")), str(summary.get("ore_class_ru") or "неизвестная"))
    talc_pct = float(summary.get("talc_fraction") or 0.0) * 100.0
    ordinary_pct = float(summary.get("ordinary_sulfide_fraction") or 0.0) * 100.0
    fine_pct = float(summary.get("fine_sulfide_fraction") or 0.0) * 100.0
    if fine_pct >= ordinary_pct:
        dominant = "тонких срастаний"
        dominant_pct = fine_pct
    else:
        dominant = "обычных срастаний"
        dominant_pct = ordinary_pct
    return (
        f"Руда классифицирована как {ore}: содержание талька — {talc_pct:.1f}%, "
        f"преобладание {dominant} — {dominant_pct:.1f}%."
    )


def metric_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, label in CLASS_LABELS_RU.items():
        value = float(summary.get(key) or 0.0)
        rows.append({"key": key, "label": label, "value": value, "percent": value * 100.0})
    rows.extend(
        [
            {"key": "component_count", "label": "Компоненты сульфидов", "value": int(summary.get("component_count") or 0), "percent": None},
            {
                "key": "analyzed_fraction",
                "label": "Доля проанализированной области",
                "value": float(summary.get("analyzed_fraction") or 0.0),
                "percent": float(summary.get("analyzed_fraction") or 0.0) * 100.0,
            },
        ]
    )
    return rows


def summary_from_final_edit(parent_sulfide: np.ndarray, final_mask: np.ndarray, analyzed_mask: np.ndarray) -> dict[str, Any]:
    analyzed = analyzed_mask > 0
    sulfide = (parent_sulfide > 0) & analyzed
    ordinary = (final_mask == 1) & analyzed
    fine = (final_mask == 2) & analyzed
    talc = (final_mask == 3) & analyzed
    image_area = int(final_mask.size)
    analysis_area = int(analyzed.sum())
    sulfide_area = int(sulfide.sum())
    ordinary_area = int(ordinary.sum())
    fine_area = int(fine.sum())
    talc_area = int(talc.sum())
    talc_fraction = talc_area / max(analysis_area, 1)
    if talc_fraction > DEFAULT_RULE_CONFIG["talc_fraction_threshold"]:
        ore_class = "talcose_ore"
        ore_class_ru = "оталькованная руда"
    elif ordinary_area >= fine_area:
        ore_class = "row_ore"
        ore_class_ru = "рядовая руда"
    else:
        ore_class = "hard_to_process_ore"
        ore_class_ru = "труднообогатимая руда"
    ordinary_fraction = ordinary_area / max(sulfide_area, 1)
    fine_fraction = fine_area / max(sulfide_area, 1)
    warnings = summary_warnings(
        sulfide_area=sulfide_area,
        analyzed_fraction=analysis_area / max(image_area, 1),
        talc_margin=talc_fraction - DEFAULT_RULE_CONFIG["talc_fraction_threshold"],
        intergrowth_margin=ordinary_fraction - fine_fraction,
    )
    return asdict(
        OreSummary(
            ore_class=ore_class,
            ore_class_ru=ore_class_ru,
            sulfide_fraction=sulfide_area / max(analysis_area, 1),
            sulfide_fraction_image=sulfide_area / max(image_area, 1),
            ordinary_sulfide_fraction=ordinary_fraction,
            fine_sulfide_fraction=fine_fraction,
            talc_fraction=talc_fraction,
            talc_fraction_image=talc_area / max(image_area, 1),
            sulfide_area_px=sulfide_area,
            ordinary_sulfide_area_px=ordinary_area,
            fine_sulfide_area_px=fine_area,
            talc_area_px=talc_area,
            image_area_px=image_area,
            analysis_area_px=analysis_area,
            analyzed_fraction=analysis_area / max(image_area, 1),
            component_count=count_components(ordinary | fine),
            ordinary_component_count=count_components(ordinary),
            fine_component_count=count_components(fine),
            talc_margin=talc_fraction - DEFAULT_RULE_CONFIG["talc_fraction_threshold"],
            intergrowth_margin=ordinary_fraction - fine_fraction,
            needs_expert_review=bool(warnings),
            warnings=warnings,
            rule_text_ru=text_output_for_summary(
                {
                    "ore_class": ore_class,
                    "ore_class_ru": ore_class_ru,
                    "talc_fraction": talc_fraction,
                    "ordinary_sulfide_fraction": ordinary_fraction,
                    "fine_sulfide_fraction": fine_fraction,
                }
            ),
        )
    )


def count_components(mask: np.ndarray) -> int:
    labels_count, _, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return max(0, int(labels_count) - 1)


class OrePipelineStore:
    def __init__(
        self,
        *,
        workspace_dir: Path,
        backend: str,
        checkpoint: Path | None,
        processing_max_side: int,
        panorama_max_side: int,
        preview_max_sides: tuple[int, ...],
    ) -> None:
        self.workspace_dir = resolve_path(workspace_dir)
        self.uploads_dir = self.workspace_dir / "uploads"
        self.runs_dir = self.workspace_dir / "runs"
        self.backend = backend
        self.checkpoint = resolve_path(checkpoint) if checkpoint else None
        self.processing_max_side = int(processing_max_side)
        self.panorama_max_side = int(panorama_max_side)
        self.preview_max_sides = preview_max_sides
        self.artifacts: dict[str, Path] = {}
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.allowed_roots = [ROOT.resolve(), self.workspace_dir.resolve()]
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def register_upload_from_bytes(self, data: bytes, original_name: str) -> dict[str, Any]:
        suffix = Path(original_name).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            raise ApiError(HTTPStatus.BAD_REQUEST, "supported image formats: PNG, JPEG, TIFF, RAW")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "uploaded image is too large")
        upload_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{hashlib.sha1(data[:1048576]).hexdigest()[:10]}"
        upload_dir = self.uploads_dir / upload_id
        upload_dir.mkdir(parents=True, exist_ok=False)
        original_path = upload_dir / safe_name(original_name)
        original_path.write_bytes(data)
        return self._register_upload_file(upload_id, upload_dir, original_path, original_name)

    def register_upload_from_path(self, path: Path, original_name: str | None = None) -> dict[str, Any]:
        original_name = original_name or path.name
        suffix = Path(original_name).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            raise ApiError(HTTPStatus.BAD_REQUEST, "supported image formats: PNG, JPEG, TIFF, RAW")
        digest = file_sha1(path)[:10]
        upload_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{digest}"
        upload_dir = self.uploads_dir / upload_id
        upload_dir.mkdir(parents=True, exist_ok=False)
        original_path = upload_dir / safe_name(original_name)
        hardlink_or_copy(path, original_path)
        return self._register_upload_file(upload_id, upload_dir, original_path, original_name)

    def _register_upload_file(self, upload_id: str, upload_dir: Path, original_path: Path, original_name: str) -> dict[str, Any]:
        width, height = image_dimensions(original_path)
        preview_dir = upload_dir / "display/original"
        previews = save_preview_pyramid(load_image_pil(original_path), preview_dir, "original", self.preview_max_sides)
        metadata = {
            "schema_version": "ore-pipeline-upload-v0.1",
            "upload_id": upload_id,
            "created_at": utc_now_iso(),
            "original_name": original_name,
            "original_path": str(original_path),
            "width": int(width),
            "height": int(height),
            "format": original_path.suffix.lower().lstrip("."),
            "display": {"original": previews},
            "preprocess": None,
        }
        self._write_json(upload_dir / "upload.json", metadata)
        return self.upload_payload(upload_id)

    def upload_payload(self, upload_id: str) -> dict[str, Any]:
        metadata = self._read_upload(upload_id)
        display = metadata.get("display", {})
        return {
            **metadata,
            "display": {key: self.preview_urls(value) for key, value in display.items()},
        }

    def prepare_upload(self, upload_id: str, preset: dict[str, bool]) -> dict[str, Any]:
        upload_dir = self.uploads_dir / upload_id
        metadata = self._read_upload(upload_id)
        original_path = resolve_path(metadata["original_path"])
        target_max_side = self.panorama_max_side if preset.get("panorama_scaling") else self.processing_max_side
        if max(int(metadata["width"]), int(metadata["height"])) > target_max_side:
            source_scaled = True
        else:
            source_scaled = False
        source = downscaled_image(original_path, max_side=target_max_side)
        preprocessed = apply_preprocessing(source, preset)
        preprocess_dir = upload_dir / "preprocessed"
        preprocessed_path = preprocess_dir / "preprocessed.png"
        save_image(preprocessed_path, preprocessed)
        previews = save_preview_pyramid(preprocessed, preprocess_dir / "display", "preprocessed", self.preview_max_sides)
        tiling = build_tiling_manifest(
            source_width=int(metadata["width"]),
            source_height=int(metadata["height"]),
            analysis_width=preprocessed.size[0],
            analysis_height=preprocessed.size[1],
            source_scaled=source_scaled,
        )
        preprocess_metadata = {
            "schema_version": "ore-pipeline-preprocess-v0.1",
            "updated_at": utc_now_iso(),
            "preset": preset,
            "preprocessed_path": str(preprocessed_path),
            "width": preprocessed.size[0],
            "height": preprocessed.size[1],
            "source_scaled_for_processing": source_scaled,
            "target_max_side": target_max_side,
            "display": previews,
            "tiling": tiling,
        }
        self._write_json(preprocess_dir / "preprocess.json", preprocess_metadata)
        metadata["preprocess"] = preprocess_metadata
        metadata.setdefault("display", {})["preprocessed"] = previews
        metadata["tiling"] = tiling
        self._write_json(upload_dir / "upload.json", metadata)
        return self.upload_payload(upload_id)

    def start_run(self, upload_id: str, preset: dict[str, bool], *, run_async: bool = True) -> dict[str, Any]:
        upload = self.prepare_upload(upload_id, preset)
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}_{hashlib.sha1(upload_id.encode()).hexdigest()[:8]}"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        self._initialize_run_from_upload(run_id, run_dir, upload, preset)
        with self.lock:
            self.jobs[run_id] = {
                "progress": 1,
                "status": "queued",
                "stage": "queued",
                "started_at": time.time(),
                "eta_seconds": None,
                "cancel_requested": False,
            }
        if run_async:
            thread = threading.Thread(target=self._run_job_guarded, args=(run_id,), daemon=True)
            thread.start()
        else:
            self._run_job_guarded(run_id)
        return self.run_payload(run_id)

    def create_edit_run(self, parent_run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        parent = self._read_run(parent_run_id)
        parent_dir = self.runs_dir / parent_run_id
        edit_layer = str(payload.get("edit_layer") or "")
        if edit_layer not in {"sulfide", "final"}:
            raise ApiError(HTTPStatus.BAD_REQUEST, "edit_layer must be sulfide or final")
        run_id = f"edit_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}_{hashlib.sha1((parent_run_id + edit_layer).encode()).hexdigest()[:8]}"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        self._copy_run_inputs(parent, parent_dir, run_dir)
        expected_shape = (int(parent["image"]["height"]), int(parent["image"]["width"]))
        mask = decode_mask_data_url(payload.get("mask_png", ""), expected_shape, final_mask=edit_layer == "final")
        comment = str(payload.get("comment") or "").strip()
        derivation = {
            "type": "edit_recalculate",
            "parent_run_id": parent_run_id,
            "edit_layer": edit_layer,
            "comment": comment,
            "created_at": utc_now_iso(),
            "operation": "recalculate_from_sulfide_edit" if edit_layer == "sulfide" else "recalculate_metrics_from_final_edit",
        }
        if edit_layer == "sulfide":
            self._write_masks_from_sulfide_edit(parent_dir, run_dir, mask)
        else:
            self._write_masks_from_final_edit(parent_dir, run_dir, mask)
        run_metadata = self._base_run_metadata(run_id, run_dir, parent["input"]["upload_id"], parent["preprocess"]["preset"])
        run_metadata["status"] = "complete"
        run_metadata["progress"] = 100
        run_metadata["backend"] = parent.get("backend", self.backend)
        run_metadata["derivation"] = derivation
        run_metadata["input"]["original_source_path"] = parent["input"].get("original_source_path")
        run_metadata["input"]["original_artifact_path"] = str(run_dir / "input/original_source" / Path(parent["input"]["original_artifact_path"]).name)
        run_metadata["tiling"] = parent.get("tiling") or {}
        self._finalize_run_metadata(run_metadata, run_dir)
        (run_dir / "edit_comment.txt").write_text(comment + "\n", encoding="utf-8")
        self._write_json(run_dir / "run.json", run_metadata)
        return self.run_payload(run_id)

    def list_runs(self) -> dict[str, Any]:
        runs = []
        for path in sorted(self.runs_dir.glob("*/run.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            summary = data.get("summary") or {}
            thumbnail = self.history_thumbnail_payload(data)
            runs.append(
                {
                    "run_id": data.get("run_id"),
                    "created_at": data.get("created_at"),
                    "status": data.get("status"),
                    "progress": data.get("progress", 0),
                    "parent_run_id": (data.get("derivation") or {}).get("parent_run_id"),
                    "edit_layer": (data.get("derivation") or {}).get("edit_layer"),
                    "ore_class_ru": summary.get("ore_class_ru"),
                    "summary": summary,
                    "metrics": data.get("metrics", []),
                    "text_output": data.get("text_output"),
                    "image": data.get("image"),
                    "thumbnail": thumbnail,
                }
            )
        return {"schema_version": "ore-pipeline-history-v0.1", "runs": runs}

    def history_thumbnail_payload(self, run_data: dict[str, Any]) -> dict[str, Any]:
        display = run_data.get("display") or {}
        previews = display.get("original") or display.get("preprocessed") or []
        if not isinstance(previews, list) or not previews:
            return {}
        thumbnail = previews[0]
        preview = previews[-1]
        thumbnail_url = self.artifact_url(thumbnail.get("path"))
        preview_url = self.artifact_url(preview.get("path"))
        if not thumbnail_url and not preview_url:
            return {}
        return {
            "thumbnail_url": thumbnail_url or preview_url,
            "preview_url": preview_url or thumbnail_url,
            "width": preview.get("width") or thumbnail.get("width"),
            "height": preview.get("height") or thumbnail.get("height"),
        }

    def delete_run(self, run_id: str) -> dict[str, Any]:
        run_dir = (self.runs_dir / run_id).resolve()
        runs_root = self.runs_dir.resolve()
        if run_dir == runs_root or not is_relative_to(run_dir, runs_root):
            raise ApiError(HTTPStatus.BAD_REQUEST, "invalid run id")
        if not (run_dir / "run.json").exists():
            raise ApiError(HTTPStatus.NOT_FOUND, f"unknown run: {run_id}")
        with self.lock:
            job_status = self.jobs.get(run_id, {}).get("status")
            if job_status in {"queued", "running", "canceling"}:
                raise ApiError(HTTPStatus.CONFLICT, "run is still running")
            self.jobs.pop(run_id, None)
        shutil.rmtree(run_dir)
        return {"removed_run_id": run_id, "history": self.list_runs()["runs"]}

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        run_dir = self.runs_dir / run_id
        run_path = run_dir / "run.json"
        if not run_path.exists():
            raise ApiError(HTTPStatus.NOT_FOUND, f"unknown run: {run_id}")
        data = json.loads(run_path.read_text(encoding="utf-8"))
        if data.get("status") in {"complete", "failed", "canceled"}:
            return self.run_payload(run_id)
        with self.lock:
            job = self.jobs.get(run_id)
            if job and job.get("status") in {"queued", "running", "canceling"}:
                progress = int(job.get("progress", data.get("progress", 0)) or 0)
                updated = {
                    **job,
                    "progress": progress,
                    "status": "canceling",
                    "stage": "canceling",
                    "eta_seconds": None,
                    "cancel_requested": True,
                }
                self.jobs[run_id] = updated
                data["status"] = "canceling"
                data["stage"] = "canceling"
                data["progress"] = progress
                data["eta_seconds"] = None
            else:
                progress = int(data.get("progress", 0) or 0)
                data["status"] = "canceled"
                data["stage"] = "canceled"
                data["progress"] = progress
                data["eta_seconds"] = None
                data["canceled_at"] = utc_now_iso()
                self.jobs[run_id] = {"status": "canceled", "progress": progress, "eta_seconds": None}
        self._write_json(run_path, data)
        return self.run_payload(run_id)

    def run_payload(self, run_id: str) -> dict[str, Any]:
        data = self._read_run(run_id)
        job = self.jobs.get(run_id)
        if job and data.get("status") not in {"complete", "failed", "canceled"}:
            data = {**data, **job}
        display = data.get("display", {})
        display_urls: dict[str, Any] = {}
        for key, value in display.items():
            if isinstance(value, list):
                display_urls[key] = self.preview_urls(value)
            elif isinstance(value, dict):
                display_urls[key] = {subkey: self.preview_urls(subvalue) for subkey, subvalue in value.items()}
        masks = {key: self.artifact_url(path) for key, path in (data.get("masks") or {}).items()}
        downloads = {
            "metrics_csv": f"/api/runs/{urllib.parse.quote(run_id)}/metrics.csv",
            "pdf_report": f"/api/runs/{urllib.parse.quote(run_id)}/report.pdf",
        }
        return {
            **data,
            "display": display_urls,
            "masks": masks,
            "downloads": downloads,
            "history": self.list_runs()["runs"],
        }

    def preview_urls(self, previews: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{**preview, "url": self.artifact_url(preview["path"])} for preview in previews]

    def artifact_url(self, path: Path | str | None) -> str | None:
        if path is None:
            return None
        resolved = resolve_path(path)
        if not resolved.exists():
            return None
        if not any(is_relative_to(resolved, allowed) for allowed in self.allowed_roots):
            raise ApiError(HTTPStatus.FORBIDDEN, f"path is outside allowed roots: {resolved}")
        artifact_id = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:16]
        self.artifacts[artifact_id] = resolved
        filename = urllib.parse.quote(resolved.name)
        version = int(resolved.stat().st_mtime)
        return f"/artifacts/{artifact_id}/{filename}?v={version}"

    def artifact_path(self, artifact_id: str) -> Path:
        path = self.artifacts.get(artifact_id)
        if path is None or not path.exists():
            raise ApiError(HTTPStatus.NOT_FOUND, "unknown artifact")
        return path

    def metrics_csv_path(self, run_id: str) -> Path:
        data = self._read_run(run_id)
        path = self.runs_dir / run_id / "reports/metrics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["metric", "key", "value", "percent"])
            writer.writeheader()
            for row in data.get("metrics", []):
                writer.writerow(
                    {
                        "metric": row["label"],
                        "key": row["key"],
                        "value": row["value"],
                        "percent": "" if row.get("percent") is None else f"{float(row['percent']):.6f}",
                    }
                )
        return path

    def pdf_report_path(self, run_id: str) -> Path:
        data = self._read_run(run_id)
        path = self.runs_dir / run_id / "reports/ore_report.pdf"
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        page = Image.new("RGB", (1240, 1754), "white")
        draw = ImageDraw.Draw(page)
        title_font = load_font(42)
        body_font = load_font(27)
        small_font = load_font(22)
        y = 80
        draw.text((80, y), "Отчет по классификации руды", fill=(20, 26, 36), font=title_font)
        y += 72
        draw.text((80, y), f"Run ID: {run_id}", fill=(58, 67, 82), font=small_font)
        y += 40
        draw.text((80, y), data.get("text_output") or "", fill=(20, 26, 36), font=body_font)
        y += 80
        for row in data.get("metrics", []):
            value = f"{float(row['percent']):.1f}%" if row.get("percent") is not None else str(row.get("value"))
            draw.text((100, y), row["label"], fill=(58, 67, 82), font=body_font)
            draw.text((730, y), value, fill=(20, 26, 36), font=body_font)
            y += 46
        preview_path = first_preview_path(data.get("display", {}).get("preprocessed"))
        if preview_path and Path(preview_path).exists():
            preview = Image.open(preview_path).convert("RGB")
            preview.thumbnail((980, 760), Image.Resampling.BILINEAR)
            page.paste(preview, (80, min(y + 40, 930)))
        page.save(path, "PDF", resolution=150.0)
        return path

    def _initialize_run_from_upload(self, run_id: str, run_dir: Path, upload: dict[str, Any], preset: dict[str, bool]) -> None:
        input_dir = run_dir / "input"
        source_path = resolve_path(upload["original_path"])
        original_artifact = input_dir / "original_source" / Path(upload["original_path"]).name
        hardlink_or_copy(source_path, original_artifact)
        preprocessed_source = resolve_path(upload["preprocess"]["preprocessed_path"])
        preprocessed_path = input_dir / "preprocessed.png"
        shutil.copy2(preprocessed_source, preprocessed_path)
        original_for_analysis = downscaled_image(source_path, size=(upload["preprocess"]["width"], upload["preprocess"]["height"]))
        save_image(input_dir / "original_for_analysis.png", original_for_analysis)
        metadata = self._base_run_metadata(run_id, run_dir, upload["upload_id"], preset)
        metadata["input"]["original_source_path"] = upload["original_path"]
        metadata["input"]["original_artifact_path"] = str(original_artifact)
        metadata["tiling"] = upload.get("tiling") or (upload.get("preprocess") or {}).get("tiling") or {}
        self._write_json(run_dir / "run.json", metadata)

    def _base_run_metadata(self, run_id: str, run_dir: Path, upload_id: str, preset: dict[str, bool]) -> dict[str, Any]:
        return {
            "schema_version": "ore-pipeline-ui-run-v0.1",
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "status": "running",
            "progress": 0,
            "backend": self.backend,
            "input": {
                "upload_id": upload_id,
                "original_artifact_path": str(run_dir / "input/original_source"),
                "original_for_analysis_path": str(run_dir / "input/original_for_analysis.png"),
                "preprocessed_path": str(run_dir / "input/preprocessed.png"),
            },
            "preprocess": {"preset": preset},
            "image": {},
            "summary": {},
            "metrics": [],
            "text_output": "",
            "display": {},
            "masks": {},
            "tiling": {},
            "derivation": None,
        }

    def _run_job_guarded(self, run_id: str) -> None:
        try:
            self._run_job(run_id)
        except RunCancelled:
            run_path = self.runs_dir / run_id / "run.json"
            data = json.loads(run_path.read_text(encoding="utf-8"))
            progress = int(data.get("progress", 0) or 0)
            data["status"] = "canceled"
            data["stage"] = "canceled"
            data["progress"] = progress
            data["eta_seconds"] = None
            data["canceled_at"] = utc_now_iso()
            self._write_json(run_path, data)
            with self.lock:
                self.jobs[run_id] = {"status": "canceled", "progress": progress, "eta_seconds": None}
        except Exception as exc:  # noqa: BLE001 - keep server alive and expose failure.
            run_path = self.runs_dir / run_id / "run.json"
            data = json.loads(run_path.read_text(encoding="utf-8"))
            data["status"] = "failed"
            data["error"] = str(exc)
            data["progress"] = 100
            self._write_json(run_path, data)
            with self.lock:
                self.jobs[run_id] = {"status": "failed", "progress": 100, "error": str(exc), "eta_seconds": None}

    def _run_job(self, run_id: str) -> None:
        run_dir = self.runs_dir / run_id
        self._set_progress(run_id, 8, "preparing immutable run artifacts")
        self._check_cancelled(run_id)
        if self.backend == "ml":
            self._run_ml_backend(run_id, run_dir)
        else:
            self._run_heuristic_backend(run_id, run_dir)
        self._check_cancelled(run_id)
        metadata = self._read_run(run_id)
        metadata["status"] = "complete"
        metadata["progress"] = 100
        metadata["completed_at"] = utc_now_iso()
        self._finalize_run_metadata(metadata, run_dir)
        self._write_json(run_dir / "run.json", metadata)
        with self.lock:
            self.jobs[run_id] = {"status": "complete", "progress": 100, "eta_seconds": 0}

    def _run_heuristic_backend(self, run_id: str, run_dir: Path) -> None:
        self._set_progress(run_id, 25, "sulfide/non-sulfide segmentation")
        self._check_cancelled(run_id)
        rgb = np.asarray(Image.open(run_dir / "input/preprocessed.png").convert("RGB"))
        result = segment_image(rgb)
        self._check_cancelled(run_id)
        sulfide_mask = (result.sulfide_mask > 0).astype(np.uint8) * 255
        talc_mask = (result.talc_candidate_mask > 0).astype(np.uint8) * 255
        analyzed_mask = build_analyzed_mask(rgb)
        self._set_progress(run_id, 58, "ordinary/fine intergrowth and talc analysis")
        self._check_cancelled(run_id)
        summary, components, classified = analyze_components(
            sulfide_mask=sulfide_mask,
            talc_mask=talc_mask,
            analyzed_mask=analyzed_mask,
            config=ComponentRuleConfig(),
        )
        final_mask = final_mask_from_classified(classified, talc_mask)
        self._write_run_outputs(
            run_dir=run_dir,
            summary=asdict(summary),
            components=components,
            sulfide_mask=sulfide_mask,
            talc_mask=talc_mask,
            analyzed_mask=analyzed_mask,
            final_mask=final_mask,
        )

    def _run_ml_backend(self, run_id: str, run_dir: Path) -> None:
        if self.checkpoint is None or not self.checkpoint.exists():
            raise ApiError(HTTPStatus.BAD_REQUEST, "ML backend requires --checkpoint")
        self._set_progress(run_id, 18, "running ML tiled inference")
        ml_dir = run_dir / "ml_pipeline"
        cmd = [
            sys.executable,
            str(ROOT / "scripts/run_ore_pipeline.py"),
            "--image",
            str(run_dir / "input/preprocessed.png"),
            "--checkpoint",
            str(self.checkpoint),
            "--out-dir",
            str(ml_dir),
            "--auto-talc-candidate",
            "--preview-max-side",
            str(max(self.preview_max_sides)),
        ]
        log_path = run_dir / "ml_pipeline.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if self._cancel_requested(run_id):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    raise RunCancelled()
                time.sleep(0.5)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, cmd)
        self._set_progress(run_id, 76, "collecting ML outputs")
        self._check_cancelled(run_id)
        ore_summary = json.loads((ml_dir / "ore_analysis/ore_summary.json").read_text(encoding="utf-8"))
        sulfide_mask = np.asarray(Image.open(ml_dir / "binary_sulfide/sulfide_mask.png").convert("L"))
        talc_path = ml_dir / "talc_candidate/talc_candidate_mask.png"
        talc_mask = np.asarray(Image.open(talc_path).convert("L")) if talc_path.exists() else np.zeros_like(sulfide_mask)
        intergrowth = np.asarray(Image.open(ml_dir / "ore_analysis/intergrowth_mask.png").convert("L"))
        analyzed_path = ml_dir / "ore_analysis/analyzed_mask.png"
        analyzed_mask = np.asarray(Image.open(analyzed_path).convert("L")) if analyzed_path.exists() else np.ones_like(sulfide_mask)
        final_mask = final_mask_from_classified(intergrowth, talc_mask)
        self._write_run_outputs(
            run_dir=run_dir,
            summary=ore_summary,
            components=[],
            sulfide_mask=sulfide_mask,
            talc_mask=talc_mask,
            analyzed_mask=analyzed_mask,
            final_mask=final_mask,
        )

    def _write_masks_from_sulfide_edit(self, parent_dir: Path, run_dir: Path, sulfide_mask: np.ndarray) -> None:
        rgb = np.asarray(Image.open(run_dir / "input/preprocessed.png").convert("RGB"))
        talc_mask = np.asarray(Image.open(parent_dir / "masks/talc_mask.png").convert("L"))
        analyzed_mask = np.asarray(Image.open(parent_dir / "masks/analyzed_mask.png").convert("L"))
        summary, components, classified = analyze_components(
            sulfide_mask=sulfide_mask,
            talc_mask=talc_mask,
            analyzed_mask=analyzed_mask,
            config=ComponentRuleConfig(),
        )
        final_mask = final_mask_from_classified(classified, talc_mask)
        self._write_run_outputs(
            run_dir=run_dir,
            summary=asdict(summary),
            components=components,
            sulfide_mask=sulfide_mask,
            talc_mask=talc_mask,
            analyzed_mask=analyzed_mask,
            final_mask=final_mask,
        )

    def _write_masks_from_final_edit(self, parent_dir: Path, run_dir: Path, final_mask: np.ndarray) -> None:
        sulfide_mask = np.asarray(Image.open(parent_dir / "masks/sulfide_mask.png").convert("L"))
        analyzed_mask = np.asarray(Image.open(parent_dir / "masks/analyzed_mask.png").convert("L"))
        talc_mask = ((final_mask == 3).astype(np.uint8) * 255)
        summary = summary_from_final_edit(sulfide_mask, final_mask, analyzed_mask)
        self._write_run_outputs(
            run_dir=run_dir,
            summary=summary,
            components=[],
            sulfide_mask=sulfide_mask,
            talc_mask=talc_mask,
            analyzed_mask=analyzed_mask,
            final_mask=final_mask,
        )

    def _copy_run_inputs(self, parent: dict[str, Any], parent_dir: Path, run_dir: Path) -> None:
        for relative in ["input/original_for_analysis.png", "input/preprocessed.png"]:
            target = run_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(parent_dir / relative, target)
        original_parent = Path(parent["input"]["original_artifact_path"])
        original_dst = run_dir / "input/original_source" / original_parent.name
        hardlink_or_copy(original_parent, original_dst)

    def _write_run_outputs(
        self,
        *,
        run_dir: Path,
        summary: dict[str, Any],
        components: list[Any],
        sulfide_mask: np.ndarray,
        talc_mask: np.ndarray,
        analyzed_mask: np.ndarray,
        final_mask: np.ndarray,
    ) -> None:
        masks_dir = run_dir / "masks"
        reports_dir = run_dir / "reports"
        masks_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray((sulfide_mask > 0).astype(np.uint8) * 255, mode="L").save(masks_dir / "sulfide_mask.png")
        Image.fromarray((talc_mask > 0).astype(np.uint8) * 255, mode="L").save(masks_dir / "talc_mask.png")
        Image.fromarray((analyzed_mask > 0).astype(np.uint8) * 255, mode="L").save(masks_dir / "analyzed_mask.png")
        Image.fromarray(final_mask.astype(np.uint8), mode="L").save(masks_dir / "final_mask.png")
        Image.fromarray(((final_mask == 1).astype(np.uint8) * 255), mode="L").save(masks_dir / "ordinary_mask.png")
        Image.fromarray(((final_mask == 2).astype(np.uint8) * 255), mode="L").save(masks_dir / "fine_mask.png")
        Image.fromarray(((final_mask == 3).astype(np.uint8) * 255), mode="L").save(masks_dir / "talc_final_mask.png")
        (reports_dir / "ore_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if components:
            write_component_csv(reports_dir / "component_features.csv", components)
        else:
            (reports_dir / "component_features.csv").write_text("", encoding="utf-8")
        run_id = Path(run_dir).name
        self._set_progress(run_id, 84, "building display layers")
        self._check_cancelled(run_id)
        self._build_display_layers(run_dir)
        self._check_cancelled(run_id)

    def _build_display_layers(self, run_dir: Path) -> None:
        display_dir = run_dir / "display"
        original = Image.open(run_dir / "input/original_for_analysis.png").convert("RGB")
        preprocessed = Image.open(run_dir / "input/preprocessed.png").convert("RGB")
        sulfide = np.asarray(Image.open(run_dir / "masks/sulfide_mask.png").convert("L"))
        final_mask = np.asarray(Image.open(run_dir / "masks/final_mask.png").convert("L"))
        layers = {
            "original": save_preview_pyramid(original, display_dir / "original", "original", self.preview_max_sides),
            "preprocessed": save_preview_pyramid(preprocessed, display_dir / "preprocessed", "preprocessed", self.preview_max_sides),
            "sulfide_overlay": save_preview_pyramid(
                colored_overlay(sulfide, None, (245, 190, 35, 145)),
                display_dir / "sulfide_overlay",
                "sulfide_overlay",
                self.preview_max_sides,
                nearest=True,
                prefer_png=True,
            ),
            "ordinary_overlay": save_preview_pyramid(
                colored_overlay(final_mask, 1, CLASS_COLORS[1]),
                display_dir / "ordinary_overlay",
                "ordinary_overlay",
                self.preview_max_sides,
                nearest=True,
                prefer_png=True,
            ),
            "fine_overlay": save_preview_pyramid(
                colored_overlay(final_mask, 2, CLASS_COLORS[2]),
                display_dir / "fine_overlay",
                "fine_overlay",
                self.preview_max_sides,
                nearest=True,
                prefer_png=True,
            ),
            "talc_overlay": save_preview_pyramid(
                colored_overlay(final_mask, 3, CLASS_COLORS[3]),
                display_dir / "talc_overlay",
                "talc_overlay",
                self.preview_max_sides,
                nearest=True,
                prefer_png=True,
            ),
        }
        display_manifest = {"schema_version": "ore-pipeline-display-v0.1", "layers": layers}
        self._write_json(display_dir / "display.json", display_manifest)

    def _finalize_run_metadata(self, metadata: dict[str, Any], run_dir: Path) -> None:
        summary = json.loads((run_dir / "reports/ore_summary.json").read_text(encoding="utf-8"))
        display = json.loads((run_dir / "display/display.json").read_text(encoding="utf-8"))["layers"]
        with Image.open(run_dir / "input/preprocessed.png") as image:
            metadata["image"] = {"width": image.size[0], "height": image.size[1], "name": Path(metadata["input"]["original_artifact_path"]).name}
        metadata["summary"] = summary
        metadata["metrics"] = metric_rows(summary)
        metadata["text_output"] = text_output_for_summary(summary)
        metadata["display"] = display
        metadata["masks"] = {
            "sulfide": str(run_dir / "masks/sulfide_mask.png"),
            "final": str(run_dir / "masks/final_mask.png"),
            "talc": str(run_dir / "masks/talc_mask.png"),
            "analyzed": str(run_dir / "masks/analyzed_mask.png"),
        }
        metadata["reports"] = {
            "summary_json": str(run_dir / "reports/ore_summary.json"),
            "component_features_csv": str(run_dir / "reports/component_features.csv"),
        }

    def _cancel_requested(self, run_id: str) -> bool:
        with self.lock:
            return bool(self.jobs.get(run_id, {}).get("cancel_requested"))

    def _check_cancelled(self, run_id: str) -> None:
        if self._cancel_requested(run_id):
            raise RunCancelled()

    def _set_progress(self, run_id: str, progress: int, status: str) -> None:
        with self.lock:
            previous = self.jobs.get(run_id, {})
            started = previous.get("started_at", time.time())
            cancel_requested = bool(previous.get("cancel_requested"))
            elapsed = max(0.0, time.time() - float(started))
            eta = None
            if progress > 1:
                eta = max(0, int(elapsed * (100 - progress) / max(progress, 1)))
            self.jobs[run_id] = {
                "progress": progress,
                "status": "canceling" if cancel_requested else "running",
                "stage": "canceling" if cancel_requested else status,
                "started_at": started,
                "eta_seconds": None if cancel_requested else eta,
                "cancel_requested": cancel_requested,
            }
        run_path = self.runs_dir / run_id / "run.json"
        if run_path.exists():
            data = json.loads(run_path.read_text(encoding="utf-8"))
            data["progress"] = progress
            data["status"] = "canceling" if cancel_requested else "running"
            data["stage"] = "canceling" if cancel_requested else status
            data["eta_seconds"] = None if cancel_requested else eta
            self._write_json(run_path, data)

    def _read_upload(self, upload_id: str) -> dict[str, Any]:
        path = self.uploads_dir / upload_id / "upload.json"
        if not path.exists():
            raise ApiError(HTTPStatus.NOT_FOUND, f"unknown upload: {upload_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_run(self, run_id: str) -> dict[str, Any]:
        path = self.runs_dir / run_id / "run.json"
        if not path.exists():
            raise ApiError(HTTPStatus.NOT_FOUND, f"unknown run: {run_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def first_preview_path(previews: Any) -> str | None:
    if not isinstance(previews, list) or not previews:
        return None
    return str(previews[-1].get("path") or "")


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


class OrePipelineHandler(BaseHTTPRequestHandler):
    server: "OrePipelineHTTPServer"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), fmt % args))

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._handle_get()
        except ApiError as exc:
            self.send_json({"error": exc.message}, status=exc.status)
        except Exception as exc:  # noqa: BLE001 - keep local app alive.
            self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._handle_post()
        except ApiError as exc:
            self.send_json({"error": exc.message}, status=exc.status)
        except Exception as exc:  # noqa: BLE001 - keep local app alive.
            self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_DELETE(self) -> None:  # noqa: N802
        try:
            self._handle_delete()
        except ApiError as exc:
            self.send_json({"error": exc.message}, status=exc.status)
        except Exception as exc:  # noqa: BLE001 - keep local app alive.
            self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_get(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/":
            self.send_redirect("/workspace")
            return
        if path in {"/workspace", "/history"}:
            self.send_html(render_html_page())
            return
        if path == "/api/runs":
            self.send_json(self.server.store.list_runs())
            return
        if path.startswith("/api/uploads/"):
            upload_id = urllib.parse.unquote(path.removeprefix("/api/uploads/"))
            self.send_json(self.server.store.upload_payload(upload_id))
            return
        if path.startswith("/api/runs/") and path.endswith("/metrics.csv"):
            run_id = urllib.parse.unquote(path.removeprefix("/api/runs/").removesuffix("/metrics.csv"))
            self.send_file(self.server.store.metrics_csv_path(run_id), content_type="text/csv; charset=utf-8")
            return
        if path.startswith("/api/runs/") and path.endswith("/report.pdf"):
            run_id = urllib.parse.unquote(path.removeprefix("/api/runs/").removesuffix("/report.pdf"))
            self.send_file(self.server.store.pdf_report_path(run_id), content_type="application/pdf")
            return
        if path.startswith("/api/runs/"):
            run_id = urllib.parse.unquote(path.removeprefix("/api/runs/"))
            self.send_json(self.server.store.run_payload(run_id))
            return
        if path.startswith("/artifacts/"):
            parts = path.split("/", 3)
            if len(parts) < 3:
                raise ApiError(HTTPStatus.NOT_FOUND, "bad artifact URL")
            self.send_file(self.server.store.artifact_path(parts[2]))
            return
        raise ApiError(HTTPStatus.NOT_FOUND, "not found")

    def _handle_post(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/api/uploads":
            self.send_json(self.handle_upload())
            return
        payload = self.read_json_payload()
        if path.startswith("/api/uploads/") and path.endswith("/preprocess"):
            upload_id = urllib.parse.unquote(path.removeprefix("/api/uploads/").removesuffix("/preprocess"))
            self.send_json(self.server.store.prepare_upload(upload_id, preset_from_payload(payload)))
            return
        if path == "/api/runs/start":
            upload_id = str(payload.get("upload_id") or "")
            if not upload_id:
                raise ApiError(HTTPStatus.BAD_REQUEST, "upload_id is required")
            self.send_json(self.server.store.start_run(upload_id, preset_from_payload(payload), run_async=True))
            return
        if path.startswith("/api/runs/") and path.endswith("/cancel"):
            run_id = urllib.parse.unquote(path.removeprefix("/api/runs/").removesuffix("/cancel"))
            self.send_json(self.server.store.cancel_run(run_id))
            return
        if path.startswith("/api/runs/") and path.endswith("/fix"):
            run_id = urllib.parse.unquote(path.removeprefix("/api/runs/").removesuffix("/fix"))
            self.send_json(self.server.store.create_edit_run(run_id, payload))
            return
        raise ApiError(HTTPStatus.NOT_FOUND, "not found")

    def _handle_delete(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path.startswith("/api/runs/"):
            run_id = urllib.parse.unquote(path.removeprefix("/api/runs/"))
            self.send_json(self.server.store.delete_run(run_id))
            return
        raise ApiError(HTTPStatus.NOT_FOUND, "not found")

    def handle_upload(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ApiError(HTTPStatus.BAD_REQUEST, "upload must use multipart/form-data")
        length = int(self.headers.get("content-length") or "0")
        if length > MAX_UPLOAD_BYTES:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "uploaded image is too large")
        body = self.rfile.read(length)
        message_bytes = (
            f"Content-Type: {content_type}\r\n"
            "MIME-Version: 1.0\r\n"
            "\r\n"
        ).encode("utf-8") + body
        message = BytesParser(policy=email_default_policy).parsebytes(message_bytes)
        if not message.is_multipart():
            raise ApiError(HTTPStatus.BAD_REQUEST, "multipart body is malformed")
        for part in message.iter_parts():
            params = dict(part.get_params(header="content-disposition") or [])
            if params.get("name") != "file":
                continue
            filename = params.get("filename")
            data = part.get_payload(decode=True)
            if not filename or data is None:
                break
            return self.server.store.register_upload_from_bytes(data, str(filename))
        raise ApiError(HTTPStatus.BAD_REQUEST, "file field is required")

    def read_json_payload(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length") or "0")
        if length > MAX_JSON_BYTES:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body is too large")
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "JSON body must be an object")
        return payload

    def send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json_response(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, markup: str) -> None:
        body = markup.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", location)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def send_file(self, path: Path, content_type: str | None = None) -> None:
        content_type = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class OrePipelineHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], store: OrePipelineStore) -> None:
        self.store = store
        super().__init__(server_address, OrePipelineHandler)


def render_html_page() -> str:
    return HTML_PAGE


def parse_preview_sides(value: str) -> tuple[int, ...]:
    sides = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        sides.append(max(256, int(part)))
    return tuple(sorted(set(sides))) or (1024, 2048, 4096)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ore pipeline upload, review, edit, and history UI.")
    parser.add_argument("--workspace-dir", type=Path, default=DEFAULT_WORKSPACE_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--backend", choices=["heuristic", "ml"], default="heuristic")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT if DEFAULT_CHECKPOINT.exists() else None)
    parser.add_argument("--processing-max-side", type=int, default=2600)
    parser.add_argument("--panorama-max-side", type=int, default=1800)
    parser.add_argument("--preview-max-sides", default="1024,2048,4096")
    args = parser.parse_args()

    store = OrePipelineStore(
        workspace_dir=args.workspace_dir,
        backend=args.backend,
        checkpoint=args.checkpoint,
        processing_max_side=args.processing_max_side,
        panorama_max_side=args.panorama_max_side,
        preview_max_sides=parse_preview_sides(args.preview_max_sides),
    )
    server = OrePipelineHTTPServer((args.host, args.port), store)
    host, port = server.server_address[:2]
    print(f"Ore pipeline UI: http://{host}:{port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


HTML_PAGE = r"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Классификатор рудного шлифа</title>
  <script>
    (() => {
      const key = 'orePipelineTheme';
      let choice = 'system';
      try { choice = localStorage.getItem(key) || 'system'; } catch (_) {}
      const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
      document.documentElement.dataset.theme = choice === 'system' ? (prefersDark ? 'dark' : 'light') : choice;
      document.documentElement.dataset.themeChoice = choice;
    })();
  </script>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --panel-alt: #fbfcfd;
      --line: #d7dce3;
      --text: #151a22;
      --muted: #657083;
      --control-bg: #ffffff;
      --drop-bg: #ffffff;
      --drop-drag-bg: #eefafa;
      --check-text: #2d3440;
      --toolbar-bg: #ffffff;
      --button-bg: #ffffff;
      --viewer-bg: #20242b;
      --viewer-border: #12151a;
      --segmented-active-bg: #e9f7f6;
      --progress-bg: #e3e7ed;
      --history-bg: #ffffff;
      --modal-shadow: 0 24px 80px rgba(10, 15, 25, .35);
      --modal-backdrop: rgba(15, 19, 27, .45);
      --accent: #167c80;
      --accent-2: #8a5d12;
      --danger: #bd3434;
      --green: #1fa25a;
      --red: #d83f45;
      --blue: #2870d8;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    :root[data-theme="dark"] {
      color-scheme: dark;
      --bg: #11151b;
      --panel: #1b2028;
      --panel-alt: #151a21;
      --line: #303846;
      --text: #edf2f7;
      --muted: #a6b0c0;
      --control-bg: #121720;
      --drop-bg: #171d25;
      --drop-drag-bg: #102b2d;
      --check-text: #d9e0ea;
      --toolbar-bg: #1b2028;
      --button-bg: #151b24;
      --viewer-bg: #0d1117;
      --viewer-border: #05070a;
      --segmented-active-bg: #12383a;
      --progress-bg: #2a3240;
      --history-bg: #151b24;
      --modal-shadow: 0 24px 80px rgba(0, 0, 0, .6);
      --modal-backdrop: rgba(0, 0, 0, .62);
      --accent: #2db6b3;
      --accent-2: #d4a64c;
      --danger: #e05858;
      --green: #32c173;
      --red: #f06267;
      --blue: #5c94f5;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); }
    header { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 14px 18px; border-bottom: 1px solid var(--line); background: var(--panel); }
    h1 { margin: 0; font-size: 18px; font-weight: 720; letter-spacing: 0; }
    button, select, input, textarea { font: inherit; }
    button { border: 1px solid var(--line); background: var(--button-bg); color: var(--text); border-radius: 6px; padding: 8px 11px; cursor: pointer; }
    button.primary { background: var(--accent); border-color: var(--accent); color: white; }
    button.danger { background: var(--danger); border-color: var(--danger); color: white; }
    #fixBtn { background: var(--danger); border-color: var(--danger); color: white; }
    button:disabled { opacity: .45; cursor: not-allowed; }
    .tabs { display: flex; gap: 8px; }
    .tab.active { border-color: var(--accent); color: var(--accent); }
    .header-actions { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .theme-control, .language-control { width: auto; min-width: 128px; }
    main { display: grid; grid-template-columns: minmax(280px, 360px) minmax(0, 1fr); min-height: calc(100vh - 57px); }
    aside { padding: 16px; border-right: 1px solid var(--line); background: var(--panel-alt); overflow: auto; }
    section.workspace { padding: 16px; min-width: 0; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; margin-bottom: 14px; }
    .panel h2 { margin: 0 0 10px; font-size: 15px; }
    .drop-zone { border: 2px dashed #9aa6b6; border-radius: 8px; padding: 18px; min-height: 132px; display: grid; place-items: center; text-align: center; background: var(--drop-bg); cursor: pointer; }
    .drop-zone.drag { border-color: var(--accent); background: var(--drop-drag-bg); }
    .drop-zone.selected { padding: 10px; place-items: stretch; text-align: left; }
    .selected-upload { display: grid; grid-template-columns: 74px minmax(0, 1fr) 32px; gap: 10px; align-items: center; width: 100%; }
    .selected-upload img { width: 74px; height: 56px; object-fit: cover; border-radius: 6px; border: 1px solid var(--line); background: var(--viewer-bg); }
    .selected-upload strong { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 14px; }
    .clear-upload { width: 32px; height: 32px; padding: 0; border-radius: 999px; font-size: 18px; line-height: 1; }
    .muted { color: var(--muted); font-size: 13px; }
    .warning { color: var(--danger); font-size: 13px; margin: 6px 0 0; font-weight: 650; }
    .upload-progress { display: grid; gap: 6px; margin-top: 8px; }
    .upload-progress .progress { height: 7px; }
    .upload-progress .muted { margin: 0; }
    .run-actions { display: flex; gap: 8px; flex-wrap: wrap; }
    .controls { display: grid; gap: 9px; }
    label.check { display: flex; align-items: center; gap: 8px; color: var(--check-text); font-size: 14px; }
    select, textarea, input[type="number"] { width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 8px; background: var(--control-bg); color: var(--text); }
    .viewer-shell { background: var(--viewer-bg); border-radius: 8px; overflow: hidden; border: 1px solid var(--viewer-border); min-height: 540px; position: relative; }
    canvas { display: block; width: 100%; height: 100%; }
    #mainCanvas { height: calc(100vh - 190px); min-height: 540px; }
    .viewer-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 10px; background: var(--toolbar-bg); border: 1px solid var(--line); border-radius: 8px 8px 0 0; border-bottom: 0; flex-wrap: wrap; }
    .viewer-mode-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .side-by-side-control { display: flex; align-items: center; gap: 7px; flex-wrap: wrap; }
    .side-divider { color: var(--muted); font-size: 13px; }
    .segmented { display: inline-flex; border: 1px solid var(--line); border-radius: 7px; overflow: hidden; background: var(--control-bg); }
    .segmented button { border: 0; border-right: 1px solid var(--line); border-radius: 0; background: transparent; padding: 7px 10px; }
    .segmented button:last-child { border-right: 0; }
    .segmented button.active { background: var(--segmented-active-bg); color: var(--accent); }
    .segmented button:disabled { opacity: .35; color: var(--muted); cursor: not-allowed; }
    .progress { height: 9px; background: var(--progress-bg); border-radius: 999px; overflow: hidden; }
    .progress > div { height: 100%; background: var(--accent); width: 0%; transition: width .2s ease; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { text-align: left; border-bottom: 1px solid var(--line); padding: 8px 6px; }
    th { color: var(--muted); font-weight: 650; }
    .result-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(300px, 420px); gap: 14px; }
    .class-toggles { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    .swatch { width: 12px; height: 12px; display: inline-block; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
    .history-row { border: 1px solid var(--line); border-radius: 7px; padding: 10px; margin-bottom: 8px; background: var(--history-bg); }
    .history-row strong { display: block; font-size: 13px; word-break: break-all; }
    .history-table-wrap { overflow: auto; }
    .history-table { min-width: 1040px; }
    .history-table td.numeric, .history-table th.numeric { text-align: right; white-space: nowrap; }
    .history-table th.thumbnail, .history-table td.thumbnail { width: 74px; text-align: center; }
    .history-table td.filename { max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .history-thumb-button { width: 56px; height: 42px; padding: 0; overflow: hidden; display: inline-flex; align-items: center; justify-content: center; background: var(--viewer-bg); border-color: var(--line); }
    .history-thumb-button img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .history-thumb-placeholder { color: var(--muted); }
    .history-actions { display: flex; gap: 6px; flex-wrap: wrap; }
    .history-actions button { padding: 6px 9px; }
    dialog { border: 0; border-radius: 8px; width: min(1180px, 96vw); max-height: 94vh; padding: 0; box-shadow: var(--modal-shadow); background: var(--panel); color: var(--text); }
    dialog::backdrop { background: var(--modal-backdrop); }
    .preview-dialog { width: min(980px, 94vw); }
    .history-preview-body { padding: 12px; background: var(--bg); display: grid; gap: 10px; }
    .history-preview-body img { max-width: 100%; max-height: min(76vh, 760px); object-fit: contain; justify-self: center; background: var(--viewer-bg); border: 1px solid var(--line); border-radius: 6px; }
    .history-preview-body .muted { margin: 0; word-break: break-word; }
    .modal-head, .modal-foot { padding: 12px 14px; border-bottom: 1px solid var(--line); background: var(--panel); display: flex; justify-content: space-between; align-items: center; gap: 10px; }
    .modal-foot { border-bottom: 0; border-top: 1px solid var(--line); }
    .editor-top-toolbar { display: flex; align-items: center; gap: 10px; padding: 10px 12px; border-bottom: 1px solid var(--line); background: var(--panel-alt); flex-wrap: wrap; }
    .editor-top-toolbar strong { font-size: 14px; }
    .modal-body { display: grid; grid-template-columns: minmax(0, 1fr) 310px; gap: 12px; padding: 12px; background: var(--bg); }
    .editor-side { display: flex; flex-direction: column; min-height: min(70vh, 720px); }
    #editLayerTabs { width: 100%; }
    #editLayerTabs button { flex: 1 1 0; min-width: 0; }
    .editor-tools { display: flex; flex-wrap: wrap; gap: 8px; margin: 0; }
    .editor-tools button.active { border-color: var(--accent); color: var(--accent); background: var(--segmented-active-bg); }
    .editor-tools button:disabled { opacity: .42; }
    .brush-size-control { display: flex; align-items: center; gap: 6px; color: var(--muted); font-size: 13px; }
    .brush-size-control input { width: 74px; }
    .editor-view { height: min(70vh, 720px); background: #1f232a; border-radius: 8px; overflow: hidden; border: 1px solid #11151b; }
    #editorCanvas { height: 100%; }
    .editor-stats { margin-top: auto; padding-top: 12px; }
    .stats-table td { font-size: 13px; padding: 6px 4px; }
    .stats-table td:last-child { text-align: right; color: var(--muted); }
    .stats-table .stat-separator td { padding: 4px 0; border-bottom: 1px solid var(--line); }
    .hidden { display: none !important; }
    @media (max-width: 980px) {
      main, .result-grid, .modal-body { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      #mainCanvas { height: 62vh; min-height: 380px; }
    }
  </style>
</head>
<body>
  <header>
    <h1 data-i18n="appTitle">Классификатор рудного шлифа</h1>
    <div class="header-actions">
      <select id="languageSelect" class="language-control" aria-label="Язык" data-i18n-aria-label="languageLabel">
        <option value="ru" data-i18n="languageRussian">Русский</option>
        <option value="en" data-i18n="languageEnglish">English</option>
      </select>
      <select id="themeSelect" class="theme-control" aria-label="Тема" data-i18n-aria-label="themeLabel">
        <option value="system" data-i18n="themeSystem">Системная</option>
        <option value="light" data-i18n="themeLight">Светлая</option>
        <option value="dark" data-i18n="themeDark">Темная</option>
      </select>
      <nav class="tabs">
        <button class="tab active" id="workspaceTab" data-i18n="workspaceTab">Рабочее место</button>
        <button class="tab" id="historyTab" data-i18n="historyTab">История</button>
      </nav>
    </div>
  </header>
  <main>
    <aside>
      <div class="panel">
        <h2 data-i18n="inputImage">Входное изображение</h2>
        <div id="dropZone" class="drop-zone" tabindex="0">
          <div id="dropPrompt">
            <strong data-i18n="dropImageHere">Перетащите изображение сюда</strong>
            <div class="muted" data-i18n="dropImageHelp">или нажмите, чтобы открыть PNG, JPEG, TIFF, RAW</div>
          </div>
          <div id="selectedUpload" class="selected-upload hidden">
            <img id="selectedThumb" alt="">
            <div>
              <strong id="selectedName"></strong>
              <div id="selectedMeta" class="muted"></div>
            </div>
            <button id="clearUploadBtn" class="clear-upload" title="Очистить изображение" aria-label="Очистить изображение" data-i18n-title="clearImage" data-i18n-aria-label="clearImage" type="button">×</button>
          </div>
        </div>
        <input id="fileInput" class="hidden" type="file" accept=".png,.jpg,.jpeg,.tif,.tiff,.raw,.dng,.cr2,.cr3,.nef,.arw,.orf,.rw2,.raf,.pef,.srw,image/png,image/jpeg,image/tiff">
        <p id="uploadInfo" class="muted" data-i18n="noImageLoaded">Изображение не загружено.</p>
        <p id="uploadWarning" class="warning hidden" role="alert"></p>
        <div id="uploadProgressWrap" class="upload-progress hidden" role="status" aria-live="polite">
          <div class="progress"><div id="uploadProgressBar"></div></div>
          <p id="uploadProgressText" class="muted"></p>
        </div>
      </div>
      <div class="panel">
        <h2 data-i18n="preprocessing">Предобработка</h2>
        <div class="controls">
          <label class="check"><input type="checkbox" id="illumination" checked> <span data-i18n="illuminationNormalization">нормализация освещения</span></label>
          <label class="check"><input type="checkbox" id="denoise" checked> <span data-i18n="denoise">шумоподавление</span></label>
          <label class="check"><input type="checkbox" id="contrast" checked> <span data-i18n="contrastCorrection">коррекция контраста</span></label>
          <label class="check"><input type="checkbox" id="panoramaScaling" checked> <span data-i18n="panoramaScaling">масштабирование для панорамных снимков</span></label>
          <button id="applyPreprocessBtn" data-i18n="applyPreprocessing">Применить предобработку</button>
        </div>
      </div>
      <div class="panel">
        <h2 data-i18n="runTitle">Запуск</h2>
        <div class="run-actions">
          <button id="startBtn" class="primary" disabled data-i18n="start">Старт</button>
          <button id="stopBtn" class="danger hidden" disabled data-i18n="stop">Стоп</button>
        </div>
        <div style="height:10px"></div>
        <div class="progress"><div id="progressBar"></div></div>
        <p id="progressText" class="muted" data-i18n="statusWaiting">Ожидание изображения.</p>
      </div>
      <div class="panel">
        <h2 data-i18n="historyTitle">История</h2>
        <div id="historyList" class="muted" data-i18n="historyNoRuns">Запусков пока нет.</div>
      </div>
    </aside>
    <section class="workspace">
      <div id="workspaceView">
        <div class="viewer-toolbar">
          <div class="viewer-mode-row">
            <div class="segmented" id="viewModeButtons">
              <button data-mode="original" class="active" data-i18n="viewOriginal">оригинал</button>
              <button data-mode="preprocessed" data-i18n="viewPreprocessed">предобработка</button>
              <button data-mode="sulfide" data-i18n="viewSulfide">сульфиды</button>
              <button data-mode="final" data-i18n="viewFinal">финал</button>
            </div>
            <span class="side-divider">&lt;---&gt;</span>
            <div class="side-by-side-control">
              <span class="muted" data-i18n="sideBySide">Сравнение:</span>
              <div class="segmented" id="sideLayerButtons">
                <button data-side-layer="none" class="active" data-i18n="sideNone">нет</button>
                <button data-side-layer="preprocessed" data-i18n="viewPreprocessed">предобработка</button>
                <button data-side-layer="sulfide" data-i18n="viewSulfide">сульфиды</button>
                <button data-side-layer="final" data-i18n="viewFinal">финал</button>
              </div>
            </div>
          </div>
          <div class="class-toggles">
            <label class="check"><input type="checkbox" id="showBackground" checked> <span data-i18n="classBackground">фон</span></label>
            <label class="check"><input type="checkbox" id="showOrdinary" checked><span class="swatch" style="background:var(--green)"></span><span data-i18n="classOrdinaryShort">обычные</span></label>
            <label class="check"><input type="checkbox" id="showFine" checked><span class="swatch" style="background:var(--red)"></span><span data-i18n="classFineShort">тонкие</span></label>
            <label class="check"><input type="checkbox" id="showTalc" checked><span class="swatch" style="background:var(--blue)"></span><span data-i18n="classTalc">тальк</span></label>
            <label class="check"><input type="checkbox" id="showTiling"> <span data-i18n="showTiling">показать тайлы</span></label>
          </div>
          <button id="fixBtn" disabled data-i18n="fixMe">Исправить</button>
        </div>
        <div class="viewer-shell"><canvas id="mainCanvas"></canvas></div>
        <div id="resultPanel" class="result-grid hidden" style="margin-top:14px">
          <div class="panel">
            <h2 data-i18n="textOutputTitle">Текстовый вывод</h2>
            <p id="textOutput"></p>
          </div>
          <div class="panel">
            <h2 data-i18n="metricsTitle">Метрики</h2>
            <table id="metricsTable"></table>
            <div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap">
              <a id="csvLink"><button data-i18n="saveCsv">Сохранить CSV</button></a>
              <a id="pdfLink"><button data-i18n="savePdf">Сохранить PDF-отчет</button></a>
            </div>
          </div>
        </div>
      </div>
      <div id="historyView" class="hidden">
        <div class="panel">
          <h2 data-i18n="historyPage">История запусков</h2>
          <div id="historyPageList"></div>
        </div>
      </div>
    </section>
  </main>
  <dialog id="historyPreviewDialog" class="preview-dialog">
    <div class="modal-head">
      <strong data-i18n="historyPreviewTitle">Превью запуска</strong>
      <button id="closeHistoryPreviewBtn" data-i18n="close">Закрыть</button>
    </div>
    <div class="history-preview-body">
      <img id="historyPreviewImage" alt="">
      <p id="historyPreviewCaption" class="muted"></p>
    </div>
  </dialog>
  <dialog id="fixDialog">
    <div class="modal-head">
      <strong data-i18n="editRecalculate">Редактирование и пересчет</strong>
      <button id="closeFixBtn" data-i18n="close">Закрыть</button>
    </div>
    <div class="editor-top-toolbar" id="editorTopToolbar">
      <strong data-i18n="tool">Инструмент</strong>
      <div class="editor-tools">
        <button id="brushToolBtn" class="active" data-i18n="brush">Кисть</button>
        <button id="panToolBtn" data-i18n="pan">Панорама</button>
        <button id="undoEditBtn" disabled data-i18n="undo">Отменить</button>
        <button id="redoEditBtn" disabled data-i18n="redo">Повторить</button>
        <button id="zoomOutEditBtn">-</button>
        <button id="zoomInEditBtn">+</button>
        <button id="fitEditBtn" data-i18n="fitView">Вписать</button>
      </div>
      <label class="brush-size-control"><span data-i18n="brushSize">Размер кисти</span><input id="brushSize" type="number" min="2" max="120" value="18"></label>
    </div>
    <div class="modal-body">
      <div class="editor-view"><canvas id="editorCanvas"></canvas></div>
      <div class="panel editor-side" style="margin:0">
        <h2 data-i18n="layer">Слой</h2>
        <div class="segmented" id="editLayerTabs">
          <button data-layer="sulfide" class="active" data-i18n="sulfideLayer">сульфиды/не сульфиды</button>
          <button data-layer="final" data-i18n="finalSegmentation">финальная сегментация</button>
        </div>
        <div style="height:10px"></div>
        <div id="classSelector">
          <h2 data-i18n="classTitle">Класс</h2>
          <select id="editClass">
            <option value="0" data-i18n="classBackground">фон</option>
            <option value="1" data-i18n="classOrdinary">обычные срастания</option>
            <option value="2" data-i18n="classFine">тонкие срастания</option>
            <option value="3" data-i18n="classTalc">тальк</option>
          </select>
        </div>
        <h2 data-i18n="comment">Комментарий</h2>
        <textarea id="editComment" rows="5" placeholder="Комментарий к изменению" data-i18n-placeholder="commentPlaceholder"></textarea>
        <p class="muted" data-i18n="editorHelp">Кисть: левая кнопка рисует, правая стирает. Панорама перемещает вид.</p>
        <div class="editor-stats">
          <h2 data-i18n="statistics">Статистика</h2>
          <table id="editorStats" class="stats-table"></table>
        </div>
      </div>
    </div>
    <div class="modal-foot">
      <span id="editStatus" class="muted" data-i18n="editNoEdits">Нет правок.</span>
      <button id="fixRestartBtn" class="primary" disabled data-i18n="fixAndRestart">Исправить и перезапустить</button>
    </div>
  </dialog>
  <script>
    const state = {
      upload: null,
      run: null,
      viewMode: 'original',
      sideLayer: 'none',
      splitter: 0.5,
      pan: {x: 0, y: 0},
      zoom: 1,
      dragging: false,
      dragSplitter: false,
      last: {x: 0, y: 0},
      images: new Map(),
      editor: {
        layer: 'sulfide',
        dirty: false,
        mask: null,
        width: 0,
        height: 0,
        drawing: false,
        panning: false,
        tool: 'brush',
        zoom: 1,
        pan: {x: 0, y: 0},
        last: {x: 0, y: 0},
        undo: [],
        redo: [],
        strokeStarted: false
      }
    };
    const $ = (id) => document.getElementById(id);
    const canvas = $('mainCanvas');
    const ctx = canvas.getContext('2d');
    const editorCanvas = $('editorCanvas');
    const editorCtx = editorCanvas.getContext('2d');
    const THEME_STORAGE_KEY = 'orePipelineTheme';
    const LANGUAGE_STORAGE_KEY = 'orePipelineLanguage';
    const PREPROCESS_STORAGE_KEY = 'orePipelinePreprocessPreset';
    const DEFAULT_LANGUAGE = 'ru';
    const DEFAULT_PREPROCESS_PRESET = {
      illumination_normalization: true,
      denoise: true,
      contrast_correction: true,
      panorama_scaling: true
    };
    let statusMessage = {key: 'statusWaiting', params: {}};
    let uploadWarningMessage = null;
    let uploadProgressMessage = null;
    let uploadProgressTimer = null;
    let activePollRunId = null;
    const ACTIVE_RUN_STATUSES = new Set(['queued', 'running', 'canceling']);
    const SUPPORTED_UPLOAD_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.raw', '.dng', '.cr2', '.cr3', '.nef', '.arw', '.orf', '.rw2', '.raf', '.pef', '.srw']);
    const I18N = {
      ru: {
        appTitle: 'Классификатор рудного шлифа',
        pageTitle: 'Классификатор рудного шлифа',
        languageLabel: 'Язык',
        languageRussian: 'Русский',
        languageEnglish: 'English',
        themeLabel: 'Тема',
        themeSystem: 'Системная',
        themeLight: 'Светлая',
        themeDark: 'Темная',
        workspaceTab: 'Рабочее место',
        historyTab: 'История',
        inputImage: 'Входное изображение',
        dropImageHere: 'Перетащите изображение сюда',
        dropImageHelp: 'или нажмите, чтобы открыть PNG, JPEG, TIFF, RAW',
        clearImage: 'Очистить изображение',
        noImageLoaded: 'Изображение не загружено.',
        selectedImage: 'Выбранное изображение',
        invalidImageFormat: 'Неподдерживаемый формат файла: {name}. Поддерживаются PNG, JPEG, TIFF, RAW.',
        uploadFailed: 'Не удалось загрузить файл: {error}',
        uploadProgressUploading: 'Загрузка файла: {progress}%',
        uploadProgressPreparing: 'Подготовка предпросмотра: {progress}%',
        uploadProgressComplete: 'Предпросмотр готов.',
        statusUploadingProgress: 'Загрузка {name} · {progress}%',
        statusPreparingPreview: 'Подготовка предпросмотра · {progress}%',
        preprocessing: 'Предобработка',
        illuminationNormalization: 'нормализация освещения',
        denoise: 'шумоподавление',
        contrastCorrection: 'коррекция контраста',
        panoramaScaling: 'масштабирование для панорамных снимков',
        applyPreprocessing: 'Применить предобработку',
        runTitle: 'Запуск',
        start: 'Старт',
        stop: 'Стоп',
        historyTitle: 'История',
        historyNoRuns: 'Запусков пока нет.',
        viewOriginal: 'оригинал',
        viewPreprocessed: 'предобработка',
        viewSulfide: 'сульфиды',
        viewFinal: 'финал',
        sideBySide: 'Сравнение:',
        sideNone: 'нет',
        classBackground: 'фон',
        classOrdinaryShort: 'обычные',
        classFineShort: 'тонкие',
        classOrdinary: 'обычные срастания',
        classFine: 'тонкие срастания',
        classTalc: 'тальк',
        showTiling: 'показать тайлы',
        fixMe: 'Исправить',
        textOutputTitle: 'Текстовый вывод',
        metricsTitle: 'Метрики',
        saveCsv: 'Сохранить CSV',
        savePdf: 'Сохранить PDF-отчет',
        historyPage: 'История запусков',
        editRecalculate: 'Редактирование и пересчет',
        close: 'Закрыть',
        tool: 'Инструмент',
        brush: 'Кисть',
        pan: 'Панорама',
        undo: 'Отменить',
        redo: 'Повторить',
        fitView: 'Вписать',
        brushSize: 'Размер кисти',
        layer: 'Слой',
        sulfideLayer: 'сульфиды/не сульфиды',
        finalSegmentation: 'финальная сегментация',
        classTitle: 'Класс',
        comment: 'Комментарий',
        commentPlaceholder: 'Комментарий к изменению',
        editorHelp: 'Кисть: левая кнопка рисует, правая стирает. Панорама перемещает вид.',
        statistics: 'Статистика',
        editNoEdits: 'Нет правок.',
        editorLoading: 'Загрузка слоя...',
        editorLoadFailed: 'Не удалось загрузить изображение или сегментацию: {error}',
        editorMissingMask: 'нет маски выбранного слоя',
        editorMissingBaseImage: 'нет изображения подложки',
        fixAndRestart: 'Исправить и перезапустить',
        statusWaiting: 'Ожидание изображения.',
        statusUploading: 'Загрузка {name}',
        statusImageLoaded: 'Изображение загружено.',
        statusPreprocessUpdated: 'Предобработанный предпросмотр обновлен.',
        statusProgress: '{stage} · {progress}%{eta}',
        statusEta: ' · осталось {seconds} с',
        statusFailed: 'Ошибка: {error}',
        statusCanceling: 'Остановка запуска...',
        statusCanceled: 'Запуск остановлен.',
        statusCancelFailed: 'Не удалось остановить запуск: {error}',
        statusRunLoaded: 'Загружен {runId}. Настройте параметры и нажмите Старт, чтобы создать новый запуск.',
        statusRunLoadedNoUpload: 'Загружен {runId}. Исходное изображение недоступно.',
        unknownError: 'неизвестная ошибка',
        stageQueued: 'в очереди',
        stageRunning: 'выполнение',
        stageCanceling: 'остановка',
        stageCanceled: 'остановлено',
        stageComplete: 'готово',
        stageFailed: 'ошибка',
        stagePreprocessing: 'предобработка',
        stageSulfide: 'сегментация сульфидов/не сульфидов',
        stageFinal: 'финальная сегментация',
        stageReport: 'расчет метрик и отчета',
        metricsHeaderMetric: 'Метрика',
        metricsHeaderValue: 'Значение',
        historyThumbnail: 'Миниатюра',
        historyPreviewTitle: 'Превью запуска',
        historyPreviewOpen: 'Открыть превью {name}',
        historyFilename: 'Файл',
        historyDate: 'Дата',
        historyOreClassification: 'Классификация руды',
        historySulfides: 'Сульфиды',
        historyNonSulfides: 'Не сульфиды',
        historyOrdinaryIntergrowth: 'Обычные срастания',
        historyFineIntergrowth: 'Тонкие срастания',
        historyTalc: 'Тальк',
        historyActions: 'Действия',
        metricSulfideFraction: 'Общая доля сульфидов',
        metricOrdinaryFraction: 'Доля обычных срастаний',
        metricFineFraction: 'Доля тонких срастаний',
        metricTalcFraction: 'Доля талька',
        metricComponentCount: 'Компоненты сульфидов',
        metricAnalyzedFraction: 'Доля проанализированной области',
        historyLoad: 'Загрузить',
        historyRemove: 'Удалить',
        confirmRemoveRun: 'Удалить запуск {runId}?',
        statusRunRemoved: 'Запуск {runId} удален из истории.',
        editUndo: 'Отмена применена.',
        editRedo: 'Повтор применен.',
        editUnsaved: 'Есть несохраненная правка.',
        editEraseStroke: 'Штрих стирания.',
        editDrawStroke: 'Штрих рисования.',
        statSulfide: 'сульфиды',
        statNonSulfide: 'не сульфиды',
        statOrdinary: 'обычные срастания',
        statFine: 'тонкие срастания',
        statTalc: 'тальк',
        oreClassTalcose: 'оталькованная',
        oreClassRow: 'рядовая',
        oreClassHard: 'труднообогатимая',
        dominantFine: 'тонких срастаний',
        dominantOrdinary: 'обычных срастаний',
        runText: 'Руда классифицирована как {ore}: содержание талька — {talcPct}%, преобладание {dominant} — {dominantPct}%.'
      },
      en: {
        appTitle: 'Ore thin-section classifier',
        pageTitle: 'Ore Pipeline UI',
        languageLabel: 'Language',
        languageRussian: 'Russian',
        languageEnglish: 'English',
        themeLabel: 'Theme',
        themeSystem: 'System',
        themeLight: 'Light',
        themeDark: 'Dark',
        workspaceTab: 'Workspace',
        historyTab: 'History',
        inputImage: 'Input image',
        dropImageHere: 'Drop image here',
        dropImageHelp: 'or click to open PNG, JPEG, TIFF, RAW',
        clearImage: 'Clear image',
        noImageLoaded: 'No image loaded.',
        selectedImage: 'Selected image',
        invalidImageFormat: 'Unsupported file format: {name}. Supported formats: PNG, JPEG, TIFF, RAW.',
        uploadFailed: 'Could not upload file: {error}',
        uploadProgressUploading: 'Uploading file: {progress}%',
        uploadProgressPreparing: 'Preparing preview: {progress}%',
        uploadProgressComplete: 'Preview is ready.',
        statusUploadingProgress: 'Uploading {name} · {progress}%',
        statusPreparingPreview: 'Preparing preview · {progress}%',
        preprocessing: 'Preprocessing',
        illuminationNormalization: 'illumination normalization',
        denoise: 'noise reduction',
        contrastCorrection: 'contrast correction',
        panoramaScaling: 'panorama image scaling',
        applyPreprocessing: 'Apply preprocessing',
        runTitle: 'Run',
        start: 'Start',
        stop: 'Stop',
        historyTitle: 'History',
        historyNoRuns: 'No runs yet.',
        viewOriginal: 'original',
        viewPreprocessed: 'preprocessed',
        viewSulfide: 'sulfide',
        viewFinal: 'final',
        sideBySide: 'Side-by-side:',
        sideNone: 'none',
        classBackground: 'background',
        classOrdinaryShort: 'ordinary',
        classFineShort: 'fine',
        classOrdinary: 'ordinary intergrowth',
        classFine: 'fine intergrowth',
        classTalc: 'talc',
        showTiling: 'show tiling',
        fixMe: 'Fix me',
        textOutputTitle: 'Text output',
        metricsTitle: 'Metrics',
        saveCsv: 'Save to CSV',
        savePdf: 'Save PDF Report',
        historyPage: 'History page',
        editRecalculate: 'Edit & Recalculate',
        close: 'Close',
        tool: 'Tool',
        brush: 'Brush',
        pan: 'Pan',
        undo: 'Undo',
        redo: 'Redo',
        fitView: 'Fit view',
        brushSize: 'Brush size',
        layer: 'Layer',
        sulfideLayer: 'sulfide/non-sulfide',
        finalSegmentation: 'final segmentation',
        classTitle: 'Class',
        comment: 'Comment',
        commentPlaceholder: 'Comment for the change',
        editorHelp: 'Brush: left draws, right erases. Pan moves the view.',
        statistics: 'Statistics',
        editNoEdits: 'No edits yet.',
        editorLoading: 'Loading layer...',
        editorLoadFailed: 'Could not load image or segmentation: {error}',
        editorMissingMask: 'selected layer mask is missing',
        editorMissingBaseImage: 'base image is missing',
        fixAndRestart: 'Fix and Restart',
        statusWaiting: 'Waiting for image.',
        statusUploading: 'Uploading {name}',
        statusImageLoaded: 'Image loaded.',
        statusPreprocessUpdated: 'Preprocessing preview updated.',
        statusProgress: '{stage} · {progress}%{eta}',
        statusEta: ' · ETA {seconds}s',
        statusFailed: 'Failed: {error}',
        statusCanceling: 'Stopping run...',
        statusCanceled: 'Run stopped.',
        statusCancelFailed: 'Could not stop run: {error}',
        statusRunLoaded: 'Loaded {runId}. Tune parameters and press Start to create a new run.',
        statusRunLoadedNoUpload: 'Loaded {runId}. Original upload is not available.',
        unknownError: 'unknown error',
        stageQueued: 'queued',
        stageRunning: 'running',
        stageCanceling: 'stopping',
        stageCanceled: 'stopped',
        stageComplete: 'complete',
        stageFailed: 'failed',
        stagePreprocessing: 'preprocessing',
        stageSulfide: 'sulfide/non-sulfide segmentation',
        stageFinal: 'final segmentation',
        stageReport: 'metrics and report calculation',
        metricsHeaderMetric: 'Metric',
        metricsHeaderValue: 'Value',
        historyThumbnail: 'Thumbnail',
        historyPreviewTitle: 'Run preview',
        historyPreviewOpen: 'Open preview for {name}',
        historyFilename: 'Filename',
        historyDate: 'Date',
        historyOreClassification: 'Ore classification',
        historySulfides: 'Sulfides',
        historyNonSulfides: 'Non-sulfides',
        historyOrdinaryIntergrowth: 'Ordinary intergrowths',
        historyFineIntergrowth: 'Fine intergrowths',
        historyTalc: 'Talc',
        historyActions: 'Actions',
        metricSulfideFraction: 'Total sulfide fraction',
        metricOrdinaryFraction: 'Ordinary intergrowth fraction',
        metricFineFraction: 'Fine intergrowth fraction',
        metricTalcFraction: 'Talc fraction',
        metricComponentCount: 'Sulfide components',
        metricAnalyzedFraction: 'Analyzed-area fraction',
        historyLoad: 'Load',
        historyRemove: 'Remove',
        confirmRemoveRun: 'Remove run {runId}?',
        statusRunRemoved: 'Run {runId} removed from history.',
        editUndo: 'Undo applied.',
        editRedo: 'Redo applied.',
        editUnsaved: 'Unsaved edit.',
        editEraseStroke: 'Erase stroke.',
        editDrawStroke: 'Draw stroke.',
        statSulfide: 'sulfide',
        statNonSulfide: 'non-sulfide',
        statOrdinary: 'ordinary intergrowth',
        statFine: 'fine intergrowth',
        statTalc: 'talc',
        oreClassTalcose: 'talcose ore',
        oreClassRow: 'ordinary ore',
        oreClassHard: 'hard-to-process ore',
        dominantFine: 'fine intergrowth',
        dominantOrdinary: 'ordinary intergrowth',
        runText: 'Ore classified as {ore}: talc content {talcPct}%, dominant {dominant} {dominantPct}%.'
      }
    };

    function currentLanguage() {
      const value = $('languageSelect') && $('languageSelect').value;
      return I18N[value] ? value : DEFAULT_LANGUAGE;
    }
    function storedLanguageChoice() {
      try {
        const value = localStorage.getItem(LANGUAGE_STORAGE_KEY) || DEFAULT_LANGUAGE;
        return I18N[value] ? value : DEFAULT_LANGUAGE;
      } catch (_) {
        return DEFAULT_LANGUAGE;
      }
    }
    function t(key, params = {}) {
      const dictionary = I18N[currentLanguage()] || I18N[DEFAULT_LANGUAGE];
      const template = dictionary[key] || I18N[DEFAULT_LANGUAGE][key] || key;
      return template.replace(/\{(\w+)\}/g, (_, name) => String(params[name] ?? ''));
    }
    function localeCode() {
      return currentLanguage() === 'ru' ? 'ru-RU' : 'en-US';
    }
    function fixedPercent(value) {
      return (Number(value || 0) * 100).toFixed(1);
    }
    function oreClassText(summary) {
      const oreClass = String((summary && summary.ore_class) || '');
      if (oreClass === 'talcose_ore') return t('oreClassTalcose');
      if (oreClass === 'row_ore') return t('oreClassRow');
      if (oreClass === 'hard_to_process_ore') return t('oreClassHard');
      return currentLanguage() === 'ru'
        ? String((summary && summary.ore_class_ru) || 'неизвестная')
        : (oreClass || 'unknown ore');
    }
    function localizedRunText(run) {
      const summary = (run && run.summary) || {};
      if (!summary || !Object.keys(summary).length) return (run && run.text_output) || '';
      const ordinary = Number(summary.ordinary_sulfide_fraction || 0);
      const fine = Number(summary.fine_sulfide_fraction || 0);
      return t('runText', {
        ore: oreClassText(summary),
        talcPct: fixedPercent(summary.talc_fraction),
        dominant: fine >= ordinary ? t('dominantFine') : t('dominantOrdinary'),
        dominantPct: fixedPercent(fine >= ordinary ? fine : ordinary)
      });
    }
    function localizedMetricLabel(row) {
      const keyMap = {
        sulfide_fraction: 'metricSulfideFraction',
        ordinary_sulfide_fraction: 'metricOrdinaryFraction',
        fine_sulfide_fraction: 'metricFineFraction',
        talc_fraction: 'metricTalcFraction',
        component_count: 'metricComponentCount',
        analyzed_fraction: 'metricAnalyzedFraction'
      };
      return keyMap[row.key] ? t(keyMap[row.key]) : (row.label || row.key || '');
    }
    function formatDate(value) {
      if (!value) return '';
      const parsed = new Date(value);
      if (Number.isNaN(parsed.getTime())) return String(value);
      return parsed.toLocaleString(localeCode());
    }
    function formatFraction(value) {
      const number = Number(value || 0);
      return `${(number * 100).toFixed(1)}%`;
    }
    function runFilename(run) {
      return (run && run.image && run.image.name) || (run && run.run_id) || '';
    }
    function runSummary(run) {
      return (run && run.summary) || {};
    }
    function applyLanguage(language) {
      const value = I18N[language] ? language : DEFAULT_LANGUAGE;
      if ($('languageSelect')) $('languageSelect').value = value;
      document.documentElement.lang = value;
      document.title = t('pageTitle');
      document.querySelectorAll('[data-i18n]').forEach(node => { node.textContent = t(node.dataset.i18n); });
      document.querySelectorAll('[data-i18n-placeholder]').forEach(node => { node.placeholder = t(node.dataset.i18nPlaceholder); });
      document.querySelectorAll('[data-i18n-title]').forEach(node => { node.title = t(node.dataset.i18nTitle); });
      document.querySelectorAll('[data-i18n-aria-label]').forEach(node => { node.setAttribute('aria-label', t(node.dataset.i18nAriaLabel)); });
      if (statusMessage) $('progressText').textContent = t(statusMessage.key, statusMessage.params);
      if (uploadWarningMessage) setUploadWarning(uploadWarningMessage.key, uploadWarningMessage.params);
      if (uploadProgressMessage) setUploadProgress(uploadProgressMessage.key, uploadProgressMessage.progress, uploadProgressMessage.params);
      if (state.editor.statusMessage) $('editStatus').textContent = t(state.editor.statusMessage.key, state.editor.statusMessage.params);
      if (state.run && state.run.downloads && state.run.metrics && !document.hidden) renderResults(state.run);
      updateEditorStats();
    }
    function setLanguage(language) {
      const value = I18N[language] ? language : DEFAULT_LANGUAGE;
      try { localStorage.setItem(LANGUAGE_STORAGE_KEY, value); } catch (_) {}
      applyLanguage(value);
      refreshHistory();
    }

    function cssColor(name) {
      return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }
    function storedThemeChoice() {
      try { return localStorage.getItem(THEME_STORAGE_KEY) || 'system'; } catch (_) { return 'system'; }
    }
    function applyThemeChoice(choice) {
      const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
      const resolved = choice === 'system' ? (prefersDark ? 'dark' : 'light') : choice;
      document.documentElement.dataset.theme = resolved;
      document.documentElement.dataset.themeChoice = choice;
      if ($('themeSelect')) $('themeSelect').value = choice;
      drawMain();
      drawEditor();
    }
    function setThemeChoice(choice) {
      try { localStorage.setItem(THEME_STORAGE_KEY, choice); } catch (_) {}
      applyThemeChoice(choice);
    }
    $('themeSelect').value = storedThemeChoice();
    $('themeSelect').addEventListener('change', (event) => setThemeChoice(event.target.value));
    if (window.matchMedia) {
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        if (storedThemeChoice() === 'system') applyThemeChoice('system');
      });
    }
    applyThemeChoice(storedThemeChoice());
    $('languageSelect').value = storedLanguageChoice();
    $('languageSelect').addEventListener('change', (event) => setLanguage(event.target.value));
    applyLanguage(storedLanguageChoice());

    function presetPayload() {
      return {
        illumination_normalization: $('illumination').checked,
        denoise: $('denoise').checked,
        contrast_correction: $('contrast').checked,
        panorama_scaling: $('panoramaScaling').checked
      };
    }
    function presetBoolean(values, primary, alias, fallback) {
      if (Object.prototype.hasOwnProperty.call(values, primary)) return Boolean(values[primary]);
      if (alias && Object.prototype.hasOwnProperty.call(values, alias)) return Boolean(values[alias]);
      return Boolean(fallback);
    }
    function normalizedPreprocessPreset(preset = {}, fallback = DEFAULT_PREPROCESS_PRESET) {
      const values = preset || {};
      return {
        illumination_normalization: presetBoolean(values, 'illumination_normalization', 'illumination', fallback.illumination_normalization),
        denoise: presetBoolean(values, 'denoise', 'noise_reduction', fallback.denoise),
        contrast_correction: presetBoolean(values, 'contrast_correction', 'contrast', fallback.contrast_correction),
        panorama_scaling: presetBoolean(values, 'panorama_scaling', 'panoramaScaling', fallback.panorama_scaling)
      };
    }
    function storedPreprocessPreset() {
      try {
        const raw = localStorage.getItem(PREPROCESS_STORAGE_KEY);
        return raw ? normalizedPreprocessPreset(JSON.parse(raw), DEFAULT_PREPROCESS_PRESET) : {...DEFAULT_PREPROCESS_PRESET};
      } catch (_) {
        return {...DEFAULT_PREPROCESS_PRESET};
      }
    }
    function savePreprocessPreset() {
      try { localStorage.setItem(PREPROCESS_STORAGE_KEY, JSON.stringify(presetPayload())); } catch (_) {}
    }
    function applyPresetToControls(preset, options = {}) {
      const values = preset || {};
      const normalized = normalizedPreprocessPreset(values, options.fallback || DEFAULT_PREPROCESS_PRESET);
      $('illumination').checked = normalized.illumination_normalization;
      $('denoise').checked = normalized.denoise;
      $('contrast').checked = normalized.contrast_correction;
      $('panoramaScaling').checked = normalized.panorama_scaling;
      if (options.save) savePreprocessPreset();
    }
    ['illumination','denoise','contrast','panoramaScaling'].forEach(id => $(id).addEventListener('change', savePreprocessPreset));
    applyPresetToControls(storedPreprocessPreset(), {save: false});
    function setStatus(key, params = {}) {
      statusMessage = {key, params};
      $('progressText').textContent = t(key, params);
    }
    function setProgress(value) { $('progressBar').style.width = `${Math.max(0, Math.min(100, value || 0))}%`; }
    function runIsActive(run) {
      return Boolean(run && ACTIVE_RUN_STATUSES.has(String(run.status || '').toLowerCase()));
    }
    function updateRunControls(run = state.run) {
      const active = runIsActive(run);
      $('startBtn').classList.toggle('hidden', active);
      $('stopBtn').classList.toggle('hidden', !active);
      $('startBtn').disabled = active || !state.upload;
      $('stopBtn').disabled = !active || String((run && run.status) || '').toLowerCase() === 'canceling';
    }
    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(320, Math.floor(rect.width * devicePixelRatio));
      canvas.height = Math.max(280, Math.floor(rect.height * devicePixelRatio));
      drawMain();
      const er = editorCanvas.getBoundingClientRect();
      editorCanvas.width = Math.max(320, Math.floor(er.width * devicePixelRatio));
      editorCanvas.height = Math.max(280, Math.floor(er.height * devicePixelRatio));
      drawEditor();
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    function bestPreview(previews) {
      if (!previews || !previews.length) return null;
      const desired = Math.max(canvas.width, canvas.height) * Math.max(1, state.zoom);
      let chosen = previews[previews.length - 1];
      for (const item of previews) {
        chosen = item;
        if (item.max_side >= desired) break;
      }
      return chosen;
    }
    async function loadImage(url) {
      if (!url) return null;
      if (state.images.has(url)) return state.images.get(url);
      const image = new Image();
      image.decoding = 'async';
      image.src = url;
      await image.decode();
      state.images.set(url, image);
      return image;
    }
    function displaySource() {
      if (state.run && state.run.display && Object.keys(state.run.display).length) return state.run;
      if (state.upload) return state.upload;
      return state.run;
    }
    function hasPreview(display, key) {
      return Boolean(display && display[key] && display[key].length);
    }
    function layerAvailable(layer) {
      const source = displaySource();
      const display = source && source.display ? source.display : {};
      if (layer === 'original') return hasPreview(display, 'original');
      if (layer === 'preprocessed') return hasPreview(display, 'preprocessed');
      if (layer === 'sulfide') return Boolean(state.run && state.run.status === 'complete' && hasPreview(display, 'sulfide_overlay'));
      if (layer === 'final') return Boolean(state.run && state.run.status === 'complete' && (hasPreview(display, 'ordinary_overlay') || hasPreview(display, 'fine_overlay') || hasPreview(display, 'talc_overlay')));
      return false;
    }
    function sideLayerAvailable(layer) {
      return layer === 'none' || layerAvailable(layer);
    }
    function tilingManifest() {
      const source = displaySource();
      const tiling = source && source.tiling;
      if (!tiling || !tiling.enabled || !Array.isArray(tiling.tiles) || !tiling.tiles.length) return null;
      return tiling;
    }
    function tilingAvailable() {
      return Boolean(tilingManifest());
    }
    function updateViewControls() {
      if (!layerAvailable(state.viewMode)) {
        state.viewMode = layerAvailable('original') ? 'original' : 'original';
      }
      if (!sideLayerAvailable(state.sideLayer)) {
        state.sideLayer = 'none';
      }
      document.querySelectorAll('#viewModeButtons button').forEach(btn => {
        const available = layerAvailable(btn.dataset.mode);
        btn.disabled = !available;
        btn.classList.toggle('active', btn.dataset.mode === state.viewMode);
      });
      document.querySelectorAll('#sideLayerButtons button').forEach(btn => {
        const available = sideLayerAvailable(btn.dataset.sideLayer);
        btn.disabled = !available;
        btn.classList.toggle('active', btn.dataset.sideLayer === state.sideLayer);
      });
      if ($('showTiling')) {
        const available = tilingAvailable();
        $('showTiling').disabled = !available;
        if (!available) $('showTiling').checked = false;
      }
    }
    async function drawMain() {
      updateViewControls();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = cssColor('--viewer-bg') || '#20242b';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      const source = displaySource();
      if (!source) return;
      const display = source.display || {};
      if (state.sideLayer !== 'none' && sideLayerAvailable(state.sideLayer)) {
        await drawSideBySide(display);
      } else {
        await drawCompositeLayer(display, state.viewMode, 0, canvas.width);
      }
      await drawTilingGrid(display);
    }
    function imageRect(image) {
      const fit = Math.min(canvas.width / image.width, canvas.height / image.height);
      const scale = fit * state.zoom;
      const w = image.width * scale;
      const h = image.height * scale;
      return {x: (canvas.width - w) / 2 + state.pan.x, y: (canvas.height - h) / 2 + state.pan.y, w, h};
    }
    async function drawLayer(display, key, clipX, clipW) {
      const preview = bestPreview(display[key]);
      const image = await loadImage(preview && preview.url);
      if (!image) return;
      const rect = imageRect(image);
      ctx.save();
      ctx.beginPath();
      ctx.rect(clipX, 0, clipW, canvas.height);
      ctx.clip();
      if (key !== 'preprocessed' || $('showBackground').checked || !state.run) {
        ctx.drawImage(image, rect.x, rect.y, rect.w, rect.h);
      }
      ctx.restore();
    }
    async function drawOverlay(previews, clipX = 0, clipW = canvas.width) {
      const preview = bestPreview(previews);
      const image = await loadImage(preview && preview.url);
      if (!image) return;
      const rect = imageRect(image);
      ctx.save();
      ctx.beginPath();
      ctx.rect(clipX, 0, clipW, canvas.height);
      ctx.clip();
      ctx.drawImage(image, rect.x, rect.y, rect.w, rect.h);
      ctx.restore();
    }
    async function drawFinalOverlays(display, clipX = 0, clipW = canvas.width) {
      if ($('showOrdinary').checked) await drawOverlay(display.ordinary_overlay, clipX, clipW);
      if ($('showFine').checked) await drawOverlay(display.fine_overlay, clipX, clipW);
      if ($('showTalc').checked) await drawOverlay(display.talc_overlay, clipX, clipW);
    }
    async function drawTilingGrid(display) {
      const tiling = $('showTiling').checked ? tilingManifest() : null;
      if (!tiling) return;
      const key = state.viewMode === 'original' ? 'original' : 'preprocessed';
      const preview = bestPreview(display[key] || display.preprocessed || display.original);
      const image = await loadImage(preview && preview.url);
      if (!image) return;
      const rect = imageRect(image);
      const useSourceCoordinates = !state.run && key === 'original';
      const coordinateWidth = Math.max(1, Number(useSourceCoordinates ? tiling.source_width : tiling.analysis_width));
      const coordinateHeight = Math.max(1, Number(useSourceCoordinates ? tiling.source_height : tiling.analysis_height));
      const scaleX = useSourceCoordinates ? Number(tiling.source_width) / Math.max(1, Number(tiling.analysis_width)) : 1;
      const scaleY = useSourceCoordinates ? Number(tiling.source_height) / Math.max(1, Number(tiling.analysis_height)) : 1;
      ctx.save();
      ctx.beginPath();
      ctx.rect(rect.x, rect.y, rect.w, rect.h);
      ctx.clip();
      ctx.lineWidth = Math.max(1.5, 1.5 * devicePixelRatio);
      ctx.strokeStyle = 'rgba(255,255,255,0.92)';
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.font = `${Math.max(10, 11 * devicePixelRatio)}px ui-sans-serif, system-ui`;
      for (let index = 0; index < tiling.tiles.length; index++) {
        const tile = tiling.tiles[index];
        const x = Number(tile.x || 0) * scaleX;
        const y = Number(tile.y || 0) * scaleY;
        const w = Number(tile.width || tiling.tile_size || 0) * scaleX;
        const h = Number(tile.height || tiling.tile_size || 0) * scaleY;
        const px = rect.x + (x / coordinateWidth) * rect.w;
        const py = rect.y + (y / coordinateHeight) * rect.h;
        const pw = (w / coordinateWidth) * rect.w;
        const ph = (h / coordinateHeight) * rect.h;
        ctx.strokeRect(px, py, pw, ph);
        if (pw > 42 * devicePixelRatio && ph > 28 * devicePixelRatio) {
          const label = String(index + 1);
          const metrics = ctx.measureText(label);
          ctx.fillRect(px + 4, py + 4, metrics.width + 8, 16 * devicePixelRatio);
          ctx.fillStyle = 'rgba(255,255,255,0.95)';
          ctx.fillText(label, px + 8, py + 16 * devicePixelRatio);
          ctx.fillStyle = 'rgba(0,0,0,0.55)';
        }
      }
      ctx.restore();
    }
    async function drawCompositeLayer(display, layer, clipX, clipW) {
      if (layer === 'original') {
        await drawLayer(display, 'original', clipX, clipW);
        return;
      }
      await drawLayer(display, 'preprocessed', clipX, clipW);
      if (layer === 'sulfide') {
        await drawOverlay(display.sulfide_overlay, clipX, clipW);
      } else if (layer === 'final') {
        await drawFinalOverlays(display, clipX, clipW);
      }
    }
    async function drawSideBySide(display) {
      const divider = Math.floor(canvas.width * state.splitter);
      await drawCompositeLayer(display, state.viewMode, 0, divider);
      await drawCompositeLayer(display, state.sideLayer, divider, canvas.width - divider);
      ctx.fillStyle = cssColor('--line') || '#ffffff';
      ctx.fillRect(divider - 2, 0, 4, canvas.height);
      ctx.fillStyle = cssColor('--accent') || '#27b8bb';
      ctx.fillRect(divider - 12, canvas.height / 2 - 32, 24, 64);
    }
    function setViewMode(mode) {
      if (!layerAvailable(mode)) {
        updateViewControls();
        return;
      }
      state.viewMode = mode;
      updateViewControls();
      drawMain();
    }
    function setSideLayer(layer) {
      if (!sideLayerAvailable(layer)) {
        updateViewControls();
        return;
      }
      state.sideLayer = layer;
      updateViewControls();
      drawMain();
    }
    document.querySelectorAll('#viewModeButtons button').forEach(btn => btn.addEventListener('click', () => setViewMode(btn.dataset.mode)));
    document.querySelectorAll('#sideLayerButtons button').forEach(btn => btn.addEventListener('click', () => setSideLayer(btn.dataset.sideLayer)));
    ['showBackground','showOrdinary','showFine','showTalc','showTiling'].forEach(id => $(id).addEventListener('change', drawMain));
    function canvasPoint(event) {
      const rect = canvas.getBoundingClientRect();
      return {x: (event.clientX - rect.left) * devicePixelRatio, y: (event.clientY - rect.top) * devicePixelRatio};
    }

    canvas.addEventListener('wheel', (event) => {
      event.preventDefault();
      const delta = event.deltaY < 0 ? 1.14 : 0.88;
      state.zoom = Math.max(0.25, Math.min(16, state.zoom * delta));
      drawMain();
    }, {passive: false});
    canvas.addEventListener('pointerdown', (event) => {
      state.dragging = true;
      state.last = canvasPoint(event);
      state.dragSplitter = state.sideLayer !== 'none' && Math.abs(state.last.x - canvas.width * state.splitter) < 18 * devicePixelRatio;
      canvas.style.cursor = state.dragSplitter ? 'col-resize' : 'grabbing';
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointermove', (event) => {
      const point = canvasPoint(event);
      if (!state.dragging) {
        const nearSplitter = state.sideLayer !== 'none' && Math.abs(point.x - canvas.width * state.splitter) < 18 * devicePixelRatio;
        canvas.style.cursor = nearSplitter ? 'col-resize' : 'grab';
        return;
      }
      if (state.dragSplitter) {
        state.splitter = Math.max(0.12, Math.min(0.88, point.x / canvas.width));
      } else {
        state.pan.x += point.x - state.last.x;
        state.pan.y += point.y - state.last.y;
      }
      state.last = point;
      drawMain();
    });
    canvas.addEventListener('pointerup', () => { state.dragging = false; state.dragSplitter = false; canvas.style.cursor = 'grab'; });

    const dropZone = $('dropZone');
    dropZone.addEventListener('click', () => $('fileInput').click());
    dropZone.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') $('fileInput').click(); });
    dropZone.addEventListener('dragover', (event) => { event.preventDefault(); dropZone.classList.add('drag'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
    dropZone.addEventListener('drop', (event) => {
      event.preventDefault();
      dropZone.classList.remove('drag');
      if (event.dataTransfer.files.length) handleSelectedFile(event.dataTransfer.files[0]);
    });
    $('fileInput').addEventListener('change', (event) => {
      if (event.target.files.length) handleSelectedFile(event.target.files[0]);
    });
    $('clearUploadBtn').addEventListener('click', (event) => {
      event.stopPropagation();
      resetPageForClearedImage();
    });
    function smallestPreviewUrl(previews) {
      if (!previews || !previews.length) return '';
      return previews[0].url || '';
    }
    function fileExtension(file) {
      const name = String((file && file.name) || '');
      const dot = name.lastIndexOf('.');
      return dot >= 0 ? name.slice(dot).toLowerCase() : '';
    }
    function isSupportedUploadFile(file) {
      return SUPPORTED_UPLOAD_EXTENSIONS.has(fileExtension(file));
    }
    function setUploadWarning(key = null, params = {}) {
      uploadWarningMessage = key ? {key, params} : null;
      if (!$('uploadWarning')) return;
      if (!key) {
        $('uploadWarning').textContent = '';
        $('uploadWarning').classList.add('hidden');
        return;
      }
      $('uploadWarning').textContent = t(key, params);
      $('uploadWarning').classList.remove('hidden');
    }
    function stopUploadProgressTimer() {
      if (uploadProgressTimer) {
        clearInterval(uploadProgressTimer);
        uploadProgressTimer = null;
      }
    }
    function setUploadProgress(key = null, progress = 0, params = {}) {
      const wrap = $('uploadProgressWrap');
      if (!wrap) return;
      if (!key) {
        uploadProgressMessage = null;
        $('uploadProgressBar').style.width = '0%';
        $('uploadProgressText').textContent = '';
        wrap.classList.add('hidden');
        return;
      }
      const safeProgress = Math.max(0, Math.min(100, Math.round(progress || 0)));
      uploadProgressMessage = {key, progress: safeProgress, params};
      $('uploadProgressBar').style.width = `${safeProgress}%`;
      $('uploadProgressText').textContent = t(key, {...params, progress: safeProgress});
      wrap.classList.remove('hidden');
    }
    function clearUploadProgress() {
      stopUploadProgressTimer();
      setUploadProgress(null);
    }
    function startPreviewPreparationProgress(start = 70, max = 96) {
      stopUploadProgressTimer();
      let progress = Math.max(1, Math.min(max, Math.round(start)));
      setUploadProgress('uploadProgressPreparing', progress);
      setProgress(progress);
      setStatus('statusPreparingPreview', {progress});
      uploadProgressTimer = setInterval(() => {
        progress = Math.min(max, progress + Math.max(1, Math.ceil((max - progress) / 10)));
        setUploadProgress('uploadProgressPreparing', progress);
        setProgress(progress);
        setStatus('statusPreparingPreview', {progress});
        if (progress >= max) stopUploadProgressTimer();
      }, 650);
    }
    function uploadFileWithProgress(file) {
      return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        const body = new FormData();
        let preparingStarted = false;
        body.append('file', file);
        function beginPreparation() {
          if (!preparingStarted) {
            preparingStarted = true;
            const current = uploadProgressMessage ? uploadProgressMessage.progress : 70;
            startPreviewPreparationProgress(Math.max(70, current), 96);
          }
        }
        xhr.upload.addEventListener('progress', event => {
          if (!event.lengthComputable) {
            setUploadProgress('uploadProgressUploading', 8);
            setProgress(8);
            setStatus('statusUploading', {name: file.name});
            return;
          }
          const progress = Math.max(1, Math.min(70, Math.round((event.loaded / Math.max(event.total, 1)) * 70)));
          setUploadProgress('uploadProgressUploading', progress);
          setProgress(progress);
          setStatus('statusUploadingProgress', {name: file.name, progress});
        });
        xhr.upload.addEventListener('load', beginPreparation);
        xhr.addEventListener('load', () => {
          stopUploadProgressTimer();
          let payload = {};
          try {
            payload = xhr.responseText ? JSON.parse(xhr.responseText) : {};
          } catch (_) {
            reject(new Error('upload failed'));
            return;
          }
          if (xhr.status < 200 || xhr.status >= 300) {
            reject(new Error(payload.error || 'upload failed'));
            return;
          }
          resolve(payload);
        });
        xhr.addEventListener('error', () => reject(new Error('upload failed')));
        xhr.addEventListener('abort', () => reject(new Error('upload aborted')));
        xhr.open('POST', '/api/uploads');
        xhr.send(body);
      });
    }
    function handleSelectedFile(file) {
      if (!file) return;
      if (!isSupportedUploadFile(file)) {
        clearUploadProgress();
        setProgress(0);
        setUploadWarning('invalidImageFormat', {name: file.name || t('selectedImage')});
        $('fileInput').value = '';
        return;
      }
      uploadFile(file).catch(error => {
        const message = error && error.message ? error.message : t('unknownError');
        setUploadWarning('uploadFailed', {error: message});
        setStatus('statusFailed', {error: message});
      });
    }
    function renderUploadCard(upload) {
      if (!upload) {
        dropZone.classList.remove('selected');
        $('dropPrompt').classList.remove('hidden');
        $('selectedUpload').classList.add('hidden');
        $('selectedThumb').removeAttribute('src');
        $('selectedThumb').alt = '';
        $('selectedName').textContent = '';
        $('selectedMeta').textContent = '';
        $('uploadInfo').textContent = t('noImageLoaded');
        return;
      }
      dropZone.classList.add('selected');
      $('dropPrompt').classList.add('hidden');
      $('selectedUpload').classList.remove('hidden');
      $('selectedThumb').src = smallestPreviewUrl(upload.display && upload.display.original);
      $('selectedThumb').alt = upload.original_name || t('selectedImage');
      $('selectedName').textContent = upload.original_name || t('selectedImage');
      $('selectedMeta').textContent = `${upload.width} × ${upload.height}`;
      $('uploadInfo').textContent = `${upload.original_name} · ${upload.width} × ${upload.height}`;
    }
    function resetPageForClearedImage() {
      state.upload = null;
      state.run = null;
      state.viewMode = 'original';
      state.sideLayer = 'none';
      state.zoom = 1;
      state.pan = {x: 0, y: 0};
      state.splitter = 0.5;
      state.images.clear();
      activePollRunId = null;
      $('fileInput').value = '';
      setUploadWarning(null);
      applyPresetToControls(storedPreprocessPreset(), {save: false});
      $('showBackground').checked = true;
      $('showOrdinary').checked = true;
      $('showFine').checked = true;
      $('showTalc').checked = true;
      $('showTiling').checked = false;
      updateRunControls(null);
      $('fixBtn').disabled = true;
      $('resultPanel').classList.add('hidden');
      $('textOutput').textContent = '';
      $('metricsTable').innerHTML = '';
      $('csvLink').removeAttribute('href');
      $('pdfLink').removeAttribute('href');
      setProgress(0);
      setStatus('statusWaiting');
      clearUploadProgress();
      renderUploadCard(null);
      setViewMode('original');
      drawMain();
    }
    async function uploadFile(file) {
      setUploadWarning(null);
      clearUploadProgress();
      state.run = null;
      activePollRunId = null;
      updateRunControls(null);
      $('startBtn').disabled = true;
      setUploadProgress('uploadProgressUploading', 1);
      setStatus('statusUploadingProgress', {name: file.name, progress: 1});
      setProgress(1);
      let payload;
      try {
        payload = await uploadFileWithProgress(file);
      } catch (error) {
        clearUploadProgress();
        setProgress(0);
        throw error;
      }
      state.upload = payload;
      state.run = null;
      activePollRunId = null;
      state.zoom = 1; state.pan = {x: 0, y: 0};
      state.sideLayer = 'none';
      renderUploadCard(payload);
      updateRunControls(null);
      setUploadProgress('uploadProgressComplete', 100);
      setTimeout(() => {
        if (uploadProgressMessage && uploadProgressMessage.key === 'uploadProgressComplete') setUploadProgress(null);
      }, 900);
      setStatus('statusImageLoaded');
      setProgress(0);
      updateViewControls();
      drawMain();
    }
    $('applyPreprocessBtn').addEventListener('click', async () => {
      if (!state.upload) return;
      savePreprocessPreset();
      $('applyPreprocessBtn').disabled = true;
      setUploadWarning(null);
      startPreviewPreparationProgress(18, 96);
      try {
        const response = await fetch(`/api/uploads/${encodeURIComponent(state.upload.upload_id)}/preprocess`, {
          method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(presetPayload())
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'preprocess failed');
        stopUploadProgressTimer();
        setUploadProgress('uploadProgressComplete', 100);
        setTimeout(() => {
          if (uploadProgressMessage && uploadProgressMessage.key === 'uploadProgressComplete') setUploadProgress(null);
        }, 900);
        state.upload = payload;
        renderUploadCard(payload);
        setViewMode('preprocessed');
        setSideLayer('none');
        setStatus('statusPreprocessUpdated');
        setProgress(0);
        drawMain();
      } catch (error) {
        clearUploadProgress();
        setProgress(0);
        const message = error && error.message ? error.message : t('unknownError');
        setUploadWarning('uploadFailed', {error: message});
        setStatus('statusFailed', {error: message});
      } finally {
        $('applyPreprocessBtn').disabled = false;
      }
    });
    $('startBtn').addEventListener('click', async () => {
      if (!state.upload) return;
      savePreprocessPreset();
      $('startBtn').disabled = true;
      setProgress(1);
      setStatus('statusProgress', {stage: stageLabel('queued'), progress: 1, eta: ''});
      try {
        const response = await fetch('/api/runs/start', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({upload_id: state.upload.upload_id, ...presetPayload()})
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'start failed');
        state.run = payload;
        activePollRunId = payload.run_id;
        $('fixBtn').disabled = true;
        updateRunControls(payload);
        updateViewControls();
        pollRun(payload.run_id);
      } catch (error) {
        const message = error && error.message ? error.message : t('unknownError');
        activePollRunId = null;
        state.run = null;
        setProgress(0);
        setStatus('statusFailed', {error: message});
        updateRunControls(null);
      }
    });
    $('stopBtn').addEventListener('click', async () => {
      if (!state.run || !state.run.run_id) return;
      $('stopBtn').disabled = true;
      setStatus('statusCanceling');
      try {
        const response = await fetch(`/api/runs/${encodeURIComponent(state.run.run_id)}/cancel`, {method: 'POST'});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'cancel failed');
        state.run = payload;
        setProgress(payload.progress || 0);
        updateRunControls(payload);
        if (payload.status === 'canceled') {
          activePollRunId = null;
          setStatus('statusCanceled');
          await refreshHistory();
        }
      } catch (error) {
        const message = error && error.message ? error.message : t('unknownError');
        setStatus('statusCancelFailed', {error: message});
        updateRunControls(state.run);
      }
    });
    function stageLabel(stage) {
      const normalized = String(stage || 'running').toLowerCase();
      if (normalized.includes('sulfide')) return t('stageSulfide');
      if (normalized.includes('final')) return t('stageFinal');
      if (normalized.includes('report') || normalized.includes('metric')) return t('stageReport');
      if (normalized.includes('preprocess')) return t('stagePreprocessing');
      if (normalized === 'queued') return t('stageQueued');
      if (normalized === 'canceling') return t('stageCanceling');
      if (normalized === 'canceled') return t('stageCanceled');
      if (normalized === 'complete') return t('stageComplete');
      if (normalized === 'failed') return t('stageFailed');
      if (normalized === 'running') return t('stageRunning');
      return stage || t('stageRunning');
    }
    async function pollRun(runId) {
      if (activePollRunId !== runId) return;
      const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
      const payload = await response.json();
      if (activePollRunId !== runId) return;
      state.run = payload;
      setProgress(payload.progress || 0);
      const eta = payload.eta_seconds == null ? '' : t('statusEta', {seconds: payload.eta_seconds});
      setStatus('statusProgress', {stage: stageLabel(payload.stage || payload.status || 'running'), progress: payload.progress || 0, eta});
      updateRunControls(payload);
      if (payload.status === 'complete') {
        activePollRunId = null;
        renderResults(payload);
        await refreshHistory();
        setSideLayer('none');
        setViewMode('final');
        updateRunControls(payload);
        $('fixBtn').disabled = false;
        return;
      }
      if (payload.status === 'failed') {
        activePollRunId = null;
        updateRunControls(payload);
        setStatus('statusFailed', {error: payload.error || t('unknownError')});
        return;
      }
      if (payload.status === 'canceled') {
        activePollRunId = null;
        updateRunControls(payload);
        $('fixBtn').disabled = true;
        setStatus('statusCanceled');
        await refreshHistory();
        return;
      }
      if (payload.status === 'canceling') {
        setStatus('statusCanceling');
      }
      setTimeout(() => pollRun(runId), 900);
    }
    function renderResults(run) {
      $('resultPanel').classList.remove('hidden');
      $('textOutput').textContent = localizedRunText(run);
      $('csvLink').href = run.downloads.metrics_csv;
      $('pdfLink').href = run.downloads.pdf_report;
      const rows = run.metrics || [];
      $('metricsTable').innerHTML = `<thead><tr><th>${escapeHtml(t('metricsHeaderMetric'))}</th><th>${escapeHtml(t('metricsHeaderValue'))}</th></tr></thead><tbody>` + rows.map(row => {
        const value = row.percent == null ? row.value : `${Number(row.percent).toFixed(1)}%`;
        return `<tr><td>${escapeHtml(localizedMetricLabel(row))}</td><td>${escapeHtml(value)}</td></tr>`;
      }).join('') + '</tbody>';
    }
    async function refreshHistory() {
      const response = await fetch('/api/runs');
      const payload = await response.json();
      const runs = payload.runs || [];
      $('historyList').innerHTML = renderCompactHistory(runs);
      $('historyPageList').innerHTML = renderHistoryTable(runs);
      document.querySelectorAll('[data-load-run]').forEach(btn => btn.addEventListener('click', () => loadRun(btn.dataset.loadRun)));
      document.querySelectorAll('[data-delete-run]').forEach(btn => btn.addEventListener('click', () => removeRun(btn.dataset.deleteRun)));
      document.querySelectorAll('[data-preview-run]').forEach(btn => btn.addEventListener('click', () => openHistoryPreview(btn.dataset.previewUrl, btn.dataset.previewTitle)));
    }
    function renderCompactHistory(runs) {
      return runs.map(run => historyRow(run)).join('') || `<p class="muted">${escapeHtml(t('historyNoRuns'))}</p>`;
    }
    function renderHistoryTable(runs) {
      if (!runs.length) return `<p class="muted">${escapeHtml(t('historyNoRuns'))}</p>`;
      const rows = runs.map(run => {
        const summary = runSummary(run);
        const sulfide = Number(summary.sulfide_fraction || 0);
        return `<tr>
          <td class="thumbnail">${renderHistoryThumbnail(run)}</td>
          <td class="filename" title="${escapeHtml(runFilename(run))}">${escapeHtml(runFilename(run))}</td>
          <td>${escapeHtml(formatDate(run.created_at))}</td>
          <td>${escapeHtml(oreClassText(summary))}</td>
          <td class="numeric">${escapeHtml(formatFraction(sulfide))}</td>
          <td class="numeric">${escapeHtml(formatFraction(Math.max(0, 1 - sulfide)))}</td>
          <td class="numeric">${escapeHtml(formatFraction(summary.ordinary_sulfide_fraction))}</td>
          <td class="numeric">${escapeHtml(formatFraction(summary.fine_sulfide_fraction))}</td>
          <td class="numeric">${escapeHtml(formatFraction(summary.talc_fraction))}</td>
          <td><div class="history-actions"><button data-load-run="${escapeHtml(run.run_id)}">${escapeHtml(t('historyLoad'))}</button><button class="danger" data-delete-run="${escapeHtml(run.run_id)}">${escapeHtml(t('historyRemove'))}</button></div></td>
        </tr>`;
      }).join('');
      return `<div class="history-table-wrap"><table class="history-table">
        <thead><tr>
          <th class="thumbnail">${escapeHtml(t('historyThumbnail'))}</th>
          <th>${escapeHtml(t('historyFilename'))}</th>
          <th>${escapeHtml(t('historyDate'))}</th>
          <th>${escapeHtml(t('historyOreClassification'))}</th>
          <th class="numeric">${escapeHtml(t('historySulfides'))}</th>
          <th class="numeric">${escapeHtml(t('historyNonSulfides'))}</th>
          <th class="numeric">${escapeHtml(t('historyOrdinaryIntergrowth'))}</th>
          <th class="numeric">${escapeHtml(t('historyFineIntergrowth'))}</th>
          <th class="numeric">${escapeHtml(t('historyTalc'))}</th>
          <th>${escapeHtml(t('historyActions'))}</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table></div>`;
    }
    function renderHistoryThumbnail(run) {
      const thumbnail = (run && run.thumbnail) || {};
      const thumbUrl = thumbnail.thumbnail_url || thumbnail.preview_url || '';
      const previewUrl = thumbnail.preview_url || thumbUrl;
      const title = runFilename(run) || (run && run.run_id) || t('historyPreviewTitle');
      if (!thumbUrl || !previewUrl) return '<span class="history-thumb-placeholder">—</span>';
      return `<button class="history-thumb-button" type="button" data-preview-run="${escapeHtml(run.run_id || '')}" data-preview-url="${escapeHtml(previewUrl)}" data-preview-title="${escapeHtml(title)}" title="${escapeHtml(t('historyPreviewOpen', {name: title}))}" aria-label="${escapeHtml(t('historyPreviewOpen', {name: title}))}"><img src="${escapeHtml(thumbUrl)}" alt="${escapeHtml(title)}"></button>`;
    }
    function openHistoryPreview(url, title) {
      if (!url) return;
      $('historyPreviewImage').src = url;
      $('historyPreviewImage').alt = title || t('historyPreviewTitle');
      $('historyPreviewCaption').textContent = title || '';
      $('historyPreviewDialog').showModal();
    }
    function historyRow(run) {
      return `<div class="history-row"><strong>${escapeHtml(run.run_id)}</strong><div class="muted">${escapeHtml(run.created_at || '')}</div><div>${escapeHtml(localizedRunText(run) || run.status || '')}</div><button data-load-run="${escapeHtml(run.run_id)}">${escapeHtml(t('historyLoad'))}</button></div>`;
    }
    $('closeHistoryPreviewBtn').addEventListener('click', () => $('historyPreviewDialog').close());
    $('historyPreviewDialog').addEventListener('click', event => {
      if (event.target === $('historyPreviewDialog')) $('historyPreviewDialog').close();
    });
    async function removeRun(runId) {
      if (!runId || !window.confirm(t('confirmRemoveRun', {runId}))) return;
      const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`, {method: 'DELETE'});
      const payload = await response.json();
      if (!response.ok) {
        window.alert(t('statusFailed', {error: payload.error || t('unknownError')}));
        return;
      }
      if (state.run && state.run.run_id === runId) {
        state.run = null;
        activePollRunId = null;
        updateRunControls(null);
        $('fixBtn').disabled = true;
        $('resultPanel').classList.add('hidden');
        $('textOutput').textContent = '';
        $('metricsTable').innerHTML = '';
        setSideLayer('none');
        setViewMode('original');
        drawMain();
      }
      setStatus('statusRunRemoved', {runId});
      await refreshHistory();
    }
    async function loadRun(runId) {
      const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'load failed');
      let upload = null;
      const uploadId = payload.input && payload.input.upload_id;
      if (uploadId) {
        const uploadResponse = await fetch(`/api/uploads/${encodeURIComponent(uploadId)}`);
        upload = await uploadResponse.json();
        if (!uploadResponse.ok) throw new Error(upload.error || 'upload load failed');
      }
      state.run = payload;
      state.upload = upload;
      activePollRunId = runIsActive(payload) ? payload.run_id : null;
      state.zoom = 1;
      state.pan = {x: 0, y: 0};
      state.sideLayer = 'none';
      state.images.clear();
      if (upload) renderUploadCard(upload);
      applyPresetToControls((payload.preprocess && payload.preprocess.preset) || {});
      updateRunControls(payload);
      $('fixBtn').disabled = payload.status !== 'complete';
      renderResults(payload);
      setSideLayer('none');
      setViewMode('final');
      setProgress(payload.status === 'complete' ? 100 : (payload.progress || 0));
      setStatus(upload ? 'statusRunLoaded' : 'statusRunLoadedNoUpload', {runId: payload.run_id});
      showWorkspace(true);
      if (runIsActive(payload)) pollRun(payload.run_id);
    }
    const PAGE_SLUGS = {workspace: '/workspace', history: '/history'};
    function pageFromLocation() {
      return window.location.pathname === PAGE_SLUGS.history ? 'history' : 'workspace';
    }
    function setPage(page, options = {}) {
      const nextPage = page === 'history' ? 'history' : 'workspace';
      const slug = PAGE_SLUGS[nextPage];
      $('workspaceView').classList.toggle('hidden', nextPage !== 'workspace');
      $('historyView').classList.toggle('hidden', nextPage !== 'history');
      $('workspaceTab').classList.toggle('active', nextPage === 'workspace');
      $('historyTab').classList.toggle('active', nextPage === 'history');
      if (options.push && window.location.pathname !== slug) {
        window.history.pushState({page: nextPage}, '', slug);
      }
      if (nextPage === 'history') refreshHistory();
      resizeCanvas();
    }
    function showWorkspace(push = false) { setPage('workspace', {push}); }
    function showHistory(push = false) { setPage('history', {push}); }
    $('workspaceTab').addEventListener('click', () => showWorkspace(true));
    $('historyTab').addEventListener('click', () => showHistory(true));
    window.addEventListener('popstate', () => setPage(pageFromLocation(), {push: false}));
    setPage(pageFromLocation(), {push: false});
    refreshHistory();

    $('fixBtn').addEventListener('click', openFixDialog);
    $('closeFixBtn').addEventListener('click', () => $('fixDialog').close());
    document.querySelectorAll('#editLayerTabs button').forEach(btn => btn.addEventListener('click', () => switchEditorLayer(btn.dataset.layer)));
    $('brushToolBtn').addEventListener('click', () => setEditorTool('brush'));
    $('panToolBtn').addEventListener('click', () => setEditorTool('pan'));
    $('undoEditBtn').addEventListener('click', undoEditor);
    $('redoEditBtn').addEventListener('click', redoEditor);
    $('zoomOutEditBtn').addEventListener('click', () => zoomEditor(0.82));
    $('zoomInEditBtn').addEventListener('click', () => zoomEditor(1.22));
    $('fitEditBtn').addEventListener('click', fitEditorView);
    $('editClass').addEventListener('change', updateEditorStats);
    editorCanvas.addEventListener('contextmenu', event => event.preventDefault());
    async function refreshRunForEditor() {
      if (!state.run || !state.run.run_id) return null;
      const response = await fetch(`/api/runs/${encodeURIComponent(state.run.run_id)}`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'run load failed');
      state.run = payload;
      updateViewControls();
      return payload;
    }
    async function openFixDialog() {
      if (!state.run) return;
      $('fixDialog').showModal();
      resizeCanvas();
      setEditorStatus('editorLoading');
      try {
        await refreshRunForEditor();
        await switchEditorLayer(state.editor.layer || 'sulfide');
      } catch (error) {
        const message = error && error.message ? error.message : t('unknownError');
        state.editor.mask = null;
        updateEditorStats();
        drawEditor();
        $('fixRestartBtn').disabled = true;
        setEditorStatus('editorLoadFailed', {error: message});
      }
      resizeCanvas();
    }
    async function switchEditorLayer(layer) {
      if (!state.run || !['sulfide', 'final'].includes(layer)) return;
      state.editor.layer = layer;
      document.querySelectorAll('#editLayerTabs button').forEach(btn => btn.classList.toggle('active', btn.dataset.layer === layer));
      $('classSelector').style.display = layer === 'final' ? 'block' : 'none';
      if (layer === 'sulfide') $('editClass').value = '1';
      const url = state.editor.layer === 'final' ? state.run.masks.final : state.run.masks.sulfide;
      if (!url) throw new Error(t('editorMissingMask'));
      const image = await loadImage(url);
      if (!image) throw new Error(t('editorMissingMask'));
      const off = document.createElement('canvas');
      off.width = image.width; off.height = image.height;
      off.getContext('2d').drawImage(image, 0, 0);
      const data = off.getContext('2d').getImageData(0, 0, off.width, off.height).data;
      state.editor.width = off.width; state.editor.height = off.height;
      state.editor.mask = new Uint8Array(off.width * off.height);
      for (let i = 0, j = 0; i < data.length; i += 4, j++) {
        state.editor.mask[j] = state.editor.layer === 'final' ? Math.min(3, data[i]) : (data[i] > 0 ? 1 : 0);
      }
      state.editor.dirty = false;
      state.editor.undo = [];
      state.editor.redo = [];
      state.editor.strokeStarted = false;
      fitEditorView(false);
      updateUndoRedoButtons();
      updateEditorStats();
      $('fixRestartBtn').disabled = true;
      setEditorStatus('editNoEdits');
      await drawEditor();
    }
    function editorBasePreview() {
      const display = (state.run && state.run.display) || {};
      return bestPreview(display.preprocessed || display.original || []);
    }
    function editorImageRect() {
      const fitScale = Math.min(editorCanvas.width / state.editor.width, editorCanvas.height / state.editor.height);
      const scale = fitScale * state.editor.zoom;
      const w = state.editor.width * scale;
      const h = state.editor.height * scale;
      return {
        x: (editorCanvas.width - w) / 2 + state.editor.pan.x,
        y: (editorCanvas.height - h) / 2 + state.editor.pan.y,
        w,
        h,
        scale
      };
    }
    async function drawEditor() {
      editorCtx.clearRect(0, 0, editorCanvas.width, editorCanvas.height);
      editorCtx.fillStyle = cssColor('--viewer-bg') || '#1f232a'; editorCtx.fillRect(0, 0, editorCanvas.width, editorCanvas.height);
      if (!state.run || !state.editor.mask) return;
      const basePreview = editorBasePreview();
      let base = null;
      try {
        base = await loadImage(basePreview && basePreview.url);
      } catch (_) {
        base = null;
      }
      if (!base) {
        setEditorStatus('editorLoadFailed', {error: t('editorMissingBaseImage')});
        return;
      }
      const rect = editorImageRect();
      editorCtx.drawImage(base, rect.x, rect.y, rect.w, rect.h);
      const overlay = document.createElement('canvas');
      overlay.width = state.editor.width; overlay.height = state.editor.height;
      const octx = overlay.getContext('2d');
      const img = octx.createImageData(overlay.width, overlay.height);
      for (let i = 0, p = 0; i < state.editor.mask.length; i++, p += 4) {
        const value = state.editor.mask[i];
        if (!value) continue;
        let color = value === 1 ? [30,185,85,145] : value === 2 ? [230,65,65,155] : [40,120,245,160];
        if (state.editor.layer === 'sulfide') color = [245,190,35,150];
        img.data[p] = color[0]; img.data[p+1] = color[1]; img.data[p+2] = color[2]; img.data[p+3] = color[3];
      }
      octx.putImageData(img, 0, 0);
      editorCtx.drawImage(overlay, rect.x, rect.y, rect.w, rect.h);
    }
    function editorPoint(event) {
      const rect = editorCanvas.getBoundingClientRect();
      const point = {x: (event.clientX - rect.left) * devicePixelRatio, y: (event.clientY - rect.top) * devicePixelRatio};
      const image = editorImageRect();
      return {x: Math.floor((point.x - image.x) / image.scale), y: Math.floor((point.y - image.y) / image.scale)};
    }
    function setEditorTool(tool) {
      state.editor.tool = tool;
      $('brushToolBtn').classList.toggle('active', tool === 'brush');
      $('panToolBtn').classList.toggle('active', tool === 'pan');
      editorCanvas.style.cursor = tool === 'pan' ? 'grab' : 'crosshair';
    }
    function fitEditorView(redraw = true) {
      state.editor.zoom = 1;
      state.editor.pan = {x: 0, y: 0};
      if (redraw) drawEditor();
    }
    function zoomEditor(factor) {
      state.editor.zoom = Math.max(0.2, Math.min(24, state.editor.zoom * factor));
      drawEditor();
    }
    function cloneEditorMask() {
      return state.editor.mask ? new Uint8Array(state.editor.mask) : null;
    }
    function pushUndoSnapshot() {
      const snapshot = cloneEditorMask();
      if (!snapshot) return;
      state.editor.undo.push(snapshot);
      if (state.editor.undo.length > 50) state.editor.undo.shift();
      state.editor.redo = [];
      updateUndoRedoButtons();
    }
    function updateUndoRedoButtons() {
      $('undoEditBtn').disabled = state.editor.undo.length === 0;
      $('redoEditBtn').disabled = state.editor.redo.length === 0;
    }
    function setEditorStatus(key, params = {}) {
      state.editor.statusMessage = {key, params};
      $('editStatus').textContent = t(key, params);
    }
    function undoEditor() {
      if (!state.editor.mask || !state.editor.undo.length) return;
      state.editor.redo.push(cloneEditorMask());
      state.editor.mask = state.editor.undo.pop();
      markEditorDirty('editUndo');
      updateUndoRedoButtons();
      drawEditor();
      updateEditorStats();
    }
    function redoEditor() {
      if (!state.editor.mask || !state.editor.redo.length) return;
      state.editor.undo.push(cloneEditorMask());
      state.editor.mask = state.editor.redo.pop();
      markEditorDirty('editRedo');
      updateUndoRedoButtons();
      drawEditor();
      updateEditorStats();
    }
    function markEditorDirty(key = 'editUnsaved') {
      state.editor.dirty = true;
      $('fixRestartBtn').disabled = false;
      setEditorStatus(key);
    }
    function updateEditorStats() {
      if (!state.editor.mask) {
        $('editorStats').innerHTML = '';
        return;
      }
      const total = Math.max(1, state.editor.mask.length);
      let sulfide = 0;
      let ordinary = 0;
      let fine = 0;
      let talc = 0;
      for (const value of state.editor.mask) {
        if (state.editor.layer === 'sulfide') {
          if (value > 0) sulfide += 1;
        } else {
          if (value === 1) ordinary += 1;
          else if (value === 2) fine += 1;
          else if (value === 3) talc += 1;
        }
      }
      if (state.editor.layer === 'final') sulfide = ordinary + fine;
      const nonSulfide = total - sulfide;
      const rows = [
        {label: t('statSulfide'), px: sulfide},
        {label: t('statNonSulfide'), px: nonSulfide},
        {separator: true},
        {label: t('statOrdinary'), px: ordinary},
        {label: t('statFine'), px: fine},
        {label: t('statTalc'), px: talc},
      ];
      $('editorStats').innerHTML = '<tbody>' + rows.map(row => {
        if (row.separator) return '<tr class="stat-separator"><td colspan="3"></td></tr>';
        const pct = row.px / total * 100;
        return `<tr><td>${escapeHtml(row.label)}</td><td>${row.px.toLocaleString(localeCode())} px</td><td>${pct.toFixed(2)}%</td></tr>`;
      }).join('') + '</tbody>';
    }
    function paintEditor(event) {
      if (!state.editor.mask) return;
      const point = editorPoint(event);
      const radius = Math.max(1, Number($('brushSize').value || 18));
      const isErase = event.button === 2 || event.buttons === 2;
      const value = isErase ? 0 : (state.editor.layer === 'final' ? Number($('editClass').value || 0) : 1);
      for (let y = point.y - radius; y <= point.y + radius; y++) {
        if (y < 0 || y >= state.editor.height) continue;
        for (let x = point.x - radius; x <= point.x + radius; x++) {
          if (x < 0 || x >= state.editor.width) continue;
          const dx = x - point.x, dy = y - point.y;
          if (dx*dx + dy*dy <= radius*radius) state.editor.mask[y * state.editor.width + x] = value;
        }
      }
      markEditorDirty(isErase ? 'editEraseStroke' : 'editDrawStroke');
      drawEditor();
      updateEditorStats();
    }
    editorCanvas.addEventListener('wheel', (event) => {
      event.preventDefault();
      zoomEditor(event.deltaY < 0 ? 1.16 : 0.86);
    }, {passive: false});
    editorCanvas.addEventListener('pointerdown', (event) => {
      editorCanvas.setPointerCapture(event.pointerId);
      state.editor.last = {x: event.clientX * devicePixelRatio, y: event.clientY * devicePixelRatio};
      if (state.editor.tool === 'pan' || event.button === 1) {
        state.editor.panning = true;
        editorCanvas.style.cursor = 'grabbing';
        return;
      }
      state.editor.drawing = true;
      state.editor.strokeStarted = true;
      pushUndoSnapshot();
      paintEditor(event);
    });
    editorCanvas.addEventListener('pointermove', (event) => {
      const point = {x: event.clientX * devicePixelRatio, y: event.clientY * devicePixelRatio};
      if (state.editor.panning) {
        state.editor.pan.x += point.x - state.editor.last.x;
        state.editor.pan.y += point.y - state.editor.last.y;
        state.editor.last = point;
        drawEditor();
        return;
      }
      if (state.editor.drawing) paintEditor(event);
    });
    editorCanvas.addEventListener('pointerup', () => {
      state.editor.drawing = false;
      state.editor.panning = false;
      state.editor.strokeStarted = false;
      editorCanvas.style.cursor = state.editor.tool === 'pan' ? 'grab' : 'crosshair';
    });
    editorCanvas.addEventListener('pointercancel', () => {
      state.editor.drawing = false;
      state.editor.panning = false;
      state.editor.strokeStarted = false;
    });
    function editorMaskDataUrl() {
      const out = document.createElement('canvas');
      out.width = state.editor.width; out.height = state.editor.height;
      const octx = out.getContext('2d');
      const img = octx.createImageData(out.width, out.height);
      for (let i = 0, p = 0; i < state.editor.mask.length; i++, p += 4) {
        const value = state.editor.layer === 'final' ? state.editor.mask[i] : (state.editor.mask[i] ? 255 : 0);
        img.data[p] = value; img.data[p+1] = value; img.data[p+2] = value; img.data[p+3] = 255;
      }
      octx.putImageData(img, 0, 0);
      return out.toDataURL('image/png');
    }
    $('fixRestartBtn').addEventListener('click', async () => {
      if (!state.run || !state.editor.dirty) return;
      const response = await fetch(`/api/runs/${encodeURIComponent(state.run.run_id)}/fix`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({edit_layer: state.editor.layer, mask_png: editorMaskDataUrl(), comment: $('editComment').value})
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'fix failed');
      $('fixDialog').close();
      state.run = payload;
      renderResults(payload);
      await refreshHistory();
      setSideLayer('none');
      setViewMode('final');
    });
    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    }
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
