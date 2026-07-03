#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import mimetypes
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import zipfile
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default as email_default_policy
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageDraw, ImageEnhance, ImageFile, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
HEURISTIC_SRC = ROOT / "heuristic_segmentation/src"
for source_root in (SRC, HEURISTIC_SRC):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from heuristic_segmentation.segmentation import segment_image  # noqa: E402
from ore_classifier.analyzed_area import build_analyzed_mask  # noqa: E402
from ore_classifier.augmentation import (  # noqa: E402
    apply_augmentation,
    augmentation_enabled,
    default_augmentation_settings,
    normalize_augmentation_settings,
)
from ore_classifier.component_analysis import (  # noqa: E402
    ComponentRuleConfig,
    OreSummary,
    analyze_components,
    summary_warnings,
    write_component_csv,
)
from ore_classifier.preprocessing import (  # noqa: E402
    apply_preprocessing,
    normalize_preprocess_settings,
)
from ore_classifier.rule_config_io import default_rule_config  # noqa: E402
from ore_classifier.tiling import iter_tiles  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_WORKSPACE_DIR = ROOT / "outputs/ore_pipeline_ui"
DEFAULT_CHECKPOINT = ROOT / "models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
MAX_JSON_BYTES = 220 * 1024 * 1024
LOG_ENTRY_LIMIT = 300
STATUS_LOG_LIMIT = 80
RUNTIME_TEST_TIMEOUT_SECONDS = 90
DISPLAY_TILE_SIZE = 1024
DISPLAY_TILE_STRIDE = 768
FULL_SIZE_PREPROCESS_MAX_PIXELS = 24_000_000
RAW_EXTENSIONS = {".raw", ".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".raf", ".pef", ".srw"}
IMAGE_EXTENSIONS = RAW_EXTENSIONS | {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
CLASS_COLORS = {
    1: (30, 185, 85, 150),
    2: (230, 65, 65, 160),
    3: (40, 120, 245, 165),
}
ARTIFACT_COLOR = (198, 60, 255, 180)
CLASS_LABELS_RU = {
    "analyzed_fraction": "Доля проанализированной области",
    "sulfide_fraction": "Общая доля сульфидов",
    "ordinary_sulfide_fraction": "Доля обычных срастаний",
    "fine_sulfide_fraction": "Доля тонких срастаний",
    "talc_fraction": "Доля талька",
    "other_fraction": "Остальное",
    "artifact_fraction_image": "Доля артефактов изображения",
    "component_count": "Компоненты сульфидов",
}
DEFAULT_RULE_CONFIG = default_rule_config()
ORE_CLASS_SHORT_RU = {
    "talcose_ore": "оталькованная",
    "row_ore": "рядовая",
    "hard_to_process_ore": "труднообогатимая",
}
REPORT_PAGE_SIZE = (1240, 1754)
REPORT_MARGIN_X = 80
REPORT_MARGIN_TOP = 76
REPORT_TEXT = (20, 26, 36)
REPORT_MUTED = (72, 83, 98)
REPORT_LINE = (204, 213, 224)
REPORT_TABLE_HEADER = (232, 238, 247)
REPORT_TABLE_ALT = (248, 250, 252)
REPORT_NON_SULFIDE_COLOR = (46, 74, 96)
REPORT_SULFIDE_COLOR = (239, 186, 43)
REPORT_MASK_BACKGROUND = (245, 247, 250)
REPORT_CLASS_SPECS = [
    (1, "Обычные срастания", "masks/ordinary_mask.png", (30, 185, 85)),
    (2, "Тонкие срастания", "masks/fine_mask.png", (230, 65, 65)),
    (3, "Тальк", "masks/talc_final_mask.png", (40, 120, 245)),
]
CURATED_METADATA_SCHEMA_VERSION = "ore-pipeline-curated-metadata-v0.1"
APP_SETTINGS_SCHEMA_VERSION = "ore-pipeline-app-settings-v0.1"
BATCH_SCHEMA_VERSION = "ore-pipeline-batch-v0.1"
BATCH_ITEM_SCHEMA_VERSION = "ore-pipeline-batch-item-v0.1"
RUNTIME_PROVENANCE_SCHEMA_VERSION = "ore-pipeline-runtime-provenance-v0.1"
ACTIVE_RUN_STATUSES = {"queued", "running", "canceling"}
RUN_TERMINAL_STATUSES = {"complete", "failed", "canceled"}
BATCH_ACTIVE_STATUSES = {"queued", "running", "canceling"}
BATCH_TERMINAL_STATUSES = {"complete", "failed", "partial", "canceled"}
PANORAMA_SCALING_MODE_MAX_SIDE = "max_side"
PANORAMA_SCALING_MODE_SCALE_FACTOR = "scale_factor"
PANORAMA_SCALING_MODES = {PANORAMA_SCALING_MODE_MAX_SIDE, PANORAMA_SCALING_MODE_SCALE_FACTOR}
DEFAULT_PANORAMA_MAX_SIDE_PX = 1800
DEFAULT_PANORAMA_SCALE_FACTOR = 0.5
MIN_PANORAMA_MAX_SIDE_PX = 64
MAX_PANORAMA_MAX_SIDE_PX = 12000
MIN_PANORAMA_SCALE_FACTOR = 0.05
MAX_PANORAMA_SCALE_FACTOR = 1.0
DEFAULT_APP_SETTINGS = {
    "schema_version": APP_SETTINGS_SCHEMA_VERSION,
    "language": "ru",
    "theme": "system",
    "show_tiling": False,
    "runtime": {
        "backend": "heuristic",
        "checkpoint": str(DEFAULT_CHECKPOINT.resolve()) if DEFAULT_CHECKPOINT.exists() else "",
    },
    "preprocess": {
        "preprocessing_enabled": True,
        "illumination_normalization": True,
        "denoise": True,
        "contrast_correction": True,
        "panorama_scaling": True,
        "panorama_scaling_mode": PANORAMA_SCALING_MODE_MAX_SIDE,
        "panorama_max_side_px": DEFAULT_PANORAMA_MAX_SIDE_PX,
        "panorama_scale_factor": DEFAULT_PANORAMA_SCALE_FACTOR,
    },
    "metadata_defaults": {},
}
SETTINGS_METADATA_DEFAULT_FIELDS = {
    "project",
    "om_instrument",
    "om_objective_magnification",
    "scale_source",
    "pixel_size_um",
    "scale_confidence",
    "review_status",
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


def load_image_pil(path: Path, max_side: int | None = None) -> Image.Image:
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
            if max_side and max(image.size) > max_side:
                try:
                    image.draft("RGB", (int(max_side), int(max_side)))
                except Exception:
                    pass
            image = ImageOps.exif_transpose(image)
            if max_side and max(image.size) > max_side:
                image.thumbnail((int(max_side), int(max_side)), Image.Resampling.BILINEAR)
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
    decode_max_side = max_side
    if decode_max_side is None and size is not None:
        decode_max_side = max(size)
    image = load_image_pil(path, max_side=decode_max_side)
    return scaled_image_copy(image, max_side=max_side, size=size)


def scaled_image_copy(image: Image.Image, max_side: int | None = None, size: tuple[int, int] | None = None) -> Image.Image:
    image = image.convert("RGB")
    if size is not None:
        if image.size != size:
            image = image.resize(size, Image.Resampling.BILINEAR)
        return image.convert("RGB")
    if max_side and max(image.size) > max_side:
        image = image.copy()
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


def read_binary_mask(path: Path, expected_shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"))
    if expected_shape_hw and mask.shape[:2] != expected_shape_hw:
        mask_image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
        mask_image = mask_image.resize((expected_shape_hw[1], expected_shape_hw[0]), Image.Resampling.NEAREST)
        mask = np.asarray(mask_image)
    return (mask > 0).astype(np.uint8) * 255


def apply_artifact_exclusion(
    *,
    artifact_mask: np.ndarray | None,
    sulfide_mask: np.ndarray,
    talc_mask: np.ndarray,
    analyzed_mask: np.ndarray,
    final_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    sulfide = (sulfide_mask > 0).astype(np.uint8) * 255
    talc = (talc_mask > 0).astype(np.uint8) * 255
    analyzed = (analyzed_mask > 0).astype(np.uint8) * 255
    final = None if final_mask is None else final_mask.astype(np.uint8).copy()
    if artifact_mask is None:
        return sulfide, talc, analyzed, final
    artifact = artifact_mask > 0
    if artifact.shape != sulfide.shape:
        artifact = read_binary_mask_from_array(artifact.astype(np.uint8) * 255, sulfide.shape) > 0
    sulfide[artifact] = 0
    talc[artifact] = 0
    analyzed[artifact] = 0
    if final is not None:
        final[artifact] = 0
    return sulfide, talc, analyzed, final


def read_binary_mask_from_array(mask: np.ndarray, expected_shape_hw: tuple[int, int]) -> np.ndarray:
    mask_image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    if mask_image.size != (expected_shape_hw[1], expected_shape_hw[0]):
        mask_image = mask_image.resize((expected_shape_hw[1], expected_shape_hw[0]), Image.Resampling.NEAREST)
    return (np.asarray(mask_image) > 0).astype(np.uint8) * 255


def payload_value(payload: dict[str, Any], key: str, aliases: tuple[str, ...] = ()) -> Any:
    for candidate in (key, *aliases):
        if candidate in payload:
            return payload[candidate]
    return None


def normalized_int(value: Any, fallback: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        parsed = int(fallback)
    return max(int(minimum), min(int(maximum), parsed))


def normalized_float(value: Any, fallback: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(fallback)
    if not math.isfinite(parsed):
        parsed = float(fallback)
    return max(float(minimum), min(float(maximum), parsed))


def normalized_panorama_scaling_mode(value: Any, fallback: str = PANORAMA_SCALING_MODE_MAX_SIDE) -> str:
    mode = str(value or fallback)
    return mode if mode in PANORAMA_SCALING_MODES else fallback


def preset_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    preset = {
        "preprocessing_enabled": bool(payload.get("preprocessing_enabled", payload.get("enabled", True))),
        "illumination_normalization": bool(payload.get("illumination_normalization") or payload.get("illumination")),
        "denoise": bool(payload.get("denoise") or payload.get("noise_reduction")),
        "contrast_correction": bool(payload.get("contrast_correction") or payload.get("contrast")),
        "panorama_scaling": bool(payload.get("panorama_scaling") or payload.get("panoramaScaling")),
        "panorama_scaling_mode": normalized_panorama_scaling_mode(
            payload_value(payload, "panorama_scaling_mode", ("panoramaScalingMode",))
        ),
    }
    max_side = payload_value(payload, "panorama_max_side_px", ("panoramaMaxSidePx", "panorama_max_side", "panoramaMaxSide"))
    if max_side is not None:
        preset["panorama_max_side_px"] = normalized_int(
            max_side,
            DEFAULT_PANORAMA_MAX_SIDE_PX,
            MIN_PANORAMA_MAX_SIDE_PX,
            MAX_PANORAMA_MAX_SIDE_PX,
        )
    scale_factor = payload_value(payload, "panorama_scale_factor", ("panoramaScaleFactor", "panorama_scaling_factor"))
    if scale_factor is not None:
        preset["panorama_scale_factor"] = normalized_float(
            scale_factor,
            DEFAULT_PANORAMA_SCALE_FACTOR,
            MIN_PANORAMA_SCALE_FACTOR,
            MAX_PANORAMA_SCALE_FACTOR,
        )
    return preset


def augmentation_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "augmentation" in payload:
        return normalize_augmentation_settings(payload.get("augmentation"))
    return normalize_augmentation_settings(
        {
            "enabled": bool(payload.get("augmentation_enabled", False)),
            "color": payload.get("augmentation_color") or {},
            "acquisition": payload.get("augmentation_acquisition") or {},
            "runtime": payload.get("augmentation_runtime") or {},
        }
    )


def json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, tuple):
        return [json_safe_value(item) for item in value]
    if isinstance(value, list):
        return [json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    return str(value)


def compact_text(value: Any, max_chars: int = 4000) -> str:
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value or "")
    if len(text) <= max_chars:
        return text
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars - 20
    return f"{text[:head_chars]}\n...[truncated]...\n{text[-tail_chars:]}"


def default_app_settings() -> dict[str, Any]:
    return json.loads(json.dumps(DEFAULT_APP_SETTINGS))


def settings_bool(payload: dict[str, Any], key: str, fallback: bool, aliases: tuple[str, ...] = ()) -> bool:
    for candidate in (key, *aliases):
        if candidate in payload:
            return bool(payload[candidate])
    return bool(fallback)


def settings_value(payload: dict[str, Any], key: str, fallback: Any, aliases: tuple[str, ...] = ()) -> Any:
    for candidate in (key, *aliases):
        if candidate in payload:
            return payload[candidate]
    return fallback


def normalize_settings_preprocess(payload: Any, base: dict[str, Any] | None = None) -> dict[str, Any]:
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ApiError(HTTPStatus.BAD_REQUEST, "settings.preprocess must be an object")
    fallback = base if isinstance(base, dict) else DEFAULT_APP_SETTINGS["preprocess"]
    return {
        "preprocessing_enabled": settings_bool(payload, "preprocessing_enabled", bool(fallback["preprocessing_enabled"]), ("enabled",)),
        "illumination_normalization": settings_bool(payload, "illumination_normalization", bool(fallback["illumination_normalization"]), ("illumination",)),
        "denoise": settings_bool(payload, "denoise", bool(fallback["denoise"]), ("noise_reduction",)),
        "contrast_correction": settings_bool(payload, "contrast_correction", bool(fallback["contrast_correction"]), ("contrast",)),
        "panorama_scaling": settings_bool(payload, "panorama_scaling", bool(fallback["panorama_scaling"]), ("panoramaScaling",)),
        "panorama_scaling_mode": normalized_panorama_scaling_mode(
            settings_value(
                payload,
                "panorama_scaling_mode",
                fallback.get("panorama_scaling_mode", PANORAMA_SCALING_MODE_MAX_SIDE),
                ("panoramaScalingMode",),
            ),
            PANORAMA_SCALING_MODE_MAX_SIDE,
        ),
        "panorama_max_side_px": normalized_int(
            settings_value(
                payload,
                "panorama_max_side_px",
                fallback.get("panorama_max_side_px", DEFAULT_PANORAMA_MAX_SIDE_PX),
                ("panoramaMaxSidePx", "panorama_max_side", "panoramaMaxSide"),
            ),
            DEFAULT_PANORAMA_MAX_SIDE_PX,
            MIN_PANORAMA_MAX_SIDE_PX,
            MAX_PANORAMA_MAX_SIDE_PX,
        ),
        "panorama_scale_factor": normalized_float(
            settings_value(
                payload,
                "panorama_scale_factor",
                fallback.get("panorama_scale_factor", DEFAULT_PANORAMA_SCALE_FACTOR),
                ("panoramaScaleFactor", "panorama_scaling_factor"),
            ),
            DEFAULT_PANORAMA_SCALE_FACTOR,
            MIN_PANORAMA_SCALE_FACTOR,
            MAX_PANORAMA_SCALE_FACTOR,
        ),
    }


def normalize_settings_runtime(payload: Any, base: dict[str, Any] | None = None, *, validate_checkpoint: bool = False) -> dict[str, Any]:
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ApiError(HTTPStatus.BAD_REQUEST, "settings.runtime must be an object")
    fallback = base if isinstance(base, dict) else DEFAULT_APP_SETTINGS["runtime"]
    backend = str(payload.get("backend", fallback.get("backend", "heuristic")) or "heuristic").lower()
    if backend not in {"heuristic", "ml"}:
        raise ApiError(HTTPStatus.BAD_REQUEST, "settings.runtime.backend must be heuristic or ml")
    checkpoint_value = str(payload.get("checkpoint", fallback.get("checkpoint", "")) or "").strip()
    if checkpoint_value:
        checkpoint_path = Path(checkpoint_value).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path
        checkpoint_value = str(checkpoint_path.resolve())
    if backend == "ml":
        if not checkpoint_value:
            raise ApiError(HTTPStatus.BAD_REQUEST, "settings.runtime.checkpoint is required for ml backend")
        if validate_checkpoint and not Path(checkpoint_value).exists():
            raise ApiError(HTTPStatus.BAD_REQUEST, f"settings.runtime.checkpoint does not exist: {checkpoint_value}")
    return {"backend": backend, "checkpoint": checkpoint_value}


def normalize_app_settings_payload(
    payload: Any,
    base: dict[str, Any] | None = None,
    *,
    validate_runtime: bool = False,
) -> dict[str, Any]:
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ApiError(HTTPStatus.BAD_REQUEST, "settings must be an object")
    fallback = default_app_settings()
    if isinstance(base, dict):
        fallback.update({key: base[key] for key in ("language", "theme", "show_tiling") if key in base})
        if isinstance(base.get("runtime"), dict):
            fallback["runtime"] = normalize_settings_runtime(base["runtime"])
        if isinstance(base.get("preprocess"), dict):
            fallback["preprocess"] = normalize_settings_preprocess(base["preprocess"])
        if isinstance(base.get("metadata_defaults"), dict):
            fallback["metadata_defaults"] = {
                str(key): json_safe_value(value)
                for key, value in base["metadata_defaults"].items()
                if str(key) in SETTINGS_METADATA_DEFAULT_FIELDS and value not in (None, "")
            }
    language = str(payload.get("language", fallback["language"]) or fallback["language"])
    if language not in {"ru", "en"}:
        raise ApiError(HTTPStatus.BAD_REQUEST, "settings.language must be ru or en")
    theme = str(payload.get("theme", fallback["theme"]) or fallback["theme"])
    if theme not in {"system", "light", "dark"}:
        raise ApiError(HTTPStatus.BAD_REQUEST, "settings.theme must be system, light, or dark")
    metadata_defaults = payload.get("metadata_defaults", fallback.get("metadata_defaults", {}))
    if metadata_defaults is None:
        metadata_defaults = {}
    if not isinstance(metadata_defaults, dict):
        raise ApiError(HTTPStatus.BAD_REQUEST, "settings.metadata_defaults must be an object")
    return {
        "schema_version": APP_SETTINGS_SCHEMA_VERSION,
        "language": language,
        "theme": theme,
        "show_tiling": bool(payload.get("show_tiling", fallback["show_tiling"])),
        "runtime": normalize_settings_runtime(
            payload.get("runtime", fallback["runtime"]),
            fallback["runtime"],
            validate_checkpoint=validate_runtime,
        ),
        "preprocess": normalize_settings_preprocess(payload.get("preprocess", fallback["preprocess"]), fallback["preprocess"]),
        "metadata_defaults": {
            str(key): json_safe_value(value)
            for key, value in metadata_defaults.items()
            if str(key) in SETTINGS_METADATA_DEFAULT_FIELDS and value not in (None, "")
        },
    }


def panorama_scaling_target(
    preset: dict[str, Any],
    *,
    preprocessing_enabled: bool,
    source_width: int,
    source_height: int,
    processing_max_side: int,
    default_panorama_max_side: int,
) -> tuple[int, dict[str, Any]]:
    source_longest_side = max(int(source_width), int(source_height))
    configured_mode = normalized_panorama_scaling_mode(
        preset.get("panorama_scaling_mode", preset.get("panoramaScalingMode")),
        PANORAMA_SCALING_MODE_MAX_SIDE,
    )
    max_side_value = preset.get("panorama_max_side_px", preset.get("panoramaMaxSidePx"))
    max_side_px = normalized_int(
        max_side_value if max_side_value is not None else default_panorama_max_side,
        int(default_panorama_max_side),
        MIN_PANORAMA_MAX_SIDE_PX,
        MAX_PANORAMA_MAX_SIDE_PX,
    )
    scale_factor = normalized_float(
        preset.get("panorama_scale_factor", preset.get("panoramaScaleFactor", DEFAULT_PANORAMA_SCALE_FACTOR)),
        DEFAULT_PANORAMA_SCALE_FACTOR,
        MIN_PANORAMA_SCALE_FACTOR,
        MAX_PANORAMA_SCALE_FACTOR,
    )
    enabled = bool(preprocessing_enabled and preset.get("panorama_scaling"))
    if not enabled:
        target_max_side = int(processing_max_side)
        mode = "off"
    elif configured_mode == PANORAMA_SCALING_MODE_SCALE_FACTOR:
        target_max_side = max(1, int(round(source_longest_side * scale_factor)))
        mode = PANORAMA_SCALING_MODE_SCALE_FACTOR
    else:
        target_max_side = max_side_px
        mode = PANORAMA_SCALING_MODE_MAX_SIDE
    return target_max_side, {
        "enabled": enabled,
        "mode": mode,
        "configured_mode": configured_mode,
        "target_max_side": int(target_max_side),
        "source_longest_side": int(source_longest_side),
        "max_side_px": int(max_side_px),
        "scale_factor": float(scale_factor),
    }


def extract_image_raw_metadata(
    path: Path,
    *,
    original_name: str,
    width: int,
    height: int,
    sha1: str | None = None,
) -> dict[str, Any]:
    stat = path.stat()
    metadata: dict[str, Any] = {
        "schema_version": "ore-pipeline-raw-image-metadata-v0.1",
        "original_name": original_name,
        "stored_path": str(path),
        "extension": path.suffix.lower(),
        "file_size_bytes": int(stat.st_size),
        "sha1": sha1 or file_sha1(path),
        "width": int(width),
        "height": int(height),
        "warnings": [],
    }
    if path.suffix.lower() in RAW_EXTENSIONS:
        metadata["warnings"].append("raw_header_metadata_limited_without_camera_decoder")
        return metadata
    try:
        with Image.open(path) as image:
            metadata.update(
                {
                    "image_format": image.format,
                    "mode": image.mode,
                    "dpi": json_safe_value(image.info.get("dpi")),
                    "jfif_unit": json_safe_value(image.info.get("jfif_unit")),
                    "jfif_density": json_safe_value(image.info.get("jfif_density")),
                    "icc_profile_present": bool(image.info.get("icc_profile")),
                    "xmp_present": any("xmp" in str(key).lower() for key in image.info),
                }
            )
            exif = image.getexif()
            metadata["exif_present"] = bool(exif)
            metadata["exif"] = {}
            if exif:
                for tag_id, value in list(exif.items())[:80]:
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    metadata["exif"][str(tag_name)] = json_safe_value(value)
            else:
                metadata["warnings"].append("exif_unavailable")
    except Exception as exc:  # noqa: BLE001 - raw metadata must not block upload.
        metadata["warnings"].append(f"raw_metadata_read_failed: {exc}")
    return metadata


def normalize_curated_metadata_payload(payload: Any) -> dict[str, Any] | None:
    if payload is None or payload == "":
        return None
    if not isinstance(payload, dict):
        raise ApiError(HTTPStatus.BAD_REQUEST, "curated_metadata must be an object")
    known = {"schema_version", "source", "generated_at", "domain", "raw_summary", "session_defaults_applied", "warnings", "extra"}
    domain = payload.get("domain") if isinstance(payload.get("domain"), dict) else {}
    raw_summary = payload.get("raw_summary") if isinstance(payload.get("raw_summary"), dict) else {}
    session_defaults = (
        payload.get("session_defaults_applied") if isinstance(payload.get("session_defaults_applied"), dict) else {}
    )
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    extra = {
        **{str(key): json_safe_value(value) for key, value in extra.items()},
        **{str(key): json_safe_value(value) for key, value in payload.items() if key not in known},
    }
    normalized: dict[str, Any] = {
        "schema_version": str(payload.get("schema_version") or CURATED_METADATA_SCHEMA_VERSION),
        "source": str(payload.get("source") or "metadata_editor"),
        "generated_at": str(payload.get("generated_at") or utc_now_iso()),
        "domain": json_safe_value(domain),
        "raw_summary": json_safe_value(raw_summary),
        "session_defaults_applied": json_safe_value(session_defaults),
        "warnings": json_safe_value(warnings),
    }
    if extra:
        normalized["extra"] = extra
    has_content = any(
        bool(normalized.get(key))
        for key in ("domain", "raw_summary", "session_defaults_applied", "warnings")
    ) or bool(extra)
    return normalized if has_content else None


def apply_preprocessing(image: Image.Image, preset: dict[str, Any]) -> Image.Image:
    # Keep a single numpy RGB buffer across steps. The previous version round-tripped
    # PIL<->numpy once per enabled step, allocating several full-size copies of large
    # images (costly for panorama-scale inputs). Output is pixel-identical; only the
    # redundant intermediate PIL images and array copies are removed.
    arr = np.asarray(image.convert("RGB"))
    if preset.get("illumination_normalization"):
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        value = hsv[..., 2]
        sigma = max(9.0, min(value.shape) / 32.0)
        background = cv2.GaussianBlur(value, (0, 0), sigmaX=sigma)
        corrected = value.astype(np.float32) - background.astype(np.float32) + float(np.median(background))
        hsv[..., 2] = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        arr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if preset.get("denoise"):
        arr = cv2.fastNlMeansDenoisingColored(arr, None, 4, 4, 7, 21)
    if preset.get("contrast_correction"):
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        corrected_l = clahe.apply(l_channel)
        corrected = cv2.merge((corrected_l, a_channel, b_channel))
        arr = cv2.cvtColor(corrected, cv2.COLOR_LAB2RGB)
    result = Image.fromarray(arr, mode="RGB")
    if preset.get("contrast_correction"):
        result = ImageEnhance.Contrast(result).enhance(1.05)
    return result


def save_image(path: Path, image: Image.Image, *, optimize: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    png_kwargs: dict[str, Any] = {"optimize": optimize}
    if not optimize:
        png_kwargs["compress_level"] = 1
    if image.mode == "RGBA":
        image.save(path, format="PNG", **png_kwargs)
    elif image.mode == "L":
        image.save(path, format="PNG", **png_kwargs)
    else:
        image.convert("RGB").save(path, format="PNG", **png_kwargs)


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
    generated: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    targets = sorted({int(max_side) for max_side in max_sides if int(max_side) > 0}, reverse=True)
    if not targets:
        targets = [max(image.size)]
    working = image.copy()
    for max_side in targets:
        if max(working.size) > max_side:
            resized = working.copy()
            resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
            resized.thumbnail((max_side, max_side), resample)
            working = resized
        preview = working.copy()
        if preview.size in seen:
            continue
        seen.add(preview.size)
        ext = ".png" if prefer_png or preview.mode in {"RGBA", "L"} else ".jpg"
        path = out_dir / f"{stem}_{max(preview.size)}{ext}"
        if ext == ".png":
            preview.save(path, format="PNG", optimize=False, compress_level=1)
        else:
            preview.convert("RGB").save(path, format="JPEG", quality=88, optimize=False)
        generated.append(
            {
                "max_side": max(preview.size),
                "width": preview.size[0],
                "height": preview.size[1],
                "path": str(path),
            }
        )
    return list(reversed(generated))


def should_defer_full_size_processing(
    *,
    source_width: int,
    source_height: int,
    target_max_side: int,
) -> bool:
    source_pixels = int(source_width) * int(source_height)
    return (
        max(int(source_width), int(source_height)) > int(target_max_side)
        and source_pixels > FULL_SIZE_PREPROCESS_MAX_PIXELS
    )


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


def masked_rgb_layer(image: Image.Image, active_mask: np.ndarray) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = active_mask.astype(bool)
    if mask.shape != rgb.shape[:2]:
        resized = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST)
        mask = np.asarray(resized) > 0
    rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = mask.astype(np.uint8) * 255
    return Image.fromarray(rgba, mode="RGBA")


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


def parse_positive_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(str(value).strip().replace(",", "."))
    except ValueError:
        return None
    if parsed <= 0:
        return None
    return parsed


def calibrated_scale_from_metadata(metadata: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any] | None:
    curated = (metadata.get("input") or {}).get("curated_metadata") or {}
    domain = curated.get("domain") if isinstance(curated.get("domain"), dict) else {}
    source_mpp = parse_positive_float(domain.get("microns_per_pixel") or domain.get("pixel_size_um"))
    if source_mpp is None:
        return None
    scale_source = str(domain.get("scale_source") or "unavailable")
    scale_confidence = str(domain.get("scale_confidence") or "none")
    if scale_confidence != "calibrated" or scale_source in {"", "unavailable", "none"}:
        return None

    tiling = metadata.get("tiling") if isinstance(metadata.get("tiling"), dict) else {}
    image = metadata.get("image") if isinstance(metadata.get("image"), dict) else {}
    source_width = parse_positive_float(tiling.get("source_width"))
    source_height = parse_positive_float(tiling.get("source_height"))
    analysis_width = parse_positive_float(tiling.get("analysis_width")) or parse_positive_float((metadata.get("image") or {}).get("width"))
    analysis_height = parse_positive_float(tiling.get("analysis_height")) or parse_positive_float((metadata.get("image") or {}).get("height"))

    if not (source_width and source_height and analysis_width and analysis_height):
        source_width = analysis_width = parse_positive_float(image.get("width")) or 1.0
        source_height = analysis_height = parse_positive_float(image.get("height")) or 1.0

    microns_per_analysis_pixel_x = source_mpp * float(source_width) / max(float(analysis_width), 1.0)
    microns_per_analysis_pixel_y = source_mpp * float(source_height) / max(float(analysis_height), 1.0)
    area_um2_per_pixel = microns_per_analysis_pixel_x * microns_per_analysis_pixel_y
    effective_mpp = area_um2_per_pixel ** 0.5
    return {
        "schema_version": "ore-pipeline-scale-v0.1",
        "available": True,
        "source_field": "microns_per_pixel" if domain.get("microns_per_pixel") not in (None, "") else "pixel_size_um",
        "microns_per_source_pixel": source_mpp,
        "microns_per_analysis_pixel_x": microns_per_analysis_pixel_x,
        "microns_per_analysis_pixel_y": microns_per_analysis_pixel_y,
        "effective_microns_per_analysis_pixel": effective_mpp,
        "area_um2_per_analysis_pixel": area_um2_per_pixel,
        "scale_source": scale_source,
        "scale_confidence": scale_confidence,
        "source_width": int(float(source_width)),
        "source_height": int(float(source_height)),
        "analysis_width": int(float(analysis_width)),
        "analysis_height": int(float(analysis_height)),
    }


def metric_rows(summary: dict[str, Any], scale: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    def area_fields(area_px: int | None) -> dict[str, Any]:
        if area_px is None:
            return {}
        fields: dict[str, Any] = {"area_px": area_px}
        if scale:
            fields["area_um2"] = area_px * float(scale["area_um2_per_analysis_pixel"])
            fields["area_mm2"] = fields["area_um2"] / 1_000_000.0
        return fields

    def row(
        key: str,
        value: float | int,
        *,
        percent: float | None,
        area_px: int | None,
        level: int,
        parent_key: str | None = None,
        denominator: str = "",
    ) -> dict[str, Any]:
        return {
            "key": key,
            "label": CLASS_LABELS_RU[key],
            "value": value,
            "percent": percent,
            "level": level,
            "parent_key": parent_key,
            "denominator": denominator,
            **area_fields(area_px),
        }

    image_area_px = int(summary.get("image_area_px") or 0)
    analyzed_area_px = int(summary.get("analysis_area_px") or 0)
    sulfide_area_px = int(summary.get("sulfide_area_px") or 0)
    ordinary_area_px = int(summary.get("ordinary_sulfide_area_px") or 0)
    fine_area_px = int(summary.get("fine_sulfide_area_px") or 0)
    talc_area_px = int(summary.get("talc_area_px") or 0)
    artifact_area_px = int(summary.get("artifact_area_px") or 0)
    other_area_px = max(analyzed_area_px - sulfide_area_px - talc_area_px, 0)
    other_fraction = other_area_px / max(analyzed_area_px, 1)
    artifact_fraction = artifact_area_px / max(image_area_px, 1)

    return [
        row(
            "analyzed_fraction",
            float(summary.get("analyzed_fraction") or 0.0),
            percent=float(summary.get("analyzed_fraction") or 0.0) * 100.0,
            area_px=analyzed_area_px,
            level=0,
            denominator="image",
        ),
        row(
            "sulfide_fraction",
            float(summary.get("sulfide_fraction") or 0.0),
            percent=float(summary.get("sulfide_fraction") or 0.0) * 100.0,
            area_px=sulfide_area_px,
            level=1,
            parent_key="analyzed_fraction",
            denominator="analyzed_area",
        ),
        row(
            "ordinary_sulfide_fraction",
            float(summary.get("ordinary_sulfide_fraction") or 0.0),
            percent=float(summary.get("ordinary_sulfide_fraction") or 0.0) * 100.0,
            area_px=ordinary_area_px,
            level=2,
            parent_key="sulfide_fraction",
            denominator="sulfides",
        ),
        row(
            "fine_sulfide_fraction",
            float(summary.get("fine_sulfide_fraction") or 0.0),
            percent=float(summary.get("fine_sulfide_fraction") or 0.0) * 100.0,
            area_px=fine_area_px,
            level=2,
            parent_key="sulfide_fraction",
            denominator="sulfides",
        ),
        row(
            "component_count",
            int(summary.get("component_count") or 0),
            percent=None,
            area_px=None,
            level=2,
            parent_key="sulfide_fraction",
            denominator="sulfides",
        ),
        row(
            "talc_fraction",
            float(summary.get("talc_fraction") or 0.0),
            percent=float(summary.get("talc_fraction") or 0.0) * 100.0,
            area_px=talc_area_px,
            level=1,
            parent_key="analyzed_fraction",
            denominator="analyzed_area",
        ),
        row(
            "other_fraction",
            other_fraction,
            percent=other_fraction * 100.0,
            area_px=other_area_px,
            level=1,
            parent_key="analyzed_fraction",
            denominator="analyzed_area",
        ),
        row(
            "artifact_fraction_image",
            artifact_fraction,
            percent=artifact_fraction * 100.0,
            area_px=artifact_area_px,
            level=0,
            denominator="image",
        ),
    ]


def add_artifact_summary_fields(summary: dict[str, Any], artifact_mask: np.ndarray | None, image_area_px: int | None = None) -> dict[str, Any]:
    enriched = dict(summary)
    resolved_image_area = int(enriched.get("image_area_px") or image_area_px or 0)
    artifact_area = int(enriched.get("artifact_area_px") or 0)
    if artifact_mask is not None:
        artifact_area = int((artifact_mask > 0).sum())
        resolved_image_area = resolved_image_area or int(artifact_mask.size)
    enriched["artifact_area_px"] = artifact_area
    enriched["artifact_fraction_image"] = artifact_area / max(resolved_image_area, 1)
    enriched["non_artifact_area_px"] = max(resolved_image_area - artifact_area, 0)
    enriched["non_artifact_fraction_image"] = enriched["non_artifact_area_px"] / max(resolved_image_area, 1)
    return enriched


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


def directory_size_summary(path: Path) -> dict[str, Any]:
    total = 0
    files = 0
    directories = 0
    errors: list[str] = []
    if not path.exists():
        return {"path": str(path), "size_bytes": 0, "files": 0, "directories": 0, "errors": []}
    stack = [path]
    while stack:
        current = stack.pop()
        directories += 1
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            files += 1
                            total += int(entry.stat(follow_symlinks=False).st_size)
                    except OSError as exc:
                        if len(errors) < 8:
                            errors.append(f"{entry.path}: {exc}")
        except OSError as exc:
            if len(errors) < 8:
                errors.append(f"{current}: {exc}")
    return {"path": str(path), "size_bytes": total, "files": files, "directories": directories, "errors": errors}


def cpu_status_payload() -> dict[str, Any]:
    logical_cpus = int(os.cpu_count() or 1)
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
        load_pct = min(999.0, max(0.0, load_1m / max(logical_cpus, 1) * 100.0))
        return {
            "logical_cpus": logical_cpus,
            "load_average_1m": load_1m,
            "load_average_5m": load_5m,
            "load_average_15m": load_15m,
            "load_percent_1m": load_pct,
        }
    except OSError:
        return {
            "logical_cpus": logical_cpus,
            "load_average_1m": None,
            "load_average_5m": None,
            "load_average_15m": None,
            "load_percent_1m": None,
        }


def memory_status_payload() -> dict[str, Any]:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        values: dict[str, int] = {}
        for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.replace(":", "").split()
            if len(parts) >= 2 and parts[1].isdigit():
                values[parts[0]] = int(parts[1]) * 1024
        total = int(values.get("MemTotal") or 0)
        available = int(values.get("MemAvailable") or values.get("MemFree") or 0)
        used = max(total - available, 0)
        return {
            "total_bytes": total,
            "available_bytes": available,
            "used_bytes": used,
            "used_percent": used / max(total, 1) * 100.0,
            "source": "/proc/meminfo",
        }
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        total = page_size * total_pages
        available = page_size * available_pages
        used = max(total - available, 0)
        return {
            "total_bytes": total,
            "available_bytes": available,
            "used_bytes": used,
            "used_percent": used / max(total, 1) * 100.0,
            "source": "sysconf",
        }
    except (AttributeError, OSError, ValueError):
        pass
    try:
        total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=1.0).strip())
        output = subprocess.check_output(["vm_stat"], text=True, timeout=1.0)
        page_size = 4096
        header = output.splitlines()[0] if output.splitlines() else ""
        if "page size of" in header:
            page_size = int(header.split("page size of", 1)[1].split("bytes", 1)[0].strip())
        pages: dict[str, int] = {}
        for line in output.splitlines()[1:]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            digits = "".join(ch for ch in value if ch.isdigit())
            if digits:
                pages[key.strip()] = int(digits)
        available_pages = pages.get("Pages free", 0) + pages.get("Pages inactive", 0) + pages.get("Pages speculative", 0)
        available = available_pages * page_size
        used = max(total - available, 0)
        return {
            "total_bytes": total,
            "available_bytes": available,
            "used_bytes": used,
            "used_percent": used / max(total, 1) * 100.0,
            "source": "vm_stat",
        }
    except (subprocess.SubprocessError, OSError, ValueError):
        return {"total_bytes": None, "available_bytes": None, "used_bytes": None, "used_percent": None, "source": "unavailable"}


def disk_status_payload(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    used = usage.total - usage.free
    return {
        "path": str(path),
        "total_bytes": int(usage.total),
        "used_bytes": int(used),
        "free_bytes": int(usage.free),
        "used_percent": used / max(usage.total, 1) * 100.0,
        "free_percent": usage.free / max(usage.total, 1) * 100.0,
    }


def _parse_optional_nvidia_number(value: str) -> float | None:
    normalized = value.strip()
    if not normalized or normalized.upper() in {"[N/A]", "N/A", "NA", "NONE", "NULL"}:
        return None
    return float(normalized)


def _parse_optional_int(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def _torch_mps_available() -> bool:
    try:
        import torch  # type: ignore[import-not-found]

        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        return bool(mps_backend and mps_backend.is_available())
    except Exception:
        return False


@lru_cache(maxsize=1)
def _apple_gpu_devices() -> list[dict[str, Any]]:
    if sys.platform != "darwin":
        return []
    system_profiler = shutil.which("system_profiler")
    if not system_profiler:
        return []
    try:
        result = subprocess.run(
            [system_profiler, "SPDisplaysDataType", "-json", "-detailLevel", "mini"],
            check=True,
            text=True,
            capture_output=True,
            timeout=3.0,
        )
        payload = json.loads(result.stdout or "{}")
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError):
        return []
    devices = []
    for index, entry in enumerate(payload.get("SPDisplaysDataType") or []):
        if not isinstance(entry, dict):
            continue
        model = str(entry.get("sppci_model") or entry.get("_name") or "").strip()
        device_type = str(entry.get("sppci_device_type") or "").lower()
        if not model or ("gpu" not in device_type and "apple" not in model.lower()):
            continue
        displays = entry.get("spdisplays_ndrvs") if isinstance(entry.get("spdisplays_ndrvs"), list) else []
        devices.append(
            {
                "index": index,
                "name": model,
                "backend": "metal",
                "source": "system_profiler",
                "mps_available": _torch_mps_available(),
                "cores": _parse_optional_int(entry.get("sppci_cores")),
                "metal_family": entry.get("spdisplays_mtlgpufamilysupport"),
                "displays": len(displays),
                "utilization_percent": None,
                "memory_total_bytes": None,
                "memory_used_bytes": None,
                "memory_used_percent": None,
                "temperature_c": None,
            }
        )
    return devices


def gpu_status_payload() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        apple_devices = _apple_gpu_devices()
        if apple_devices:
            return {
                "available": True,
                "source": "system_profiler",
                "message": "Apple Metal GPU detected; utilization metrics unavailable via nvidia-smi",
                "devices": apple_devices,
            }
        return {"available": False, "source": "nvidia-smi", "message": "nvidia-smi not found", "devices": []}
    command = [
        nvidia_smi,
        "--query-gpu=index,name,utilization.gpu,memory.total,memory.used,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True, timeout=2.0)
    except (subprocess.SubprocessError, OSError) as exc:
        return {"available": False, "source": "nvidia-smi", "message": str(exc), "devices": []}
    devices = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if len(row) < 6:
            continue
        index, name, util, memory_total, memory_used, temperature = [part.strip() for part in row[:6]]
        total_mib = _parse_optional_nvidia_number(memory_total)
        used_mib = _parse_optional_nvidia_number(memory_used)
        memory_total_bytes = int(total_mib * 1024 * 1024) if total_mib is not None else None
        memory_used_bytes = int(used_mib * 1024 * 1024) if used_mib is not None else None
        memory_used_percent = (
            used_mib / total_mib * 100.0
            if used_mib is not None and total_mib is not None and total_mib > 0
            else None
        )
        devices.append(
            {
                "index": int(_parse_optional_nvidia_number(index) or 0),
                "name": name,
                "utilization_percent": _parse_optional_nvidia_number(util),
                "memory_total_bytes": memory_total_bytes,
                "memory_used_bytes": memory_used_bytes,
                "memory_used_percent": memory_used_percent,
                "temperature_c": _parse_optional_nvidia_number(temperature),
            }
        )
    return {"available": bool(devices), "source": "nvidia-smi", "message": "" if devices else "no devices", "devices": devices}


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
        self.batches_dir = self.workspace_dir / "batches"
        self.settings_dir = self.workspace_dir / "settings"
        self.settings_path = self.settings_dir / "app_settings.json"
        self.backend = backend
        self.checkpoint = resolve_path(checkpoint) if checkpoint else None
        self.processing_max_side = int(processing_max_side)
        self.panorama_max_side = int(panorama_max_side)
        self.preview_max_sides = preview_max_sides
        self.started_at = time.time()
        self.started_at_iso = utc_now_iso()
        self.artifacts: dict[str, Path] = {}
        self.jobs: dict[str, dict[str, Any]] = {}
        self.batch_jobs: dict[str, dict[str, Any]] = {}
        self.foreground_operations: dict[str, dict[str, Any]] = {}
        self.system_log: deque[dict[str, Any]] = deque(maxlen=LOG_ENTRY_LIMIT)
        self.lock = threading.RLock()
        self.allowed_roots = [ROOT.resolve(), self.workspace_dir.resolve()]
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.batches_dir.mkdir(parents=True, exist_ok=True)
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        self._load_persisted_runtime_settings()
        self.record_system_event(
            "info",
            "service initialized",
            backend=self.backend,
            workspace_dir=str(self.workspace_dir),
            checkpoint=str(self.checkpoint) if self.checkpoint else None,
        )

    def record_system_event(self, level: str, message: str, **fields: Any) -> None:
        entry = {
            "timestamp": utc_now_iso(),
            "level": str(level or "info").lower(),
            "message": str(message),
        }
        details = {key: json_safe_value(value) for key, value in fields.items() if value is not None}
        if details:
            entry["details"] = details
        with self.lock:
            self.system_log.append(entry)

    def system_log_payload(self, limit: int = STATUS_LOG_LIMIT) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), LOG_ENTRY_LIMIT))
        with self.lock:
            return list(reversed(list(self.system_log)[-limit:]))

    def begin_foreground_operation(self, kind: str, label: str, **fields: Any) -> str:
        operation_id = f"op_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
        operation = {
            "operation_id": operation_id,
            "kind": str(kind or "operation"),
            "label": str(label or kind or "operation"),
            "status": "running",
            "started_at": time.time(),
            "started_at_iso": utc_now_iso(),
        }
        details = {key: json_safe_value(value) for key, value in fields.items() if value is not None}
        operation.update(details)
        with self.lock:
            self.foreground_operations[operation_id] = operation
        return operation_id

    def finish_foreground_operation(self, operation_id: str) -> None:
        with self.lock:
            self.foreground_operations.pop(operation_id, None)

    def active_foreground_operations_payload(self) -> list[dict[str, Any]]:
        now = time.time()
        with self.lock:
            operations = []
            for operation in self.foreground_operations.values():
                item = {key: value for key, value in operation.items() if key != "started_at"}
                item["elapsed_seconds"] = max(0.0, now - float(operation.get("started_at", now)))
                operations.append(item)
        return sorted(operations, key=lambda item: str(item.get("started_at_iso") or ""))

    def current_runtime_settings(self) -> dict[str, Any]:
        return normalize_settings_runtime(
            {
                "backend": self.backend,
                "checkpoint": str(self.checkpoint) if self.checkpoint else "",
            }
        )

    def _app_settings_base(self) -> dict[str, Any]:
        settings = default_app_settings()
        settings["runtime"] = self.current_runtime_settings()
        return settings

    def _active_runtime_jobs(self) -> list[str]:
        active: list[str] = []
        with self.lock:
            active.extend(
                run_id
                for run_id, job in self.jobs.items()
                if str(job.get("status") or "").lower() in ACTIVE_RUN_STATUSES
            )
            active.extend(
                batch_id
                for batch_id, job in self.batch_jobs.items()
                if str(job.get("status") or "").lower() in BATCH_ACTIVE_STATUSES
            )
            active.extend(
                operation_id
                for operation_id, operation in self.foreground_operations.items()
                if str(operation.get("status") or "").lower() == "running"
            )
        return active

    def _apply_runtime_settings(self, runtime: dict[str, Any], *, validate_checkpoint: bool = True) -> dict[str, Any]:
        normalized = normalize_settings_runtime(runtime, base=self.current_runtime_settings(), validate_checkpoint=validate_checkpoint)
        with self.lock:
            self.backend = normalized["backend"]
            self.checkpoint = Path(normalized["checkpoint"]) if normalized["checkpoint"] else None
        return self.current_runtime_settings()

    def _load_persisted_runtime_settings(self) -> None:
        if not self.settings_path.exists():
            return
        try:
            payload = json.loads(self.settings_path.read_text(encoding="utf-8"))
            settings = normalize_app_settings_payload(payload, base=self._app_settings_base(), validate_runtime=True)
            applied = self._apply_runtime_settings(settings["runtime"], validate_checkpoint=True)
        except (json.JSONDecodeError, ApiError, OSError) as exc:
            self.record_system_event("warning", "runtime settings ignored", error=str(exc))
            return
        self.record_system_event("info", "runtime settings loaded", **applied)

    def _runtime_checkpoint_path(self, checkpoint: str | Path | None) -> str | None:
        if checkpoint is None:
            return None
        checkpoint_text = str(checkpoint).strip()
        if not checkpoint_text:
            return None
        try:
            checkpoint_path = Path(checkpoint_text).expanduser()
            if not checkpoint_path.is_absolute():
                checkpoint_path = ROOT / checkpoint_path
            return str(checkpoint_path.resolve())
        except OSError:
            return checkpoint_text

    def _initial_runtime_provenance(
        self,
        *,
        backend: str | None = None,
        checkpoint: str | Path | None = None,
    ) -> dict[str, Any]:
        backend_value = str(backend or self.backend or "heuristic").lower()
        checkpoint_path = self._runtime_checkpoint_path(checkpoint if checkpoint is not None else self.checkpoint)
        binary_checkpoint = checkpoint_path if backend_value == "ml" else None
        talc_backend = "auto_candidate" if backend_value == "ml" else "heuristic_candidate"
        return {
            "schema_version": RUNTIME_PROVENANCE_SCHEMA_VERSION,
            "backend": backend_value,
            "recorded_at": utc_now_iso(),
            "python_executable": sys.executable,
            "checkpoints": {
                "binary_sulfide": binary_checkpoint,
                "talc": None,
                "final_segmentation": None,
            },
            "models": {
                "binary_sulfide": {
                    "backend": backend_value,
                    "checkpoint": binary_checkpoint,
                    "role": "sulfide/non-sulfide segmentation",
                    "source": "ML checkpoint" if backend_value == "ml" else "heuristic_segmentation",
                },
                "talc": {
                    "backend": talc_backend,
                    "checkpoint": None,
                    "role": "talc candidate detection",
                },
                "final_segmentation": {
                    "backend": "component_rules",
                    "checkpoint": None,
                    "role": "ordinary/fine intergrowth and final class metrics",
                    "rule_config": json_safe_value(DEFAULT_RULE_CONFIG),
                },
            },
        }

    def _runtime_provenance_from_metadata(self, metadata: dict[str, Any], run_dir: Path | None = None) -> dict[str, Any]:
        runtime = metadata.get("runtime") if isinstance(metadata.get("runtime"), dict) else {}
        backend = str(runtime.get("backend") or metadata.get("backend") or self.backend or "heuristic").lower()
        if run_dir is not None and (run_dir / "ml_pipeline/binary_sulfide/summary.json").exists():
            backend = "ml"
        checkpoints = runtime.get("checkpoints") if isinstance(runtime.get("checkpoints"), dict) else {}
        checkpoint = (
            checkpoints.get("binary_sulfide")
            or runtime.get("checkpoint")
            or metadata.get("checkpoint")
            or (str(self.checkpoint) if self.checkpoint else None)
        )
        provenance = self._initial_runtime_provenance(backend=backend, checkpoint=checkpoint)
        for key, value in runtime.items():
            if key not in {"schema_version", "backend", "checkpoints", "models", "recorded_at", "python_executable"}:
                provenance[key] = json_safe_value(value)
        if runtime.get("recorded_at"):
            provenance["recorded_at"] = json_safe_value(runtime["recorded_at"])
        if runtime.get("python_executable"):
            provenance["python_executable"] = str(runtime["python_executable"])
        return provenance

    def _read_optional_json(self, path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _finalize_runtime_provenance(self, metadata: dict[str, Any], run_dir: Path) -> dict[str, Any]:
        runtime = self._runtime_provenance_from_metadata(metadata, run_dir)
        binary_summary = self._read_optional_json(run_dir / "ml_pipeline/binary_sulfide/summary.json")
        pipeline_summary = self._read_optional_json(run_dir / "ml_pipeline/pipeline_summary.json")
        if binary_summary:
            checkpoint = self._runtime_checkpoint_path(binary_summary.get("checkpoint") or runtime["checkpoints"].get("binary_sulfide"))
            runtime["backend"] = "ml"
            runtime["checkpoints"]["binary_sulfide"] = checkpoint
            runtime["models"]["binary_sulfide"] = {
                **runtime["models"].get("binary_sulfide", {}),
                "backend": "ml",
                "checkpoint": checkpoint,
                "source": "ML checkpoint",
                "schema_version": binary_summary.get("schema_version"),
                "checkpoint_meta": json_safe_value(binary_summary.get("checkpoint_meta") or {}),
                "device": binary_summary.get("device"),
                "tile_size": binary_summary.get("tile_size"),
                "stride": binary_summary.get("stride"),
                "threshold": binary_summary.get("threshold"),
                "tiles": binary_summary.get("tiles"),
            }
        if pipeline_summary:
            runtime["pipeline"] = {
                "schema_version": pipeline_summary.get("schema_version"),
                "image": pipeline_summary.get("image"),
                "talc_source": pipeline_summary.get("talc_source"),
                "rule_config": json_safe_value(pipeline_summary.get("rule_config") or {}),
            }
            runtime["models"]["talc"] = {
                **runtime["models"].get("talc", {}),
                "backend": pipeline_summary.get("talc_source") or runtime["models"].get("talc", {}).get("backend"),
                "checkpoint": None,
                "role": "talc candidate detection",
            }
            runtime["models"]["final_segmentation"] = {
                **runtime["models"].get("final_segmentation", {}),
                "backend": "component_rules",
                "checkpoint": None,
                "rule_config": json_safe_value(pipeline_summary.get("rule_config") or DEFAULT_RULE_CONFIG),
            }
        runtime["completed_at"] = metadata.get("completed_at") or runtime.get("completed_at")
        runtime["backend"] = str(runtime.get("backend") or metadata.get("backend") or self.backend or "heuristic").lower()
        metadata["runtime"] = runtime
        metadata["backend"] = runtime["backend"]
        metadata["checkpoint"] = runtime["checkpoints"].get("binary_sulfide")
        runtime_path = run_dir / "reports/runtime.json"
        self._write_json(runtime_path, runtime)
        metadata.setdefault("reports", {})["runtime_json"] = str(runtime_path)
        return runtime

    def _ensure_runtime_provenance(self, run_id: str, data: dict[str, Any]) -> dict[str, Any]:
        run_dir = self.runs_dir / run_id
        runtime = data.get("runtime")
        runtime_report_path = run_dir / "reports/runtime.json"
        if isinstance(runtime, dict) and runtime.get("schema_version") == RUNTIME_PROVENANCE_SCHEMA_VERSION and runtime_report_path.exists():
            return data
        try:
            self._finalize_runtime_provenance(data, run_dir)
            self._write_json(run_dir / "run.json", data)
        except Exception as exc:  # noqa: BLE001 - compatibility layer should not block run loading.
            self.record_system_event("warning", "runtime provenance regeneration failed", run_id=run_id, error=str(exc))
        return data

    def register_upload_from_bytes(self, data: bytes, original_name: str) -> dict[str, Any]:
        suffix = Path(original_name).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            raise ApiError(HTTPStatus.BAD_REQUEST, "supported image formats: PNG, JPEG, TIFF, RAW")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "uploaded image is too large")
        upload_id, upload_dir = self._create_upload_dir(hashlib.sha1(data[:1048576]).hexdigest()[:10])
        original_path = upload_dir / safe_name(original_name)
        original_path.write_bytes(data)
        return self._register_upload_file(upload_id, upload_dir, original_path, original_name)

    def register_upload_from_path(self, path: Path, original_name: str | None = None) -> dict[str, Any]:
        original_name = original_name or path.name
        suffix = Path(original_name).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            raise ApiError(HTTPStatus.BAD_REQUEST, "supported image formats: PNG, JPEG, TIFF, RAW")
        sha1 = file_sha1(path)
        upload_id, upload_dir = self._create_upload_dir(sha1[:10])
        original_path = upload_dir / safe_name(original_name)
        hardlink_or_copy(path, original_path)
        return self._register_upload_file(upload_id, upload_dir, original_path, original_name, sha1=sha1)

    def _create_upload_dir(self, digest: str) -> tuple[str, Path]:
        with self.lock:
            for _ in range(16):
                upload_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}_{digest}"
                upload_dir = self.uploads_dir / upload_id
                try:
                    upload_dir.mkdir(parents=True, exist_ok=False)
                    return upload_id, upload_dir
                except FileExistsError:
                    time.sleep(0)
        raise ApiError(HTTPStatus.CONFLICT, "could not allocate unique upload id")

    def _register_upload_file(
        self,
        upload_id: str,
        upload_dir: Path,
        original_path: Path,
        original_name: str,
        *,
        sha1: str | None = None,
    ) -> dict[str, Any]:
        width, height = image_dimensions(original_path)
        preview_dir = upload_dir / "display/original"
        previews = save_preview_pyramid(
            load_image_pil(original_path, max_side=max(self.preview_max_sides)),
            preview_dir,
            "original",
            self.preview_max_sides,
        )
        raw_metadata = extract_image_raw_metadata(
            original_path,
            original_name=original_name,
            width=width,
            height=height,
            sha1=sha1,
        )
        metadata = {
            "schema_version": "ore-pipeline-upload-v0.1",
            "upload_id": upload_id,
            "created_at": utc_now_iso(),
            "original_name": original_name,
            "original_path": str(original_path),
            "width": int(width),
            "height": int(height),
            "format": original_path.suffix.lower().lstrip("."),
            "file_size_bytes": raw_metadata["file_size_bytes"],
            "sha1": raw_metadata["sha1"],
            "raw_metadata": raw_metadata,
            "display": {"original": previews},
            "preprocess": None,
        }
        self._write_json(upload_dir / "upload.json", metadata)
        return self.upload_payload(upload_id)

    def upload_payload(self, upload_id: str) -> dict[str, Any]:
        metadata = self._read_upload(upload_id)
        display = metadata.get("display", {})
        payload = {
            **metadata,
            "display": {key: self.preview_urls(value) for key, value in display.items()},
        }
        artifact = metadata.get("artifact_mask") if isinstance(metadata.get("artifact_mask"), dict) else None
        if artifact and artifact.get("mask_path"):
            payload["artifact_mask"] = {**artifact, "mask_url": self.artifact_url(artifact.get("mask_path"))}
        return payload

    def save_upload_artifact_mask(self, upload_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        upload_dir = self.uploads_dir / upload_id
        metadata = self._read_upload(upload_id)
        preprocess = metadata.get("preprocess") if isinstance(metadata.get("preprocess"), dict) else None
        if not preprocess:
            raise ApiError(HTTPStatus.BAD_REQUEST, "upload must be prepared before saving an artifact mask")
        expected_shape = (int(preprocess["height"]), int(preprocess["width"]))
        mask = decode_mask_data_url(str(payload.get("mask_png") or ""), expected_shape)
        artifact_dir = upload_dir / "artifacts"
        mask_path = artifact_dir / "artifact_mask.png"
        save_image(mask_path, Image.fromarray(mask, mode="L"))
        comment = str(payload.get("comment") or "").strip()
        metadata["artifact_mask"] = {
            "schema_version": "ore-pipeline-artifact-mask-v0.1",
            "updated_at": utc_now_iso(),
            "mask_path": str(mask_path),
            "width": int(expected_shape[1]),
            "height": int(expected_shape[0]),
            "comment": comment,
        }
        self._write_json(upload_dir / "upload.json", metadata)
        return self.upload_payload(upload_id)

    def prepare_upload(
        self,
        upload_id: str,
        preset: dict[str, Any],
        augmentation_settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        upload_dir = self.uploads_dir / upload_id
        metadata = self._read_upload(upload_id)
        original_path = resolve_path(metadata["original_path"])
        augmentation = normalize_augmentation_settings(augmentation_settings or default_augmentation_settings())
        preprocessing_enabled = bool(preset.get("preprocessing_enabled", preset.get("enabled", True)))
        target_max_side, panorama_scaling = panorama_scaling_target(
            preset,
            preprocessing_enabled=preprocessing_enabled,
            source_width=int(metadata["width"]),
            source_height=int(metadata["height"]),
            processing_max_side=self.processing_max_side,
            default_panorama_max_side=self.panorama_max_side,
        )
        if max(int(metadata["width"]), int(metadata["height"])) > target_max_side:
            source_scaled = True
        else:
            source_scaled = False
        full_size_processing_deferred = should_defer_full_size_processing(
            source_width=int(metadata["width"]),
            source_height=int(metadata["height"]),
            target_max_side=target_max_side,
        )
        decode_max_side = target_max_side if full_size_processing_deferred else None
        source = load_image_pil(original_path, max_side=decode_max_side)
        augmented_image = apply_augmentation(source, augmentation) if augmentation_enabled(augmentation) else source
        if augmentation_enabled(augmentation):
            augmentation_dir = upload_dir / "augmentation"
            augmented_path = augmentation_dir / "augmented.png"
            save_image(augmented_path, augmented_image)
            augmented_previews = save_preview_pyramid(
                augmented_image,
                augmentation_dir / "display",
                "augmented",
                self.preview_max_sides,
            )
            augmentation_metadata = {
                "schema_version": "ore-pipeline-runtime-augmentation-v0.1",
                "updated_at": utc_now_iso(),
                "enabled": True,
                "settings": augmentation,
                "augmented_path": str(augmented_path),
                "width": augmented_image.size[0],
                "height": augmented_image.size[1],
                "source_width": int(metadata["width"]),
                "source_height": int(metadata["height"]),
                "source_scaled_for_processing": full_size_processing_deferred,
                "display": augmented_previews,
            }
            self._write_json(augmentation_dir / "augmentation.json", augmentation_metadata)
        else:
            augmented_previews = []
            augmentation_metadata = {
                "schema_version": "ore-pipeline-runtime-augmentation-v0.1",
                "updated_at": utc_now_iso(),
                "enabled": False,
                "settings": augmentation,
            }
        preprocessed_stage = apply_preprocessing(augmented_image, preset) if preprocessing_enabled else augmented_image
        preprocess_dir = upload_dir / "preprocessed"
        preprocessed_full_path = preprocess_dir / "preprocessed_full.png"
        if preprocessing_enabled and not full_size_processing_deferred:
            save_image(preprocessed_full_path, preprocessed_stage)
        analysis_image = scaled_image_copy(preprocessed_stage, max_side=target_max_side)
        preprocessed_path = preprocess_dir / "preprocessed.png"
        save_image(preprocessed_path, analysis_image)
        previews = (
            save_preview_pyramid(
                analysis_image if full_size_processing_deferred else preprocessed_stage,
                preprocess_dir / "display",
                "preprocessed",
                self.preview_max_sides,
            )
            if preprocessing_enabled
            else []
        )
        tiling = build_tiling_manifest(
            source_width=int(metadata["width"]),
            source_height=int(metadata["height"]),
            analysis_width=analysis_image.size[0],
            analysis_height=analysis_image.size[1],
            source_scaled=source_scaled,
        )
        preprocess_metadata = {
            "schema_version": "ore-pipeline-preprocess-v0.1",
            "updated_at": utc_now_iso(),
            "enabled": preprocessing_enabled,
            "preset": preset,
            "preprocessed_path": str(preprocessed_path),
            "analysis_path": str(preprocessed_path),
            "width": analysis_image.size[0],
            "height": analysis_image.size[1],
            "full_width": preprocessed_stage.size[0],
            "full_height": preprocessed_stage.size[1],
            "source_width": int(metadata["width"]),
            "source_height": int(metadata["height"]),
            "source_scaled_for_processing": source_scaled,
            "full_size_processing_deferred": full_size_processing_deferred,
            "full_size_preprocess_max_pixels": FULL_SIZE_PREPROCESS_MAX_PIXELS,
            "target_max_side": target_max_side,
            "panorama_scaling": panorama_scaling,
            "display": previews,
            "tiling": tiling,
        }
        if preprocessing_enabled and not full_size_processing_deferred:
            preprocess_metadata["preprocessed_full_path"] = str(preprocessed_full_path)
        self._write_json(preprocess_dir / "preprocess.json", preprocess_metadata)
        metadata["augmentation"] = augmentation_metadata
        metadata["preprocess"] = preprocess_metadata
        self._sync_upload_artifact_mask(upload_dir, metadata, analysis_image.size)
        if augmentation_enabled(augmentation):
            metadata.setdefault("display", {})["augmented"] = augmented_previews
        else:
            metadata.setdefault("display", {}).pop("augmented", None)
        if preprocessing_enabled:
            metadata.setdefault("display", {})["preprocessed"] = previews
        else:
            metadata.setdefault("display", {}).pop("preprocessed", None)
        metadata["tiling"] = tiling
        self._write_json(upload_dir / "upload.json", metadata)
        return self.upload_payload(upload_id)

    def _sync_upload_artifact_mask(self, upload_dir: Path, metadata: dict[str, Any], size: tuple[int, int]) -> None:
        artifact = metadata.get("artifact_mask") if isinstance(metadata.get("artifact_mask"), dict) else None
        if not artifact or not artifact.get("mask_path"):
            return
        mask_path = resolve_path(artifact["mask_path"])
        if not mask_path.exists():
            metadata.pop("artifact_mask", None)
            return
        expected_shape = (int(size[1]), int(size[0]))
        mask = read_binary_mask(mask_path, expected_shape)
        artifact_dir = upload_dir / "artifacts"
        synced_path = artifact_dir / "artifact_mask.png"
        save_image(synced_path, Image.fromarray(mask, mode="L"))
        metadata["artifact_mask"] = {
            **artifact,
            "schema_version": "ore-pipeline-artifact-mask-v0.1",
            "updated_at": utc_now_iso(),
            "mask_path": str(synced_path),
            "width": int(size[0]),
            "height": int(size[1]),
        }

    def start_run(
        self,
        upload_id: str,
        preset: dict[str, Any],
        *,
        run_async: bool = True,
        curated_metadata: Any = None,
        augmentation_settings: dict[str, Any] | None = None,
        batch_link: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_curated_metadata = normalize_curated_metadata_payload(curated_metadata)
        upload = self.prepare_upload(upload_id, preset, augmentation_settings)
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}_{hashlib.sha1(upload_id.encode()).hexdigest()[:8]}"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        self._initialize_run_from_upload(
            run_id,
            run_dir,
            upload,
            preset,
            curated_metadata=normalized_curated_metadata,
            batch_link=batch_link,
        )
        with self.lock:
            self.jobs[run_id] = {
                "progress": 1,
                "status": "queued",
                "stage": "queued",
                "started_at": time.time(),
                "eta_seconds": None,
                "cancel_requested": False,
            }
        self.record_system_event("info", "run queued", run_id=run_id, upload_id=upload_id)
        if run_async:
            thread = threading.Thread(target=self._run_job_guarded, args=(run_id,), daemon=True)
            thread.start()
        else:
            self._run_job_guarded(run_id)
        return self.run_payload(run_id)

    def prepare_run_from_apply(
        self,
        run_id: str,
        preset: dict[str, Any],
        *,
        augmentation_settings: dict[str, Any] | None = None,
        changed_step: str,
    ) -> dict[str, Any]:
        if changed_step not in {"augmentation", "preprocess"}:
            raise ApiError(HTTPStatus.BAD_REQUEST, "changed_step must be augmentation or preprocess")
        parent = self._read_run(run_id)
        status = str(parent.get("status") or "").lower()
        if status not in {"complete", "prepared"}:
            raise ApiError(HTTPStatus.CONFLICT, "run must be complete or prepared before applying pipeline settings")
        upload_id = str((parent.get("input") or {}).get("upload_id") or "")
        if not upload_id:
            raise ApiError(HTTPStatus.BAD_REQUEST, "run has no upload_id")
        upload = self.prepare_upload(upload_id, preset, augmentation_settings)
        if status == "complete":
            target_run_id = f"apply_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}_{hashlib.sha1((run_id + changed_step).encode()).hexdigest()[:8]}"
            target_run_dir = self.runs_dir / target_run_id
            target_run_dir.mkdir(parents=True, exist_ok=False)
            parent_run_id = run_id
        else:
            target_run_id = run_id
            target_run_dir = self.runs_dir / target_run_id
            parent_run_id = str((parent.get("derivation") or {}).get("parent_run_id") or run_id)
            for relative in ("input", "display", "masks", "reports", "ml_pipeline"):
                shutil.rmtree(target_run_dir / relative, ignore_errors=True)
            with self.lock:
                self.jobs.pop(target_run_id, None)

        self._initialize_run_from_upload(
            target_run_id,
            target_run_dir,
            upload,
            preset,
            curated_metadata=(parent.get("input") or {}).get("curated_metadata"),
        )
        metadata = self._read_run(target_run_id)
        metadata["status"] = "prepared"
        metadata["stage"] = "prepared"
        metadata["progress"] = 0
        metadata["eta_seconds"] = None
        metadata["backend"] = parent.get("backend", self.backend)
        metadata["checkpoint"] = parent.get("checkpoint", str(self.checkpoint) if self.checkpoint else None)
        metadata["runtime"] = self._runtime_provenance_from_metadata(parent, self.runs_dir / str(parent.get("run_id") or run_id))
        metadata["derivation"] = {
            "type": "apply_pipeline_settings",
            "parent_run_id": parent_run_id,
            "changed_step": changed_step,
            "created_at": utc_now_iso(),
            "operation": "prepare_from_augmentation_apply" if changed_step == "augmentation" else "prepare_from_preprocessing_apply",
            "mutable_until_start": True,
        }
        self._preserve_parent_artifact_mask(parent, target_run_dir, metadata)
        self._finalize_prepared_run_metadata(
            metadata,
            target_run_dir,
            preprocessing_enabled=bool((upload.get("preprocess") or {}).get("enabled", True)),
        )
        self._write_json(target_run_dir / "run.json", metadata)
        self.record_system_event(
            "info",
            "prepared run updated" if status == "prepared" else "prepared run created",
            run_id=target_run_id,
            parent_run_id=parent_run_id,
            changed_step=changed_step,
        )
        return self.run_payload(target_run_id)

    def start_prepared_run(
        self,
        run_id: str,
        *,
        run_async: bool = True,
        curated_metadata: Any = None,
    ) -> dict[str, Any]:
        run_dir = self.runs_dir / run_id
        metadata = self._read_run(run_id)
        if str(metadata.get("status") or "").lower() != "prepared":
            raise ApiError(HTTPStatus.CONFLICT, "run is not prepared")
        normalized_curated_metadata = normalize_curated_metadata_payload(curated_metadata)
        if normalized_curated_metadata:
            shutil.rmtree(run_dir / "metadata", ignore_errors=True)
            metadata.get("input", {}).pop("curated_metadata", None)
            metadata.get("input", {}).pop("curated_metadata_json", None)
            self._attach_curated_metadata(metadata, run_dir, normalized_curated_metadata)
        metadata["status"] = "queued"
        metadata["stage"] = "queued"
        metadata["progress"] = 1
        metadata["eta_seconds"] = None
        runtime = self._runtime_provenance_from_metadata(metadata, run_dir)
        runtime["started_at"] = utc_now_iso()
        metadata["runtime"] = runtime
        self._write_json(run_dir / "run.json", metadata)
        with self.lock:
            self.jobs[run_id] = {
                "progress": 1,
                "status": "queued",
                "stage": "queued",
                "started_at": time.time(),
                "eta_seconds": None,
                "cancel_requested": False,
            }
        self.record_system_event("info", "prepared run queued", run_id=run_id)
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
        if edit_layer not in {"artifact", "sulfide", "final"}:
            raise ApiError(HTTPStatus.BAD_REQUEST, "edit_layer must be artifact, sulfide, or final")
        run_id = f"edit_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}_{hashlib.sha1((parent_run_id + edit_layer).encode()).hexdigest()[:8]}"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        self._copy_run_inputs(parent, parent_dir, run_dir)
        expected_shape = (int(parent["image"]["height"]), int(parent["image"]["width"]))
        mask = decode_mask_data_url(payload.get("mask_png", ""), expected_shape, final_mask=edit_layer == "final")
        comment = str(payload.get("comment") or "").strip()
        operation_by_layer = {
            "artifact": "recalculate_from_artifact_edit",
            "sulfide": "recalculate_from_sulfide_edit",
            "final": "recalculate_metrics_from_final_edit",
        }
        derivation = {
            "type": "edit_recalculate",
            "parent_run_id": parent_run_id,
            "edit_layer": edit_layer,
            "comment": comment,
            "created_at": utc_now_iso(),
            "operation": operation_by_layer[edit_layer],
        }
        if edit_layer == "artifact":
            self._write_masks_from_artifact_edit(parent_dir, run_dir, mask)
        elif edit_layer == "sulfide":
            self._write_masks_from_sulfide_edit(parent_dir, run_dir, mask)
        else:
            self._write_masks_from_final_edit(parent_dir, run_dir, mask)
        run_metadata = self._base_run_metadata(run_id, run_dir, parent["input"]["upload_id"], parent["preprocess"]["preset"])
        run_metadata["status"] = "complete"
        run_metadata["progress"] = 100
        run_metadata["backend"] = parent.get("backend", self.backend)
        run_metadata["checkpoint"] = parent.get("checkpoint", str(self.checkpoint) if self.checkpoint else None)
        run_metadata["runtime"] = self._runtime_provenance_from_metadata(parent, parent_dir)
        run_metadata["runtime"]["derived_from_run_id"] = parent_run_id
        run_metadata["preprocess"]["enabled"] = bool((parent.get("preprocess") or {}).get("enabled", True))
        run_metadata["augmentation"] = parent.get("augmentation") or {"enabled": False, "settings": default_augmentation_settings()}
        run_metadata["derivation"] = derivation
        run_metadata["input"]["original_source_path"] = parent["input"].get("original_source_path")
        run_metadata["input"]["original_artifact_path"] = str(run_dir / "input/original_source" / Path(parent["input"]["original_artifact_path"]).name)
        if (run_dir / "input/preprocessed_full.png").exists():
            run_metadata["input"]["preprocessed_full_path"] = str(run_dir / "input/preprocessed_full.png")
        if (run_dir / "input/augmented.png").exists():
            run_metadata["input"]["augmented_path"] = str(run_dir / "input/augmented.png")
        if (run_dir / "input/artifact_mask.png").exists():
            run_metadata["input"]["artifact_mask_path"] = str(run_dir / "input/artifact_mask.png")
        self._attach_curated_metadata(run_metadata, run_dir, (parent.get("input") or {}).get("curated_metadata"))
        run_metadata["tiling"] = parent.get("tiling") or {}
        self._finalize_run_metadata(run_metadata, run_dir)
        (run_dir / "edit_comment.txt").write_text(comment + "\n", encoding="utf-8")
        self._write_json(run_dir / "run.json", run_metadata)
        self.record_system_event("info", "edit recalculated", run_id=run_id, parent_run_id=parent_run_id, edit_layer=edit_layer)
        return self.run_payload(run_id)

    def list_runs(self) -> dict[str, Any]:
        runs = []
        for path in sorted(self.runs_dir.glob("*/run.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            run_id = str(data.get("run_id") or path.parent.name)
            data = self._merge_active_run_job(run_id, data)
            summary = data.get("summary") or {}
            thumbnail = self.history_thumbnail_payload(data)
            runs.append(
                {
                    "run_id": run_id,
                    "created_at": data.get("created_at"),
                    "status": data.get("status"),
                    "progress": data.get("progress", 0),
                    "stage": data.get("stage"),
                    "eta_seconds": data.get("eta_seconds"),
                    "tile_progress": data.get("tile_progress"),
                    "parent_run_id": (data.get("derivation") or {}).get("parent_run_id"),
                    "edit_layer": (data.get("derivation") or {}).get("edit_layer"),
                    "ore_class_ru": summary.get("ore_class_ru"),
                    "summary": summary,
                    "metrics": data.get("metrics", []),
                    "text_output": data.get("text_output"),
                    "image": data.get("image"),
                    "batch": data.get("batch"),
                    "thumbnail": thumbnail,
                }
            )
        return {"schema_version": "ore-pipeline-history-v0.1", "runs": runs}

    def status_payload(self, *, access_log: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        runs_payload = self.list_runs()
        batches_payload = self.list_batches()
        runs = runs_payload.get("runs", [])
        batches = batches_payload.get("batches", [])
        run_status_counts: dict[str, int] = {}
        for run in runs:
            status = str(run.get("status") or "unknown")
            run_status_counts[status] = run_status_counts.get(status, 0) + 1
        batch_status_counts: dict[str, int] = {}
        for batch in batches:
            status = str(batch.get("status") or "unknown")
            batch_status_counts[status] = batch_status_counts.get(status, 0) + 1
        with self.lock:
            active_runs = [
                {"run_id": run_id, "status": job.get("status"), "progress": job.get("progress", 0)}
                for run_id, job in self.jobs.items()
                if str(job.get("status") or "").lower() in ACTIVE_RUN_STATUSES
            ]
            active_batches = [
                {"batch_id": batch_id, "status": job.get("status"), "progress": job.get("progress", 0)}
                for batch_id, job in self.batch_jobs.items()
                if str(job.get("status") or "").lower() in BATCH_ACTIVE_STATUSES
            ]
        active_operations = self.active_foreground_operations_payload()
        runs_size = directory_size_summary(self.runs_dir)
        batches_size = directory_size_summary(self.batches_dir)
        uploads_size = directory_size_summary(self.uploads_dir)
        disk = disk_status_payload(self.workspace_dir)
        memory = memory_status_payload()
        cpu = cpu_status_payload()
        gpu = gpu_status_payload()
        checks: list[dict[str, Any]] = []
        if os.access(self.workspace_dir, os.W_OK):
            checks.append({"key": "workspace_writable", "status": "ok", "message": str(self.workspace_dir)})
        else:
            checks.append({"key": "workspace_writable", "status": "error", "message": str(self.workspace_dir)})
        if self.backend == "ml" and (not self.checkpoint or not self.checkpoint.exists()):
            checks.append({"key": "checkpoint", "status": "error", "message": str(self.checkpoint or "")})
        elif self.backend == "ml":
            checks.append({"key": "checkpoint", "status": "ok", "message": str(self.checkpoint)})
        else:
            checks.append({"key": "backend", "status": "ok", "message": self.backend})
        if disk["free_percent"] < 3:
            checks.append({"key": "flash_free", "status": "error", "message": f"{disk['free_percent']:.1f}%"})
        elif disk["free_percent"] < 10:
            checks.append({"key": "flash_free", "status": "warning", "message": f"{disk['free_percent']:.1f}%"})
        else:
            checks.append({"key": "flash_free", "status": "ok", "message": f"{disk['free_percent']:.1f}%"})
        memory_available = memory.get("available_bytes")
        memory_total = memory.get("total_bytes")
        if isinstance(memory_available, int) and isinstance(memory_total, int) and memory_total > 0:
            available_percent = memory_available / memory_total * 100.0
            if available_percent < 5:
                checks.append({"key": "ram_available", "status": "error", "message": f"{available_percent:.1f}%"})
            elif available_percent < 12:
                checks.append({"key": "ram_available", "status": "warning", "message": f"{available_percent:.1f}%"})
            else:
                checks.append({"key": "ram_available", "status": "ok", "message": f"{available_percent:.1f}%"})
        if cpu.get("load_percent_1m") is not None and float(cpu["load_percent_1m"]) > 200.0:
            checks.append({"key": "cpu_load", "status": "warning", "message": f"{float(cpu['load_percent_1m']):.1f}%"})
        if active_runs or active_batches or active_operations:
            operation_labels = ", ".join(str(operation.get("label") or operation.get("kind") or "") for operation in active_operations[:3])
            operation_text = f", {len(active_operations)} foreground"
            if operation_labels:
                operation_text += f" ({operation_labels})"
            checks.append(
                {
                    "key": "active_jobs",
                    "status": "warning",
                    "message": f"{len(active_runs)} runs, {len(active_batches)} series{operation_text}",
                }
            )
        overall = "ok"
        if any(check["status"] == "error" for check in checks):
            overall = "error"
        elif any(check["status"] == "warning" for check in checks):
            overall = "warning"
        history_size = int(runs_size["size_bytes"]) + int(batches_size["size_bytes"])
        return {
            "schema_version": "ore-pipeline-status-v0.1",
            "generated_at": utc_now_iso(),
            "app": {
                "started_at": self.started_at_iso,
                "uptime_seconds": max(0.0, time.time() - self.started_at),
                "backend": self.backend,
                "checkpoint": str(self.checkpoint) if self.checkpoint else None,
                "checkpoint_exists": bool(self.checkpoint and self.checkpoint.exists()),
                "workspace_dir": str(self.workspace_dir),
            },
            "health": {"overall": overall, "checks": checks},
            "cpu": cpu,
            "gpu": gpu,
            "ram": memory,
            "flash": disk,
            "history": {
                "runs_total": len(runs),
                "batches_total": len(batches),
                "run_status_counts": run_status_counts,
                "batch_status_counts": batch_status_counts,
                "runs_size_bytes": int(runs_size["size_bytes"]),
                "batches_size_bytes": int(batches_size["size_bytes"]),
                "uploads_size_bytes": int(uploads_size["size_bytes"]),
                "history_size_bytes": history_size,
                "total_workspace_size_bytes": history_size + int(uploads_size["size_bytes"]),
                "active_runs": active_runs,
                "active_batches": active_batches,
                "active_operations": active_operations,
            },
            "storage_scan": {"runs": runs_size, "batches": batches_size, "uploads": uploads_size},
            "logs": {
                "system": self.system_log_payload(),
                "access": access_log or [],
                "limit": STATUS_LOG_LIMIT,
            },
        }

    def app_settings(self) -> dict[str, Any]:
        if not self.settings_path.exists():
            return normalize_app_settings_payload({}, base=self._app_settings_base())
        try:
            payload = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return normalize_app_settings_payload({}, base=self._app_settings_base())
        return normalize_app_settings_payload(payload, base=self._app_settings_base())

    def save_app_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        settings = normalize_app_settings_payload(payload, base=self.app_settings(), validate_runtime=True)
        runtime_before = self.current_runtime_settings()
        runtime_after = normalize_settings_runtime(settings.get("runtime"), base=runtime_before, validate_checkpoint=True)
        if runtime_after != runtime_before:
            active_jobs = self._active_runtime_jobs()
            if active_jobs:
                raise ApiError(HTTPStatus.CONFLICT, f"runtime backend cannot be changed while jobs are active: {', '.join(active_jobs)}")
            settings["runtime"] = self._apply_runtime_settings(runtime_after, validate_checkpoint=True)
            self.record_system_event("info", "runtime settings changed", **settings["runtime"])
        settings["updated_at"] = utc_now_iso()
        self._write_json(self.settings_path, settings)
        return settings

    def test_runtime(self, payload: dict[str, Any]) -> dict[str, Any]:
        runtime_payload: Any = payload.get("runtime")
        settings_payload = payload.get("settings")
        if runtime_payload is None and isinstance(settings_payload, dict):
            runtime_payload = settings_payload.get("runtime")
        if runtime_payload is None:
            runtime_payload = payload
        runtime = normalize_settings_runtime(runtime_payload, base=self.current_runtime_settings(), validate_checkpoint=True)
        started = time.time()
        base_result: dict[str, Any] = {
            "schema_version": "ore-pipeline-runtime-test-v0.1",
            "generated_at": utc_now_iso(),
            "backend": runtime["backend"],
            "checkpoint": runtime["checkpoint"] or None,
            "ok": False,
            "status": "error",
            "seconds": 0.0,
        }
        if runtime["backend"] == "heuristic":
            result = {
                **base_result,
                "ok": True,
                "status": "ok",
                "message": "heuristic backend is available",
                "seconds": round(time.time() - started, 3),
                "details": {"module": "heuristic_segmentation.segmentation", "function": "segment_image"},
            }
            self.record_system_event("info", "runtime test ok", backend="heuristic")
            return result

        active_jobs = self._active_runtime_jobs()
        if active_jobs:
            raise ApiError(HTTPStatus.CONFLICT, f"runtime test cannot run while jobs are active: {', '.join(active_jobs)}")

        checkpoint = Path(runtime["checkpoint"])
        probe_script = r"""
import json
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
sys.path.insert(0, str(root / "src"))
started = time.time()
import torch
try:
    import transformers
    transformers_version = transformers.__version__
except Exception:
    transformers_version = "unavailable"
from ore_classifier.model_io import load_binary_segmentation_checkpoint, resolve_device
device = resolve_device("auto")
model, checkpoint_meta = load_binary_segmentation_checkpoint(checkpoint, device)
parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
print(json.dumps({
    "device": str(device),
    "torch": torch.__version__,
    "transformers": transformers_version,
    "checkpoint_meta": checkpoint_meta,
    "parameter_count": parameter_count,
    "seconds": round(time.time() - started, 3),
}, default=str))
"""
        command = [sys.executable, "-c", probe_script, str(ROOT), str(checkpoint)]
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=RUNTIME_TEST_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = round(time.time() - started, 3)
            result = {
                **base_result,
                "seconds": elapsed,
                "message": f"ML runtime probe timed out after {elapsed:.1f}s",
                "details": {
                    "timeout_seconds": RUNTIME_TEST_TIMEOUT_SECONDS,
                    "stdout": compact_text(exc.stdout),
                    "stderr": compact_text(exc.stderr),
                },
            }
            self.record_system_event("warning", "runtime test failed", backend="ml", error=result["message"])
            return result

        elapsed = round(time.time() - started, 3)
        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        if completed.returncode:
            error_text = compact_text(stderr or stdout or f"runtime probe exited with code {completed.returncode}")
            result = {
                **base_result,
                "seconds": elapsed,
                "message": error_text,
                "details": {"returncode": completed.returncode, "stdout": compact_text(stdout), "stderr": compact_text(stderr)},
            }
            self.record_system_event("warning", "runtime test failed", backend="ml", error=error_text)
            return result

        try:
            details = json.loads(stdout.splitlines()[-1]) if stdout else {}
        except (json.JSONDecodeError, IndexError):
            details = {"stdout": compact_text(stdout)}
        checkpoint_meta = details.get("checkpoint_meta") if isinstance(details.get("checkpoint_meta"), dict) else {}
        model_name = str(checkpoint_meta.get("model") or "unknown")
        device = str(details.get("device") or "cpu")
        result = {
            **base_result,
            "ok": True,
            "status": "ok",
            "seconds": elapsed,
            "message": f"ML checkpoint loaded: model={model_name}, device={device}",
            "details": json_safe_value(details),
        }
        self.record_system_event("info", "runtime test ok", backend="ml", checkpoint=str(checkpoint), model=model_name, seconds=elapsed)
        return result

    def create_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        batch_id = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
        batch_dir = self.batches_dir / batch_id
        batch_dir.mkdir(parents=True, exist_ok=False)
        summary = {
            "schema_version": BATCH_SCHEMA_VERSION,
            "batch_id": batch_id,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "status": "draft",
            "progress": 0,
            "current_item_id": None,
            "settings": self._batch_settings_from_payload(payload),
            "item_counts": {},
            "items": [],
            "reports": {},
        }
        self._write_batch(summary)
        if self._batch_upload_refs(payload):
            return self.add_batch_items(batch_id, payload)
        return self.batch_payload(batch_id)

    def list_batches(self) -> dict[str, Any]:
        batches = []
        for path in sorted(self.batches_dir.glob("*/batch_summary.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            batches.append(
                {
                    "batch_id": data.get("batch_id"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "status": data.get("status"),
                    "progress": data.get("progress", 0),
                    "item_counts": data.get("item_counts") or {},
                    "items_count": len(data.get("items") or []),
                }
            )
        return {"schema_version": "ore-pipeline-batch-history-v0.1", "batches": batches}

    def add_batch_items(self, batch_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        summary = self._read_batch(batch_id)
        if summary.get("status") != "draft":
            raise ApiError(HTTPStatus.CONFLICT, "items can only be added to a draft batch")
        refs = self._batch_upload_refs(payload)
        if not refs:
            raise ApiError(HTTPStatus.BAD_REQUEST, "upload_ids are required")
        items = summary.setdefault("items", [])
        for ref in refs:
            upload_id = ref["upload_id"]
            upload = self.upload_payload(upload_id)
            index = len(items) + 1
            item_digest = hashlib.sha1(f"{batch_id}:{index}:{upload_id}:{time.time_ns()}".encode("utf-8")).hexdigest()[:8]
            item = {
                "schema_version": BATCH_ITEM_SCHEMA_VERSION,
                "item_id": f"item_{index:04d}_{item_digest}",
                "index": index,
                "upload_id": upload_id,
                "original_name": upload.get("original_name"),
                "width": upload.get("width"),
                "height": upload.get("height"),
                "sha1": upload.get("sha1"),
                "status": "draft",
                "progress": 0,
                "stage": "draft",
                "run_id": None,
                "error": None,
                "curated_metadata": normalize_curated_metadata_payload(ref.get("curated_metadata")),
                "created_at": utc_now_iso(),
                "updated_at": utc_now_iso(),
            }
            items.append(item)
        self._finalize_batch_summary(summary)
        summary["updated_at"] = utc_now_iso()
        self._write_batch(summary)
        return self.batch_payload(batch_id)

    def update_batch_settings(self, batch_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        summary = self._read_batch(batch_id)
        if summary.get("status") != "draft":
            raise ApiError(HTTPStatus.CONFLICT, "settings can only be changed before the batch starts")
        summary["settings"] = self._batch_settings_from_payload(payload)
        summary["updated_at"] = utc_now_iso()
        self._write_batch(summary)
        return self.batch_payload(batch_id)

    def update_batch_item_metadata(self, batch_id: str, item_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        summary = self._read_batch(batch_id)
        if summary.get("status") in BATCH_ACTIVE_STATUSES:
            raise ApiError(HTTPStatus.CONFLICT, "metadata cannot be edited while the batch is running")
        item = self._batch_item(summary, item_id)
        if item.get("run_id"):
            raise ApiError(HTTPStatus.CONFLICT, "metadata cannot be edited after the item has a run")
        item["curated_metadata"] = normalize_curated_metadata_payload(payload.get("curated_metadata"))
        item["updated_at"] = utc_now_iso()
        summary["updated_at"] = utc_now_iso()
        self._write_batch(summary)
        return self.batch_payload(batch_id)

    def remove_batch_item(self, batch_id: str, item_id: str) -> dict[str, Any]:
        summary = self._read_batch(batch_id)
        if summary.get("status") in BATCH_ACTIVE_STATUSES:
            raise ApiError(HTTPStatus.CONFLICT, "items cannot be removed while the batch is running")
        items = summary.get("items") or []
        item = self._batch_item(summary, item_id)
        if item.get("run_id"):
            raise ApiError(HTTPStatus.CONFLICT, "items with immutable runs cannot be removed")
        summary["items"] = [candidate for candidate in items if candidate.get("item_id") != item_id]
        for index, candidate in enumerate(summary["items"], start=1):
            candidate["index"] = index
        self._finalize_batch_summary(summary)
        summary["updated_at"] = utc_now_iso()
        self._write_batch(summary)
        return self.batch_payload(batch_id)

    def delete_batch(self, batch_id: str) -> dict[str, Any]:
        summary = self._read_batch(batch_id)
        if summary.get("status") in BATCH_ACTIVE_STATUSES:
            raise ApiError(HTTPStatus.CONFLICT, "batch is still running")
        with self.lock:
            job_status = self.batch_jobs.get(batch_id, {}).get("status")
            if job_status in BATCH_ACTIVE_STATUSES:
                raise ApiError(HTTPStatus.CONFLICT, "batch is still running")
        batch_dir = (self.batches_dir / batch_id).resolve()
        batches_root = self.batches_dir.resolve()
        if batch_dir == batches_root or not is_relative_to(batch_dir, batches_root):
            raise ApiError(HTTPStatus.BAD_REQUEST, "invalid batch id")
        if not (batch_dir / "batch_summary.json").exists():
            raise ApiError(HTTPStatus.NOT_FOUND, f"unknown batch: {batch_id}")

        run_ids = [str(item.get("run_id")) for item in summary.get("items", []) if item.get("run_id")]
        run_dirs: list[tuple[str, Path]] = []
        runs_root = self.runs_dir.resolve()
        for run_id in run_ids:
            run_dir = (self.runs_dir / run_id).resolve()
            if run_dir == runs_root or not is_relative_to(run_dir, runs_root):
                raise ApiError(HTTPStatus.BAD_REQUEST, f"invalid child run id: {run_id}")
            if not (run_dir / "run.json").exists():
                continue
            with self.lock:
                job_status = self.jobs.get(run_id, {}).get("status")
                if job_status in BATCH_ACTIVE_STATUSES:
                    raise ApiError(HTTPStatus.CONFLICT, "child run is still running")
            run_dirs.append((run_id, run_dir))

        with self.lock:
            self.batch_jobs.pop(batch_id, None)
            for run_id, _ in run_dirs:
                self.jobs.pop(run_id, None)
        for _, run_dir in run_dirs:
            shutil.rmtree(run_dir)
        shutil.rmtree(batch_dir)
        return {
            "removed_batch_id": batch_id,
            "removed_run_ids": [run_id for run_id, _ in run_dirs],
            "batches": self.list_batches()["batches"],
            "history": self.list_runs()["runs"],
        }

    def delete_history(self) -> dict[str, Any]:
        active_jobs = self._active_runtime_jobs()
        if active_jobs:
            raise ApiError(HTTPStatus.CONFLICT, f"history cannot be removed while jobs are active: {', '.join(active_jobs)}")

        runs_root = self.runs_dir.resolve()
        batches_root = self.batches_dir.resolve()
        run_dirs = [
            path.resolve()
            for path in self.runs_dir.iterdir()
            if path.is_dir() and path.resolve() != runs_root and is_relative_to(path.resolve(), runs_root)
        ]
        batch_dirs = [
            path.resolve()
            for path in self.batches_dir.iterdir()
            if path.is_dir() and path.resolve() != batches_root and is_relative_to(path.resolve(), batches_root)
        ]
        removed_run_ids = sorted(path.name for path in run_dirs if (path / "run.json").exists())
        removed_batch_ids = sorted(path.name for path in batch_dirs if (path / "batch_summary.json").exists())

        with self.lock:
            active_jobs = self._active_runtime_jobs()
            if active_jobs:
                raise ApiError(HTTPStatus.CONFLICT, f"history cannot be removed while jobs are active: {', '.join(active_jobs)}")
            self.jobs.clear()
            self.batch_jobs.clear()
            self.artifacts = {
                key: value
                for key, value in self.artifacts.items()
                if not is_relative_to(value.resolve(), runs_root) and not is_relative_to(value.resolve(), batches_root)
            }

        for run_dir in run_dirs:
            shutil.rmtree(run_dir, ignore_errors=True)
        for batch_dir in batch_dirs:
            shutil.rmtree(batch_dir, ignore_errors=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.batches_dir.mkdir(parents=True, exist_ok=True)
        self.record_system_event(
            "warning",
            "history removed",
            removed_runs=len(removed_run_ids),
            removed_batches=len(removed_batch_ids),
        )
        return {
            "schema_version": "ore-pipeline-history-delete-v0.1",
            "removed_run_ids": removed_run_ids,
            "removed_batch_ids": removed_batch_ids,
            "runs": [],
            "batches": [],
        }

    def run_batch(self, batch_id: str, payload: dict[str, Any] | None = None, *, run_async: bool = True) -> dict[str, Any]:
        summary = self._read_batch(batch_id)
        if summary.get("status") in BATCH_ACTIVE_STATUSES:
            raise ApiError(HTTPStatus.CONFLICT, "batch is already running")
        if summary.get("status") != "draft":
            raise ApiError(HTTPStatus.CONFLICT, "only draft batches can be started")
        if not summary.get("items"):
            raise ApiError(HTTPStatus.BAD_REQUEST, "batch has no images")
        if payload:
            summary["settings"] = self._batch_settings_from_payload(payload)
        for item in summary["items"]:
            item["status"] = "queued"
            item["progress"] = 0
            item["stage"] = "queued"
            item["error"] = None
            item["updated_at"] = utc_now_iso()
        summary["status"] = "queued"
        summary["progress"] = 1
        summary["queued_at"] = utc_now_iso()
        summary["updated_at"] = utc_now_iso()
        self._finalize_batch_summary(summary)
        self._write_batch(summary)
        with self.lock:
            self.batch_jobs[batch_id] = {
                "status": "queued",
                "progress": 1,
                "cancel_requested": False,
                "active_run_id": None,
                "started_at": time.time(),
            }
        self.record_system_event("info", "series queued", batch_id=batch_id, item_count=len(summary.get("items", [])))
        if run_async:
            thread = threading.Thread(target=self._run_batch_guarded, args=(batch_id, True), daemon=True)
            thread.start()
        else:
            self._run_batch_guarded(batch_id, False)
        return self.batch_payload(batch_id)

    def cancel_batch(self, batch_id: str) -> dict[str, Any]:
        summary = self._read_batch(batch_id)
        active_run_id = None
        with self.lock:
            job = self.batch_jobs.get(batch_id)
            if job:
                job["status"] = "canceling"
                job["cancel_requested"] = True
                active_run_id = job.get("active_run_id")
                self.batch_jobs[batch_id] = job
        if active_run_id:
            self.cancel_run(str(active_run_id))
        if summary.get("status") not in BATCH_ACTIVE_STATUSES:
            return self.batch_payload(batch_id)
        summary["status"] = "canceling"
        summary["updated_at"] = utc_now_iso()
        for item in summary.get("items", []):
            if item.get("status") in {"draft", "queued"}:
                item["status"] = "canceled"
                item["stage"] = "canceled"
                item["updated_at"] = utc_now_iso()
            elif item.get("status") == "running":
                item["status"] = "canceling"
                item["stage"] = "canceling"
                item["updated_at"] = utc_now_iso()
        self._finalize_batch_summary(summary)
        self._write_batch(summary)
        self.record_system_event("warning", "series cancellation requested", batch_id=batch_id, active_run_id=active_run_id)
        return self.batch_payload(batch_id)

    def batch_payload(self, batch_id: str) -> dict[str, Any]:
        summary = self._read_batch(batch_id)
        items = [self._batch_item_payload(item) for item in summary.get("items", [])]
        payload = {**summary, "items": items}
        with self.lock:
            job = self.batch_jobs.get(batch_id)
        if job and payload.get("status") not in BATCH_TERMINAL_STATUSES:
            payload["status"] = job.get("status", payload.get("status"))
            payload["progress"] = job.get("progress", payload.get("progress", 0))
        if (self.batches_dir / batch_id / "reports/batch_results.csv").exists():
            payload.setdefault("downloads", {})["results_csv"] = f"/api/batches/{urllib.parse.quote(batch_id)}/results.csv"
        return payload

    def batch_results_csv_path(self, batch_id: str) -> Path:
        summary = self._read_batch(batch_id)
        path = self.batches_dir / batch_id / "reports/batch_results.csv"
        if not path.exists():
            self._write_batch_results_csv(summary)
        return path

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
        self.record_system_event("warning", "run cancellation requested", run_id=run_id, status=data.get("status"))
        return self.run_payload(run_id)

    def run_payload(self, run_id: str) -> dict[str, Any]:
        data = self._read_run(run_id)
        data = self._ensure_non_sulfide_display_layer(run_id, data)
        data = self._ensure_runtime_provenance(run_id, data)
        data = self._merge_active_run_job(run_id, data)
        raw_summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
        summary = add_artifact_summary_fields(raw_summary, self._artifact_mask_for_summary(run_id)) if raw_summary else {}
        scale = data.get("scale") or None
        metrics = metric_rows(summary, scale) if summary else data.get("metrics", [])
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
            "files": f"/api/runs/{urllib.parse.quote(run_id)}/files",
            "artifacts_zip": f"/api/runs/{urllib.parse.quote(run_id)}/artifacts.zip",
        }
        return {
            **data,
            "summary": summary,
            "metrics": metrics,
            "display": display_urls,
            "masks": masks,
            "downloads": downloads,
            "history": self.list_runs()["runs"],
        }

    def _merge_active_run_job(self, run_id: str, data: dict[str, Any]) -> dict[str, Any]:
        if str(data.get("status") or "").lower() in RUN_TERMINAL_STATUSES:
            return data
        with self.lock:
            job = self.jobs.get(run_id)
            if not job:
                return data
            return {**data, **job}

    def _ensure_non_sulfide_display_layer(self, run_id: str, data: dict[str, Any]) -> dict[str, Any]:
        if str(data.get("status") or "").lower() != "complete":
            return data
        display = data.get("display") if isinstance(data.get("display"), dict) else {}
        if display.get("non_sulfide_base"):
            return data
        run_dir = self.runs_dir / run_id
        required = [run_dir / "input/preprocessed.png", run_dir / "masks/sulfide_mask.png", run_dir / "masks/final_mask.png"]
        if not all(path.exists() for path in required):
            return data
        try:
            self._build_display_layers(
                run_dir,
                preprocessing_enabled=bool((data.get("preprocess") or {}).get("enabled", True)),
            )
            refreshed_display = json.loads((run_dir / "display/display.json").read_text(encoding="utf-8"))["layers"]
        except Exception as exc:  # noqa: BLE001 - compatibility layer should not break run loading.
            self.record_system_event("warning", "non-sulfide display regeneration failed", run_id=run_id, error=str(exc))
            return data
        data["display"] = refreshed_display
        self._write_json(run_dir / "run.json", data)
        return data

    def run_files_payload(self, run_id: str) -> dict[str, Any]:
        run_dir = self._existing_run_dir(run_id)
        self._ensure_runtime_provenance(run_id, self._read_run(run_id))
        files = []
        for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
            relative_path = path.relative_to(run_dir).as_posix()
            if relative_path == "reports/run_artifacts.zip":
                continue
            files.append(self._run_file_entry(path, relative_path))
        return {
            "run_id": run_id,
            "file_count": len(files),
            "total_size_bytes": sum(int(file["size_bytes"]) for file in files),
            "files": files,
            "downloads": {
                "artifacts_zip": f"/api/runs/{urllib.parse.quote(run_id)}/artifacts.zip",
            },
        }

    def run_zip_path(self, run_id: str) -> Path:
        run_dir = self._existing_run_dir(run_id)
        self._ensure_runtime_provenance(run_id, self._read_run(run_id))
        zip_path = run_dir / "reports/run_artifacts.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        files = []
        for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
            relative_path = path.relative_to(run_dir).as_posix()
            if relative_path == "reports/run_artifacts.zip":
                continue
            files.append((path, relative_path))
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path, relative_path in files:
                archive.write(path, relative_path)
        return zip_path

    def _existing_run_dir(self, run_id: str) -> Path:
        run_dir = (self.runs_dir / run_id).resolve()
        runs_root = self.runs_dir.resolve()
        if run_dir == runs_root or not is_relative_to(run_dir, runs_root):
            raise ApiError(HTTPStatus.BAD_REQUEST, "invalid run id")
        if not (run_dir / "run.json").exists():
            raise ApiError(HTTPStatus.NOT_FOUND, f"unknown run: {run_id}")
        return run_dir

    def _run_file_entry(self, path: Path, relative_path: str) -> dict[str, Any]:
        stat = path.stat()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        entry: dict[str, Any] = {
            "path": relative_path,
            "name": path.name,
            "size_bytes": int(stat.st_size),
            "content_type": content_type,
            "view_url": self.artifact_url(path),
            "is_image": False,
        }
        try:
            with Image.open(path) as image:
                entry.update(
                    {
                        "is_image": True,
                        "width": int(image.size[0]),
                        "height": int(image.size[1]),
                        "format": image.format or path.suffix.lower().lstrip("."),
                    }
                )
        except Exception:
            pass
        return entry

    def _artifact_mask_for_summary(self, run_id: str) -> np.ndarray | None:
        mask_path = self.runs_dir / run_id / "masks/artifact_mask.png"
        if not mask_path.exists():
            return None
        return np.asarray(Image.open(mask_path).convert("L"))

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
        scale = data.get("scale") or {}
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "metric",
                    "key",
                    "level",
                    "parent_key",
                    "denominator",
                    "value",
                    "percent",
                    "area_px",
                    "area_um2",
                    "area_mm2",
                    "microns_per_pixel",
                    "effective_microns_per_analysis_pixel",
                    "scale_source",
                    "scale_confidence",
                ],
            )
            writer.writeheader()
            raw_summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
            summary = add_artifact_summary_fields(raw_summary, self._artifact_mask_for_summary(run_id)) if raw_summary else {}
            metrics = metric_rows(summary, scale) if summary else data.get("metrics", [])
            for row in metrics:
                writer.writerow(
                    {
                        "metric": row["label"],
                        "key": row["key"],
                        "level": row.get("level", ""),
                        "parent_key": row.get("parent_key", ""),
                        "denominator": row.get("denominator", ""),
                        "value": row["value"],
                        "percent": "" if row.get("percent") is None else f"{float(row['percent']):.6f}",
                        "area_px": "" if row.get("area_px") is None else int(row["area_px"]),
                        "area_um2": "" if row.get("area_um2") is None else f"{float(row['area_um2']):.6f}",
                        "area_mm2": "" if row.get("area_mm2") is None else f"{float(row['area_mm2']):.9f}",
                        "microns_per_pixel": "" if not scale else f"{float(scale['microns_per_source_pixel']):.9f}",
                        "effective_microns_per_analysis_pixel": ""
                        if not scale
                        else f"{float(scale['effective_microns_per_analysis_pixel']):.9f}",
                        "scale_source": scale.get("scale_source", ""),
                        "scale_confidence": scale.get("scale_confidence", ""),
                    }
                )
        return path

    def pdf_report_path(self, run_id: str) -> Path:
        data = self._read_run(run_id)
        run_dir = self.runs_dir / run_id
        path = self.runs_dir / run_id / "reports/ore_report.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        pages = build_pdf_report_pages(data, run_dir)
        pages[0].save(path, "PDF", resolution=150.0, save_all=True, append_images=pages[1:])
        return path

    def _initialize_run_from_upload(
        self,
        run_id: str,
        run_dir: Path,
        upload: dict[str, Any],
        preset: dict[str, Any],
        *,
        curated_metadata: Any = None,
        batch_link: dict[str, Any] | None = None,
    ) -> None:
        input_dir = run_dir / "input"
        source_path = resolve_path(upload["original_path"])
        original_artifact = input_dir / "original_source" / Path(upload["original_path"]).name
        hardlink_or_copy(source_path, original_artifact)
        preprocessed_source = resolve_path(upload["preprocess"]["preprocessed_path"])
        preprocessed_path = input_dir / "preprocessed.png"
        shutil.copy2(preprocessed_source, preprocessed_path)
        preprocessed_full_source = upload["preprocess"].get("preprocessed_full_path")
        preprocessed_full_path = input_dir / "preprocessed_full.png"
        if preprocessed_full_source:
            shutil.copy2(resolve_path(preprocessed_full_source), preprocessed_full_path)
        original_for_analysis = downscaled_image(source_path, size=(upload["preprocess"]["width"], upload["preprocess"]["height"]))
        save_image(input_dir / "original_for_analysis.png", original_for_analysis)
        metadata = self._base_run_metadata(run_id, run_dir, upload["upload_id"], preset)
        metadata["input"]["original_source_path"] = upload["original_path"]
        metadata["input"]["original_artifact_path"] = str(original_artifact)
        if preprocessed_full_source:
            metadata["input"]["preprocessed_full_path"] = str(preprocessed_full_path)
        upload_augmentation = upload.get("augmentation") or {"enabled": False, "settings": default_augmentation_settings()}
        if upload_augmentation.get("enabled") and upload_augmentation.get("augmented_path"):
            augmented_source = resolve_path(upload_augmentation["augmented_path"])
            augmented_path = input_dir / "augmented.png"
            shutil.copy2(augmented_source, augmented_path)
            metadata["input"]["augmented_path"] = str(augmented_path)
            metadata["augmentation"] = {**upload_augmentation, "augmented_path": str(augmented_path)}
        else:
            metadata["augmentation"] = upload_augmentation
        upload_artifact = upload.get("artifact_mask") if isinstance(upload.get("artifact_mask"), dict) else None
        if upload_artifact and upload_artifact.get("mask_path"):
            artifact_source = resolve_path(upload_artifact["mask_path"])
            if artifact_source.exists():
                artifact_path = input_dir / "artifact_mask.png"
                shutil.copy2(artifact_source, artifact_path)
                metadata["input"]["artifact_mask_path"] = str(artifact_path)
        metadata["preprocess"]["enabled"] = bool((upload.get("preprocess") or {}).get("enabled", True))
        if batch_link:
            metadata["batch"] = json_safe_value(batch_link)
        self._attach_curated_metadata(metadata, run_dir, curated_metadata)
        metadata["tiling"] = upload.get("tiling") or (upload.get("preprocess") or {}).get("tiling") or {}
        self._write_json(run_dir / "run.json", metadata)

    def _base_run_metadata(self, run_id: str, run_dir: Path, upload_id: str, preset: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": "ore-pipeline-ui-run-v0.1",
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "status": "running",
            "progress": 0,
            "backend": self.backend,
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "runtime": self._initial_runtime_provenance(backend=self.backend, checkpoint=self.checkpoint),
            "input": {
                "upload_id": upload_id,
                "original_artifact_path": str(run_dir / "input/original_source"),
                "original_for_analysis_path": str(run_dir / "input/original_for_analysis.png"),
                "preprocessed_path": str(run_dir / "input/preprocessed.png"),
            },
            "preprocess": {"enabled": bool(preset.get("preprocessing_enabled", preset.get("enabled", True))), "preset": preset},
            "image": {},
            "summary": {},
            "metrics": [],
            "text_output": "",
            "display": {},
            "masks": {},
            "tiling": {},
            "augmentation": {"enabled": False, "settings": default_augmentation_settings()},
            "derivation": None,
        }

    def _attach_curated_metadata(self, run_metadata: dict[str, Any], run_dir: Path, curated_metadata: Any) -> None:
        normalized = normalize_curated_metadata_payload(curated_metadata)
        if not normalized:
            return
        metadata_path = run_dir / "metadata/curated_metadata.json"
        self._write_json(metadata_path, normalized)
        run_metadata.setdefault("input", {})["curated_metadata"] = normalized
        run_metadata["input"]["curated_metadata_json"] = str(metadata_path)

    def _run_job_guarded(self, run_id: str) -> None:
        try:
            self._run_job(run_id)
            self.record_system_event("info", "run complete", run_id=run_id)
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
            self.record_system_event("warning", "run canceled", run_id=run_id, progress=progress)
        except Exception as exc:  # noqa: BLE001 - keep server alive and expose failure.
            run_path = self.runs_dir / run_id / "run.json"
            data = json.loads(run_path.read_text(encoding="utf-8"))
            data["status"] = "failed"
            data["error"] = str(exc)
            data["progress"] = 100
            try:
                runtime = self._runtime_provenance_from_metadata(data, self.runs_dir / run_id)
                runtime["failed_at"] = utc_now_iso()
                data["runtime"] = runtime
                data["backend"] = runtime["backend"]
                data["checkpoint"] = runtime["checkpoints"].get("binary_sulfide")
                self._write_json(self.runs_dir / run_id / "reports/runtime.json", runtime)
                data.setdefault("reports", {})["runtime_json"] = str(self.runs_dir / run_id / "reports/runtime.json")
            except Exception as provenance_exc:  # noqa: BLE001 - preserve original failure.
                self.record_system_event(
                    "warning",
                    "runtime provenance write failed after run error",
                    run_id=run_id,
                    error=str(provenance_exc),
                )
            self._write_json(run_path, data)
            with self.lock:
                self.jobs[run_id] = {"status": "failed", "progress": 100, "error": str(exc), "eta_seconds": None}
            self.record_system_event("error", "run failed", run_id=run_id, error=str(exc))

    def _run_job(self, run_id: str) -> None:
        run_dir = self.runs_dir / run_id
        self._set_progress(run_id, 8, "preparing immutable run artifacts")
        self._check_cancelled(run_id)
        run_metadata = self._read_run(run_id)
        runtime = self._runtime_provenance_from_metadata(run_metadata, run_dir)
        runtime["started_at"] = utc_now_iso()
        run_metadata["runtime"] = runtime
        run_metadata["backend"] = runtime["backend"]
        run_metadata["checkpoint"] = runtime["checkpoints"].get("binary_sulfide")
        self._write_json(run_dir / "run.json", run_metadata)
        run_backend = runtime["backend"]
        run_checkpoint = runtime["checkpoints"].get("binary_sulfide") or run_metadata.get(
            "checkpoint",
            str(self.checkpoint) if self.checkpoint else None,
        )
        if run_backend == "ml":
            self._run_ml_backend(run_id, run_dir, checkpoint=run_checkpoint)
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
        artifact_mask = self._run_artifact_mask(run_dir, rgb.shape[:2])
        result = segment_image(rgb)
        self._check_cancelled(run_id)
        sulfide_mask = (result.sulfide_mask > 0).astype(np.uint8) * 255
        talc_mask = (result.talc_candidate_mask > 0).astype(np.uint8) * 255
        analyzed_mask = build_analyzed_mask(rgb)
        sulfide_mask, talc_mask, analyzed_mask, _ = apply_artifact_exclusion(
            artifact_mask=artifact_mask,
            sulfide_mask=sulfide_mask,
            talc_mask=talc_mask,
            analyzed_mask=analyzed_mask,
        )
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
            artifact_mask=artifact_mask,
        )

    def _run_ml_backend(self, run_id: str, run_dir: Path, *, checkpoint: str | Path | None = None) -> None:
        effective_checkpoint = Path(checkpoint).expanduser() if checkpoint else self.checkpoint
        if effective_checkpoint and not effective_checkpoint.is_absolute():
            effective_checkpoint = ROOT / effective_checkpoint
        effective_checkpoint = effective_checkpoint.resolve() if effective_checkpoint else None
        if effective_checkpoint is None or not effective_checkpoint.exists():
            raise ApiError(HTTPStatus.BAD_REQUEST, "ML backend requires --checkpoint")
        self._set_progress(run_id, 18, "running ML tiled inference")
        ml_dir = run_dir / "ml_pipeline"
        tile_progress_path = ml_dir / "binary_sulfide/progress.json"
        cmd = [
            sys.executable,
            str(ROOT / "scripts/run_ore_pipeline.py"),
            "--image",
            str(run_dir / "input/preprocessed.png"),
            "--checkpoint",
            str(effective_checkpoint),
            "--out-dir",
            str(ml_dir),
            "--auto-talc-candidate",
            "--preview-max-side",
            str(max(self.preview_max_sides)),
            "--progress-json",
            str(tile_progress_path),
        ]
        log_path = run_dir / "ml_pipeline.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                self._update_ml_tile_progress(run_id, tile_progress_path)
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
        self._update_ml_tile_progress(run_id, tile_progress_path)
        self._set_progress(run_id, 76, "collecting ML outputs")
        self._check_cancelled(run_id)
        ore_summary = json.loads((ml_dir / "ore_analysis/ore_summary.json").read_text(encoding="utf-8"))
        sulfide_mask = np.asarray(Image.open(ml_dir / "binary_sulfide/sulfide_mask.png").convert("L"))
        talc_path = ml_dir / "talc_candidate/talc_candidate_mask.png"
        talc_mask = np.asarray(Image.open(talc_path).convert("L")) if talc_path.exists() else np.zeros_like(sulfide_mask)
        intergrowth = np.asarray(Image.open(ml_dir / "ore_analysis/intergrowth_mask.png").convert("L"))
        analyzed_path = ml_dir / "ore_analysis/analyzed_mask.png"
        analyzed_mask = np.asarray(Image.open(analyzed_path).convert("L")) if analyzed_path.exists() else np.ones_like(sulfide_mask)
        artifact_mask = self._run_artifact_mask(run_dir, sulfide_mask.shape)
        sulfide_mask, talc_mask, analyzed_mask, intergrowth = apply_artifact_exclusion(
            artifact_mask=artifact_mask,
            sulfide_mask=sulfide_mask,
            talc_mask=talc_mask,
            analyzed_mask=analyzed_mask,
            final_mask=intergrowth,
        )
        summary, components, classified = analyze_components(
            sulfide_mask=sulfide_mask,
            talc_mask=talc_mask,
            analyzed_mask=analyzed_mask,
            config=ComponentRuleConfig(),
        )
        final_mask = final_mask_from_classified(classified, talc_mask)
        self._write_run_outputs(
            run_dir=run_dir,
            summary={**ore_summary, **asdict(summary)},
            components=components,
            sulfide_mask=sulfide_mask,
            talc_mask=talc_mask,
            analyzed_mask=analyzed_mask,
            final_mask=final_mask,
            artifact_mask=artifact_mask,
        )

    def _run_artifact_mask(self, run_dir: Path, expected_shape_hw: tuple[int, int]) -> np.ndarray | None:
        mask_path = run_dir / "input/artifact_mask.png"
        if not mask_path.exists():
            return None
        return read_binary_mask(mask_path, expected_shape_hw)

    def _write_masks_from_artifact_edit(self, parent_dir: Path, run_dir: Path, artifact_mask: np.ndarray) -> None:
        save_image(run_dir / "input/artifact_mask.png", Image.fromarray(artifact_mask, mode="L"))
        sulfide_mask = np.asarray(Image.open(parent_dir / "masks/sulfide_mask.png").convert("L"))
        talc_mask = np.asarray(Image.open(parent_dir / "masks/talc_mask.png").convert("L"))
        analyzed_mask = np.asarray(Image.open(parent_dir / "masks/analyzed_mask.png").convert("L"))
        final_mask = np.asarray(Image.open(parent_dir / "masks/final_mask.png").convert("L"))
        parent_metadata = json.loads((parent_dir / "run.json").read_text(encoding="utf-8"))
        sulfide_mask, talc_mask, analyzed_mask, final_mask = apply_artifact_exclusion(
            artifact_mask=artifact_mask,
            sulfide_mask=sulfide_mask,
            talc_mask=talc_mask,
            analyzed_mask=analyzed_mask,
            final_mask=final_mask,
        )
        summary = summary_from_final_edit(sulfide_mask, final_mask, analyzed_mask)
        self._write_run_outputs(
            run_dir=run_dir,
            summary=summary,
            components=[],
            sulfide_mask=sulfide_mask,
            talc_mask=talc_mask,
            analyzed_mask=analyzed_mask,
            final_mask=final_mask,
            artifact_mask=artifact_mask,
            preprocessing_enabled=bool((parent_metadata.get("preprocess") or {}).get("enabled", True)),
        )

    def _write_masks_from_sulfide_edit(self, parent_dir: Path, run_dir: Path, sulfide_mask: np.ndarray) -> None:
        rgb = np.asarray(Image.open(run_dir / "input/preprocessed.png").convert("RGB"))
        talc_mask = np.asarray(Image.open(parent_dir / "masks/talc_mask.png").convert("L"))
        analyzed_mask = np.asarray(Image.open(parent_dir / "masks/analyzed_mask.png").convert("L"))
        artifact_mask = self._run_artifact_mask(run_dir, sulfide_mask.shape)
        sulfide_mask, talc_mask, analyzed_mask, _ = apply_artifact_exclusion(
            artifact_mask=artifact_mask,
            sulfide_mask=sulfide_mask,
            talc_mask=talc_mask,
            analyzed_mask=analyzed_mask,
        )
        parent_metadata = json.loads((parent_dir / "run.json").read_text(encoding="utf-8"))
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
            artifact_mask=artifact_mask,
            preprocessing_enabled=bool((parent_metadata.get("preprocess") or {}).get("enabled", True)),
        )

    def _write_masks_from_final_edit(self, parent_dir: Path, run_dir: Path, final_mask: np.ndarray) -> None:
        sulfide_mask = np.asarray(Image.open(parent_dir / "masks/sulfide_mask.png").convert("L"))
        analyzed_mask = np.asarray(Image.open(parent_dir / "masks/analyzed_mask.png").convert("L"))
        parent_metadata = json.loads((parent_dir / "run.json").read_text(encoding="utf-8"))
        artifact_mask = self._run_artifact_mask(run_dir, final_mask.shape)
        if artifact_mask is not None:
            final_mask = final_mask.astype(np.uint8).copy()
            final_mask[artifact_mask > 0] = 0
        talc_mask = ((final_mask == 3).astype(np.uint8) * 255)
        sulfide_mask, talc_mask, analyzed_mask, final_mask = apply_artifact_exclusion(
            artifact_mask=artifact_mask,
            sulfide_mask=sulfide_mask,
            talc_mask=talc_mask,
            analyzed_mask=analyzed_mask,
            final_mask=final_mask,
        )
        summary = summary_from_final_edit(sulfide_mask, final_mask, analyzed_mask)
        self._write_run_outputs(
            run_dir=run_dir,
            summary=summary,
            components=[],
            sulfide_mask=sulfide_mask,
            talc_mask=talc_mask,
            analyzed_mask=analyzed_mask,
            final_mask=final_mask,
            artifact_mask=artifact_mask,
            preprocessing_enabled=bool((parent_metadata.get("preprocess") or {}).get("enabled", True)),
        )

    def _copy_run_inputs(self, parent: dict[str, Any], parent_dir: Path, run_dir: Path) -> None:
        for relative in [
            "input/original_for_analysis.png",
            "input/preprocessed.png",
            "input/preprocessed_full.png",
            "input/augmented.png",
            "input/artifact_mask.png",
        ]:
            if not (parent_dir / relative).exists():
                continue
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
        artifact_mask: np.ndarray | None = None,
        preprocessing_enabled: bool | None = None,
    ) -> None:
        masks_dir = run_dir / "masks"
        reports_dir = run_dir / "reports"
        masks_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        if artifact_mask is None:
            artifact_mask = self._run_artifact_mask(run_dir, sulfide_mask.shape)
        if artifact_mask is not None and artifact_mask.shape != sulfide_mask.shape:
            artifact_mask = read_binary_mask_from_array((artifact_mask > 0).astype(np.uint8) * 255, sulfide_mask.shape)
        summary = add_artifact_summary_fields(summary, artifact_mask, int(sulfide_mask.size))
        sulfide_mask, talc_mask, analyzed_mask, final_mask = apply_artifact_exclusion(
            artifact_mask=artifact_mask,
            sulfide_mask=sulfide_mask,
            talc_mask=talc_mask,
            analyzed_mask=analyzed_mask,
            final_mask=final_mask,
        )
        Image.fromarray((sulfide_mask > 0).astype(np.uint8) * 255, mode="L").save(masks_dir / "sulfide_mask.png")
        Image.fromarray((talc_mask > 0).astype(np.uint8) * 255, mode="L").save(masks_dir / "talc_mask.png")
        Image.fromarray((analyzed_mask > 0).astype(np.uint8) * 255, mode="L").save(masks_dir / "analyzed_mask.png")
        Image.fromarray(final_mask.astype(np.uint8), mode="L").save(masks_dir / "final_mask.png")
        if artifact_mask is not None:
            Image.fromarray((artifact_mask > 0).astype(np.uint8) * 255, mode="L").save(masks_dir / "artifact_mask.png")
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
        self._build_display_layers(run_dir, preprocessing_enabled=preprocessing_enabled)
        self._check_cancelled(run_id)

    def _build_display_layers(self, run_dir: Path, preprocessing_enabled: bool | None = None) -> None:
        display_dir = run_dir / "display"
        original = Image.open(run_dir / "input/original_for_analysis.png").convert("RGB")
        preprocessed = Image.open(run_dir / "input/preprocessed.png").convert("RGB")
        sulfide = np.asarray(Image.open(run_dir / "masks/sulfide_mask.png").convert("L"))
        final_mask = np.asarray(Image.open(run_dir / "masks/final_mask.png").convert("L"))
        analyzed_path = run_dir / "masks/analyzed_mask.png"
        analyzed_mask = np.asarray(Image.open(analyzed_path).convert("L")) if analyzed_path.exists() else None
        artifact_path = run_dir / "masks/artifact_mask.png"
        artifact_mask = np.asarray(Image.open(artifact_path).convert("L")) if artifact_path.exists() else None
        if preprocessing_enabled is None:
            run_metadata_path = run_dir / "run.json"
            if run_metadata_path.exists():
                run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
                preprocessing_enabled = bool((run_metadata.get("preprocess") or {}).get("enabled", True))
            else:
                preprocessing_enabled = True
        layers = {
            "original": save_preview_pyramid(original, display_dir / "original", "original", self.preview_max_sides),
            "non_sulfide_base": save_preview_pyramid(
                masked_rgb_layer(preprocessed, self._non_sulfide_display_mask(sulfide, analyzed_mask, artifact_mask)),
                display_dir / "non_sulfide_base",
                "non_sulfide_base",
                self.preview_max_sides,
                prefer_png=True,
            ),
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
        if artifact_mask is not None:
            layers["artifact_overlay"] = save_preview_pyramid(
                colored_overlay(artifact_mask, None, ARTIFACT_COLOR),
                display_dir / "artifact_overlay",
                "artifact_overlay",
                self.preview_max_sides,
                nearest=True,
                prefer_png=True,
            )
        augmented_path = run_dir / "input/augmented.png"
        if augmented_path.exists():
            layers = {
                "original": layers["original"],
                "augmented": save_preview_pyramid(
                    Image.open(augmented_path).convert("RGB"),
                    display_dir / "augmented",
                    "augmented",
                    self.preview_max_sides,
                ),
                **{key: value for key, value in layers.items() if key != "original"},
            }
        if preprocessing_enabled:
            layers["preprocessed"] = save_preview_pyramid(
                preprocessed,
                display_dir / "preprocessed",
                "preprocessed",
                self.preview_max_sides,
            )
        display_manifest = {"schema_version": "ore-pipeline-display-v0.1", "layers": layers}
        self._write_json(display_dir / "display.json", display_manifest)

    def _non_sulfide_display_mask(
        self,
        sulfide_mask: np.ndarray,
        analyzed_mask: np.ndarray | None,
        artifact_mask: np.ndarray | None,
    ) -> np.ndarray:
        active = sulfide_mask <= 0
        if analyzed_mask is not None:
            if analyzed_mask.shape != active.shape:
                analyzed_mask = read_binary_mask_from_array((analyzed_mask > 0).astype(np.uint8) * 255, active.shape)
            active &= analyzed_mask > 0
        if artifact_mask is not None:
            if artifact_mask.shape != active.shape:
                artifact_mask = read_binary_mask_from_array((artifact_mask > 0).astype(np.uint8) * 255, active.shape)
            active &= artifact_mask <= 0
        return active

    def _build_prepared_display_layers(self, run_dir: Path, *, preprocessing_enabled: bool) -> None:
        display_dir = run_dir / "display"
        shutil.rmtree(display_dir, ignore_errors=True)
        original = Image.open(run_dir / "input/original_for_analysis.png").convert("RGB")
        layers = {
            "original": save_preview_pyramid(original, display_dir / "original", "original", self.preview_max_sides),
        }
        augmented_path = run_dir / "input/augmented.png"
        if augmented_path.exists():
            layers["augmented"] = save_preview_pyramid(
                Image.open(augmented_path).convert("RGB"),
                display_dir / "augmented",
                "augmented",
                self.preview_max_sides,
            )
        if preprocessing_enabled:
            layers["preprocessed"] = save_preview_pyramid(
                Image.open(run_dir / "input/preprocessed.png").convert("RGB"),
                display_dir / "preprocessed",
                "preprocessed",
                self.preview_max_sides,
            )
        display_manifest = {"schema_version": "ore-pipeline-display-v0.1", "layers": layers}
        self._write_json(display_dir / "display.json", display_manifest)

    def _preserve_parent_artifact_mask(self, parent: dict[str, Any], run_dir: Path, metadata: dict[str, Any]) -> None:
        parent_input = parent.get("input") or {}
        candidates = [
            parent_input.get("artifact_mask_path"),
            self.runs_dir / str(parent.get("run_id") or "") / "masks/artifact_mask.png",
        ]
        source_path = None
        for candidate in candidates:
            if not candidate:
                continue
            path = resolve_path(candidate)
            if path.exists():
                source_path = path
                break
        if source_path is None:
            return
        with Image.open(run_dir / "input/preprocessed.png") as image:
            expected_shape = (image.size[1], image.size[0])
        artifact_mask = read_binary_mask(source_path, expected_shape)
        artifact_path = run_dir / "input/artifact_mask.png"
        save_image(artifact_path, Image.fromarray(artifact_mask, mode="L"))
        metadata.setdefault("input", {})["artifact_mask_path"] = str(artifact_path)

    def _finalize_prepared_run_metadata(
        self,
        metadata: dict[str, Any],
        run_dir: Path,
        *,
        preprocessing_enabled: bool,
    ) -> None:
        self._build_prepared_display_layers(run_dir, preprocessing_enabled=preprocessing_enabled)
        display = json.loads((run_dir / "display/display.json").read_text(encoding="utf-8"))["layers"]
        with Image.open(run_dir / "input/preprocessed.png") as image:
            metadata["image"] = {"width": image.size[0], "height": image.size[1], "name": Path(metadata["input"]["original_artifact_path"]).name}
        metadata["summary"] = {}
        metadata["metrics"] = []
        metadata["text_output"] = ""
        metadata["display"] = display
        metadata["masks"] = {}
        metadata["reports"] = {}
        metadata.pop("scale", None)

    def _finalize_run_metadata(self, metadata: dict[str, Any], run_dir: Path) -> None:
        summary = json.loads((run_dir / "reports/ore_summary.json").read_text(encoding="utf-8"))
        display = json.loads((run_dir / "display/display.json").read_text(encoding="utf-8"))["layers"]
        with Image.open(run_dir / "input/preprocessed.png") as image:
            metadata["image"] = {"width": image.size[0], "height": image.size[1], "name": Path(metadata["input"]["original_artifact_path"]).name}
        metadata["summary"] = summary
        scale = calibrated_scale_from_metadata(metadata, summary)
        if scale:
            metadata["scale"] = scale
        else:
            metadata.pop("scale", None)
        metadata["metrics"] = metric_rows(summary, scale)
        metadata["text_output"] = text_output_for_summary(summary)
        metadata["display"] = display
        metadata["masks"] = {
            "sulfide": str(run_dir / "masks/sulfide_mask.png"),
            "final": str(run_dir / "masks/final_mask.png"),
            "talc": str(run_dir / "masks/talc_mask.png"),
            "analyzed": str(run_dir / "masks/analyzed_mask.png"),
        }
        if (run_dir / "masks/artifact_mask.png").exists():
            metadata["masks"]["artifact"] = str(run_dir / "masks/artifact_mask.png")
            metadata.setdefault("input", {})["artifact_mask_path"] = str(run_dir / "input/artifact_mask.png")
        metadata["reports"] = {
            "summary_json": str(run_dir / "reports/ore_summary.json"),
            "component_features_csv": str(run_dir / "reports/component_features.csv"),
        }
        self._finalize_runtime_provenance(metadata, run_dir)

    def _cancel_requested(self, run_id: str) -> bool:
        with self.lock:
            return bool(self.jobs.get(run_id, {}).get("cancel_requested"))

    def _check_cancelled(self, run_id: str) -> None:
        if self._cancel_requested(run_id):
            raise RunCancelled()

    def _update_ml_tile_progress(self, run_id: str, progress_path: Path) -> None:
        try:
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return
        tiles_total = int(payload.get("tiles_total") or payload.get("tiles") or 0)
        tiles_processed = int(payload.get("tiles_processed") or 0)
        if tiles_total <= 0:
            return
        tiles_processed = max(0, min(tiles_processed, tiles_total))
        fraction = tiles_processed / max(tiles_total, 1)
        progress = min(74, max(18, 18 + int(round(fraction * 56))))
        tile_progress = {
            "schema_version": "ore-pipeline-tile-progress-v0.1",
            "stage": str(payload.get("stage") or "running"),
            "tiles_processed": tiles_processed,
            "tiles_total": tiles_total,
            "progress_fraction": fraction,
        }
        self._set_progress(
            run_id,
            progress,
            f"running ML tiled inference ({tiles_processed}/{tiles_total} tiles)",
            extra={"tile_progress": tile_progress},
        )

    def _set_progress(self, run_id: str, progress: int, status: str, *, extra: dict[str, Any] | None = None) -> None:
        extra = json_safe_value(extra or {})
        with self.lock:
            previous = self.jobs.get(run_id, {})
            started = previous.get("started_at", time.time())
            cancel_requested = bool(previous.get("cancel_requested"))
            elapsed = max(0.0, time.time() - float(started))
            eta = None
            if progress > 1:
                eta = max(0, int(elapsed * (100 - progress) / max(progress, 1)))
            job_payload = {
                "progress": progress,
                "status": "canceling" if cancel_requested else "running",
                "stage": "canceling" if cancel_requested else status,
                "started_at": started,
                "eta_seconds": None if cancel_requested else eta,
                "cancel_requested": cancel_requested,
            }
            if isinstance(extra, dict):
                job_payload.update(extra)
            self.jobs[run_id] = job_payload
        run_path = self.runs_dir / run_id / "run.json"
        if run_path.exists():
            data = json.loads(run_path.read_text(encoding="utf-8"))
            data["progress"] = progress
            data["status"] = "canceling" if cancel_requested else "running"
            data["stage"] = "canceling" if cancel_requested else status
            data["eta_seconds"] = None if cancel_requested else eta
            if isinstance(extra, dict):
                data.update(extra)
            self._write_json(run_path, data)

    def _run_batch_guarded(self, batch_id: str, child_run_async: bool = True) -> None:
        try:
            self._run_batch(batch_id, child_run_async=child_run_async)
        except Exception as exc:  # noqa: BLE001 - keep server alive and expose batch failure.
            summary = self._read_batch(batch_id)
            summary["status"] = "failed"
            summary["error"] = str(exc)
            summary["progress"] = 100
            summary["completed_at"] = utc_now_iso()
            summary["updated_at"] = utc_now_iso()
            for item in summary.get("items", []):
                if item.get("status") in {"queued", "running", "canceling"}:
                    item["status"] = "failed"
                    item["stage"] = "failed"
                    item["error"] = str(exc)
                    item["progress"] = 100
                    item["updated_at"] = utc_now_iso()
            self._finalize_batch_summary(summary)
            self._write_batch(summary)
            with self.lock:
                self.batch_jobs[batch_id] = {"status": "failed", "progress": 100, "error": str(exc), "cancel_requested": False}
            self.record_system_event("error", "series failed", batch_id=batch_id, error=str(exc))

    def _run_batch(self, batch_id: str, *, child_run_async: bool = True) -> None:
        summary = self._read_batch(batch_id)
        settings = summary.get("settings") or self._batch_settings_from_payload({})
        summary["status"] = "running"
        summary["started_at"] = utc_now_iso()
        summary["updated_at"] = utc_now_iso()
        self._write_batch(summary)
        with self.lock:
            job = self.batch_jobs.get(batch_id, {})
            job.update({"status": "running", "progress": 1})
            self.batch_jobs[batch_id] = job

        for item in summary.get("items", []):
            if self._batch_cancel_requested(batch_id):
                break
            if item.get("status") not in {"queued", "draft"}:
                continue
            item["status"] = "running"
            item["progress"] = 1
            item["stage"] = "queued"
            item["started_at"] = utc_now_iso()
            item["updated_at"] = utc_now_iso()
            summary["current_item_id"] = item["item_id"]
            summary["status"] = "running"
            self._finalize_batch_summary(summary)
            self._write_batch(summary)
            try:
                run = self.start_run(
                    str(item["upload_id"]),
                    settings.get("preprocess") or {},
                    run_async=child_run_async,
                    curated_metadata=item.get("curated_metadata"),
                    augmentation_settings=settings.get("augmentation"),
                    batch_link={
                        "batch_id": batch_id,
                        "item_id": item["item_id"],
                        "index": item["index"],
                    },
                )
                run_id = run["run_id"]
                item["run_id"] = run_id
                item["links"] = {"load_run": f"/api/runs/{urllib.parse.quote(run_id)}"}
                item["updated_at"] = utc_now_iso()
                with self.lock:
                    job = self.batch_jobs.get(batch_id, {})
                    job["active_run_id"] = run_id
                    self.batch_jobs[batch_id] = job

                while True:
                    child = self.run_payload(run_id)
                    child_status = str(child.get("status") or "running")
                    if self._batch_cancel_requested(batch_id) and child_status in BATCH_ACTIVE_STATUSES:
                        self.cancel_run(run_id)
                        child = self.run_payload(run_id)
                        child_status = str(child.get("status") or "canceling")
                    item["progress"] = int(child.get("progress", item.get("progress", 0)) or 0)
                    item["stage"] = child.get("stage") or child_status
                    item["status"] = "running" if child_status in {"queued", "running"} else child_status
                    item["updated_at"] = utc_now_iso()
                    if child.get("error"):
                        item["error"] = child.get("error")
                    self._finalize_batch_summary(summary)
                    self._write_batch(summary)
                    with self.lock:
                        job = self.batch_jobs.get(batch_id, {})
                        job.update({"status": summary.get("status", "running"), "progress": summary.get("progress", 0)})
                        self.batch_jobs[batch_id] = job
                    if child_status not in BATCH_ACTIVE_STATUSES:
                        break
                    time.sleep(0.25)

                if child_status == "complete":
                    item["status"] = "complete"
                    item["stage"] = "complete"
                    item["progress"] = 100
                    item["completed_at"] = utc_now_iso()
                elif child_status == "canceled":
                    item["status"] = "canceled"
                    item["stage"] = "canceled"
                    item["canceled_at"] = utc_now_iso()
                else:
                    item["status"] = "failed"
                    item["stage"] = "failed"
                    item["progress"] = 100
                    item["error"] = item.get("error") or child.get("error") or "run failed"
                    item["completed_at"] = utc_now_iso()
            except Exception as exc:  # noqa: BLE001 - a failed item should not hide completed siblings.
                item["status"] = "failed"
                item["stage"] = "failed"
                item["progress"] = 100
                item["error"] = str(exc)
                item["completed_at"] = utc_now_iso()
            finally:
                item["updated_at"] = utc_now_iso()
                with self.lock:
                    job = self.batch_jobs.get(batch_id, {})
                    job["active_run_id"] = None
                    self.batch_jobs[batch_id] = job
                self._finalize_batch_summary(summary)
                self._write_batch(summary)

        for item in summary.get("items", []):
            if item.get("status") in {"draft", "queued", "canceling"}:
                item["status"] = "canceled"
                item["stage"] = "canceled"
                item["updated_at"] = utc_now_iso()
        summary["current_item_id"] = None
        summary["status"] = self._batch_status_from_items(summary.get("items", []))
        summary["progress"] = 100
        summary["completed_at"] = utc_now_iso()
        summary["updated_at"] = utc_now_iso()
        self._finalize_batch_summary(summary)
        self._write_batch_results_csv(summary)
        self._write_batch(summary)
        with self.lock:
            self.batch_jobs[batch_id] = {"status": summary["status"], "progress": 100, "cancel_requested": False, "active_run_id": None}
        self.record_system_event("info", "series complete", batch_id=batch_id, status=summary["status"])

    def _batch_cancel_requested(self, batch_id: str) -> bool:
        with self.lock:
            return bool(self.batch_jobs.get(batch_id, {}).get("cancel_requested"))

    def _batch_settings_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        values = payload.get("settings") if isinstance(payload.get("settings"), dict) else payload
        values = values if isinstance(values, dict) else {}
        preprocess_payload = values.get("preprocess") if isinstance(values.get("preprocess"), dict) else values
        augmentation_payload = values.get("augmentation") if isinstance(values.get("augmentation"), dict) else values
        return {
            "schema_version": "ore-pipeline-batch-settings-v0.1",
            "preprocess": preset_from_payload(preprocess_payload if isinstance(preprocess_payload, dict) else {}),
            "augmentation": normalize_augmentation_settings(augmentation_payload),
            "backend": self.backend,
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
        }

    def _batch_upload_refs(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        upload_id = payload.get("upload_id")
        if upload_id:
            refs.append({"upload_id": str(upload_id), "curated_metadata": payload.get("curated_metadata")})
        upload_ids = payload.get("upload_ids")
        if isinstance(upload_ids, list):
            refs.extend({"upload_id": str(item), "curated_metadata": None} for item in upload_ids if item)
        uploads = payload.get("uploads")
        if isinstance(uploads, list):
            for item in uploads:
                if isinstance(item, dict) and item.get("upload_id"):
                    refs.append({"upload_id": str(item["upload_id"]), "curated_metadata": item.get("curated_metadata")})
                elif item:
                    refs.append({"upload_id": str(item), "curated_metadata": None})
        return refs

    def _batch_item_payload(self, item: dict[str, Any]) -> dict[str, Any]:
        payload = {**item}
        try:
            upload = self.upload_payload(str(item["upload_id"]))
        except ApiError as exc:
            payload["upload_error"] = exc.message
            upload = {}
        if upload:
            payload["upload"] = upload
            payload["display"] = upload.get("display") or {}
            payload["raw_metadata"] = upload.get("raw_metadata") or {}
        if item.get("run_id"):
            payload.setdefault("links", {})["load_run"] = f"/api/runs/{urllib.parse.quote(str(item['run_id']))}"
        return payload

    def _batch_item(self, summary: dict[str, Any], item_id: str) -> dict[str, Any]:
        for item in summary.get("items", []):
            if item.get("item_id") == item_id:
                return item
        raise ApiError(HTTPStatus.NOT_FOUND, f"unknown batch item: {item_id}")

    def _batch_status_from_items(self, items: list[dict[str, Any]]) -> str:
        statuses = [str(item.get("status") or "draft") for item in items]
        if not statuses:
            return "draft"
        if any(status in BATCH_ACTIVE_STATUSES for status in statuses):
            return "running"
        if all(status == "complete" for status in statuses):
            return "complete"
        if all(status == "canceled" for status in statuses):
            return "canceled"
        if all(status == "failed" for status in statuses):
            return "failed"
        if any(status == "complete" for status in statuses):
            return "partial"
        if any(status in {"failed", "canceled"} for status in statuses):
            return "partial"
        return "draft"

    def _finalize_batch_summary(self, summary: dict[str, Any]) -> None:
        items = summary.get("items") or []
        counts: dict[str, int] = {}
        progress_total = 0
        for item in items:
            status = str(item.get("status") or "draft")
            counts[status] = counts.get(status, 0) + 1
            progress_total += int(item.get("progress", 0) or 0)
        summary["item_counts"] = counts
        if items:
            summary["progress"] = max(int(summary.get("progress", 0) or 0), int(progress_total / len(items)))
        else:
            summary["progress"] = 0
        if summary.get("status") in BATCH_TERMINAL_STATUSES:
            summary["progress"] = 100

    def _write_batch_results_csv(self, summary: dict[str, Any]) -> None:
        path = self.batches_dir / summary["batch_id"] / "reports/batch_results.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "batch_id",
                    "item_id",
                    "index",
                    "original_name",
                    "status",
                    "run_id",
                    "ore_class",
                    "talc_fraction",
                    "ordinary_sulfide_fraction",
                    "fine_sulfide_fraction",
                    "error",
                ],
            )
            writer.writeheader()
            for item in summary.get("items", []):
                run_summary = {}
                if item.get("run_id"):
                    try:
                        run_summary = (self._read_run(str(item["run_id"])).get("summary") or {})
                    except ApiError:
                        run_summary = {}
                writer.writerow(
                    {
                        "batch_id": summary["batch_id"],
                        "item_id": item.get("item_id"),
                        "index": item.get("index"),
                        "original_name": item.get("original_name"),
                        "status": item.get("status"),
                        "run_id": item.get("run_id") or "",
                        "ore_class": run_summary.get("ore_class", ""),
                        "talc_fraction": run_summary.get("talc_fraction", ""),
                        "ordinary_sulfide_fraction": run_summary.get("ordinary_sulfide_fraction", ""),
                        "fine_sulfide_fraction": run_summary.get("fine_sulfide_fraction", ""),
                        "error": item.get("error") or "",
                    }
                )

    def _batch_summary_path(self, batch_id: str) -> Path:
        batch_dir = (self.batches_dir / batch_id).resolve()
        batches_root = self.batches_dir.resolve()
        if batch_dir == batches_root or not is_relative_to(batch_dir, batches_root):
            raise ApiError(HTTPStatus.BAD_REQUEST, "invalid batch id")
        return batch_dir / "batch_summary.json"

    def _read_batch(self, batch_id: str) -> dict[str, Any]:
        path = self._batch_summary_path(batch_id)
        if not path.exists():
            raise ApiError(HTTPStatus.NOT_FOUND, f"unknown batch: {batch_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_batch(self, summary: dict[str, Any]) -> None:
        self._write_json(self._batch_summary_path(str(summary["batch_id"])), summary)

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
        tmp_path = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(path)


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


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0])


def wrap_text_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    *,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    for raw_line in str(text).splitlines() or [""]:
        current = ""
        for word in raw_line.split(" "):
            candidate = word if not current else f"{current} {word}"
            if text_width(draw, candidate, font) <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
                current = word
                if text_width(draw, current, font) <= max_width:
                    continue
            chunks: list[str] = []
            chunk = ""
            for character in word:
                candidate = f"{chunk}{character}"
                if not chunk or text_width(draw, candidate, font) <= max_width:
                    chunk = candidate
                else:
                    chunks.append(chunk)
                    chunk = character
            if chunk:
                chunks.append(chunk)
            if chunks:
                lines.extend(chunks[:-1])
                current = chunks[-1]
        lines.append(current)
    return lines


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    fill: tuple[int, int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    line_spacing: int = 8,
) -> int:
    x, y = xy
    line_bbox = draw.textbbox((0, 0), "Ag", font=font)
    line_height = int(line_bbox[3] - line_bbox[1])
    for line in wrap_text_lines(draw, text, font, max_width=max_width):
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height + line_spacing
    return y - line_spacing


def new_report_page(title: str, page_no: int) -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
    page = Image.new("RGB", REPORT_PAGE_SIZE, "white")
    draw = ImageDraw.Draw(page)
    title_font = load_font(34)
    y = REPORT_MARGIN_TOP
    draw.text((REPORT_MARGIN_X, y), title, fill=REPORT_TEXT, font=title_font)
    draw.line((REPORT_MARGIN_X, 128, REPORT_PAGE_SIZE[0] - REPORT_MARGIN_X, 128), fill=REPORT_LINE, width=2)
    return page, draw, 158


def add_report_footers(pages: list[Image.Image]) -> None:
    footer_font = load_font(18)
    total = len(pages)
    for index, page in enumerate(pages, start=1):
        draw = ImageDraw.Draw(page)
        y = REPORT_PAGE_SIZE[1] - 58
        draw.line((REPORT_MARGIN_X, y - 18, REPORT_PAGE_SIZE[0] - REPORT_MARGIN_X, y - 18), fill=REPORT_LINE, width=1)
        draw.text(
            (REPORT_MARGIN_X, y),
            "Отчет сформирован автоматически. Не является аккредитованным протоколом испытаний.",
            fill=REPORT_MUTED,
            font=footer_font,
        )
        page_text = f"Страница {index} из {total}"
        draw.text(
            (REPORT_PAGE_SIZE[0] - REPORT_MARGIN_X - text_width(draw, page_text, footer_font), y),
            page_text,
            fill=REPORT_MUTED,
            font=footer_font,
        )


def fit_report_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    fitted = image.convert("RGB").copy()
    fitted.thumbnail((int(max_width), int(max_height)), Image.Resampling.BILINEAR)
    return fitted


def draw_image_card(
    page: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    image: Image.Image,
    caption: str = "",
) -> int:
    title_font = load_font(24)
    small_font = load_font(18)
    draw.text((x, y), title, fill=REPORT_TEXT, font=title_font)
    image_top = y + 40
    caption_height = 62 if caption else 0
    image_box_height = max(120, height - 48 - caption_height)
    draw.rectangle(
        (x, image_top, x + width, image_top + image_box_height),
        outline=REPORT_LINE,
        width=2,
    )
    fitted = fit_report_image(image, width - 18, image_box_height - 18)
    paste_x = x + (width - fitted.size[0]) // 2
    paste_y = image_top + (image_box_height - fitted.size[1]) // 2
    page.paste(fitted, (paste_x, paste_y))
    bottom = image_top + image_box_height
    if caption:
        bottom = draw_wrapped_text(
            draw,
            (x, bottom + 12),
            caption,
            fill=REPORT_MUTED,
            font=small_font,
            max_width=width,
            line_spacing=5,
        )
    return bottom


def draw_report_legend(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    entries: list[tuple[str, tuple[int, int, int]]],
) -> int:
    font = load_font(18)
    cursor_x = x
    cursor_y = y
    for label, color in entries:
        label_width = text_width(draw, label, font)
        if cursor_x + 26 + label_width > REPORT_PAGE_SIZE[0] - REPORT_MARGIN_X:
            cursor_x = x
            cursor_y += 34
        draw.rectangle((cursor_x, cursor_y + 4, cursor_x + 18, cursor_y + 22), fill=color, outline=REPORT_LINE)
        draw.text((cursor_x + 26, cursor_y), label, fill=REPORT_MUTED, font=font)
        cursor_x += 46 + label_width
    return cursor_y + 30


def report_image_from_path(path: Path, fallback_size: tuple[int, int] = (640, 480)) -> Image.Image:
    if path.exists():
        return Image.open(path).convert("RGB")
    return Image.new("RGB", fallback_size, REPORT_MASK_BACKGROUND)


def report_mask_array(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if size and mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask)


def sulfide_non_sulfide_image(sulfide_mask_path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    mask = report_mask_array(sulfide_mask_path, size=size) > 0
    image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    image[:, :] = np.array(REPORT_NON_SULFIDE_COLOR, dtype=np.uint8)
    image[mask] = np.array(REPORT_SULFIDE_COLOR, dtype=np.uint8)
    return Image.fromarray(image, mode="RGB")


def class_mask_image(final_mask: np.ndarray, class_id: int, color: tuple[int, int, int]) -> Image.Image:
    image = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
    image[:, :] = np.array(REPORT_MASK_BACKGROUND, dtype=np.uint8)
    image[final_mask == class_id] = np.array(color, dtype=np.uint8)
    return Image.fromarray(image, mode="RGB")


def final_overlay_image(
    base: Image.Image,
    final_mask: np.ndarray,
    *,
    class_ids: set[int] | None = None,
    artifact_mask: np.ndarray | None = None,
) -> Image.Image:
    base_rgba = base.convert("RGBA")
    if final_mask.shape != (base_rgba.size[1], base_rgba.size[0]):
        resized = Image.fromarray(final_mask.astype(np.uint8), mode="L").resize(base_rgba.size, Image.Resampling.NEAREST)
        final_mask = np.asarray(resized)
    overlay = np.zeros((final_mask.shape[0], final_mask.shape[1], 4), dtype=np.uint8)
    allowed = class_ids or {1, 2, 3}
    for class_id, rgba in CLASS_COLORS.items():
        if class_id in allowed:
            overlay[final_mask == class_id] = np.array(rgba, dtype=np.uint8)
    if artifact_mask is not None and not class_ids:
        if artifact_mask.shape != final_mask.shape:
            artifact_mask = np.asarray(
                Image.fromarray(artifact_mask.astype(np.uint8), mode="L").resize(base_rgba.size, Image.Resampling.NEAREST)
            )
        overlay[artifact_mask > 0] = np.array(ARTIFACT_COLOR, dtype=np.uint8)
    return Image.alpha_composite(base_rgba, Image.fromarray(overlay, mode="RGBA")).convert("RGB")


def report_preprocess_lines(data: dict[str, Any]) -> list[str]:
    preprocess = data.get("preprocess") if isinstance(data.get("preprocess"), dict) else {}
    preset = preprocess.get("preset") if isinstance(preprocess.get("preset"), dict) else {}
    tiling = data.get("tiling") if isinstance(data.get("tiling"), dict) else {}
    enabled = bool(preprocess.get("enabled", preset.get("preprocessing_enabled", preset.get("enabled", True))))
    lines = [f"Предобработка: {'включена' if enabled else 'отключена'}."]
    if enabled:
        lines.append(f"Нормализация освещения: {'да' if preset.get('illumination_normalization') else 'нет'}.")
        lines.append(f"Подавление шума: {'да' if preset.get('denoise') else 'нет'}.")
        lines.append(f"Коррекция контраста: {'да' if preset.get('contrast_correction') else 'нет'}.")
    else:
        lines.append("Для анализа использована масштабированная копия исходного изображения без фильтров предобработки.")
    if preset.get("panorama_scaling"):
        mode = str(preset.get("panorama_scaling_mode") or PANORAMA_SCALING_MODE_MAX_SIDE)
        if mode == PANORAMA_SCALING_MODE_SCALE_FACTOR:
            value = preset.get("panorama_scale_factor", DEFAULT_PANORAMA_SCALE_FACTOR)
            lines.append(f"Масштабирование панорамы: коэффициент {float(value):.2f}.")
        else:
            value = preset.get("panorama_max_side_px")
            suffix = f" до {int(value)} px по длинной стороне" if value else ""
            lines.append(f"Масштабирование панорамы: включено{suffix}.")
    else:
        lines.append("Масштабирование панорамы: отключено.")
    image = data.get("image") if isinstance(data.get("image"), dict) else {}
    if image.get("width") and image.get("height"):
        lines.append(f"Размер анализа: {int(image['width'])} x {int(image['height'])} px.")
    if tiling.get("tile_count"):
        lines.append(f"Тайлинг: {int(tiling['tile_count'])} плиток.")
    return lines


def report_domain(data: dict[str, Any]) -> dict[str, Any]:
    curated = (data.get("input") or {}).get("curated_metadata") if isinstance(data.get("input"), dict) else {}
    return curated.get("domain") if isinstance(curated, dict) and isinstance(curated.get("domain"), dict) else {}


def report_field(value: Any, default: str = "не указано") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def report_source_image_name(data: dict[str, Any]) -> str:
    input_data = data.get("input") if isinstance(data.get("input"), dict) else {}
    for key in ("original_artifact_path", "original_source_path", "preprocessed_path"):
        value = input_data.get(key)
        if value:
            path = Path(str(value))
            if path.name:
                return path.name
    image = data.get("image") if isinstance(data.get("image"), dict) else {}
    return report_field(image.get("name"))


def report_scale_text(data: dict[str, Any]) -> str:
    scale = data.get("scale") if isinstance(data.get("scale"), dict) else None
    if scale:
        return (
            f"{float(scale['microns_per_source_pixel']):.6g} мкм/px; "
            f"источник: {report_field(scale.get('scale_source'))}; "
            f"статус: {report_field(scale.get('scale_confidence'))}"
        )
    return "не задана; физические площади не рассчитываются"


def report_algorithm_text(data: dict[str, Any]) -> str:
    backend = report_field(data.get("backend"), "heuristic")
    if backend == "ml":
        return "ML-сегментация с подключенным checkpoint; компонентные правила классификации"
    return "эвристическая сегментация + компонентные правила классификации"


def report_review_status(data: dict[str, Any], summary: dict[str, Any]) -> str:
    domain = report_domain(data)
    if domain.get("review_status"):
        return report_field(domain.get("review_status"))
    if summary.get("needs_expert_review"):
        return "автоматическое заключение; требуется экспертная проверка"
    return "автоматическое заключение; экспертная проверка не выполнена"


def report_passport_rows(data: dict[str, Any], summary: dict[str, Any]) -> list[tuple[str, str]]:
    domain = report_domain(data)
    image = data.get("image") if isinstance(data.get("image"), dict) else {}
    source_context = " / ".join(
        part
        for part in [
            report_field(domain.get("deposit"), ""),
            report_field(domain.get("area"), ""),
            report_field(domain.get("task_label") or domain.get("ore_type"), ""),
        ]
        if part
    )
    return [
        ("Номер отчета / run_id", report_field(data.get("run_id"))),
        ("Дата формирования", report_field(data.get("completed_at") or data.get("created_at"))),
        ("Образец / sample_id", report_field(domain.get("sample_id") or domain.get("run_label"))),
        ("Источник изображения", report_source_image_name(data)),
        ("Размер анализа", f"{int(image.get('width') or 0)} x {int(image.get('height') or 0)} px"),
        ("Тип препарата", report_field(domain.get("preparation_type"), "полированный шлиф / аншлиф")),
        ("Месторождение / участок / тип руды", report_field(source_context)),
        ("Метод", "OM, автоматизированный анализ изображения"),
        ("Модель / версия / параметры", report_algorithm_text(data)),
        ("Масштаб / калибровка", report_scale_text(data)),
        ("Статус экспертной проверки", report_review_status(data, summary)),
    ]


def draw_key_value_table(
    draw: ImageDraw.ImageDraw,
    rows: list[tuple[str, str]],
    *,
    x: int,
    y: int,
    width: int,
) -> int:
    key_font = load_font(19)
    value_font = load_font(19)
    key_width = 360
    row_height = 43
    for index, (key, value) in enumerate(rows):
        fill = REPORT_TABLE_ALT if index % 2 else (255, 255, 255)
        draw.rectangle((x, y, x + width, y + row_height), fill=fill, outline=REPORT_LINE)
        draw.text((x + 12, y + 10), key, fill=REPORT_MUTED, font=key_font)
        draw_wrapped_text(
            draw,
            (x + key_width + 12, y + 10),
            value,
            fill=REPORT_TEXT,
            font=value_font,
            max_width=width - key_width - 26,
            line_spacing=3,
        )
        y += row_height
    return y


def report_mineralogical_conclusion(summary: dict[str, Any]) -> str:
    ore_class = report_field(summary.get("ore_class_ru"), "тип руды не определен")
    talc_pct = float(summary.get("talc_fraction") or 0.0) * 100.0
    ordinary_pct = float(summary.get("ordinary_sulfide_fraction") or 0.0) * 100.0
    fine_pct = float(summary.get("fine_sulfide_fraction") or 0.0) * 100.0
    sulfide_pct = float(summary.get("sulfide_fraction") or 0.0) * 100.0
    return (
        f"Вещественный/минералогический вывод: изображение отнесено к классу \"{ore_class}\". "
        f"В анализируемой области: сульфидные включения {sulfide_pct:.1f}%, "
        f"тальковая зона {talc_pct:.1f}%; среди сульфидов обычные срастания {ordinary_pct:.1f}%, "
        f"тонкие срастания {fine_pct:.1f}%. Вывод основан на OM RGB-изображении и масках автоматического анализа."
    )


def report_method_lines(data: dict[str, Any]) -> list[str]:
    lines = [
        "Исходные данные: цифровое OM-изображение полированного шлифа.",
        "Предобработка: " + " ".join(report_preprocess_lines(data)),
        "Сегментация: выделение сульфидной/несульфидной области и итоговых классов.",
        "Классификация: компонентный анализ сульфидных включений и детерминированное правило типа руды.",
        "Цвета классов: зеленый - обычные срастания, красный - тонкие срастания, синий - тальк.",
    ]
    return lines


def report_qc_lines(data: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    lines = [
        "Документ является демонстрационным автоматическим отчетом, не аккредитованным протоколом испытаний.",
        "Химическое подтверждение EDS/WDS/XRF не выполнялось и не заявляется.",
    ]
    if not data.get("scale"):
        lines.append("Калиброванный масштаб не задан: физические площади не рассчитываются.")
    warnings = summary.get("warnings") if isinstance(summary.get("warnings"), list) else []
    if warnings:
        lines.append("Предупреждения алгоритма: " + "; ".join(str(item) for item in warnings))
    talc_margin = summary.get("talc_margin")
    if talc_margin is not None and abs(float(talc_margin)) < 0.03:
        lines.append("Доля талька близка к порогу классификации; рекомендуется экспертная проверка.")
    if summary.get("needs_expert_review"):
        lines.append("Алгоритм выставил флаг необходимости экспертной проверки.")
    else:
        lines.append("Флаг обязательной экспертной проверки алгоритмом не выставлен; ручная проверка все равно не выполнялась.")
    return lines


def report_artifact_lines(data: dict[str, Any]) -> list[str]:
    run_id = report_field(data.get("run_id"))
    return [
        f"Идентификатор запуска: {run_id}",
        "В составе run-артефактов: run.json, reports/metrics.csv, reports/ore_summary.json, маски PNG и изображения предпросмотра.",
        "В UI доступны View files и Download ZIP для проверки воспроизводимости.",
    ]


def report_denominator_label(value: str) -> str:
    return {
        "image": "все изображение",
        "analyzed_area": "анализируемая область",
        "sulfides": "сульфиды",
    }.get(value, value or "")


def report_metric_value(row: dict[str, Any]) -> str:
    if row.get("percent") is not None:
        return f"{float(row['percent']):.1f}%"
    return str(row.get("value", ""))


def report_metric_area(row: dict[str, Any]) -> str:
    parts: list[str] = []
    if row.get("area_px") is not None:
        parts.append(f"{int(row['area_px'])} px")
    if row.get("area_um2") is not None:
        parts.append(f"{float(row['area_um2']):.1f} мкм²")
    return "; ".join(parts)


def draw_metrics_table(
    draw: ImageDraw.ImageDraw,
    rows: list[dict[str, Any]],
    *,
    x: int,
    y: int,
    width: int,
) -> int:
    header_font = load_font(20)
    body_font = load_font(19)
    columns = [440, 160, 220, width - 820]
    headers = ["Показатель", "Доля/значение", "Площадь", "База расчета"]
    row_height = 50
    draw.rectangle((x, y, x + width, y + row_height), fill=REPORT_TABLE_HEADER, outline=REPORT_LINE)
    cursor_x = x
    for header, column_width in zip(headers, columns):
        draw.text((cursor_x + 12, y + 13), header, fill=REPORT_TEXT, font=header_font)
        cursor_x += column_width
    y += row_height
    for index, row in enumerate(rows):
        fill = REPORT_TABLE_ALT if index % 2 else (255, 255, 255)
        draw.rectangle((x, y, x + width, y + row_height), fill=fill, outline=REPORT_LINE)
        values = [
            "  " * int(row.get("level") or 0) + str(row.get("label") or row.get("key") or ""),
            report_metric_value(row),
            report_metric_area(row),
            report_denominator_label(str(row.get("denominator") or "")),
        ]
        cursor_x = x
        for value, column_width in zip(values, columns):
            draw_wrapped_text(
                draw,
                (cursor_x + 12, y + 10),
                value,
                fill=REPORT_TEXT,
                font=body_font,
                max_width=column_width - 20,
                line_spacing=3,
            )
            cursor_x += column_width
        y += row_height
    return y


def build_pdf_report_pages(data: dict[str, Any], run_dir: Path) -> list[Image.Image]:
    title_font = load_font(31)
    body_font = load_font(24)
    small_font = load_font(20)
    pages: list[Image.Image] = []

    raw_summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    artifact_path = run_dir / "masks/artifact_mask.png"
    artifact_mask = report_mask_array(artifact_path) if artifact_path.exists() else None
    summary = add_artifact_summary_fields(raw_summary, artifact_mask) if raw_summary else {}
    metrics = metric_rows(summary, data.get("scale") or None) if summary else data.get("metrics", [])

    original = report_image_from_path(run_dir / "input/original_for_analysis.png")
    preprocessed_path = run_dir / "input/preprocessed.png"
    preprocessed = report_image_from_path(preprocessed_path, fallback_size=original.size)
    sulfide_mask_path = run_dir / "masks/sulfide_mask.png"
    final_mask_path = run_dir / "masks/final_mask.png"
    final_mask = report_mask_array(final_mask_path, size=preprocessed.size) if final_mask_path.exists() else np.zeros((preprocessed.size[1], preprocessed.size[0]), dtype=np.uint8)
    if artifact_mask is not None and artifact_mask.shape != final_mask.shape:
        artifact_mask = np.asarray(Image.fromarray(artifact_mask.astype(np.uint8), mode="L").resize(preprocessed.size, Image.Resampling.NEAREST))

    page, draw, y = new_report_page("Демонстрационный отчет автоматизированного анализа шлифа", 1)
    draw.text((REPORT_MARGIN_X, y), "Паспорт исследования", fill=REPORT_TEXT, font=title_font)
    y += 42
    y = draw_key_value_table(
        draw,
        report_passport_rows(data, summary),
        x=REPORT_MARGIN_X,
        y=y,
        width=REPORT_PAGE_SIZE[0] - REPORT_MARGIN_X * 2,
    )
    y += 34
    draw.text((REPORT_MARGIN_X, y), "Заключение", fill=REPORT_TEXT, font=title_font)
    y += 42
    y = draw_wrapped_text(
        draw,
        (REPORT_MARGIN_X, y),
        f"Заключение: {data.get('text_output') or ''}",
        fill=REPORT_TEXT,
        font=body_font,
        max_width=REPORT_PAGE_SIZE[0] - REPORT_MARGIN_X * 2,
        line_spacing=9,
    )
    rule_text = summary.get("rule_text_ru") if isinstance(summary, dict) else ""
    if rule_text:
        y += 26
        y = draw_wrapped_text(
            draw,
            (REPORT_MARGIN_X, y),
            str(rule_text),
            fill=REPORT_MUTED,
            font=small_font,
            max_width=REPORT_PAGE_SIZE[0] - REPORT_MARGIN_X * 2,
            line_spacing=7,
        )
    y += 36
    draw.text((REPORT_MARGIN_X, y), "Результаты количественного анализа", fill=REPORT_TEXT, font=title_font)
    y += 46
    y = draw_metrics_table(draw, metrics, x=REPORT_MARGIN_X, y=y, width=REPORT_PAGE_SIZE[0] - REPORT_MARGIN_X * 2)
    y += 28
    draw_wrapped_text(
        draw,
        (REPORT_MARGIN_X, y),
        report_mineralogical_conclusion(summary),
        fill=REPORT_MUTED,
        font=small_font,
        max_width=REPORT_PAGE_SIZE[0] - REPORT_MARGIN_X * 2,
        line_spacing=7,
    )
    pages.append(page)

    page, draw, y = new_report_page("Фотодокументация: исходные данные и предобработка", 2)
    draw.text((REPORT_MARGIN_X, y), "Выполненные операции", fill=REPORT_TEXT, font=title_font)
    y += 44
    for line in report_preprocess_lines(data):
        y = draw_wrapped_text(draw, (REPORT_MARGIN_X, y), f"- {line}", fill=REPORT_MUTED, font=small_font, max_width=1080, line_spacing=6)
        y += 10
    y += 26
    card_width = 520
    card_height = 770
    draw_image_card(
        page,
        draw,
        x=REPORT_MARGIN_X,
        y=y,
        width=card_width,
        height=card_height,
        title="Исходное изображение",
        image=original,
        caption="Анализируемая копия исходного изображения.",
    )
    draw_image_card(
        page,
        draw,
        x=REPORT_MARGIN_X + card_width + 40,
        y=y,
        width=card_width,
        height=card_height,
        title="После предобработки",
        image=preprocessed,
        caption="Изображение, использованное для сегментации и расчета метрик.",
    )
    pages.append(page)

    page, draw, y = new_report_page("Фотодокументация: карты сегментации", 3)
    if sulfide_mask_path.exists():
        sulfide_image = sulfide_non_sulfide_image(sulfide_mask_path, size=preprocessed.size)
    else:
        sulfide_image = Image.new("RGB", preprocessed.size, REPORT_NON_SULFIDE_COLOR)
    final_image = final_overlay_image(preprocessed, final_mask, artifact_mask=artifact_mask)
    y = draw_report_legend(
        draw,
        REPORT_MARGIN_X,
        y,
        [
            ("Сульфиды", REPORT_SULFIDE_COLOR),
            ("Не сульфиды", REPORT_NON_SULFIDE_COLOR),
            ("Обычные срастания", (30, 185, 85)),
            ("Тонкие срастания", (230, 65, 65)),
            ("Тальк", (40, 120, 245)),
            ("Артефакты", ARTIFACT_COLOR[:3]),
        ],
    )
    y += 34
    draw_image_card(
        page,
        draw,
        x=REPORT_MARGIN_X,
        y=y,
        width=card_width,
        height=900,
        title="Сульфиды / не сульфиды",
        image=sulfide_image,
        caption="Двухцветная карта бинарной сегментации.",
    )
    draw_image_card(
        page,
        draw,
        x=REPORT_MARGIN_X + card_width + 40,
        y=y,
        width=card_width,
        height=900,
        title="Итоговая карта классов",
        image=final_image,
        caption="Цветная итоговая сегментация поверх предобработанного изображения.",
    )
    pages.append(page)

    page, draw, y = new_report_page("Классы итоговой сегментации", 4)
    row_height = 445
    small_card_width = 500
    small_card_height = 370
    for class_id, label, _mask_name, color in REPORT_CLASS_SPECS:
        draw.text((REPORT_MARGIN_X, y), label, fill=REPORT_TEXT, font=title_font)
        y += 42
        overlay = final_overlay_image(preprocessed, final_mask, class_ids={class_id})
        mask = class_mask_image(final_mask, class_id, color)
        draw_image_card(
            page,
            draw,
            x=REPORT_MARGIN_X,
            y=y,
            width=small_card_width,
            height=small_card_height,
            title=f"Итоговое изображение: {label.lower()}",
            image=overlay,
        )
        draw_image_card(
            page,
            draw,
            x=REPORT_MARGIN_X + small_card_width + 60,
            y=y,
            width=small_card_width,
            height=small_card_height,
            title=f"Маска: {label.lower()}",
            image=mask,
        )
        y += row_height
    pages.append(page)
    page, draw, y = new_report_page("Методика, контроль качества, экспертная проверка", 5)
    draw.text((REPORT_MARGIN_X, y), "Методика автоматизированного анализа", fill=REPORT_TEXT, font=title_font)
    y += 44
    for line in report_method_lines(data):
        y = draw_wrapped_text(draw, (REPORT_MARGIN_X, y), f"- {line}", fill=REPORT_MUTED, font=small_font, max_width=1080, line_spacing=6)
        y += 10
    y += 18
    draw.text((REPORT_MARGIN_X, y), "Контроль качества и ограничения", fill=REPORT_TEXT, font=title_font)
    y += 44
    for line in report_qc_lines(data, summary):
        y = draw_wrapped_text(draw, (REPORT_MARGIN_X, y), f"- {line}", fill=REPORT_MUTED, font=small_font, max_width=1080, line_spacing=6)
        y += 10
    y += 18
    draw.text((REPORT_MARGIN_X, y), "Артефакты и воспроизводимость", fill=REPORT_TEXT, font=title_font)
    y += 44
    for line in report_artifact_lines(data):
        y = draw_wrapped_text(draw, (REPORT_MARGIN_X, y), f"- {line}", fill=REPORT_MUTED, font=small_font, max_width=1080, line_spacing=6)
        y += 10
    y += 18
    draw.text((REPORT_MARGIN_X, y), "Экспертная проверка", fill=REPORT_TEXT, font=title_font)
    y += 44
    draw_key_value_table(
        draw,
        [
            ("Статус", report_review_status(data, summary)),
            ("ФИО эксперта", "не заполнено"),
            ("Дата проверки", "не заполнено"),
            ("Комментарий", "не заполнено"),
        ],
        x=REPORT_MARGIN_X,
        y=y,
        width=REPORT_PAGE_SIZE[0] - REPORT_MARGIN_X * 2,
    )
    pages.append(page)
    add_report_footers(pages)
    return pages


class OrePipelineHandler(BaseHTTPRequestHandler):
    server: "OrePipelineHTTPServer"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), fmt % args))

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        self.server.record_access_event(
            client=self.client_address[0],
            method=getattr(self, "command", ""),
            path=getattr(self, "path", ""),
            status=code,
            size=size,
        )
        self.log_message('"%s" %s %s', getattr(self, "requestline", ""), str(code), str(size))

    def _record_handler_error(self, exc: Exception, status: int) -> None:
        parsed = urllib.parse.urlparse(getattr(self, "path", ""))
        self.server.store.record_system_event(
            "error" if int(status) >= 500 else "warning",
            "request failed",
            method=getattr(self, "command", ""),
            path=parsed.path,
            status=int(status),
            error=str(exc),
        )

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._handle_get()
        except ApiError as exc:
            self._record_handler_error(exc, exc.status)
            self.send_json({"error": exc.message}, status=exc.status)
        except Exception as exc:  # noqa: BLE001 - keep local app alive.
            self._record_handler_error(exc, HTTPStatus.INTERNAL_SERVER_ERROR)
            self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._handle_post()
        except ApiError as exc:
            self._record_handler_error(exc, exc.status)
            self.send_json({"error": exc.message}, status=exc.status)
        except Exception as exc:  # noqa: BLE001 - keep local app alive.
            self._record_handler_error(exc, HTTPStatus.INTERNAL_SERVER_ERROR)
            self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_PUT(self) -> None:  # noqa: N802
        try:
            self._handle_put()
        except ApiError as exc:
            self._record_handler_error(exc, exc.status)
            self.send_json({"error": exc.message}, status=exc.status)
        except Exception as exc:  # noqa: BLE001 - keep local app alive.
            self._record_handler_error(exc, HTTPStatus.INTERNAL_SERVER_ERROR)
            self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_DELETE(self) -> None:  # noqa: N802
        try:
            self._handle_delete()
        except ApiError as exc:
            self._record_handler_error(exc, exc.status)
            self.send_json({"error": exc.message}, status=exc.status)
        except Exception as exc:  # noqa: BLE001 - keep local app alive.
            self._record_handler_error(exc, HTTPStatus.INTERNAL_SERVER_ERROR)
            self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_get(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/":
            self.send_redirect("/workspace")
            return
        if path in {"/workspace", "/history", "/history_series", "/settings", "/status", "/api", "/batch"} or path.startswith("/batch/"):
            self.send_html(render_html_page())
            return
        if path == "/api/settings":
            self.send_json(self.server.store.app_settings())
            return
        if path == "/api/status":
            self.send_json(self.server.status_payload())
            return
        if path == "/api/batches":
            self.send_json(self.server.store.list_batches())
            return
        if path.startswith("/api/batches/") and path.endswith("/results.csv"):
            batch_id = urllib.parse.unquote(path.removeprefix("/api/batches/").removesuffix("/results.csv"))
            self.send_file(
                self.server.store.batch_results_csv_path(batch_id),
                content_type="text/csv; charset=utf-8",
                download_name=f"{batch_id}_results.csv",
            )
            return
        if path.startswith("/api/batches/"):
            batch_id = urllib.parse.unquote(path.removeprefix("/api/batches/"))
            self.send_json(self.server.store.batch_payload(batch_id))
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
            self.send_file(
                self.server.store.metrics_csv_path(run_id),
                content_type="text/csv; charset=utf-8",
                download_name=f"{run_id}_metrics.csv",
            )
            return
        if path.startswith("/api/runs/") and path.endswith("/report.pdf"):
            run_id = urllib.parse.unquote(path.removeprefix("/api/runs/").removesuffix("/report.pdf"))
            self.send_file(
                self.server.store.pdf_report_path(run_id),
                content_type="application/pdf",
                download_name=f"{run_id}_report.pdf",
            )
            return
        if path.startswith("/api/runs/") and path.endswith("/files"):
            run_id = urllib.parse.unquote(path.removeprefix("/api/runs/").removesuffix("/files"))
            self.send_json(self.server.store.run_files_payload(run_id))
            return
        if path.startswith("/api/runs/") and path.endswith("/artifacts.zip"):
            run_id = urllib.parse.unquote(path.removeprefix("/api/runs/").removesuffix("/artifacts.zip"))
            self.send_file(
                self.server.store.run_zip_path(run_id),
                content_type="application/zip",
                download_name=f"{run_id}_artifacts.zip",
            )
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
            operation_id = self.server.store.begin_foreground_operation("upload", "receiving upload", path=path)
            try:
                self.send_json(self.handle_upload())
            finally:
                self.server.store.finish_foreground_operation(operation_id)
            return
        payload = self.read_json_payload()
        if path == "/api/batches":
            self.send_json(self.server.store.create_batch(payload))
            return
        if path == "/api/runtime/test":
            self.send_json(self.server.store.test_runtime(payload))
            return
        if path.startswith("/api/batches/") and path.endswith("/items"):
            batch_id = urllib.parse.unquote(path.removeprefix("/api/batches/").removesuffix("/items"))
            self.send_json(self.server.store.add_batch_items(batch_id, payload))
            return
        if path.startswith("/api/batches/") and path.endswith("/run"):
            batch_id = urllib.parse.unquote(path.removeprefix("/api/batches/").removesuffix("/run"))
            self.send_json(self.server.store.run_batch(batch_id, payload, run_async=True))
            return
        if path.startswith("/api/batches/") and path.endswith("/cancel"):
            batch_id = urllib.parse.unquote(path.removeprefix("/api/batches/").removesuffix("/cancel"))
            self.send_json(self.server.store.cancel_batch(batch_id))
            return
        if path.startswith("/api/uploads/") and path.endswith("/preprocess"):
            upload_id = urllib.parse.unquote(path.removeprefix("/api/uploads/").removesuffix("/preprocess"))
            operation_id = self.server.store.begin_foreground_operation(
                "preprocess",
                "preparing upload",
                path=path,
                upload_id=upload_id,
            )
            try:
                self.send_json(self.server.store.prepare_upload(upload_id, preset_from_payload(payload), augmentation_from_payload(payload)))
            finally:
                self.server.store.finish_foreground_operation(operation_id)
            return
        if path.startswith("/api/uploads/") and path.endswith("/artifact-mask"):
            upload_id = urllib.parse.unquote(path.removeprefix("/api/uploads/").removesuffix("/artifact-mask"))
            self.send_json(self.server.store.save_upload_artifact_mask(upload_id, payload))
            return
        if path == "/api/runs/start":
            upload_id = str(payload.get("upload_id") or "")
            if not upload_id:
                raise ApiError(HTTPStatus.BAD_REQUEST, "upload_id is required")
            operation_id = self.server.store.begin_foreground_operation(
                "run_prepare",
                "preparing run",
                path=path,
                upload_id=upload_id,
            )
            try:
                self.send_json(
                    self.server.store.start_run(
                        upload_id,
                        preset_from_payload(payload),
                        run_async=True,
                        curated_metadata=payload.get("curated_metadata"),
                        augmentation_settings=augmentation_from_payload(payload),
                    )
                )
            finally:
                self.server.store.finish_foreground_operation(operation_id)
            return
        if path.startswith("/api/runs/") and path.endswith("/prepare"):
            run_id = urllib.parse.unquote(path.removeprefix("/api/runs/").removesuffix("/prepare"))
            operation_id = self.server.store.begin_foreground_operation(
                "run_prepare",
                "preparing derived run",
                path=path,
                run_id=run_id,
                changed_step=str(payload.get("changed_step") or ""),
            )
            try:
                self.send_json(
                    self.server.store.prepare_run_from_apply(
                        run_id,
                        preset_from_payload(payload),
                        augmentation_settings=augmentation_from_payload(payload),
                        changed_step=str(payload.get("changed_step") or ""),
                    )
                )
            finally:
                self.server.store.finish_foreground_operation(operation_id)
            return
        if path.startswith("/api/runs/") and path.endswith("/start"):
            run_id = urllib.parse.unquote(path.removeprefix("/api/runs/").removesuffix("/start"))
            self.send_json(
                self.server.store.start_prepared_run(
                    run_id,
                    run_async=True,
                    curated_metadata=payload.get("curated_metadata"),
                )
            )
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

    def _handle_put(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        payload = self.read_json_payload()
        if path == "/api/settings":
            self.send_json(self.server.store.save_app_settings(payload))
            return
        if path.startswith("/api/batches/") and path.endswith("/settings"):
            batch_id = urllib.parse.unquote(path.removeprefix("/api/batches/").removesuffix("/settings"))
            self.send_json(self.server.store.update_batch_settings(batch_id, payload))
            return
        if path.startswith("/api/batches/") and path.endswith("/metadata"):
            parts = [urllib.parse.unquote(part) for part in path.strip("/").split("/")]
            if len(parts) == 6 and parts[:2] == ["api", "batches"] and parts[3] == "items" and parts[5] == "metadata":
                self.send_json(self.server.store.update_batch_item_metadata(parts[2], parts[4], payload))
                return
        raise ApiError(HTTPStatus.NOT_FOUND, "not found")

    def _handle_delete(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/api/history":
            self.send_json(self.server.store.delete_history())
            return
        if path.startswith("/api/batches/"):
            parts = [urllib.parse.unquote(part) for part in path.strip("/").split("/")]
            if len(parts) == 5 and parts[:2] == ["api", "batches"] and parts[3] == "items":
                self.send_json(self.server.store.remove_batch_item(parts[2], parts[4]))
                return
            if len(parts) == 3 and parts[:2] == ["api", "batches"]:
                self.send_json(self.server.store.delete_batch(parts[2]))
                return
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

    def send_file(self, path: Path, content_type: str | None = None, download_name: str | None = None) -> None:
        content_type = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if download_name:
            quoted = urllib.parse.quote(download_name)
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quoted}")
        self.end_headers()
        self.wfile.write(body)


class OrePipelineHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], store: OrePipelineStore) -> None:
        self.store = store
        self.access_log: deque[dict[str, Any]] = deque(maxlen=LOG_ENTRY_LIMIT)
        self.log_lock = threading.RLock()
        super().__init__(server_address, OrePipelineHandler)
        self.store.record_system_event("info", "http server listening", host=str(server_address[0]), port=int(self.server_address[1]))

    def record_access_event(self, *, client: str, method: str, path: str, status: int | str, size: int | str) -> None:
        parsed = urllib.parse.urlparse(path or "")
        try:
            status_value: int | str = int(status)
        except (TypeError, ValueError):
            status_value = str(status)
        try:
            size_value: int | str = int(size)
        except (TypeError, ValueError):
            size_value = str(size)
        entry = {
            "timestamp": utc_now_iso(),
            "client": str(client or ""),
            "method": str(method or ""),
            "path": parsed.path or "/",
            "status": status_value,
            "size_bytes": size_value,
        }
        with self.log_lock:
            self.access_log.append(entry)

    def access_log_payload(self, limit: int = STATUS_LOG_LIMIT) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), LOG_ENTRY_LIMIT))
        with self.log_lock:
            return list(reversed(list(self.access_log)[-limit:]))

    def status_payload(self) -> dict[str, Any]:
        return self.store.status_payload(access_log=self.access_log_payload())


_STATIC_DIR = Path(__file__).resolve().parent / "static"
_HTML_PAGE_PATH = _STATIC_DIR / "ore_pipeline_ui.html"
_html_page_cache: str | None = None


def render_html_page() -> str:
    """Return the single-page UI HTML, loaded once from apps/static/."""
    global _html_page_cache
    if _html_page_cache is None:
        _html_page_cache = _HTML_PAGE_PATH.read_text(encoding="utf-8")
    return _html_page_cache


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


if __name__ == "__main__":
    raise SystemExit(main())
