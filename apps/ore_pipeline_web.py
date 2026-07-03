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
CURATED_METADATA_SCHEMA_VERSION = "ore-pipeline-curated-metadata-v0.1"
APP_SETTINGS_SCHEMA_VERSION = "ore-pipeline-app-settings-v0.1"
BATCH_SCHEMA_VERSION = "ore-pipeline-batch-v0.1"
BATCH_ITEM_SCHEMA_VERSION = "ore-pipeline-batch-item-v0.1"
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


def normalize_app_settings_payload(payload: Any, base: dict[str, Any] | None = None) -> dict[str, Any]:
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ApiError(HTTPStatus.BAD_REQUEST, "settings must be an object")
    fallback = default_app_settings()
    if isinstance(base, dict):
        fallback.update({key: base[key] for key in ("language", "theme", "show_tiling") if key in base})
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


def extract_image_raw_metadata(path: Path, *, original_name: str, width: int, height: int) -> dict[str, Any]:
    stat = path.stat()
    metadata: dict[str, Any] = {
        "schema_version": "ore-pipeline-raw-image-metadata-v0.1",
        "original_name": original_name,
        "stored_path": str(path),
        "extension": path.suffix.lower(),
        "file_size_bytes": int(stat.st_size),
        "sha1": file_sha1(path),
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
        self.artifacts: dict[str, Path] = {}
        self.jobs: dict[str, dict[str, Any]] = {}
        self.batch_jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.allowed_roots = [ROOT.resolve(), self.workspace_dir.resolve()]
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.batches_dir.mkdir(parents=True, exist_ok=True)
        self.settings_dir.mkdir(parents=True, exist_ok=True)

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
        raw_metadata = extract_image_raw_metadata(original_path, original_name=original_name, width=width, height=height)
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
        source = load_image_pil(original_path)
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
        preprocessed_full = apply_preprocessing(augmented_image, preset) if preprocessing_enabled else augmented_image
        preprocess_dir = upload_dir / "preprocessed"
        preprocessed_full_path = preprocess_dir / "preprocessed_full.png"
        if preprocessing_enabled:
            save_image(preprocessed_full_path, preprocessed_full)
        analysis_image = scaled_image_copy(preprocessed_full, max_side=target_max_side)
        preprocessed_path = preprocess_dir / "preprocessed.png"
        save_image(preprocessed_path, analysis_image)
        previews = (
            save_preview_pyramid(preprocessed_full, preprocess_dir / "display", "preprocessed", self.preview_max_sides)
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
            "full_width": preprocessed_full.size[0],
            "full_height": preprocessed_full.size[1],
            "source_width": int(metadata["width"]),
            "source_height": int(metadata["height"]),
            "source_scaled_for_processing": source_scaled,
            "target_max_side": target_max_side,
            "panorama_scaling": panorama_scaling,
            "display": previews,
            "tiling": tiling,
        }
        if preprocessing_enabled:
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
                    "batch": data.get("batch"),
                    "thumbnail": thumbnail,
                }
            )
        return {"schema_version": "ore-pipeline-history-v0.1", "runs": runs}

    def app_settings(self) -> dict[str, Any]:
        if not self.settings_path.exists():
            return default_app_settings()
        try:
            payload = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return default_app_settings()
        return normalize_app_settings_payload(payload)

    def save_app_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        settings = normalize_app_settings_payload(payload, base=self.app_settings())
        settings["updated_at"] = utc_now_iso()
        self._write_json(self.settings_path, settings)
        return settings

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
        return self.run_payload(run_id)

    def run_payload(self, run_id: str) -> dict[str, Any]:
        data = self._read_run(run_id)
        job = self.jobs.get(run_id)
        if job and data.get("status") not in {"complete", "failed", "canceled"}:
            data = {**data, **job}
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

    def run_files_payload(self, run_id: str) -> dict[str, Any]:
        run_dir = self._existing_run_dir(run_id)
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
        raw_summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
        summary = add_artifact_summary_fields(raw_summary, self._artifact_mask_for_summary(run_id)) if raw_summary else {}
        metrics = metric_rows(summary, data.get("scale") or None) if summary else data.get("metrics", [])
        for row in metrics:
            value = f"{float(row['percent']):.1f}%" if row.get("percent") is not None else str(row.get("value"))
            if row.get("area_um2") is not None:
                value = f"{value}; {float(row['area_um2']):.1f} мкм²"
            elif row.get("area_px") is not None:
                value = f"{value}; {int(row['area_px'])} px"
            draw.text((100 + int(row.get("level") or 0) * 28, y), row["label"], fill=(58, 67, 82), font=body_font)
            draw.text((730, y), value, fill=(20, 26, 36), font=body_font)
            y += 46
        display = data.get("display", {})
        preview_path = first_preview_path(display.get("preprocessed")) or first_preview_path(display.get("original"))
        if preview_path and Path(preview_path).exists():
            preview = Image.open(preview_path).convert("RGB")
            preview.thumbnail((980, 760), Image.Resampling.BILINEAR)
            page.paste(preview, (80, min(y + 40, 930)))
        page.save(path, "PDF", resolution=150.0)
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

    def do_PUT(self) -> None:  # noqa: N802
        try:
            self._handle_put()
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
        if path in {"/workspace", "/history", "/settings", "/batch"} or path.startswith("/batch/"):
            self.send_html(render_html_page())
            return
        if path == "/api/settings":
            self.send_json(self.server.store.app_settings())
            return
        if path == "/api/batches":
            self.send_json(self.server.store.list_batches())
            return
        if path.startswith("/api/batches/") and path.endswith("/results.csv"):
            batch_id = urllib.parse.unquote(path.removeprefix("/api/batches/").removesuffix("/results.csv"))
            self.send_file(self.server.store.batch_results_csv_path(batch_id), content_type="text/csv; charset=utf-8")
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
            self.send_file(self.server.store.metrics_csv_path(run_id), content_type="text/csv; charset=utf-8")
            return
        if path.startswith("/api/runs/") and path.endswith("/report.pdf"):
            run_id = urllib.parse.unquote(path.removeprefix("/api/runs/").removesuffix("/report.pdf"))
            self.send_file(self.server.store.pdf_report_path(run_id), content_type="application/pdf")
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
            self.send_json(self.handle_upload())
            return
        payload = self.read_json_payload()
        if path == "/api/batches":
            self.send_json(self.server.store.create_batch(payload))
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
            self.send_json(self.server.store.prepare_upload(upload_id, preset_from_payload(payload), augmentation_from_payload(payload)))
            return
        if path.startswith("/api/uploads/") and path.endswith("/artifact-mask"):
            upload_id = urllib.parse.unquote(path.removeprefix("/api/uploads/").removesuffix("/artifact-mask"))
            self.send_json(self.server.store.save_upload_artifact_mask(upload_id, payload))
            return
        if path == "/api/runs/start":
            upload_id = str(payload.get("upload_id") or "")
            if not upload_id:
                raise ApiError(HTTPStatus.BAD_REQUEST, "upload_id is required")
            self.send_json(
                self.server.store.start_run(
                    upload_id,
                    preset_from_payload(payload),
                    run_async=True,
                    curated_metadata=payload.get("curated_metadata"),
                    augmentation_settings=augmentation_from_payload(payload),
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
        if path.startswith("/api/batches/"):
            parts = [urllib.parse.unquote(part) for part in path.strip("/").split("/")]
            if len(parts) == 5 and parts[:2] == ["api", "batches"] and parts[3] == "items":
                self.send_json(self.server.store.remove_batch_item(parts[2], parts[4]))
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
      --button-disabled-bg: #edf1f5;
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
      --sulfide: #d79b10;
      --non-sulfide: #7c8796;
      --green: #1fa25a;
      --red: #d83f45;
      --blue: #2870d8;
      --artifact: #c63cff;
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
      --button-disabled-bg: #121720;
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
      --sulfide: #f1c44d;
      --non-sulfide: #9aa7b8;
      --green: #32c173;
      --red: #f06267;
      --blue: #5c94f5;
      --artifact: #d16bff;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); }
    header { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 14px 18px; border-bottom: 1px solid var(--line); background: var(--panel); flex-wrap: wrap; }
    h1 { margin: 0; font-size: 18px; font-weight: 720; letter-spacing: 0; }
    button, select, input, textarea { font: inherit; }
    button { border: 1px solid var(--line); background: var(--button-bg); color: var(--text); border-radius: 6px; padding: 8px 11px; cursor: pointer; }
    button.primary { background: var(--accent); border-color: var(--accent); color: white; }
    button.danger { background: var(--danger); border-color: var(--danger); color: white; }
    #fixBtn { background: var(--danger); border-color: var(--danger); color: white; }
    button:disabled { opacity: 1; cursor: not-allowed; background: var(--button-disabled-bg); border-color: var(--line); color: var(--muted); }
    button.primary:disabled, button.danger:disabled, #fixBtn:disabled { background: var(--button-disabled-bg); border-color: var(--line); color: var(--muted); }
    .tabs { display: flex; gap: 8px; flex-wrap: wrap; min-width: 0; }
    .tab { min-width: 0; }
    .tab.active { border-color: var(--accent); color: var(--accent); }
    .header-actions { display: flex; align-items: center; justify-content: flex-end; gap: 10px; flex-wrap: wrap; min-width: 0; }
    .theme-control, .language-control { width: auto; min-width: 128px; }
    main { display: grid; grid-template-columns: minmax(280px, 360px) minmax(0, 1fr); min-height: calc(100vh - 57px); }
    aside { padding: 16px; border-right: 1px solid var(--line); background: var(--panel-alt); overflow: auto; }
    section.workspace { padding: 16px; min-width: 0; }
    body[data-page="batch"] main, body[data-page="history"] main, body[data-page="settings"] main { grid-template-columns: 1fr; }
    body[data-page="batch"] aside, body[data-page="history"] aside, body[data-page="settings"] aside { display: none; }
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
    select, textarea, input[type="number"], input[type="text"] { width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 8px; background: var(--control-bg); color: var(--text); }
    .viewer-shell { background: var(--viewer-bg); border-radius: 8px; overflow: hidden; border: 1px solid var(--viewer-border); min-height: 420px; position: relative; }
    canvas { display: block; width: 100%; height: 100%; }
    #mainCanvas { height: min(72vh, 760px); min-height: 420px; }
    .viewer-toolbar { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; padding: 10px; background: var(--toolbar-bg); border: 1px solid var(--line); border-radius: 8px 8px 0 0; border-bottom: 0; flex-wrap: wrap; }
    .viewer-mode-row { display: flex; align-items: flex-start; gap: 10px; flex-wrap: wrap; min-width: 0; max-width: 100%; }
    .primary-view-controls { display: grid; gap: 8px; width: auto; min-width: 0; max-width: 100%; }
    .side-by-side-control { display: flex; align-items: center; gap: 7px; flex-wrap: wrap; min-width: 0; max-width: 100%; }
    .segmented { display: inline-flex; max-width: 100%; border: 1px solid var(--line); border-radius: 7px; overflow-x: auto; overflow-y: hidden; background: var(--control-bg); scrollbar-width: thin; }
    .segmented button { border: 0; border-right: 1px solid var(--line); border-radius: 0; background: transparent; padding: 7px 10px; white-space: nowrap; flex: 0 0 auto; }
    .segmented button:last-child { border-right: 0; }
    .segmented button.active { background: var(--segmented-active-bg); color: var(--accent); }
    .segmented button:disabled { opacity: .35; color: var(--muted); cursor: not-allowed; }
    .viewer-toolbar .segmented { overflow: visible; flex-wrap: nowrap; scrollbar-width: none; }
    .viewer-toolbar .segmented::-webkit-scrollbar { display: none; }
    .progress { height: 9px; background: var(--progress-bg); border-radius: 999px; overflow: hidden; }
    .progress > div { height: 100%; background: var(--accent); width: 0%; transition: width .2s ease; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { text-align: left; border-bottom: 1px solid var(--line); padding: 8px 6px; }
    th { color: var(--muted); font-weight: 650; }
    .result-grid { display: grid; grid-template-columns: minmax(0, 1fr); gap: 14px; }
    .metrics-panel { overflow-x: auto; }
    .metrics-table { min-width: 720px; }
    .metrics-table .metric-label { padding-left: calc(6px + var(--metric-level, 0) * 24px); }
    .metrics-table tr.metric-level-0 .metric-label { font-weight: 750; }
    .metrics-table tr.metric-level-1 .metric-label { font-weight: 650; }
    .metrics-table tr.metric-level-2 .metric-label { color: var(--muted); }
    .layer-toggle-row, .viewer-options-row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 8px; }
    .segmentation-legend-overlay { position: absolute; z-index: 4; top: 10px; left: 10px; right: 10px; display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; pointer-events: none; }
    .segmentation-legend-panel { max-width: min(48%, 760px); display: grid; gap: 6px; padding: 8px 10px; border: 1px solid color-mix(in srgb, var(--line) 78%, transparent); border-radius: 7px; background: color-mix(in srgb, var(--panel) 88%, transparent); box-shadow: 0 8px 24px rgba(0,0,0,.22); backdrop-filter: blur(8px); pointer-events: auto; }
    .segmentation-legend-panel.right { justify-items: end; margin-left: auto; }
    .segmentation-legend-panel[hidden] { display: none; }
    .segmentation-legend-title { color: var(--muted); font-size: 11px; font-weight: 750; text-transform: uppercase; letter-spacing: 0; }
    .class-toggles { display: flex; flex-direction: column; gap: 6px; align-items: flex-start; padding: 0; }
    .class-toggles[hidden] { display: none; }
    .segmentation-legend-panel.right .class-toggles { align-items: flex-start; justify-content: flex-start; }
    .overlay-opacity-control { display: flex; align-items: center; gap: 7px; color: var(--muted); font-size: 13px; }
    .overlay-opacity-control input { width: 116px; }
    .decision-rationale, .metrics-note { margin: 8px 0 0; line-height: 1.35; }
    .swatch { width: 12px; height: 12px; display: inline-block; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
    .history-row { display: grid; grid-template-columns: 68px minmax(0, 1fr); gap: 10px; align-items: start; border: 1px solid var(--line); border-radius: 7px; padding: 10px; margin-bottom: 8px; background: var(--history-bg); }
    .history-row-media { display: grid; gap: 6px; align-content: start; }
    .history-row-media .history-thumb-button, .history-row-media .history-thumb-placeholder { width: 100%; height: 48px; }
    .history-row-media .history-thumb-placeholder { display: grid; place-items: center; border: 1px solid var(--line); border-radius: 6px; background: var(--viewer-bg); }
    .history-row-load { width: 100%; padding: 5px 6px; font-size: 12px; }
    .history-row-text { min-width: 0; display: grid; gap: 4px; }
    .history-row-title { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; font-weight: 750; }
    .history-row-run { color: var(--muted); font-size: 12px; word-break: break-all; }
    .history-row-summary { font-size: 13px; line-height: 1.32; }
    .history-page-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 10px; }
    .history-page-head h2 { margin-bottom: 0; }
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
    .run-files-dialog { width: min(980px, 94vw); }
    .run-files-body { padding: 12px 14px; background: var(--bg); display: grid; gap: 10px; }
    .run-files-table-wrap { max-height: min(66vh, 620px); overflow: auto; border: 1px solid var(--line); border-radius: 7px; background: var(--panel); }
    .run-files-table { min-width: 760px; }
    .run-files-table td:first-child { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; word-break: break-all; }
    .run-files-table td.numeric, .run-files-table th.numeric { text-align: right; white-space: nowrap; }
    .metadata-entry { display: grid; gap: 6px; margin-top: 10px; }
    .metadata-entry button { width: 100%; }
    .settings-page { display: grid; gap: 14px; max-width: 1120px; }
    .settings-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
    .settings-field { display: grid; gap: 5px; }
    .settings-field.full { grid-column: 1 / -1; }
    .settings-field span { font-size: 13px; color: var(--muted); font-weight: 650; }
    .settings-actions { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .batch-page { display: grid; gap: 14px; }
    .batch-head { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; flex-wrap: wrap; }
    .batch-actions { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .batch-summary { display: flex; gap: 10px; flex-wrap: wrap; color: var(--muted); font-size: 13px; }
    .batch-panel-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }
    .batch-panel-head h2 { margin: 0; }
    .batch-gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px; }
    .batch-card { border: 1px solid var(--line); border-radius: 8px; background: var(--history-bg); overflow: hidden; display: grid; grid-template-rows: 142px auto; min-width: 0; }
    .batch-thumb { width: 100%; height: 142px; background: var(--viewer-bg); display: grid; place-items: center; overflow: hidden; }
    .batch-thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .batch-thumb-placeholder { color: var(--muted); }
    .batch-card-body { padding: 10px; display: grid; gap: 8px; min-width: 0; }
    .batch-card-title { font-weight: 750; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .batch-card-meta, .batch-card-status { color: var(--muted); font-size: 12px; }
    .batch-card-error { color: var(--danger); font-size: 12px; }
    .batch-progress { display: grid; gap: 5px; }
    .batch-card-actions { display: flex; gap: 7px; flex-wrap: wrap; }
    .batch-card-actions button { padding: 6px 8px; font-size: 12px; }
    .preprocess-compact { display: grid; gap: 8px; }
    .preprocess-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; flex-wrap: wrap; }
    .preprocess-actions { display: flex; gap: 8px; flex-wrap: wrap; }
    .preprocess-dialog, .augmentation-dialog { width: min(760px, 92vw); }
    .preprocess-settings, .augmentation-settings { display: grid; gap: 10px; padding: 12px 14px; background: var(--bg); }
    .preprocess-option { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
    .preprocess-option.with-extra { display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: start; }
    .preprocess-option label.check { flex: 1 1 auto; min-width: 0; }
    .panorama-scaling-controls, .settings-panorama-controls { grid-column: 1 / -1; display: grid; grid-template-columns: minmax(170px, 1fr) minmax(150px, 1fr) minmax(140px, 1fr); gap: 8px; }
    .panorama-scaling-controls { padding-left: 24px; }
    .panorama-scale-field { display: grid; gap: 5px; }
    .panorama-scale-field span { font-size: 12px; color: var(--muted); font-weight: 650; }
    .panorama-scale-field input, .panorama-scale-field select { width: 100%; }
    .settings-group { border: 1px solid var(--line); border-radius: 8px; padding: 10px; display: grid; gap: 10px; }
    .settings-group legend { color: var(--muted); font-size: 12px; font-weight: 800; padding: 0 4px; }
    .settings-preprocess-defaults { display: grid; gap: 10px; }
    .settings-section-divider { height: 1px; background: var(--line); }
    .settings-scale-group { display: grid; gap: 10px; }
    .range-field { display: grid; grid-template-columns: minmax(150px, 1fr) minmax(160px, 2fr) 58px; align-items: center; gap: 8px; }
    .range-field input[type="number"] { width: 100%; }
    .help-dot { position: relative; width: 24px; height: 24px; padding: 0; border-radius: 999px; flex: 0 0 auto; display: inline-grid; place-items: center; color: var(--accent); font-size: 12px; font-weight: 750; line-height: 1; }
    .help-dot::after { content: attr(data-tooltip); position: absolute; left: 50%; bottom: calc(100% + 8px); width: min(260px, 72vw); transform: translate(-50%, 4px); opacity: 0; pointer-events: none; z-index: 30; padding: 8px 10px; border: 1px solid var(--line); border-radius: 7px; background: var(--panel); color: var(--text); box-shadow: var(--modal-shadow); font-size: 12px; font-weight: 500; line-height: 1.35; text-align: left; transition: opacity .12s ease, transform .12s ease; }
    .help-dot:hover::after, .help-dot:focus-visible::after { opacity: 1; transform: translate(-50%, 0); }
    .metadata-dialog { width: min(980px, 94vw); }
    .metadata-body { padding: 12px 14px; background: var(--bg); display: grid; gap: 12px; }
    .metadata-tabs { width: max-content; max-width: 100%; }
    .metadata-panel { display: grid; gap: 12px; }
    .metadata-panel[hidden] { display: none; }
    .metadata-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
    .metadata-section-title { grid-column: 1 / -1; color: var(--accent); font-size: 13px; font-weight: 750; text-transform: uppercase; letter-spacing: 0; margin-top: 2px; }
    .metadata-field { display: grid; gap: 5px; }
    .metadata-field.full { grid-column: 1 / -1; }
    .metadata-field span { font-size: 13px; color: var(--muted); font-weight: 650; }
    .metadata-raw-wrap { max-height: 360px; overflow: auto; border: 1px solid var(--line); border-radius: 7px; background: var(--panel); }
    .metadata-raw-wrap table { margin: 0; }
    .metadata-warning { border: 1px solid color-mix(in srgb, var(--danger) 45%, var(--line)); background: color-mix(in srgb, var(--danger) 10%, transparent); border-radius: 7px; padding: 9px; color: var(--danger); }
    .modal-head, .modal-foot { padding: 12px 14px; border-bottom: 1px solid var(--line); background: var(--panel); display: flex; justify-content: space-between; align-items: center; gap: 10px; }
    .modal-foot { border-bottom: 0; border-top: 1px solid var(--line); }
    .editor-top-toolbar { display: flex; align-items: center; gap: 10px; padding: 10px 12px; border-bottom: 1px solid var(--line); background: var(--panel-alt); flex-wrap: wrap; }
    .editor-top-toolbar strong { font-size: 14px; }
    .modal-body { display: grid; grid-template-columns: minmax(0, 1fr) 310px; gap: 12px; padding: 12px; background: var(--bg); }
    .editor-side { display: flex; flex-direction: column; min-height: min(70vh, 720px); }
    #editLayerTabs { width: 100%; }
    #editLayerTabs button { flex: 1 1 0; min-width: 0; white-space: normal; line-height: 1.2; }
    .editor-tools { display: flex; flex-wrap: wrap; gap: 8px; margin: 0; }
    .editor-tools button.active { border-color: var(--accent); color: var(--accent); background: var(--segmented-active-bg); }
    .editor-tools button:disabled { opacity: 1; }
    .brush-size-control { display: flex; align-items: center; gap: 6px; color: var(--muted); font-size: 13px; }
    .brush-size-control input { width: 74px; }
    .editor-view { height: min(70vh, 720px); background: #1f232a; border-radius: 8px; overflow: hidden; border: 1px solid #11151b; }
    #editorCanvas { height: 100%; }
    .editor-stats { margin-top: auto; padding-top: 12px; }
    .stats-table td { font-size: 13px; padding: 6px 4px; overflow-wrap: anywhere; }
    .stats-table td:nth-child(2), .stats-table td:last-child { text-align: right; white-space: nowrap; }
    .stats-table td:last-child { color: var(--muted); }
    .stats-table .stat-separator td { padding: 4px 0; border-bottom: 1px solid var(--line); }
    .hidden { display: none !important; }
    @media (max-width: 980px) {
      main, .result-grid, .modal-body { grid-template-columns: 1fr; }
      .metadata-grid, .settings-grid, .batch-gallery { grid-template-columns: 1fr; }
      .panorama-scaling-controls, .settings-panorama-controls { grid-template-columns: 1fr; }
      .panorama-scaling-controls { padding-left: 0; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      #mainCanvas { height: 62vh; min-height: 380px; }
    }
    @media (max-width: 700px) {
      header { display: grid; grid-template-columns: 1fr; }
      .header-actions { width: 100%; display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); align-items: stretch; }
      .theme-control, .language-control { width: 100%; min-width: 0; }
      .tabs { grid-column: 1 / -1; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .tab { white-space: normal; }
      section.workspace, aside { padding: 10px; }
      .viewer-toolbar { padding: 8px; }
      .viewer-mode-row, .primary-view-controls, .side-by-side-control { width: 100%; }
      .viewer-toolbar .segmented { width: 100%; }
      .range-field { grid-template-columns: 1fr; }
      .overlay-opacity-control { flex-wrap: wrap; }
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
        <button class="tab" id="batchTab" data-i18n="batchTab">Серии</button>
        <button class="tab" id="historyTab" data-i18n="historyTab">История</button>
        <button class="tab" id="settingsTab" data-i18n="settingsTab">Настройки</button>
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
        <div class="metadata-entry">
          <button id="metadataBtn" type="button" disabled data-i18n="editMetadata">Редактировать метаданные...</button>
        </div>
      </div>
      <div class="panel">
        <div class="preprocess-compact">
          <div class="preprocess-row">
            <label class="check"><input type="checkbox" id="augmentationEnabled"> <strong data-i18n="augmentation">Аугментация</strong></label>
            <div class="preprocess-actions">
              <button id="editAugmentationBtn" type="button" data-i18n="editAugmentation">Настроить...</button>
              <button id="applyAugmentationBtn" type="button" data-i18n="applyAugmentation">Применить</button>
            </div>
          </div>
          <p id="augmentationSummary" class="muted"></p>
        </div>
      </div>
      <div class="panel">
        <div class="preprocess-compact">
          <div class="preprocess-row">
            <label class="check"><input type="checkbox" id="preprocessingEnabled" checked> <strong data-i18n="preprocessing">Предобработка</strong></label>
            <div class="preprocess-actions">
              <button id="editPreprocessBtn" type="button" data-i18n="editPreprocessing">Настроить...</button>
              <button id="applyPreprocessBtn" type="button" data-i18n="applyPreprocessing">Применить</button>
            </div>
          </div>
          <p id="preprocessSummary" class="muted"></p>
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
            <div class="primary-view-controls">
              <div class="segmented" id="viewModeButtons">
                <button data-mode="original" class="active" data-i18n="viewOriginal">оригинал</button>
                <button data-mode="augmented" data-i18n="viewAugmented">аугментированное</button>
                <button data-mode="preprocessed" data-i18n="viewPreprocessed">предобработка</button>
                <button data-mode="sulfide" data-i18n="viewSulfide">сульфиды</button>
                <button data-mode="final" data-i18n="viewFinal">финал</button>
              </div>
            </div>
            <div class="side-by-side-control">
              <span class="muted" data-i18n="sideBySide">Сравнение:</span>
              <div class="segmented" id="sideLayerButtons">
                <button data-side-layer="none" class="active" data-i18n="sideNone">нет</button>
                <button data-side-layer="augmented" data-i18n="viewAugmented">аугментированное</button>
                <button data-side-layer="preprocessed" data-i18n="viewPreprocessed">предобработка</button>
                <button data-side-layer="sulfide" data-i18n="viewSulfide">сульфиды</button>
                <button data-side-layer="final" data-i18n="viewFinal">финал</button>
              </div>
            </div>
          </div>
          <button id="fixBtn" disabled data-i18n="fixMe">Исправить</button>
        </div>
        <div class="viewer-shell">
          <div id="segmentationClassToggles" class="segmentation-legend-overlay hidden">
            <div id="primaryClassLegend" class="segmentation-legend-panel left" hidden>
              <div class="segmentation-legend-title" data-i18n="leftViewLegend">Левый слой</div>
              <div id="primarySulfideClassToggles" class="class-toggles" data-legend-layer="sulfide" hidden>
                <label class="check"><input type="checkbox" data-legend-toggle="showSulfide" checked><span class="swatch" style="background:var(--sulfide)"></span><span data-i18n="classSulfides">сульфиды</span></label>
                <label class="check"><input type="checkbox" data-legend-toggle="showNonSulfide" checked><span class="swatch" style="background:var(--non-sulfide)"></span><span data-i18n="classNonSulfides">не-сульфиды</span></label>
                <label class="check"><input type="checkbox" data-legend-toggle="showSulfideArtifacts" checked><span class="swatch" style="background:var(--artifact)"></span><span data-i18n="classArtefacts">артефакты</span></label>
              </div>
              <div id="primaryFinalClassToggles" class="class-toggles" data-legend-layer="final" hidden>
                <label class="check"><input type="checkbox" data-legend-toggle="showOrdinary" checked><span class="swatch" style="background:var(--green)"></span><span data-i18n="classOrdinaryShort">обычные</span></label>
                <label class="check"><input type="checkbox" data-legend-toggle="showFine" checked><span class="swatch" style="background:var(--red)"></span><span data-i18n="classFineShort">тонкие</span></label>
                <label class="check"><input type="checkbox" data-legend-toggle="showTalc" checked><span class="swatch" style="background:var(--blue)"></span><span data-i18n="classTalc">тальк</span></label>
                <label class="check"><input type="checkbox" data-legend-toggle="showFinalArtifacts" checked><span class="swatch" style="background:var(--artifact)"></span><span data-i18n="classArtefacts">артефакты</span></label>
                <label class="check"><input type="checkbox" data-legend-toggle="showBackground" checked> <span data-i18n="classBackground">фон</span></label>
              </div>
            </div>
            <div id="sideClassLegend" class="segmentation-legend-panel right" hidden>
              <div class="segmentation-legend-title" data-i18n="rightViewLegend">Правый слой</div>
              <div id="sideSulfideClassToggles" class="class-toggles" data-legend-layer="sulfide" hidden>
                <label class="check"><input type="checkbox" data-legend-toggle="showSulfide" checked><span class="swatch" style="background:var(--sulfide)"></span><span data-i18n="classSulfides">сульфиды</span></label>
                <label class="check"><input type="checkbox" data-legend-toggle="showNonSulfide" checked><span class="swatch" style="background:var(--non-sulfide)"></span><span data-i18n="classNonSulfides">не-сульфиды</span></label>
                <label class="check"><input type="checkbox" data-legend-toggle="showSulfideArtifacts" checked><span class="swatch" style="background:var(--artifact)"></span><span data-i18n="classArtefacts">артефакты</span></label>
              </div>
              <div id="sideFinalClassToggles" class="class-toggles" data-legend-layer="final" hidden>
                <label class="check"><input type="checkbox" data-legend-toggle="showOrdinary" checked><span class="swatch" style="background:var(--green)"></span><span data-i18n="classOrdinaryShort">обычные</span></label>
                <label class="check"><input type="checkbox" data-legend-toggle="showFine" checked><span class="swatch" style="background:var(--red)"></span><span data-i18n="classFineShort">тонкие</span></label>
                <label class="check"><input type="checkbox" data-legend-toggle="showTalc" checked><span class="swatch" style="background:var(--blue)"></span><span data-i18n="classTalc">тальк</span></label>
                <label class="check"><input type="checkbox" data-legend-toggle="showFinalArtifacts" checked><span class="swatch" style="background:var(--artifact)"></span><span data-i18n="classArtefacts">артефакты</span></label>
                <label class="check"><input type="checkbox" data-legend-toggle="showBackground" checked> <span data-i18n="classBackground">фон</span></label>
              </div>
            </div>
          </div>
          <canvas id="mainCanvas"></canvas>
        </div>
        <div class="viewer-options-row">
          <label class="check"><input type="checkbox" id="showTiling"> <span data-i18n="showTiling">показать тайлы</span></label>
          <label class="check"><input type="checkbox" id="boundaryOnly"> <span data-i18n="boundaryOnly">только контуры</span></label>
          <label class="overlay-opacity-control"><span data-i18n="overlayOpacity">прозрачность</span><input id="overlayOpacity" type="range" min="0.2" max="1" step="0.05" value="0.65"><output id="overlayOpacityValue">65%</output></label>
        </div>
        <div id="resultPanel" class="result-grid hidden" style="margin-top:14px">
          <div class="panel">
            <button id="backToBatchBtn" type="button" class="hidden" style="margin-bottom:10px" data-i18n="batchBack">Назад к серии</button>
            <h2 data-i18n="textOutputTitle">Текстовый вывод</h2>
            <p id="textOutput"></p>
            <p id="decisionRationale" class="decision-rationale muted"></p>
          </div>
          <div class="panel metrics-panel">
            <h2 data-i18n="metricsTitle">Метрики</h2>
            <table id="metricsTable" class="metrics-table"></table>
            <p id="metricsDenominatorNote" class="metrics-note muted"></p>
            <div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap">
              <a id="csvLink"><button data-i18n="saveCsv">Сохранить CSV</button></a>
              <a id="pdfLink"><button data-i18n="savePdf">Сохранить PDF-отчет</button></a>
              <button id="runFilesBtn" type="button" disabled data-i18n="viewRunFiles">Просмотреть файлы</button>
            </div>
          </div>
        </div>
      </div>
      <div id="batchView" class="hidden">
        <div class="batch-page">
          <div class="panel">
            <div class="batch-head">
              <div>
                <h2 data-i18n="batchPage">Серии</h2>
                <div id="batchSummary" class="batch-summary"></div>
              </div>
              <div class="batch-actions">
                <input id="batchFileInput" class="hidden" type="file" multiple accept=".png,.jpg,.jpeg,.tif,.tiff,.raw,.dng,.cr2,.cr3,.nef,.arw,.orf,.rw2,.raf,.pef,.srw,image/png,image/jpeg,image/tiff">
                <button id="newBatchBtn" type="button" data-i18n="batchNew">Новая серия</button>
                <button id="runBatchBtn" type="button" class="primary" disabled data-i18n="batchRun">Запустить серию</button>
                <button id="stopBatchBtn" type="button" class="danger hidden" disabled data-i18n="stop">Стоп</button>
              </div>
            </div>
          </div>
          <div class="panel">
            <h2 data-i18n="batchSharedSettings">Общие настройки</h2>
            <div id="batchSettingsSummary" class="batch-summary"></div>
          </div>
          <div class="panel">
            <div class="batch-panel-head">
              <h2 data-i18n="batchGallery">Галерея</h2>
              <button id="addBatchImagesBtn" type="button" data-i18n="batchAddImages">Добавить изображения</button>
            </div>
            <div id="batchStatus" class="muted"></div>
            <div id="batchGallery" class="batch-gallery"></div>
          </div>
        </div>
      </div>
      <div id="historyView" class="hidden">
        <div class="panel">
          <div class="history-page-head">
            <h2 data-i18n="historyPage">История запусков</h2>
            <div class="segmented" id="historyModeButtons">
              <button data-history-mode="all" class="active" data-i18n="historyModeAllRuns">все запуски</button>
              <button data-history-mode="single" data-i18n="historyModeSingleRuns">одиночные запуски</button>
              <button data-history-mode="batches" data-i18n="historyModeBatches">серии</button>
            </div>
          </div>
          <div id="historyPageList"></div>
        </div>
      </div>
      <div id="settingsView" class="hidden">
        <div class="settings-page">
          <div class="panel">
            <h2 data-i18n="settingsPage">Настройки</h2>
            <p class="muted" data-i18n="settingsIntro">Эти настройки сохраняются на сервере приложения и применяются для всех браузеров, открывающих этот рабочий каталог.</p>
          </div>
          <div class="panel">
            <h2 data-i18n="settingsUiDefaults">Интерфейс</h2>
            <div class="settings-grid">
              <label class="settings-field"><span data-i18n="languageLabel">Язык</span><select id="settingsLanguage">
                <option value="ru" data-i18n="languageRussian">Русский</option>
                <option value="en" data-i18n="languageEnglish">English</option>
              </select></label>
              <label class="settings-field"><span data-i18n="themeLabel">Тема</span><select id="settingsTheme">
                <option value="system" data-i18n="themeSystem">Системная</option>
                <option value="light" data-i18n="themeLight">Светлая</option>
                <option value="dark" data-i18n="themeDark">Темная</option>
              </select></label>
              <label class="check settings-field"><input type="checkbox" id="settingsShowTiling"> <span data-i18n="settingsShowTilingDefault">показывать тайлы по умолчанию</span></label>
            </div>
          </div>
          <div class="panel">
            <h2 data-i18n="settingsPreprocessDefaults">Предобработка по умолчанию</h2>
            <div class="settings-preprocess-defaults">
              <div class="settings-grid settings-preprocess-main">
                <label class="check settings-field"><input type="checkbox" id="settingsPreprocessingEnabled"> <span data-i18n="preprocessing">Предобработка</span></label>
                <label class="check settings-field"><input type="checkbox" id="settingsIllumination"> <span data-i18n="illuminationNormalization">нормализация освещения</span></label>
                <label class="check settings-field"><input type="checkbox" id="settingsDenoise"> <span data-i18n="denoise">шумоподавление</span></label>
                <label class="check settings-field"><input type="checkbox" id="settingsContrast"> <span data-i18n="contrastCorrection">коррекция контраста</span></label>
              </div>
              <div class="settings-section-divider" aria-hidden="true"></div>
              <div class="settings-scale-group">
                <label class="check settings-field"><input type="checkbox" id="settingsPanoramaScaling"> <span data-i18n="panoramaScaling">масштабирование для панорамных снимков</span></label>
                <div class="settings-panorama-controls">
                  <label class="panorama-scale-field"><span data-i18n="panoramaScalingMode">режим масштабирования</span><select id="settingsPanoramaScalingMode">
                    <option value="max_side" data-i18n="panoramaScalingModeMaxSide">граница по длинной стороне</option>
                    <option value="scale_factor" data-i18n="panoramaScalingModeFactor">коэффициент</option>
                  </select></label>
                  <label class="panorama-scale-field"><span data-i18n="panoramaScalingMaxSide">Длинная сторона, px</span><input id="settingsPanoramaMaxSidePx" type="number" min="64" max="12000" step="1" value="1800"></label>
                  <label class="panorama-scale-field"><span data-i18n="panoramaScalingFactor">Коэффициент, x</span><input id="settingsPanoramaScaleFactor" type="number" min="0.05" max="1" step="0.05" value="0.5"></label>
                </div>
              </div>
            </div>
          </div>
          <div class="panel">
            <h2 data-i18n="settingsMetadataDefaults">Метаданные сессии по умолчанию</h2>
            <div class="settings-grid">
              <label class="settings-field"><span data-i18n="metadataProject">Проект</span><input id="settingsMetaProject" type="text"></label>
              <label class="settings-field"><span data-i18n="metadataInstrument">Микроскоп/камера</span><input id="settingsMetaInstrument" type="text"></label>
              <label class="settings-field"><span data-i18n="metadataObjective">Объектив</span><input id="settingsMetaObjective" type="text"></label>
              <label class="settings-field"><span data-i18n="metadataScaleSource">Источник масштаба</span><select id="settingsMetaScaleSource">
                <option value="unavailable" data-i18n="metadataScaleUnavailable">недоступен</option>
                <option value="manual" data-i18n="metadataScaleManual">ручной ввод</option>
                <option value="visible_scale_bar" data-i18n="metadataScaleBar">видимая линейка</option>
                <option value="instrument_sidecar" data-i18n="metadataScaleSidecar">служебный файл прибора</option>
                <option value="calibration_slide" data-i18n="metadataScaleSlide">калибровочное стекло</option>
              </select></label>
              <label class="settings-field"><span data-i18n="metadataScaleValue">Масштаб, мкм/пиксель</span><input id="settingsMetaPixelSize" type="text" inputmode="decimal"></label>
              <label class="settings-field"><span data-i18n="metadataScaleConfidence">Доверие масштаба</span><select id="settingsMetaScaleConfidence">
                <option value="none" data-i18n="metadataConfidenceNone">нет</option>
                <option value="weak" data-i18n="metadataConfidenceWeak">слабое</option>
                <option value="calibrated" data-i18n="metadataConfidenceCalibrated">калиброванное</option>
              </select></label>
              <label class="settings-field"><span data-i18n="metadataReviewStatus">Статус проверки</span><select id="settingsMetaReviewStatus">
                <option value="unreviewed" data-i18n="metadataReviewUnreviewed">не проверено</option>
                <option value="reviewed" data-i18n="metadataReviewReviewed">проверено</option>
                <option value="needs_manual_review" data-i18n="metadataReviewNeeds">нужна ручная проверка</option>
                <option value="bad_image" data-i18n="metadataReviewBad">плохое изображение</option>
              </select></label>
            </div>
          </div>
          <div class="panel">
            <div class="settings-actions">
              <button id="saveSettingsBtn" class="primary" type="button" data-i18n="settingsSave">Сохранить настройки</button>
              <button id="resetSettingsBtn" type="button" data-i18n="settingsReset">Сбросить по умолчанию</button>
              <span id="settingsStatus" class="muted"></span>
            </div>
          </div>
        </div>
      </div>
    </section>
  </main>
  <dialog id="preprocessDialog" class="preprocess-dialog">
    <div class="modal-head">
      <strong data-i18n="preprocessingSettingsTitle">Настройки предобработки</strong>
      <button id="closePreprocessBtn" data-i18n="close">Закрыть</button>
    </div>
    <div class="preprocess-settings">
      <p class="muted" data-i18n="preprocessingSettingsIntro">Эти параметры применяются только когда включена предобработка.</p>
      <div class="preprocess-option">
        <label class="check"><input type="checkbox" id="illumination" checked> <span data-i18n="illuminationNormalization">нормализация освещения</span></label>
        <button class="help-dot" type="button" title="Выравнивает неравномерное освещение перед сегментацией." aria-label="Выравнивает неравномерное освещение перед сегментацией." data-tooltip="Выравнивает неравномерное освещение перед сегментацией." data-i18n-title="illuminationNormalizationHelp" data-i18n-aria-label="illuminationNormalizationHelp" data-i18n-tooltip="illuminationNormalizationHelp">(?)</button>
      </div>
      <div class="preprocess-option">
        <label class="check"><input type="checkbox" id="denoise" checked> <span data-i18n="denoise">шумоподавление</span></label>
        <button class="help-dot" type="button" title="Подавляет мелкий шум, сохраняя крупные структуры руды." aria-label="Подавляет мелкий шум, сохраняя крупные структуры руды." data-tooltip="Подавляет мелкий шум, сохраняя крупные структуры руды." data-i18n-title="denoiseHelp" data-i18n-aria-label="denoiseHelp" data-i18n-tooltip="denoiseHelp">(?)</button>
      </div>
      <div class="preprocess-option">
        <label class="check"><input type="checkbox" id="contrast" checked> <span data-i18n="contrastCorrection">коррекция контраста</span></label>
        <button class="help-dot" type="button" title="Мягко усиливает тональный контраст для проверки сульфидов и матрицы." aria-label="Мягко усиливает тональный контраст для проверки сульфидов и матрицы." data-tooltip="Мягко усиливает тональный контраст для проверки сульфидов и матрицы." data-i18n-title="contrastCorrectionHelp" data-i18n-aria-label="contrastCorrectionHelp" data-i18n-tooltip="contrastCorrectionHelp">(?)</button>
      </div>
      <div class="preprocess-option with-extra">
        <label class="check"><input type="checkbox" id="panoramaScaling" checked> <span data-i18n="panoramaScaling">масштабирование для панорамных снимков</span></label>
        <button class="help-dot" type="button" title="Включает явное уменьшение панорам: до заданной длинной стороны или по коэффициенту. Если выключено, применяется обычный рабочий размер, а тайлинг остается независимым." aria-label="Включает явное уменьшение панорам: до заданной длинной стороны или по коэффициенту. Если выключено, применяется обычный рабочий размер, а тайлинг остается независимым." data-tooltip="Включает явное уменьшение панорам: до заданной длинной стороны или по коэффициенту. Если выключено, применяется обычный рабочий размер, а тайлинг остается независимым." data-i18n-title="panoramaScalingHelp" data-i18n-aria-label="panoramaScalingHelp" data-i18n-tooltip="panoramaScalingHelp">(?)</button>
        <div class="panorama-scaling-controls">
          <label class="panorama-scale-field"><span data-i18n="panoramaScalingMode">режим масштабирования</span><select id="panoramaScalingMode">
            <option value="max_side" data-i18n="panoramaScalingModeMaxSide">граница по длинной стороне</option>
            <option value="scale_factor" data-i18n="panoramaScalingModeFactor">коэффициент</option>
          </select></label>
          <label class="panorama-scale-field"><span data-i18n="panoramaScalingMaxSide">Длинная сторона, px</span><input id="panoramaMaxSidePx" type="number" min="64" max="12000" step="1" value="1800"></label>
          <label class="panorama-scale-field"><span data-i18n="panoramaScalingFactor">Коэффициент, x</span><input id="panoramaScaleFactor" type="number" min="0.05" max="1" step="0.05" value="0.5"></label>
        </div>
      </div>
    </div>
    <div class="modal-foot">
      <span class="muted" data-i18n="preprocessingDialogHint">Нажмите «Применить» в боковой панели, чтобы обновить предпросмотр.</span>
      <button id="donePreprocessBtn" type="button" class="primary" data-i18n="done">Готово</button>
    </div>
  </dialog>
  <dialog id="augmentationDialog" class="augmentation-dialog">
    <div class="modal-head">
      <strong data-i18n="augmentationSettingsTitle">Настройки аугментации</strong>
      <button id="closeAugmentationBtn" data-i18n="close">Закрыть</button>
    </div>
    <div class="augmentation-settings">
      <p class="muted" data-i18n="augmentationSettingsIntro">Эти параметры создают один детерминированный вариант изображения без изменения геометрии перед предобработкой.</p>
      <fieldset class="settings-group">
        <legend data-i18n="augmentationColorGroup">Color and tone</legend>
        <label class="range-field"><span data-i18n="augBrightness">Brightness</span><input id="augBrightness" type="range" min="-50" max="50" step="1"><output id="augBrightnessValue"></output></label>
        <label class="range-field"><span data-i18n="augContrast">Contrast</span><input id="augContrast" type="range" min="-50" max="80" step="1"><output id="augContrastValue"></output></label>
        <label class="range-field"><span data-i18n="augSaturation">Saturation</span><input id="augSaturation" type="range" min="-60" max="80" step="1"><output id="augSaturationValue"></output></label>
        <label class="range-field"><span data-i18n="augHue">Hue</span><input id="augHue" type="range" min="-30" max="30" step="1"><output id="augHueValue"></output></label>
        <label class="range-field"><span data-i18n="augGamma">Gamma</span><input id="augGamma" type="range" min="0.5" max="2" step="0.05"><output id="augGammaValue"></output></label>
      </fieldset>
      <fieldset class="settings-group">
        <legend data-i18n="augmentationAcquisitionGroup">Acquisition noise</legend>
        <label class="range-field"><span data-i18n="augBlur">Blur radius</span><input id="augBlur" type="range" min="0" max="3" step="0.1"><output id="augBlurValue"></output></label>
        <label class="range-field"><span data-i18n="augNoise">Gaussian noise</span><input id="augNoise" type="range" min="0" max="20" step="1"><output id="augNoiseValue"></output></label>
        <label class="range-field"><span data-i18n="augSeed">Seed</span><input id="augSeed" type="number" min="0" max="2147483647" step="1"><output></output></label>
      </fieldset>
      <fieldset class="settings-group">
        <legend data-i18n="augmentationSurfaceGroup">Артефакты шлифовки/полировки</legend>
        <label class="range-field"><span data-i18n="augScratchCount">Scratches</span><input id="augScratchCount" type="range" min="0" max="80" step="1"><output id="augScratchCountValue"></output></label>
        <label class="range-field"><span data-i18n="augScratchIntensity">Scratch intensity</span><input id="augScratchIntensity" type="range" min="0" max="60" step="1"><output id="augScratchIntensityValue"></output></label>
        <label class="range-field"><span data-i18n="augPolishingHaze">Polishing haze</span><input id="augPolishingHaze" type="range" min="0" max="50" step="1"><output id="augPolishingHazeValue"></output></label>
        <label class="range-field"><span data-i18n="augPitCount">Pits/dust specks</span><input id="augPitCount" type="range" min="0" max="300" step="1"><output id="augPitCountValue"></output></label>
        <label class="range-field"><span data-i18n="augPitIntensity">Pit/dust intensity</span><input id="augPitIntensity" type="range" min="0" max="60" step="1"><output id="augPitIntensityValue"></output></label>
      </fieldset>
    </div>
    <div class="modal-foot">
      <span class="muted" data-i18n="augmentationDialogHint">Нажмите «Применить» в предобработке, чтобы обновить отладочные превью перед Стартом.</span>
      <button id="doneAugmentationBtn" type="button" class="primary" data-i18n="done">Готово</button>
    </div>
  </dialog>
  <dialog id="metadataDialog" class="metadata-dialog">
    <div class="modal-head">
      <strong data-i18n="metadataTitle">Метаданные изображения</strong>
      <button id="closeMetadataBtn" data-i18n="close">Закрыть</button>
    </div>
    <div class="metadata-body">
      <div class="segmented metadata-tabs" id="metadataTabs">
        <button data-metadata-tab="domain" class="active" data-i18n="metadataDomainTab">Домен</button>
        <button data-metadata-tab="raw" data-i18n="metadataRawTab">Raw</button>
        <button data-metadata-tab="defaults" data-i18n="metadataDefaultsTab">Шаблон сессии</button>
      </div>
      <section id="metadataDomainPanel" class="metadata-panel">
        <p class="muted" data-i18n="metadataIntro">Заполните только известные поля. DPI и подсказки 5x/10x не используются как калиброванный масштаб.</p>
        <div id="metadataScaleWarning" class="metadata-warning hidden" data-i18n="metadataScaleWarning">Масштаб задан без калиброванного источника. В отчетах следует использовать пиксельные площади и доли.</div>
        <div class="metadata-grid">
          <div class="metadata-section-title" data-i18n="metadataSessionSpecific">Для сессии</div>
          <label class="metadata-field"><span data-i18n="metadataProject">Проект</span><input type="text" data-metadata-field="project"></label>
          <label class="metadata-field"><span data-i18n="metadataInstrument">Микроскоп/камера</span><input type="text" data-metadata-field="om_instrument"></label>
          <label class="metadata-field"><span data-i18n="metadataObjective">Объектив</span><input type="text" data-metadata-field="om_objective_magnification"></label>
          <label class="metadata-field"><span data-i18n="metadataScaleSource">Источник масштаба</span><select data-metadata-field="scale_source">
            <option value="unavailable" data-i18n="metadataScaleUnavailable">недоступен</option>
            <option value="manual" data-i18n="metadataScaleManual">ручной ввод</option>
            <option value="visible_scale_bar" data-i18n="metadataScaleBar">видимая линейка</option>
            <option value="instrument_sidecar" data-i18n="metadataScaleSidecar">служебный файл прибора</option>
            <option value="calibration_slide" data-i18n="metadataScaleSlide">калибровочное стекло</option>
          </select></label>
          <label class="metadata-field"><span data-i18n="metadataScaleValue">Масштаб, мкм/пиксель</span><input type="text" inputmode="decimal" data-metadata-field="pixel_size_um"></label>
          <label class="metadata-field"><span data-i18n="metadataScaleConfidence">Доверие масштаба</span><select data-metadata-field="scale_confidence">
            <option value="none" data-i18n="metadataConfidenceNone">нет</option>
            <option value="weak" data-i18n="metadataConfidenceWeak">слабое</option>
            <option value="calibrated" data-i18n="metadataConfidenceCalibrated">калиброванное</option>
          </select></label>
          <div class="metadata-section-title" data-i18n="metadataSampleSpecific">Для образца</div>
          <label class="metadata-field"><span data-i18n="metadataSampleId">ID образца</span><input type="text" data-metadata-field="sample_id"></label>
          <label class="metadata-field"><span data-i18n="metadataRunLabel">Метка запуска</span><input type="text" data-metadata-field="run_label"></label>
          <label class="metadata-field"><span data-i18n="metadataSourceRole">Роль источника</span><select data-metadata-field="source_role">
            <option value=""></option>
            <option value="original_image" data-i18n="metadataSourceOriginal">оригинал</option>
            <option value="panorama" data-i18n="metadataSourcePanorama">панорама</option>
            <option value="annotation_image" data-i18n="metadataSourceAnnotation">аннотация</option>
            <option value="unknown" data-i18n="metadataSourceUnknown">неизвестно</option>
          </select></label>
          <label class="metadata-field"><span data-i18n="metadataTaskLabel">Метка задачи</span><select data-metadata-field="task_label">
            <option value=""></option>
            <option value="ordinary_intergrowth" data-i18n="metadataTaskOrdinary">обычные срастания</option>
            <option value="fine_intergrowth" data-i18n="metadataTaskFine">тонкие срастания</option>
            <option value="talcose" data-i18n="metadataTaskTalcose">оталькованная руда</option>
            <option value="unknown" data-i18n="metadataSourceUnknown">неизвестно</option>
          </select></label>
          <label class="metadata-field"><span data-i18n="metadataFilenameHint">Подсказка из имени файла</span><input type="text" data-metadata-field="filename_magnification_hint"></label>
          <label class="metadata-field"><span data-i18n="metadataReviewStatus">Статус проверки</span><select data-metadata-field="review_status">
            <option value="unreviewed" data-i18n="metadataReviewUnreviewed">не проверено</option>
            <option value="reviewed" data-i18n="metadataReviewReviewed">проверено</option>
            <option value="needs_manual_review" data-i18n="metadataReviewNeeds">нужна ручная проверка</option>
            <option value="bad_image" data-i18n="metadataReviewBad">плохое изображение</option>
          </select></label>
          <label class="check metadata-field"><input type="checkbox" data-metadata-field="exclude_from_training"> <span data-i18n="metadataExcludeTraining">исключить изображение из обучения/валидации</span></label>
          <label class="metadata-field full"><span data-i18n="metadataNotes">Заметки</span><textarea rows="4" data-metadata-field="sample_notes"></textarea></label>
        </div>
      </section>
      <section id="metadataRawPanel" class="metadata-panel" hidden>
        <p class="muted" data-i18n="metadataRawIntro">Raw-метаданные доступны только для просмотра и не меняют исходный файл.</p>
        <div class="metadata-raw-wrap">
          <table id="metadataRawTable"></table>
        </div>
      </section>
      <section id="metadataDefaultsPanel" class="metadata-panel" hidden>
        <p class="muted" data-i18n="metadataDefaultsIntro">Шаблон хранится только в браузере и применяется к повторяющимся полям новых образцов.</p>
        <div class="metadata-raw-wrap">
          <table id="metadataDefaultsTable"></table>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap">
          <button id="saveMetadataDefaultsBtn" type="button" data-i18n="metadataSaveDefaults">Сохранить текущие как шаблон</button>
          <button id="clearMetadataDefaultsBtn" type="button" data-i18n="metadataClearDefaults">Очистить шаблон</button>
        </div>
      </section>
    </div>
    <div class="modal-foot">
      <button id="applyMetadataDefaultsBtn" type="button" data-i18n="metadataApplyDefaults">Применить шаблон</button>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <button id="cancelMetadataBtn" type="button" data-i18n="cancel">Отмена</button>
        <button id="saveMetadataBtn" type="button" class="primary" data-i18n="metadataSave">Сохранить метаданные</button>
      </div>
    </div>
  </dialog>
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
  <dialog id="runFilesDialog" class="run-files-dialog">
    <div class="modal-head">
      <strong data-i18n="runFilesTitle">Файлы запуска</strong>
      <button id="closeRunFilesBtn" data-i18n="close">Закрыть</button>
    </div>
    <div class="run-files-body">
      <p id="runFilesStatus" class="muted" data-i18n="runFilesLoading">Загрузка списка файлов...</p>
      <div class="run-files-table-wrap">
        <table id="runFilesTable" class="run-files-table"></table>
      </div>
    </div>
    <div class="modal-foot">
      <span id="runFilesSummary" class="muted"></span>
      <a id="runFilesZipLink"><button type="button" data-i18n="downloadZip">Скачать ZIP</button></a>
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
        <button id="panToolBtn" data-i18n="pan">Перемещение</button>
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
          <button data-layer="artifact" class="active" data-i18n="artefactsLayerShort">артефакты</button>
          <button data-layer="sulfide" data-i18n="sulfideLayerShort">сульфиды</button>
          <button data-layer="final" data-i18n="finalLayerShort">финал</button>
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
        <p class="muted" id="editorHelpText" data-i18n="editorHelp">Кисть: левая кнопка рисует, правая стирает. Перемещение двигает вид.</p>
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
      curatedMetadata: null,
      settings: null,
      batch: null,
      batchPollId: null,
      historyMode: 'all',
      historyRuns: [],
      historyBatches: [],
      metadataTarget: {type: 'workspace', itemId: null},
      returnToBatchId: null,
      viewMode: 'original',
      sideLayer: 'none',
      splitter: 0.5,
      pan: {x: 0, y: 0},
      zoom: 1,
      dragging: false,
      dragSplitter: false,
      last: {x: 0, y: 0},
      images: new Map(),
      boundaryImages: new Map(),
      overlayOpacity: 0.65,
      boundaryOnly: false,
      classVisibility: {
        showSulfide: true,
        showNonSulfide: true,
        showSulfideArtifacts: true,
        showBackground: true,
        showOrdinary: true,
        showFine: true,
        showTalc: true,
        showFinalArtifacts: true
      },
      editor: {
        layer: 'artifact',
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
    const AUGMENTATION_STORAGE_KEY = 'orePipelineAugmentationSettings';
    const METADATA_STORAGE_KEY = 'orePipelineMetadataDefaults';
    const DEFAULT_LANGUAGE = 'ru';
    const DEFAULT_PREPROCESS_PRESET = {
      preprocessing_enabled: true,
      illumination_normalization: true,
      denoise: true,
      contrast_correction: true,
      panorama_scaling: true,
      panorama_scaling_mode: 'max_side',
      panorama_max_side_px: 1800,
      panorama_scale_factor: 0.5
    };
    const DEFAULT_AUGMENTATION_SETTINGS = {
      schema_version: 'ore-pipeline-augmentation-v0.1',
      enabled: false,
      color: {
        brightness_pct: 4,
        contrast_pct: 6,
        saturation_pct: 3,
        hue_degrees: 0,
        gamma: 1
      },
      acquisition: {
        blur_radius: 0,
        gaussian_noise_std: 0
      },
      surface_artifacts: {
        scratch_count: 6,
        scratch_intensity_pct: 14,
        polishing_haze_pct: 7,
        pit_count: 18,
        pit_intensity_pct: 12
      },
      runtime: {
        geometry_preserving: true,
        coordinate_mode: 'original',
        random_seed: 0
      }
    };
    const DEFAULT_APP_SETTINGS = {
      language: DEFAULT_LANGUAGE,
      theme: 'system',
      show_tiling: false,
      preprocess: {...DEFAULT_PREPROCESS_PRESET},
      metadata_defaults: {}
    };
    let statusMessage = {key: 'statusWaiting', params: {}};
    let settingsStatusMessage = null;
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
        batchTab: 'Серии',
        historyTab: 'История',
        settingsTab: 'Настройки',
        batchPage: 'Серии',
        batchNew: 'Новая серия',
        batchAddImages: 'Добавить изображения',
        batchRun: 'Запустить серию',
        batchSharedSettings: 'Общие настройки',
        batchGallery: 'Галерея',
        batchBack: 'Назад к серии',
        batchEditMetadata: 'Редактировать метаданные...',
        batchLoad: 'Загрузить',
        batchRemoveImage: 'Удалить',
        batchRemoveImageConfirm: 'Удалить изображение «{name}» из серии?',
        batchImageRemoved: 'Изображение удалено из серии.',
        batchRemoveFailed: 'Не удалось удалить изображение: {error}',
        batchNoImages: 'Добавьте изображения, чтобы собрать серию.',
        batchNoBatch: 'Серия еще не создана.',
        batchItemsSummary: '{count} изображений · статус: {status}',
        batchSettingsSummary: 'Предобработка: {preprocess}. Аугментация: {augmentation}.',
        batchProgressLabel: '{stage} · {progress}%',
        batchUploading: 'Загрузка {done}/{total}: {name}',
        batchAddingImages: 'Добавление изображений...',
        batchMetadataSaved: 'Метаданные сохранены.',
        batchRunStarted: 'Серия запущена.',
        batchRunFailed: 'Не удалось запустить серию: {error}',
        batchUploadFailed: 'Не удалось добавить изображения: {error}',
        batchStopFailed: 'Не удалось остановить серию: {error}',
        batchStatusDraft: 'черновик',
        batchStatusQueued: 'в очереди',
        batchStatusRunning: 'выполнение',
        batchStatusCanceling: 'остановка',
        batchStatusCanceled: 'остановлено',
        batchStatusComplete: 'готово',
        batchStatusPartial: 'частично',
        batchStatusFailed: 'ошибка',
        settingsPage: 'Настройки',
        settingsIntro: 'Эти настройки сохраняются на сервере приложения и применяются для всех браузеров, открывающих этот рабочий каталог.',
        settingsUiDefaults: 'Интерфейс',
        settingsShowTilingDefault: 'показывать тайлы по умолчанию',
        settingsPreprocessDefaults: 'Предобработка по умолчанию',
        settingsMetadataDefaults: 'Метаданные сессии по умолчанию',
        settingsSave: 'Сохранить настройки',
        settingsReset: 'Сбросить по умолчанию',
        settingsLoaded: 'Настройки загружены.',
        settingsSaved: 'Настройки сохранены.',
        settingsResetDone: 'Настройки сброшены.',
        settingsLoadFailed: 'Не удалось загрузить настройки: {error}',
        settingsSaveFailed: 'Не удалось сохранить настройки: {error}',
        settingsResetConfirm: 'Сбросить системные настройки по умолчанию?',
        inputImage: 'Входное изображение',
        dropImageHere: 'Перетащите изображение сюда',
        dropImageHelp: 'или нажмите, чтобы открыть PNG, JPEG, TIFF, RAW',
        clearImage: 'Очистить изображение',
        noImageLoaded: 'Изображение не загружено.',
        selectedImage: 'Выбранное изображение',
        editMetadata: 'Редактировать метаданные...',
        metadataTitle: 'Метаданные изображения',
        metadataDomainTab: 'Домен',
        metadataRawTab: 'Raw',
        metadataDefaultsTab: 'Шаблон сессии',
        metadataIntro: 'Заполните только известные поля. DPI и подсказки 5x/10x не используются как калиброванный масштаб.',
        metadataScaleWarning: 'Масштаб задан без калиброванного источника. В отчетах следует использовать пиксельные площади и доли.',
        metadataSessionSpecific: 'Для сессии',
        metadataSampleSpecific: 'Для образца',
        metadataSampleId: 'ID образца',
        metadataRunLabel: 'Метка запуска',
        metadataProject: 'Проект',
        metadataSourceRole: 'Роль источника',
        metadataSourceOriginal: 'оригинал',
        metadataSourcePanorama: 'панорама',
        metadataSourceAnnotation: 'аннотация',
        metadataSourceUnknown: 'неизвестно',
        metadataTaskLabel: 'Метка задачи',
        metadataTaskOrdinary: 'обычные срастания',
        metadataTaskFine: 'тонкие срастания',
        metadataTaskTalcose: 'оталькованная руда',
        metadataInstrument: 'Микроскоп/камера',
        metadataObjective: 'Объектив',
        metadataFilenameHint: 'Подсказка из имени файла',
        metadataScaleValue: 'Масштаб, мкм/пиксель',
        metadataScaleSource: 'Источник масштаба',
        metadataScaleUnavailable: 'недоступен',
        metadataScaleManual: 'ручной ввод',
        metadataScaleBar: 'видимая линейка',
        metadataScaleSidecar: 'служебный файл прибора',
        metadataScaleSlide: 'калибровочное стекло',
        metadataScaleConfidence: 'Доверие масштаба',
        metadataConfidenceNone: 'нет',
        metadataConfidenceWeak: 'слабое',
        metadataConfidenceCalibrated: 'калиброванное',
        metadataReviewStatus: 'Статус проверки',
        metadataReviewUnreviewed: 'не проверено',
        metadataReviewReviewed: 'проверено',
        metadataReviewNeeds: 'нужна ручная проверка',
        metadataReviewBad: 'плохое изображение',
        metadataExcludeTraining: 'исключить изображение из обучения/валидации',
        metadataNotes: 'Заметки',
        metadataRawIntro: 'Raw-метаданные доступны только для просмотра и не меняют исходный файл.',
        metadataDefaultsIntro: 'Шаблон берется из системных настроек и применяется к повторяющимся полям новых образцов.',
        metadataSaveDefaults: 'Сохранить текущие как шаблон',
        metadataClearDefaults: 'Очистить шаблон',
        metadataApplyDefaults: 'Применить шаблон',
        metadataSave: 'Сохранить метаданные',
        metadataNoDefaults: 'Шаблон сессии пуст.',
        metadataField: 'Поле',
        metadataValue: 'Значение',
        cancel: 'Отмена',
        invalidImageFormat: 'Неподдерживаемый формат файла: {name}. Поддерживаются PNG, JPEG, TIFF, RAW.',
        uploadFailed: 'Не удалось загрузить файл: {error}',
        uploadProgressUploading: 'Загрузка файла: {progress}%',
        uploadProgressPreparing: 'Подготовка предпросмотра: {progress}%',
        uploadProgressComplete: 'Предпросмотр готов.',
        statusUploadingProgress: 'Загрузка {name} · {progress}%',
        statusPreparingPreview: 'Подготовка предпросмотра · {progress}%',
        augmentation: 'Аугментация',
        editAugmentation: 'Настроить...',
        augmentationSettingsTitle: 'Настройки аугментации',
        augmentationSettingsIntro: 'Эти параметры создают один детерминированный вариант изображения без изменения геометрии перед предобработкой.',
        augmentationColorGroup: 'Цвет и тон',
        augmentationAcquisitionGroup: 'Шум съемки',
        augmentationSurfaceGroup: 'Артефакты шлифовки/полировки',
        augmentationSummaryDisabled: 'Аугментация выключена.',
        augmentationSummaryEnabled: 'Включено: {items}.',
        augmentationDialogHint: 'Нажмите «Применить» в блоке аугментации, чтобы обновить отладочный предпросмотр перед Стартом.',
        augBrightness: 'яркость',
        augContrast: 'контраст',
        augSaturation: 'насыщенность',
        augHue: 'оттенок',
        augGamma: 'гамма',
        augBlur: 'радиус размытия',
        augNoise: 'гауссов шум',
        augSeed: 'зерно',
        augScratchCount: 'царапины',
        augScratchIntensity: 'интенсивность царапин',
        augPolishingHaze: 'полировочная дымка',
        augPitCount: 'ямки/пылинки',
        augPitIntensity: 'интенсивность ямок/пылинок',
        applyAugmentation: 'Применить',
        preprocessing: 'Предобработка',
        editPreprocessing: 'Настроить...',
        preprocessingSettingsTitle: 'Настройки предобработки',
        preprocessingSettingsIntro: 'Эти параметры применяются только когда включена предобработка.',
        preprocessingDialogHint: 'Нажмите «Применить» в боковой панели, чтобы обновить предпросмотр.',
        preprocessingSummaryDisabled: 'Предобработка будет пропущена при запуске.',
        preprocessingSummaryEnabled: 'Включено: {items}.',
        preprocessingSummaryNone: 'без дополнительных фильтров',
        done: 'Готово',
        illuminationNormalization: 'нормализация освещения',
        illuminationNormalizationHelp: 'Выравнивает неравномерное освещение перед сегментацией.',
        denoise: 'шумоподавление',
        denoiseHelp: 'Подавляет мелкий шум, сохраняя крупные структуры руды.',
        contrastCorrection: 'коррекция контраста',
        contrastCorrectionHelp: 'Мягко усиливает тональный контраст для проверки сульфидов и матрицы.',
        panoramaScaling: 'масштабирование для панорамных снимков',
        panoramaScalingHelp: 'Включает явное уменьшение панорам: до заданной длинной стороны или по коэффициенту. Если выключено, применяется обычный рабочий размер, а тайлинг остается независимым.',
        panoramaScalingMode: 'режим масштабирования',
        panoramaScalingModeMaxSide: 'граница по длинной стороне',
        panoramaScalingModeFactor: 'коэффициент',
        panoramaScalingMaxSide: 'Длинная сторона, px',
        panoramaScalingFactor: 'Коэффициент, x',
        panoramaScalingMaxSideSummary: 'панорама до {value} px',
        panoramaScalingFactorSummary: 'панорама {value}x',
        applyPreprocessing: 'Применить',
        runTitle: 'Запуск',
        start: 'Старт',
        stop: 'Стоп',
        historyTitle: 'История',
        historyNoRuns: 'Запусков пока нет.',
        viewOriginal: 'оригинал',
        viewAugmented: 'аугментированное',
        viewArtefacts: 'артефакты',
        viewPreprocessed: 'предобработка',
        viewSulfide: 'сульфиды',
        viewFinal: 'финал',
        sideBySide: 'Сравнение:',
        sideNone: 'нет',
        leftViewLegend: 'Левый слой',
        rightViewLegend: 'Правый слой',
        classBackground: 'фон',
        classSulfides: 'сульфиды',
        classNonSulfides: 'не-сульфиды',
        classOrdinaryShort: 'обычные',
        classFineShort: 'тонкие',
        classOrdinary: 'обычные срастания',
        classFine: 'тонкие срастания',
        classTalc: 'тальк',
        classArtefacts: 'артефакты',
        showTiling: 'показать тайлы',
        overlayOpacity: 'прозрачность',
        boundaryOnly: 'только контуры',
        fixMe: 'Исправить',
        textOutputTitle: 'Текстовый вывод',
        metricsTitle: 'Метрики',
        saveCsv: 'Сохранить CSV',
        savePdf: 'Сохранить PDF-отчет',
        viewRunFiles: 'Просмотреть файлы',
        runFilesTitle: 'Файлы запуска',
        runFilesLoading: 'Загрузка списка файлов...',
        runFilesEmpty: 'Файлов нет.',
        runFilesLoadFailed: 'Не удалось загрузить список файлов: {error}',
        runFilesHeaderPath: 'Файл',
        runFilesHeaderKind: 'Тип',
        runFilesHeaderSize: 'Размер',
        runFilesHeaderImageSize: 'Изображение',
        runFilesSummary: '{count} файлов · {size}',
        runFilesImageKind: 'изображение',
        runFilesFileKind: 'файл',
        downloadZip: 'Скачать ZIP',
        historyPage: 'История запусков',
        historyModeAllRuns: 'все запуски',
        historyModeSingleRuns: 'одиночные запуски',
        historyModeBatches: 'серии',
        historyNoBatches: 'Серий пока нет.',
        historyBatchId: 'Серия',
        historyBatchStatus: 'Статус',
        historyBatchImages: 'Изображения',
        historyBatchProgress: 'Прогресс',
        historyBatchCounts: 'Итоги',
        historyOpenBatch: 'Открыть',
        editRecalculate: 'Редактирование и пересчет',
        close: 'Закрыть',
        tool: 'Инструмент',
        brush: 'Кисть',
        pan: 'Перемещение',
        undo: 'Отменить',
        redo: 'Повторить',
        fitView: 'Вписать',
        brushSize: 'Размер кисти',
        layer: 'Слой',
        artefactsLayer: 'артефакты',
        artefactsLayerShort: 'артефакты',
        sulfideLayer: 'сульфиды/не сульфиды',
        sulfideLayerShort: 'сульфиды',
        finalSegmentation: 'финальная сегментация',
        finalLayerShort: 'финал',
        classTitle: 'Класс',
        comment: 'Комментарий',
        commentPlaceholder: 'Комментарий к изменению',
        editorHelp: 'Кисть: левая кнопка рисует, правая стирает. Перемещение двигает вид.',
        editorArtifactHelp: 'Артефакты шлифовки и полировки: фиолетовая кисть помечает области, исключенные из всех шагов.',
        statistics: 'Статистика',
        editNoEdits: 'Нет правок.',
        editSavedArtefacts: 'Артефакты сохранены. Нажмите Старт, чтобы запустить расчет с исключением этих областей.',
        editorLoading: 'Загрузка слоя...',
        editorLoadFailed: 'Не удалось загрузить изображение или сегментацию: {error}',
        editorMissingMask: 'нет маски выбранного слоя',
        editorMissingBaseImage: 'нет изображения подложки',
        fixAndRestart: 'Исправить и перезапустить',
        saveArtefacts: 'Сохранить артефакты',
        statusWaiting: 'Ожидание изображения.',
        statusUploading: 'Загрузка {name}',
        statusImageLoaded: 'Изображение загружено.',
        statusAugmentationUpdated: 'Аугментированный предпросмотр обновлен.',
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
        metricsHeaderAreaPx: 'Площадь, px',
        metricsHeaderPhysicalArea: 'Физическая площадь',
        metricsDenominatorNote: 'Сульфиды, тальк и остальное считаются от проанализированной области; обычные и тонкие срастания — от площади сульфидов; артефакты — от всего изображения.',
        decisionRationaleText: 'Основание: сульфиды — {sulfidePct}% проанализированной области; тальк — {talcPct}%; обычные/тонкие срастания — {ordinaryPct}%/{finePct}% от сульфидов; отрыв тонких от обычных — {intergrowthMargin} п.п.; запас по тальку — {talcMargin} п.п.{warnings}',
        decisionWarnings: '; предупреждения: {warnings}',
        notAvailable: 'н/д',
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
        metricOtherFraction: 'Остальное',
        metricArtifactFraction: 'Доля артефактов изображения',
        historyLoad: 'Загрузить',
        historyRemove: 'Удалить',
        confirmRemoveRun: 'Удалить запуск {runId}?',
        statusRunRemoved: 'Запуск {runId} удален из истории.',
        editUndo: 'Отмена применена.',
        editRedo: 'Повтор применен.',
        editUnsaved: 'Есть несохраненная правка.',
        editEraseStroke: 'Штрих стирания.',
        editDrawStroke: 'Штрих рисования.',
        statArtefacts: 'артефакты',
        statCleanArea: 'область без артефактов',
        statSulfide: 'сульфиды',
        statNonSulfide: 'не сульфиды',
        statOrdinary: 'обычные срастания',
        statFine: 'тонкие срастания',
        statTalc: 'тальк',
        statOfImage: 'от изображения',
        statOfSulfides: 'от сульфидов',
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
        batchTab: 'Series',
        historyTab: 'History',
        settingsTab: 'Settings',
        batchPage: 'Series',
        batchNew: 'New Series',
        batchAddImages: 'Add images',
        batchRun: 'Run Series',
        batchSharedSettings: 'Shared settings',
        batchGallery: 'Gallery',
        batchBack: 'Back to Series',
        batchEditMetadata: 'Edit Metadata...',
        batchLoad: 'Load',
        batchRemoveImage: 'Remove',
        batchRemoveImageConfirm: 'Remove image "{name}" from this Series?',
        batchImageRemoved: 'Image removed from this Series.',
        batchRemoveFailed: 'Could not remove image: {error}',
        batchNoImages: 'Add images to build a Series.',
        batchNoBatch: 'No Series created yet.',
        batchItemsSummary: '{count} images · status: {status}',
        batchSettingsSummary: 'Preprocessing: {preprocess}. Augmentation: {augmentation}.',
        batchProgressLabel: '{stage} · {progress}%',
        batchUploading: 'Uploading {done}/{total}: {name}',
        batchAddingImages: 'Adding images...',
        batchMetadataSaved: 'Metadata saved.',
        batchRunStarted: 'Series started.',
        batchRunFailed: 'Could not start Series: {error}',
        batchUploadFailed: 'Could not add images: {error}',
        batchStopFailed: 'Could not stop Series: {error}',
        batchStatusDraft: 'draft',
        batchStatusQueued: 'queued',
        batchStatusRunning: 'running',
        batchStatusCanceling: 'stopping',
        batchStatusCanceled: 'stopped',
        batchStatusComplete: 'complete',
        batchStatusPartial: 'partial',
        batchStatusFailed: 'failed',
        settingsPage: 'Settings',
        settingsIntro: 'These settings are saved by the app server and apply to every browser that opens this workspace.',
        settingsUiDefaults: 'Interface',
        settingsShowTilingDefault: 'show tiling by default',
        settingsPreprocessDefaults: 'Default preprocessing',
        settingsMetadataDefaults: 'Default session metadata',
        settingsSave: 'Save settings',
        settingsReset: 'Reset to defaults',
        settingsLoaded: 'Settings loaded.',
        settingsSaved: 'Settings saved.',
        settingsResetDone: 'Settings reset.',
        settingsLoadFailed: 'Could not load settings: {error}',
        settingsSaveFailed: 'Could not save settings: {error}',
        settingsResetConfirm: 'Reset system settings to defaults?',
        inputImage: 'Input image',
        dropImageHere: 'Drop image here',
        dropImageHelp: 'or click to open PNG, JPEG, TIFF, RAW',
        clearImage: 'Clear image',
        noImageLoaded: 'No image loaded.',
        selectedImage: 'Selected image',
        editMetadata: 'Edit Metadata...',
        metadataTitle: 'Image metadata',
        metadataDomainTab: 'Domain',
        metadataRawTab: 'Raw',
        metadataDefaultsTab: 'Session Defaults',
        metadataIntro: 'Fill only known fields. DPI and 5x/10x filename hints are not used as calibrated scale.',
        metadataScaleWarning: 'Scale value is set without a calibrated scale source. Reports should use pixel areas and fractions.',
        metadataSessionSpecific: 'Session specific',
        metadataSampleSpecific: 'Sample specific',
        metadataSampleId: 'Sample ID',
        metadataRunLabel: 'Run label',
        metadataProject: 'Project',
        metadataSourceRole: 'Source role',
        metadataSourceOriginal: 'original',
        metadataSourcePanorama: 'panorama',
        metadataSourceAnnotation: 'annotation',
        metadataSourceUnknown: 'unknown',
        metadataTaskLabel: 'Task label',
        metadataTaskOrdinary: 'ordinary intergrowth',
        metadataTaskFine: 'fine intergrowth',
        metadataTaskTalcose: 'talcose ore',
        metadataInstrument: 'Microscope/camera',
        metadataObjective: 'Objective',
        metadataFilenameHint: 'Filename hint',
        metadataScaleValue: 'Scale value, µm/px',
        metadataScaleSource: 'Scale source',
        metadataScaleUnavailable: 'unavailable',
        metadataScaleManual: 'manual',
        metadataScaleBar: 'visible scale bar',
        metadataScaleSidecar: 'instrument sidecar',
        metadataScaleSlide: 'calibration slide',
        metadataScaleConfidence: 'Scale confidence',
        metadataConfidenceNone: 'none',
        metadataConfidenceWeak: 'weak',
        metadataConfidenceCalibrated: 'calibrated',
        metadataReviewStatus: 'Review status',
        metadataReviewUnreviewed: 'unreviewed',
        metadataReviewReviewed: 'reviewed',
        metadataReviewNeeds: 'needs manual review',
        metadataReviewBad: 'bad image',
        metadataExcludeTraining: 'Exclude this image from training/validation sets',
        metadataNotes: 'Notes',
        metadataRawIntro: 'Raw metadata is read-only and does not modify the source file.',
        metadataDefaultsIntro: 'The template comes from system settings and applies to repeated fields for new samples.',
        metadataSaveDefaults: 'Save current as template',
        metadataClearDefaults: 'Clear template',
        metadataApplyDefaults: 'Apply template',
        metadataSave: 'Save metadata',
        metadataNoDefaults: 'Session template is empty.',
        metadataField: 'Field',
        metadataValue: 'Value',
        cancel: 'Cancel',
        invalidImageFormat: 'Unsupported file format: {name}. Supported formats: PNG, JPEG, TIFF, RAW.',
        uploadFailed: 'Could not upload file: {error}',
        uploadProgressUploading: 'Uploading file: {progress}%',
        uploadProgressPreparing: 'Preparing preview: {progress}%',
        uploadProgressComplete: 'Preview is ready.',
        statusUploadingProgress: 'Uploading {name} · {progress}%',
        statusPreparingPreview: 'Preparing preview · {progress}%',
        augmentation: 'Augmentation',
        editAugmentation: 'Edit',
        augmentationSettingsTitle: 'Augmentation settings',
        augmentationSettingsIntro: 'These settings create one deterministic, geometry-preserving augmented image before preprocessing.',
        augmentationColorGroup: 'Color and tone',
        augmentationAcquisitionGroup: 'Acquisition noise',
        augmentationSurfaceGroup: 'Grinding/polishing artifacts',
        augmentationSummaryDisabled: 'Augmentation is off.',
        augmentationSummaryEnabled: 'Enabled: {items}.',
        augmentationDialogHint: 'Press Apply in Augmentation to refresh the debug preview before Start.',
        augBrightness: 'brightness',
        augContrast: 'contrast',
        augSaturation: 'saturation',
        augHue: 'hue',
        augGamma: 'gamma',
        augBlur: 'blur radius',
        augNoise: 'Gaussian noise',
        augSeed: 'seed',
        augScratchCount: 'scratches',
        augScratchIntensity: 'scratch intensity',
        augPolishingHaze: 'polishing haze',
        augPitCount: 'pits/dust specks',
        augPitIntensity: 'pit/dust intensity',
        applyAugmentation: 'Apply',
        preprocessing: 'Preprocessing',
        editPreprocessing: 'Edit...',
        preprocessingSettingsTitle: 'Preprocessing settings',
        preprocessingSettingsIntro: 'These settings apply only when preprocessing is enabled.',
        preprocessingDialogHint: 'Press Apply in the sidebar to refresh the preview.',
        preprocessingSummaryDisabled: 'Preprocessing will be skipped on Start.',
        preprocessingSummaryEnabled: 'Enabled: {items}.',
        preprocessingSummaryNone: 'no additional filters',
        done: 'Done',
        illuminationNormalization: 'illumination normalization',
        illuminationNormalizationHelp: 'Balances uneven lighting before segmentation.',
        denoise: 'noise reduction',
        denoiseHelp: 'Suppresses small image noise while preserving larger ore structures.',
        contrastCorrection: 'contrast correction',
        contrastCorrectionHelp: 'Gently increases tonal separation for sulfide and matrix inspection.',
        panoramaScaling: 'panorama image scaling',
        panoramaScalingHelp: 'Enables explicit panorama downscaling: to a longest-side bound or by a scale factor. When off, normal processing size is used and tiling remains independent.',
        panoramaScalingMode: 'scaling mode',
        panoramaScalingModeMaxSide: 'longest side bound',
        panoramaScalingModeFactor: 'scale factor',
        panoramaScalingMaxSide: 'Longest side, px',
        panoramaScalingFactor: 'Scale factor, x',
        panoramaScalingMaxSideSummary: 'panorama to {value} px',
        panoramaScalingFactorSummary: 'panorama {value}x',
        applyPreprocessing: 'Apply',
        runTitle: 'Run',
        start: 'Start',
        stop: 'Stop',
        historyTitle: 'History',
        historyNoRuns: 'No runs yet.',
        viewOriginal: 'original',
        viewAugmented: 'augmented',
        viewArtefacts: 'artefacts',
        viewPreprocessed: 'preprocessed',
        viewSulfide: 'sulfide',
        viewFinal: 'final',
        sideBySide: 'Side-by-side:',
        sideNone: 'none',
        leftViewLegend: 'Left layer',
        rightViewLegend: 'Right layer',
        classBackground: 'background',
        classSulfides: 'sulfides',
        classNonSulfides: 'non-sulfides',
        classOrdinaryShort: 'ordinary',
        classFineShort: 'fine',
        classOrdinary: 'ordinary intergrowth',
        classFine: 'fine intergrowth',
        classTalc: 'talc',
        classArtefacts: 'artefacts',
        showTiling: 'show tiling',
        overlayOpacity: 'opacity',
        boundaryOnly: 'contours only',
        fixMe: 'Fix me',
        textOutputTitle: 'Text output',
        metricsTitle: 'Metrics',
        saveCsv: 'Save to CSV',
        savePdf: 'Save PDF Report',
        viewRunFiles: 'View files',
        runFilesTitle: 'Run files',
        runFilesLoading: 'Loading file list...',
        runFilesEmpty: 'No files.',
        runFilesLoadFailed: 'Could not load file list: {error}',
        runFilesHeaderPath: 'File',
        runFilesHeaderKind: 'Type',
        runFilesHeaderSize: 'Size',
        runFilesHeaderImageSize: 'Image',
        runFilesSummary: '{count} files · {size}',
        runFilesImageKind: 'image',
        runFilesFileKind: 'file',
        downloadZip: 'Download ZIP',
        historyPage: 'History page',
        historyModeAllRuns: 'all runs',
        historyModeSingleRuns: 'single runs',
        historyModeBatches: 'series',
        historyNoBatches: 'No series yet.',
        historyBatchId: 'Series',
        historyBatchStatus: 'Status',
        historyBatchImages: 'Images',
        historyBatchProgress: 'Progress',
        historyBatchCounts: 'Counts',
        historyOpenBatch: 'Open',
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
        artefactsLayer: 'artefacts',
        artefactsLayerShort: 'artefacts',
        sulfideLayer: 'sulfide/non-sulfide',
        sulfideLayerShort: 'sulfides',
        finalSegmentation: 'final segmentation',
        finalLayerShort: 'final',
        classTitle: 'Class',
        comment: 'Comment',
        commentPlaceholder: 'Comment for the change',
        editorHelp: 'Brush: left draws, right erases. Pan moves the view.',
        editorArtifactHelp: 'Grinding and polishing artefacts: the violet brush marks regions excluded from every pipeline step.',
        statistics: 'Statistics',
        editNoEdits: 'No edits yet.',
        editSavedArtefacts: 'Artefacts saved. Press Start to run with these regions excluded.',
        editorLoading: 'Loading layer...',
        editorLoadFailed: 'Could not load image or segmentation: {error}',
        editorMissingMask: 'selected layer mask is missing',
        editorMissingBaseImage: 'base image is missing',
        fixAndRestart: 'Fix and Restart',
        saveArtefacts: 'Save Artefacts',
        statusWaiting: 'Waiting for image.',
        statusUploading: 'Uploading {name}',
        statusImageLoaded: 'Image loaded.',
        statusAugmentationUpdated: 'Augmented preview updated.',
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
        metricsHeaderAreaPx: 'Area, px',
        metricsHeaderPhysicalArea: 'Physical area',
        metricsDenominatorNote: 'Sulfides, talc, and other use analyzed area as denominator; ordinary and fine intergrowths use sulfide area; artefacts use the whole image.',
        decisionRationaleText: 'Rationale: sulfides {sulfidePct}% of analyzed area; talc {talcPct}%; ordinary/fine intergrowths {ordinaryPct}%/{finePct}% of sulfides; fine-vs-ordinary margin {intergrowthMargin} pp; talc margin {talcMargin} pp{warnings}',
        decisionWarnings: '; warnings: {warnings}',
        notAvailable: 'n/a',
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
        metricOtherFraction: 'Other',
        metricArtifactFraction: 'Image artefact fraction',
        historyLoad: 'Load',
        historyRemove: 'Remove',
        confirmRemoveRun: 'Remove run {runId}?',
        statusRunRemoved: 'Run {runId} removed from history.',
        editUndo: 'Undo applied.',
        editRedo: 'Redo applied.',
        editUnsaved: 'Unsaved edit.',
        editEraseStroke: 'Erase stroke.',
        editDrawStroke: 'Draw stroke.',
        statArtefacts: 'artefacts',
        statCleanArea: 'non-artefact area',
        statSulfide: 'sulfide',
        statNonSulfide: 'non-sulfide',
        statOrdinary: 'ordinary intergrowth',
        statFine: 'fine intergrowth',
        statTalc: 'talc',
        statOfImage: 'of image',
        statOfSulfides: 'of sulfides',
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
    function formatBytes(bytes) {
      const value = Number(bytes || 0);
      if (!Number.isFinite(value) || value <= 0) return '0 B';
      const units = ['B', 'KiB', 'MiB', 'GiB'];
      let scaled = value;
      let unitIndex = 0;
      while (scaled >= 1024 && unitIndex < units.length - 1) {
        scaled /= 1024;
        unitIndex += 1;
      }
      const precision = unitIndex === 0 ? 0 : (scaled >= 10 ? 1 : 2);
      return `${scaled.toFixed(precision)} ${units[unitIndex]}`;
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
        analyzed_fraction: 'metricAnalyzedFraction',
        other_fraction: 'metricOtherFraction',
        artifact_fraction_image: 'metricArtifactFraction'
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
    function formatMarginPercentPoints(value) {
      if (value == null || value === '') return t('notAvailable');
      const number = Number(value);
      return Number.isFinite(number) ? (number * 100).toFixed(1) : t('notAvailable');
    }
    function decisionRationale(run) {
      const summary = runSummary(run);
      if (!summary || !Object.keys(summary).length) return '';
      const warnings = Array.isArray(summary.warnings) && summary.warnings.length
        ? t('decisionWarnings', {warnings: summary.warnings.join(', ')})
        : '';
      return t('decisionRationaleText', {
        sulfidePct: fixedPercent(summary.sulfide_fraction),
        talcPct: fixedPercent(summary.talc_fraction),
        ordinaryPct: fixedPercent(summary.ordinary_sulfide_fraction),
        finePct: fixedPercent(summary.fine_sulfide_fraction),
        intergrowthMargin: formatMarginPercentPoints(summary.intergrowth_margin),
        talcMargin: formatMarginPercentPoints(summary.talc_margin),
        warnings
      });
    }
    function formatPhysicalArea(row) {
      if (row.area_um2 == null || row.area_um2 === '') return '';
      const areaUm2 = Number(row.area_um2);
      if (!Number.isFinite(areaUm2)) return '';
      if (areaUm2 >= 1000000) return `${(areaUm2 / 1000000).toFixed(6)} mm²`;
      return `${areaUm2.toFixed(3)} µm²`;
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
      document.querySelectorAll('[data-i18n-tooltip]').forEach(node => { node.dataset.tooltip = t(node.dataset.i18nTooltip); });
      document.querySelectorAll('[data-i18n-aria-label]').forEach(node => { node.setAttribute('aria-label', t(node.dataset.i18nAriaLabel)); });
      if (statusMessage) $('progressText').textContent = t(statusMessage.key, statusMessage.params);
      if (settingsStatusMessage) $('settingsStatus').textContent = t(settingsStatusMessage.key, settingsStatusMessage.params);
      if (uploadWarningMessage) setUploadWarning(uploadWarningMessage.key, uploadWarningMessage.params);
      if (uploadProgressMessage) setUploadProgress(uploadProgressMessage.key, uploadProgressMessage.progress, uploadProgressMessage.params);
      updateAugmentationValueLabels();
      updateAugmentationSummary();
      updatePreprocessSummary();
      renderBatch();
      renderSettingsForm(currentAppSettings());
      renderMetadataStatus();
      if ($('metadataDialog') && $('metadataDialog').open) {
        renderMetadataRawTable();
        renderMetadataDefaultsTable();
      }
      if (state.editor.statusMessage) $('editStatus').textContent = t(state.editor.statusMessage.key, state.editor.statusMessage.params);
      if ($('fixDialog') && $('fixDialog').open) {
        updateFixRestartLabel();
        $('editorHelpText').textContent = t(state.editor.layer === 'artifact' ? 'editorArtifactHelp' : 'editorHelp');
      }
      updateOverlayOpacityLabel();
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
    function rgbaComponentsFromCssColor(value, alpha = 180) {
      const color = String(value || '').trim();
      const hex = color.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
      if (hex) {
        const raw = hex[1].length === 3 ? hex[1].split('').map(ch => ch + ch).join('') : hex[1];
        return [parseInt(raw.slice(0, 2), 16), parseInt(raw.slice(2, 4), 16), parseInt(raw.slice(4, 6), 16), alpha];
      }
      const rgb = color.match(/^rgba?\(([^)]+)\)$/i);
      if (rgb) {
        const parts = rgb[1].split(',').map(part => Number.parseFloat(part.trim())).filter(value => Number.isFinite(value));
        if (parts.length >= 3) return [parts[0], parts[1], parts[2], alpha];
      }
      return [198, 60, 255, alpha];
    }
    function artifactOverlayColor(alpha = 180) {
      return rgbaComponentsFromCssColor(cssColor('--artifact') || '#c63cff', alpha);
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

    function clampNumberInput(value, fallback, min, max) {
      const parsed = Number.parseFloat(String(value ?? '').replace(',', '.'));
      const number = Number.isFinite(parsed) ? parsed : fallback;
      return Math.min(max, Math.max(min, number));
    }
    function normalizedPanoramaMode(value, fallback = DEFAULT_PREPROCESS_PRESET.panorama_scaling_mode) {
      return ['max_side', 'scale_factor'].includes(value) ? value : fallback;
    }
    function normalizedPanoramaMaxSide(value, fallback = DEFAULT_PREPROCESS_PRESET.panorama_max_side_px) {
      return Math.round(clampNumberInput(value, fallback, 64, 12000));
    }
    function normalizedPanoramaScaleFactor(value, fallback = DEFAULT_PREPROCESS_PRESET.panorama_scale_factor) {
      return clampNumberInput(value, fallback, 0.05, 1);
    }
    function formatScaleFactor(value) {
      return normalizedPanoramaScaleFactor(value).toFixed(2).replace(/\.?0+$/, '');
    }
    function preprocessPresetValue(values, primary, aliases = [], fallback = undefined) {
      if (Object.prototype.hasOwnProperty.call(values, primary)) return values[primary];
      for (const alias of aliases) {
        if (Object.prototype.hasOwnProperty.call(values, alias)) return values[alias];
      }
      return fallback;
    }
    function presetPayload() {
      return {
        preprocessing_enabled: $('preprocessingEnabled').checked,
        illumination_normalization: $('illumination').checked,
        denoise: $('denoise').checked,
        contrast_correction: $('contrast').checked,
        panorama_scaling: $('panoramaScaling').checked,
        panorama_scaling_mode: normalizedPanoramaMode($('panoramaScalingMode').value),
        panorama_max_side_px: normalizedPanoramaMaxSide($('panoramaMaxSidePx').value),
        panorama_scale_factor: normalizedPanoramaScaleFactor($('panoramaScaleFactor').value)
      };
    }
    function presetBoolean(values, primary, alias, fallback) {
      if (Object.prototype.hasOwnProperty.call(values, primary)) return Boolean(values[primary]);
      if (alias && Object.prototype.hasOwnProperty.call(values, alias)) return Boolean(values[alias]);
      return Boolean(fallback);
    }
    function normalizedPreprocessPreset(preset = {}, fallback = DEFAULT_PREPROCESS_PRESET) {
      const values = preset || {};
      const fallbackValues = fallback || DEFAULT_PREPROCESS_PRESET;
      return {
        preprocessing_enabled: presetBoolean(values, 'preprocessing_enabled', 'enabled', fallbackValues.preprocessing_enabled),
        illumination_normalization: presetBoolean(values, 'illumination_normalization', 'illumination', fallbackValues.illumination_normalization),
        denoise: presetBoolean(values, 'denoise', 'noise_reduction', fallbackValues.denoise),
        contrast_correction: presetBoolean(values, 'contrast_correction', 'contrast', fallbackValues.contrast_correction),
        panorama_scaling: presetBoolean(values, 'panorama_scaling', 'panoramaScaling', fallbackValues.panorama_scaling),
        panorama_scaling_mode: normalizedPanoramaMode(
          preprocessPresetValue(values, 'panorama_scaling_mode', ['panoramaScalingMode'], fallbackValues.panorama_scaling_mode)
        ),
        panorama_max_side_px: normalizedPanoramaMaxSide(
          preprocessPresetValue(values, 'panorama_max_side_px', ['panoramaMaxSidePx', 'panorama_max_side', 'panoramaMaxSide'], fallbackValues.panorama_max_side_px)
        ),
        panorama_scale_factor: normalizedPanoramaScaleFactor(
          preprocessPresetValue(values, 'panorama_scale_factor', ['panoramaScaleFactor', 'panorama_scaling_factor'], fallbackValues.panorama_scale_factor)
        )
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
    function metadataDefaultsFromLocalStorage() {
      try {
        const raw = localStorage.getItem(METADATA_STORAGE_KEY);
        return raw ? JSON.parse(raw) : {};
      } catch (_) {
        return {};
      }
    }
    function normalizedAppSettings(settings = {}) {
      const values = settings && typeof settings === 'object' ? settings : {};
      const language = I18N[values.language] ? values.language : DEFAULT_APP_SETTINGS.language;
      const theme = ['system', 'light', 'dark'].includes(values.theme) ? values.theme : DEFAULT_APP_SETTINGS.theme;
      const metadataDefaults = values.metadata_defaults && typeof values.metadata_defaults === 'object' ? values.metadata_defaults : {};
      return {
        schema_version: 'ore-pipeline-app-settings-v0.1',
        language,
        theme,
        show_tiling: Boolean(values.show_tiling),
        preprocess: normalizedPreprocessPreset(values.preprocess || {}, DEFAULT_APP_SETTINGS.preprocess),
        metadata_defaults: {...metadataDefaults}
      };
    }
    function currentAppSettings() {
      if (state.settings) return normalizedAppSettings(state.settings);
      return normalizedAppSettings({
        language: storedLanguageChoice(),
        theme: storedThemeChoice(),
        show_tiling: false,
        preprocess: storedPreprocessPreset(),
        metadata_defaults: metadataDefaultsFromLocalStorage()
      });
    }
    function persistSettingsLocally(settings) {
      const normalized = normalizedAppSettings(settings);
      try { localStorage.setItem(LANGUAGE_STORAGE_KEY, normalized.language); } catch (_) {}
      try { localStorage.setItem(THEME_STORAGE_KEY, normalized.theme); } catch (_) {}
      try { localStorage.setItem(PREPROCESS_STORAGE_KEY, JSON.stringify(normalized.preprocess)); } catch (_) {}
      try { localStorage.setItem(METADATA_STORAGE_KEY, JSON.stringify(normalized.metadata_defaults || {})); } catch (_) {}
    }
    function applyShowTilingDefault() {
      if (!$('showTiling')) return;
      $('showTiling').checked = Boolean(currentAppSettings().show_tiling);
      updateViewControls();
    }
    function savePreprocessPreset() {
      try { localStorage.setItem(PREPROCESS_STORAGE_KEY, JSON.stringify(presetPayload())); } catch (_) {}
    }
    function numericControl(id, fallback) {
      const value = Number($(id).value);
      return Number.isFinite(value) ? value : fallback;
    }
    function clampNumber(value, min, max) {
      const number = Number(value);
      if (!Number.isFinite(number)) return min;
      return Math.max(min, Math.min(max, number));
    }
    function augmentationPayload() {
      return normalizedAugmentationSettings({
        enabled: $('augmentationEnabled').checked,
        color: {
          brightness_pct: numericControl('augBrightness', DEFAULT_AUGMENTATION_SETTINGS.color.brightness_pct),
          contrast_pct: numericControl('augContrast', DEFAULT_AUGMENTATION_SETTINGS.color.contrast_pct),
          saturation_pct: numericControl('augSaturation', DEFAULT_AUGMENTATION_SETTINGS.color.saturation_pct),
          hue_degrees: numericControl('augHue', DEFAULT_AUGMENTATION_SETTINGS.color.hue_degrees),
          gamma: numericControl('augGamma', DEFAULT_AUGMENTATION_SETTINGS.color.gamma)
        },
        acquisition: {
          blur_radius: numericControl('augBlur', DEFAULT_AUGMENTATION_SETTINGS.acquisition.blur_radius),
          gaussian_noise_std: numericControl('augNoise', DEFAULT_AUGMENTATION_SETTINGS.acquisition.gaussian_noise_std)
        },
        surface_artifacts: {
          scratch_count: Math.round(numericControl('augScratchCount', DEFAULT_AUGMENTATION_SETTINGS.surface_artifacts.scratch_count)),
          scratch_intensity_pct: numericControl('augScratchIntensity', DEFAULT_AUGMENTATION_SETTINGS.surface_artifacts.scratch_intensity_pct),
          polishing_haze_pct: numericControl('augPolishingHaze', DEFAULT_AUGMENTATION_SETTINGS.surface_artifacts.polishing_haze_pct),
          pit_count: Math.round(numericControl('augPitCount', DEFAULT_AUGMENTATION_SETTINGS.surface_artifacts.pit_count)),
          pit_intensity_pct: numericControl('augPitIntensity', DEFAULT_AUGMENTATION_SETTINGS.surface_artifacts.pit_intensity_pct)
        },
        runtime: {
          geometry_preserving: true,
          coordinate_mode: 'original',
          random_seed: Math.round(numericControl('augSeed', DEFAULT_AUGMENTATION_SETTINGS.runtime.random_seed))
        }
      });
    }
    function normalizedAugmentationSettings(settings = {}, fallback = DEFAULT_AUGMENTATION_SETTINGS) {
      const values = settings && typeof settings === 'object' ? settings : {};
      const color = values.color && typeof values.color === 'object' ? values.color : {};
      const acquisition = values.acquisition && typeof values.acquisition === 'object' ? values.acquisition : {};
      const surface = values.surface_artifacts && typeof values.surface_artifacts === 'object' ? values.surface_artifacts : {};
      const runtime = values.runtime && typeof values.runtime === 'object' ? values.runtime : {};
      const fallbackColor = fallback.color || DEFAULT_AUGMENTATION_SETTINGS.color;
      const fallbackAcquisition = fallback.acquisition || DEFAULT_AUGMENTATION_SETTINGS.acquisition;
      const fallbackSurface = fallback.surface_artifacts || DEFAULT_AUGMENTATION_SETTINGS.surface_artifacts;
      const fallbackRuntime = fallback.runtime || DEFAULT_AUGMENTATION_SETTINGS.runtime;
      return {
        schema_version: 'ore-pipeline-augmentation-v0.1',
        enabled: Boolean(values.enabled),
        color: {
          brightness_pct: clampNumber(color.brightness_pct ?? fallbackColor.brightness_pct, -50, 50),
          contrast_pct: clampNumber(color.contrast_pct ?? fallbackColor.contrast_pct, -50, 80),
          saturation_pct: clampNumber(color.saturation_pct ?? fallbackColor.saturation_pct, -60, 80),
          hue_degrees: clampNumber(color.hue_degrees ?? fallbackColor.hue_degrees, -30, 30),
          gamma: clampNumber(color.gamma ?? fallbackColor.gamma, 0.5, 2)
        },
        acquisition: {
          blur_radius: clampNumber(acquisition.blur_radius ?? fallbackAcquisition.blur_radius, 0, 3),
          gaussian_noise_std: clampNumber(acquisition.gaussian_noise_std ?? fallbackAcquisition.gaussian_noise_std, 0, 20)
        },
        surface_artifacts: {
          scratch_count: Math.round(clampNumber(surface.scratch_count ?? fallbackSurface.scratch_count, 0, 120)),
          scratch_intensity_pct: clampNumber(surface.scratch_intensity_pct ?? fallbackSurface.scratch_intensity_pct, 0, 70),
          polishing_haze_pct: clampNumber(surface.polishing_haze_pct ?? fallbackSurface.polishing_haze_pct, 0, 60),
          pit_count: Math.round(clampNumber(surface.pit_count ?? fallbackSurface.pit_count, 0, 600)),
          pit_intensity_pct: clampNumber(surface.pit_intensity_pct ?? fallbackSurface.pit_intensity_pct, 0, 70)
        },
        runtime: {
          geometry_preserving: true,
          coordinate_mode: 'original',
          random_seed: Math.round(clampNumber(runtime.random_seed ?? fallbackRuntime.random_seed, 0, 2147483647))
        }
      };
    }
    function storedAugmentationSettings() {
      try {
        const raw = localStorage.getItem(AUGMENTATION_STORAGE_KEY);
        return raw ? normalizedAugmentationSettings(JSON.parse(raw), DEFAULT_AUGMENTATION_SETTINGS) : {...DEFAULT_AUGMENTATION_SETTINGS};
      } catch (_) {
        return {...DEFAULT_AUGMENTATION_SETTINGS};
      }
    }
    function saveAugmentationSettings() {
      try { localStorage.setItem(AUGMENTATION_STORAGE_KEY, JSON.stringify(augmentationPayload())); } catch (_) {}
    }
    function formatSignedPercent(value) {
      const number = Number(value || 0);
      return `${number > 0 ? '+' : ''}${number.toFixed(0)}%`;
    }
    function updateAugmentationValueLabels() {
      if (!$('augBrightnessValue')) return;
      $('augBrightnessValue').textContent = formatSignedPercent($('augBrightness').value);
      $('augContrastValue').textContent = formatSignedPercent($('augContrast').value);
      $('augSaturationValue').textContent = formatSignedPercent($('augSaturation').value);
      $('augHueValue').textContent = `${Number($('augHue').value || 0).toFixed(0)}°`;
      $('augGammaValue').textContent = Number($('augGamma').value || 1).toFixed(2);
      $('augBlurValue').textContent = Number($('augBlur').value || 0).toFixed(1);
      $('augNoiseValue').textContent = Number($('augNoise').value || 0).toFixed(0);
      $('augScratchCountValue').textContent = Number($('augScratchCount').value || 0).toFixed(0);
      $('augScratchIntensityValue').textContent = formatSignedPercent($('augScratchIntensity').value);
      $('augPolishingHazeValue').textContent = formatSignedPercent($('augPolishingHaze').value);
      $('augPitCountValue').textContent = Number($('augPitCount').value || 0).toFixed(0);
      $('augPitIntensityValue').textContent = formatSignedPercent($('augPitIntensity').value);
    }
    function applyAugmentationToControls(settings, options = {}) {
      const normalized = normalizedAugmentationSettings(settings || {}, options.fallback || DEFAULT_AUGMENTATION_SETTINGS);
      $('augmentationEnabled').checked = normalized.enabled;
      $('augBrightness').value = normalized.color.brightness_pct;
      $('augContrast').value = normalized.color.contrast_pct;
      $('augSaturation').value = normalized.color.saturation_pct;
      $('augHue').value = normalized.color.hue_degrees;
      $('augGamma').value = normalized.color.gamma;
      $('augBlur').value = normalized.acquisition.blur_radius;
      $('augNoise').value = normalized.acquisition.gaussian_noise_std;
      $('augScratchCount').value = normalized.surface_artifacts.scratch_count;
      $('augScratchIntensity').value = normalized.surface_artifacts.scratch_intensity_pct;
      $('augPolishingHaze').value = normalized.surface_artifacts.polishing_haze_pct;
      $('augPitCount').value = normalized.surface_artifacts.pit_count;
      $('augPitIntensity').value = normalized.surface_artifacts.pit_intensity_pct;
      $('augSeed').value = normalized.runtime.random_seed;
      updateAugmentationValueLabels();
      updateAugmentationSummary();
      renderBatch();
      if (options.save) saveAugmentationSettings();
    }
    function updateAugmentationSummary() {
      if (!$('augmentationSummary')) return;
      if (!$('augmentationEnabled').checked) {
        $('augmentationSummary').textContent = t('augmentationSummaryDisabled');
        return;
      }
      const payload = augmentationPayload();
      const items = [
        `${t('augBrightness')} ${formatSignedPercent(payload.color.brightness_pct)}`,
        `${t('augContrast')} ${formatSignedPercent(payload.color.contrast_pct)}`,
        `${t('augSaturation')} ${formatSignedPercent(payload.color.saturation_pct)}`,
      ];
      if (Math.abs(payload.color.hue_degrees) > 0.001) items.push(`${t('augHue')} ${payload.color.hue_degrees}°`);
      if (Math.abs(payload.color.gamma - 1) > 0.001) items.push(`${t('augGamma')} ${payload.color.gamma.toFixed(2)}`);
      if (payload.acquisition.blur_radius > 0) items.push(`${t('augBlur')} ${payload.acquisition.blur_radius}`);
      if (payload.acquisition.gaussian_noise_std > 0) items.push(`${t('augNoise')} ${payload.acquisition.gaussian_noise_std}`);
      if (payload.surface_artifacts.scratch_count > 0 && payload.surface_artifacts.scratch_intensity_pct > 0) items.push(`${t('augScratchCount')} ${payload.surface_artifacts.scratch_count}`);
      if (payload.surface_artifacts.polishing_haze_pct > 0) items.push(`${t('augPolishingHaze')} ${payload.surface_artifacts.polishing_haze_pct}%`);
      if (payload.surface_artifacts.pit_count > 0 && payload.surface_artifacts.pit_intensity_pct > 0) items.push(`${t('augPitCount')} ${payload.surface_artifacts.pit_count}`);
      $('augmentationSummary').textContent = t('augmentationSummaryEnabled', {items: items.join(', ')});
    }
    ['augmentationEnabled','augBrightness','augContrast','augSaturation','augHue','augGamma','augBlur','augNoise','augScratchCount','augScratchIntensity','augPolishingHaze','augPitCount','augPitIntensity','augSeed'].forEach(id => $(id).addEventListener('change', () => {
      updateAugmentationValueLabels();
      updateAugmentationSummary();
      renderBatch();
      saveAugmentationSettings();
      updateViewControls();
      drawMain();
    }));
    ['augBrightness','augContrast','augSaturation','augHue','augGamma','augBlur','augNoise','augScratchCount','augScratchIntensity','augPolishingHaze','augPitCount','augPitIntensity'].forEach(id => $(id).addEventListener('input', () => {
      updateAugmentationValueLabels();
      updateAugmentationSummary();
    }));
    applyAugmentationToControls(storedAugmentationSettings(), {save: false});
    function panoramaScalingSummaryItem(preset) {
      const normalized = normalizedPreprocessPreset(preset);
      if (!normalized.panorama_scaling) return null;
      if (normalized.panorama_scaling_mode === 'scale_factor') {
        return t('panoramaScalingFactorSummary', {value: formatScaleFactor(normalized.panorama_scale_factor)});
      }
      return t('panoramaScalingMaxSideSummary', {value: normalized.panorama_max_side_px});
    }
    function updatePanoramaScalingControls(prefix = '') {
      const enabledId = prefix ? `${prefix}PanoramaScaling` : 'panoramaScaling';
      const modeId = prefix ? `${prefix}PanoramaScalingMode` : 'panoramaScalingMode';
      const maxSideId = prefix ? `${prefix}PanoramaMaxSidePx` : 'panoramaMaxSidePx';
      const factorId = prefix ? `${prefix}PanoramaScaleFactor` : 'panoramaScaleFactor';
      const enabled = $(enabledId).checked;
      const mode = normalizedPanoramaMode($(modeId).value);
      $(modeId).disabled = !enabled;
      $(maxSideId).disabled = !enabled || mode !== 'max_side';
      $(factorId).disabled = !enabled || mode !== 'scale_factor';
    }
    function applyPresetToControls(preset, options = {}) {
      const values = preset || {};
      const normalized = normalizedPreprocessPreset(values, options.fallback || DEFAULT_PREPROCESS_PRESET);
      $('preprocessingEnabled').checked = normalized.preprocessing_enabled;
      $('illumination').checked = normalized.illumination_normalization;
      $('denoise').checked = normalized.denoise;
      $('contrast').checked = normalized.contrast_correction;
      $('panoramaScaling').checked = normalized.panorama_scaling;
      $('panoramaScalingMode').value = normalized.panorama_scaling_mode;
      $('panoramaMaxSidePx').value = normalized.panorama_max_side_px;
      $('panoramaScaleFactor').value = formatScaleFactor(normalized.panorama_scale_factor);
      updatePanoramaScalingControls();
      if (options.save) savePreprocessPreset();
      updatePreprocessSummary();
      renderBatch();
    }
    function updatePreprocessSummary() {
      if (!$('preprocessSummary')) return;
      if (!$('preprocessingEnabled').checked) {
        $('preprocessSummary').textContent = t('preprocessingSummaryDisabled');
        return;
      }
      const items = [];
      if ($('illumination').checked) items.push(t('illuminationNormalization'));
      if ($('denoise').checked) items.push(t('denoise'));
      if ($('contrast').checked) items.push(t('contrastCorrection'));
      const panoramaItem = panoramaScalingSummaryItem(presetPayload());
      if (panoramaItem) items.push(panoramaItem);
      $('preprocessSummary').textContent = t('preprocessingSummaryEnabled', {items: items.join(', ') || t('preprocessingSummaryNone')});
    }
    ['preprocessingEnabled','illumination','denoise','contrast','panoramaScaling','panoramaScalingMode','panoramaMaxSidePx','panoramaScaleFactor'].forEach(id => $(id).addEventListener('change', () => {
      updatePanoramaScalingControls();
      savePreprocessPreset();
      updatePreprocessSummary();
      renderBatch();
      updateViewControls();
      drawMain();
    }));
    ['panoramaMaxSidePx','panoramaScaleFactor'].forEach(id => $(id).addEventListener('input', () => {
      updatePreprocessSummary();
    }));
    applyPresetToControls(storedPreprocessPreset(), {save: false});
    function renderSettingsForm(settings = currentAppSettings()) {
      const normalized = normalizedAppSettings(settings);
      if (!$('settingsLanguage')) return;
      $('settingsLanguage').value = normalized.language;
      $('settingsTheme').value = normalized.theme;
      $('settingsShowTiling').checked = normalized.show_tiling;
      $('settingsPreprocessingEnabled').checked = normalized.preprocess.preprocessing_enabled;
      $('settingsIllumination').checked = normalized.preprocess.illumination_normalization;
      $('settingsDenoise').checked = normalized.preprocess.denoise;
      $('settingsContrast').checked = normalized.preprocess.contrast_correction;
      $('settingsPanoramaScaling').checked = normalized.preprocess.panorama_scaling;
      $('settingsPanoramaScalingMode').value = normalized.preprocess.panorama_scaling_mode;
      $('settingsPanoramaMaxSidePx').value = normalized.preprocess.panorama_max_side_px;
      $('settingsPanoramaScaleFactor').value = formatScaleFactor(normalized.preprocess.panorama_scale_factor);
      updatePanoramaScalingControls('settings');
      const defaults = normalized.metadata_defaults || {};
      $('settingsMetaProject').value = defaults.project || '';
      $('settingsMetaInstrument').value = defaults.om_instrument || '';
      $('settingsMetaObjective').value = defaults.om_objective_magnification || '';
      $('settingsMetaScaleSource').value = defaults.scale_source || 'unavailable';
      $('settingsMetaPixelSize').value = defaults.pixel_size_um || '';
      $('settingsMetaScaleConfidence').value = defaults.scale_confidence || 'none';
      $('settingsMetaReviewStatus').value = defaults.review_status || 'unreviewed';
    }
    function collectSettingsMetadataDefaults() {
      const defaults = {
        project: $('settingsMetaProject').value.trim(),
        om_instrument: $('settingsMetaInstrument').value.trim(),
        om_objective_magnification: $('settingsMetaObjective').value.trim(),
        scale_source: $('settingsMetaScaleSource').value,
        pixel_size_um: $('settingsMetaPixelSize').value.trim(),
        scale_confidence: $('settingsMetaScaleConfidence').value,
        review_status: $('settingsMetaReviewStatus').value
      };
      return Object.fromEntries(Object.entries(defaults).filter(([, value]) => value !== '' && value != null));
    }
    function collectSettingsForm() {
      return normalizedAppSettings({
        language: $('settingsLanguage').value,
        theme: $('settingsTheme').value,
        show_tiling: $('settingsShowTiling').checked,
        preprocess: {
          preprocessing_enabled: $('settingsPreprocessingEnabled').checked,
          illumination_normalization: $('settingsIllumination').checked,
          denoise: $('settingsDenoise').checked,
          contrast_correction: $('settingsContrast').checked,
          panorama_scaling: $('settingsPanoramaScaling').checked,
          panorama_scaling_mode: normalizedPanoramaMode($('settingsPanoramaScalingMode').value),
          panorama_max_side_px: normalizedPanoramaMaxSide($('settingsPanoramaMaxSidePx').value),
          panorama_scale_factor: normalizedPanoramaScaleFactor($('settingsPanoramaScaleFactor').value)
        },
        metadata_defaults: collectSettingsMetadataDefaults()
      });
    }
    function setSettingsStatus(key = null, params = {}) {
      settingsStatusMessage = key ? {key, params} : null;
      if (!$('settingsStatus')) return;
      $('settingsStatus').textContent = key ? t(key, params) : '';
    }
    function applyAppSettings(settings) {
      const normalized = normalizedAppSettings(settings);
      state.settings = normalized;
      persistSettingsLocally(normalized);
      setLanguage(normalized.language);
      setThemeChoice(normalized.theme);
      applyPresetToControls(normalized.preprocess, {save: true});
      applyShowTilingDefault();
      renderSettingsForm(normalized);
      renderMetadataDefaultsTable();
    }
    async function saveSettingsObject(settings, successKey = 'settingsSaved') {
      const response = await fetch('/api/settings', {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(settings)
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'settings save failed');
      applyAppSettings(payload);
      setSettingsStatus(successKey);
      return payload;
    }
    async function loadAppSettings() {
      try {
        const response = await fetch('/api/settings');
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'settings load failed');
        applyAppSettings(payload);
      } catch (error) {
        setSettingsStatus('settingsLoadFailed', {error: error.message || t('unknownError')});
        renderSettingsForm(currentAppSettings());
      }
    }
    async function saveSettingsFromPage() {
      try {
        await saveSettingsObject(collectSettingsForm(), 'settingsSaved');
      } catch (error) {
        setSettingsStatus('settingsSaveFailed', {error: error.message || t('unknownError')});
      }
    }
    async function resetSettingsFromPage() {
      if (!window.confirm(t('settingsResetConfirm'))) return;
      try {
        await saveSettingsObject(DEFAULT_APP_SETTINGS, 'settingsResetDone');
      } catch (error) {
        setSettingsStatus('settingsSaveFailed', {error: error.message || t('unknownError')});
      }
    }
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
      $('fixBtn').disabled = active || !(state.upload || (run && run.status === 'complete'));
    }
    function previewForCanvasAspect() {
      const source = displaySource();
      const display = source && source.display ? source.display : {};
      const keys = state.sideLayer !== 'none'
        ? [state.viewMode, state.sideLayer, baseLayerKey(display), 'original']
        : [state.viewMode, baseLayerKey(display), 'original'];
      for (const key of keys) {
        const previews = display[key];
        if (previews && previews.length) return previews[previews.length - 1];
      }
      return null;
    }
    function adjustMainCanvasHeight() {
      const shell = canvas.parentElement;
      if (!shell) return;
      const preview = previewForCanvasAspect();
      const viewportLimit = Math.max(320, Math.min(window.innerHeight * 0.72, 760));
      const minHeight = window.innerWidth < 980 ? 340 : 420;
      let target = viewportLimit;
      if (preview && preview.width && preview.height) {
        const width = Math.max(320, shell.clientWidth || canvas.clientWidth || 800);
        const aspect = Math.max(0.05, Number(preview.width) / Math.max(1, Number(preview.height)));
        target = Math.max(minHeight, Math.min(viewportLimit, width / aspect));
      }
      const pixels = `${Math.round(target)}px`;
      canvas.style.height = pixels;
      shell.style.minHeight = pixels;
    }
    function resizeCanvas() {
      adjustMainCanvasHeight();
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
    function boundaryCanvasForImage(image, key) {
      if (!image || !key) return image;
      const cacheKey = `${key}:${image.width}x${image.height}`;
      if (state.boundaryImages.has(cacheKey)) return state.boundaryImages.get(cacheKey);
      const source = document.createElement('canvas');
      source.width = image.width;
      source.height = image.height;
      const sourceCtx = source.getContext('2d', {willReadFrequently: true});
      sourceCtx.drawImage(image, 0, 0);
      const input = sourceCtx.getImageData(0, 0, source.width, source.height);
      const output = sourceCtx.createImageData(source.width, source.height);
      const w = source.width;
      const h = source.height;
      const alphaAt = (x, y) => {
        if (x < 0 || y < 0 || x >= w || y >= h) return 0;
        return input.data[(y * w + x) * 4 + 3];
      };
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const index = (y * w + x) * 4;
          if (input.data[index + 3] <= 24) continue;
          if (alphaAt(x - 1, y) > 24 && alphaAt(x + 1, y) > 24 && alphaAt(x, y - 1) > 24 && alphaAt(x, y + 1) > 24) continue;
          output.data[index] = input.data[index];
          output.data[index + 1] = input.data[index + 1];
          output.data[index + 2] = input.data[index + 2];
          output.data[index + 3] = 235;
        }
      }
      const boundary = document.createElement('canvas');
      boundary.width = w;
      boundary.height = h;
      boundary.getContext('2d').putImageData(output, 0, 0);
      state.boundaryImages.set(cacheKey, boundary);
      return boundary;
    }
    function tintedOverlayCanvasForImage(image, key, color) {
      if (!image || !key || !color) return image;
      const cacheKey = `${key}:tint:${color}:${image.width}x${image.height}`;
      if (state.boundaryImages.has(cacheKey)) return state.boundaryImages.get(cacheKey);
      const tinted = document.createElement('canvas');
      tinted.width = image.width;
      tinted.height = image.height;
      const tintCtx = tinted.getContext('2d');
      tintCtx.drawImage(image, 0, 0);
      tintCtx.globalCompositeOperation = 'source-in';
      tintCtx.fillStyle = color;
      tintCtx.fillRect(0, 0, tinted.width, tinted.height);
      tintCtx.globalCompositeOperation = 'source-over';
      state.boundaryImages.set(cacheKey, tinted);
      return tinted;
    }
    function displaySource() {
      if (state.run && state.run.display && Object.keys(state.run.display).length) return state.run;
      if (state.upload) return state.upload;
      return state.run;
    }
    function hasPreview(display, key) {
      return Boolean(display && display[key] && display[key].length);
    }
    function preprocessingEnabledForView() {
      return !$('preprocessingEnabled') || $('preprocessingEnabled').checked;
    }
    function augmentationEnabledForView() {
      if (state.run && state.run.augmentation && state.run.augmentation.enabled) return true;
      return Boolean($('augmentationEnabled') && $('augmentationEnabled').checked);
    }
    function layerAvailable(layer) {
      const source = displaySource();
      const display = source && source.display ? source.display : {};
      if (layer === 'original') return hasPreview(display, 'original');
      if (layer === 'augmented') return augmentationEnabledForView() && hasPreview(display, 'augmented');
      if (layer === 'preprocessed') return preprocessingEnabledForView() && hasPreview(display, 'preprocessed');
      if (layer === 'sulfide') return Boolean(state.run && state.run.status === 'complete' && hasPreview(display, 'sulfide_overlay'));
      if (layer === 'final') return Boolean(state.run && state.run.status === 'complete' && (hasPreview(display, 'ordinary_overlay') || hasPreview(display, 'fine_overlay') || hasPreview(display, 'talc_overlay')));
      return false;
    }
    function baseLayerKey(display) {
      if (preprocessingEnabledForView() && hasPreview(display, 'preprocessed')) return 'preprocessed';
      if (augmentationEnabledForView() && hasPreview(display, 'augmented')) return 'augmented';
      return 'original';
    }
    function sideLayerAvailable(layer) {
      return layer === 'none' || layerAvailable(layer);
    }
    function visibleCompositeLayers() {
      const layers = [state.viewMode];
      if (state.sideLayer !== 'none' && sideLayerAvailable(state.sideLayer)) layers.push(state.sideLayer);
      return layers;
    }
    function classVisible(key) {
      return state.classVisibility[key] !== false;
    }
    function syncClassVisibilityControls() {
      document.querySelectorAll('[data-legend-toggle]').forEach(input => {
        input.checked = classVisible(input.dataset.legendToggle);
      });
    }
    function resetClassVisibility() {
      Object.keys(state.classVisibility).forEach(key => { state.classVisibility[key] = true; });
      syncClassVisibilityControls();
    }
    function setLegendPanel(panelId, sulfideId, finalId, layer) {
      const panel = $(panelId);
      const sulfide = $(sulfideId);
      const final = $(finalId);
      const showSulfide = layer === 'sulfide';
      const showFinal = layer === 'final';
      panel.hidden = !(showSulfide || showFinal);
      sulfide.hidden = !showSulfide;
      final.hidden = !showFinal;
    }
    function updateSegmentationToggleVisibility() {
      const primaryLayer = ['sulfide', 'final'].includes(state.viewMode) ? state.viewMode : null;
      const rightLayer = state.sideLayer !== 'none' && sideLayerAvailable(state.sideLayer) && ['sulfide', 'final'].includes(state.sideLayer)
        ? state.sideLayer
        : null;
      $('segmentationClassToggles').classList.toggle('hidden', !(primaryLayer || rightLayer));
      setLegendPanel('primaryClassLegend', 'primarySulfideClassToggles', 'primaryFinalClassToggles', primaryLayer);
      setLegendPanel('sideClassLegend', 'sideSulfideClassToggles', 'sideFinalClassToggles', rightLayer);
      syncClassVisibilityControls();
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
      updateSegmentationToggleVisibility();
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
    async function drawLayer(display, key, clipX, clipW, options = {}) {
      const preview = bestPreview(display[key]);
      const image = await loadImage(preview && preview.url);
      if (!image) return;
      const rect = imageRect(image);
      ctx.save();
      ctx.beginPath();
      ctx.rect(clipX, 0, clipW, canvas.height);
      ctx.clip();
      const shouldDraw = options.showImage == null ? true : Boolean(options.showImage);
      if (shouldDraw) {
        ctx.drawImage(image, rect.x, rect.y, rect.w, rect.h);
      }
      ctx.restore();
    }
    async function drawOverlay(previews, clipX = 0, clipW = canvas.width, options = {}) {
      const preview = bestPreview(previews);
      const image = await loadImage(preview && preview.url);
      if (!image) return;
      const rect = imageRect(image);
      const tintKey = options.tintColor ? `${preview.url}:tint:${options.tintColor}` : preview.url;
      const tinted = options.tintColor ? tintedOverlayCanvasForImage(image, preview.url, options.tintColor) : image;
      const source = options.boundaryOnly ? boundaryCanvasForImage(tinted, tintKey) : tinted;
      ctx.save();
      ctx.beginPath();
      ctx.rect(clipX, 0, clipW, canvas.height);
      ctx.clip();
      ctx.globalAlpha = options.opacity == null ? 1 : Number(options.opacity);
      ctx.drawImage(source, rect.x, rect.y, rect.w, rect.h);
      ctx.restore();
    }
    async function drawFinalOverlays(display, clipX = 0, clipW = canvas.width) {
      const overlayOptions = {opacity: state.overlayOpacity, boundaryOnly: state.boundaryOnly};
      if (classVisible('showOrdinary')) await drawOverlay(display.ordinary_overlay, clipX, clipW, overlayOptions);
      if (classVisible('showFine')) await drawOverlay(display.fine_overlay, clipX, clipW, overlayOptions);
      if (classVisible('showTalc')) await drawOverlay(display.talc_overlay, clipX, clipW, overlayOptions);
      if (classVisible('showFinalArtifacts')) await drawOverlay(display.artifact_overlay, clipX, clipW, {...overlayOptions, tintColor: cssColor('--artifact') || '#c63cff'});
    }
    async function drawTilingGrid(display) {
      const tiling = $('showTiling').checked ? tilingManifest() : null;
      if (!tiling) return;
      const key = ['original', 'augmented', 'preprocessed'].includes(state.viewMode) ? state.viewMode : baseLayerKey(display);
      const preview = bestPreview(display[key] || display[baseLayerKey(display)] || display.preprocessed || display.augmented || display.original);
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
      if (layer === 'augmented') {
        await drawLayer(display, 'augmented', clipX, clipW);
        return;
      }
      if (layer === 'preprocessed') {
        await drawLayer(display, 'preprocessed', clipX, clipW);
        return;
      }
      if (layer === 'sulfide') {
        await drawLayer(display, baseLayerKey(display), clipX, clipW, {showImage: classVisible('showNonSulfide')});
        if (classVisible('showSulfide')) {
          await drawOverlay(display.sulfide_overlay, clipX, clipW, {opacity: state.overlayOpacity, boundaryOnly: state.boundaryOnly});
        }
        if (classVisible('showSulfideArtifacts')) {
          await drawOverlay(display.artifact_overlay, clipX, clipW, {opacity: state.overlayOpacity, boundaryOnly: state.boundaryOnly, tintColor: cssColor('--artifact') || '#c63cff'});
        }
      } else if (layer === 'final') {
        await drawLayer(display, baseLayerKey(display), clipX, clipW, {showImage: classVisible('showBackground')});
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
    document.querySelectorAll('[data-legend-toggle]').forEach(input => {
      input.addEventListener('change', event => {
        state.classVisibility[event.target.dataset.legendToggle] = event.target.checked;
        syncClassVisibilityControls();
        drawMain();
      });
    });
    $('showTiling').addEventListener('change', drawMain);
    function updateOverlayOpacityLabel() {
      if (!$('overlayOpacityValue')) return;
      $('overlayOpacityValue').textContent = `${Math.round(state.overlayOpacity * 100)}%`;
    }
    $('overlayOpacity').addEventListener('input', () => {
      state.overlayOpacity = Math.max(0.2, Math.min(1, Number($('overlayOpacity').value || 0.65)));
      updateOverlayOpacityLabel();
      drawMain();
    });
    $('boundaryOnly').addEventListener('change', () => {
      state.boundaryOnly = $('boundaryOnly').checked;
      drawMain();
    });
    updateOverlayOpacityLabel();
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
      uploadProgressTimer = setInterval(() => {
        progress = Math.min(max, progress + Math.max(1, Math.ceil((max - progress) / 10)));
        setUploadProgress('uploadProgressPreparing', progress);
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
            return;
          }
          const progress = Math.max(1, Math.min(70, Math.round((event.loaded / Math.max(event.total, 1)) * 70)));
          setUploadProgress('uploadProgressUploading', progress);
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
        renderMetadataStatus();
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
      renderMetadataStatus();
    }
    function batchItemById(itemId) {
      return ((state.batch && state.batch.items) || []).find(item => item.item_id === itemId) || null;
    }
    function activeMetadataItem() {
      if (state.metadataTarget && state.metadataTarget.type === 'batch') return batchItemById(state.metadataTarget.itemId);
      return null;
    }
    function activeMetadataUpload() {
      const item = activeMetadataItem();
      if (item) return item.upload || item;
      return state.upload || {};
    }
    function activeCuratedMetadata() {
      const item = activeMetadataItem();
      return item ? item.curated_metadata : state.curatedMetadata;
    }
    function setActiveCuratedMetadata(payload) {
      const item = activeMetadataItem();
      if (item) {
        item.curated_metadata = payload;
        renderBatch();
        return;
      }
      state.curatedMetadata = payload;
      renderMetadataStatus();
    }
    function rawMetadataSummaryForUpload(upload, saved = {}) {
      upload = upload || {};
      return {
        upload_id: upload.upload_id || saved.upload_id || '',
        original_name: upload.original_name || saved.original_name || '',
        original_path: upload.original_path || saved.original_path || '',
        width: Number(upload.width || saved.width || 0),
        height: Number(upload.height || saved.height || 0),
        format: upload.format || saved.format || '',
        file_size_bytes: upload.file_size_bytes || saved.file_size_bytes || null,
        sha1: upload.sha1 || saved.sha1 || '',
        raw_metadata: upload.raw_metadata || saved.raw_metadata || {}
      };
    }
    function currentRawMetadataSummary() {
      const saved = (activeCuratedMetadata() && activeCuratedMetadata().raw_summary) || {};
      return rawMetadataSummaryForUpload(activeMetadataUpload(), saved);
    }
    function normalizeMetadataPayload(payload) {
      if (!payload || typeof payload !== 'object') return null;
      return {
        schema_version: payload.schema_version || 'ore-pipeline-curated-metadata-v0.1',
        source: payload.source || 'metadata_editor',
        generated_at: payload.generated_at || new Date().toISOString(),
        domain: payload.domain && typeof payload.domain === 'object' ? payload.domain : {},
        raw_summary: payload.raw_summary && typeof payload.raw_summary === 'object' ? payload.raw_summary : currentRawMetadataSummary(),
        session_defaults_applied: payload.session_defaults_applied && typeof payload.session_defaults_applied === 'object' ? payload.session_defaults_applied : {},
        warnings: Array.isArray(payload.warnings) ? payload.warnings : []
      };
    }
    function renderMetadataStatus() {
      if (!$('metadataBtn')) return;
      $('metadataBtn').disabled = !state.upload;
    }
    function metadataFields() {
      return Array.from(document.querySelectorAll('[data-metadata-field]'));
    }
    function collectMetadataDomain() {
      const domain = {};
      for (const field of metadataFields()) {
        const key = field.dataset.metadataField;
        if (!key) continue;
        if (field.type === 'checkbox') {
          if (field.checked) domain[key] = true;
          continue;
        }
        const value = String(field.value || '').trim();
        if (value) domain[key] = value;
      }
      return domain;
    }
    function applyMetadataDomain(domain = {}) {
      for (const field of metadataFields()) {
        const key = field.dataset.metadataField;
        if (!key) continue;
        if (field.type === 'checkbox') {
          field.checked = Boolean(domain[key]);
        } else {
          field.value = domain[key] == null ? '' : String(domain[key]);
        }
      }
      updateMetadataScaleWarning();
    }
    function metadataWarningsForDomain(domain) {
      const warnings = [];
      const pixelSize = String(domain.pixel_size_um || '').trim();
      const source = String(domain.scale_source || 'unavailable');
      const confidence = String(domain.scale_confidence || 'none');
      if (pixelSize && (source === 'unavailable' || confidence !== 'calibrated')) {
        warnings.push({
          code: 'pixel_size_without_calibrated_scale',
          message: 'pixel_size_um is present without calibrated scale_source/scale_confidence'
        });
      }
      return warnings;
    }
    function updateMetadataScaleWarning() {
      const warning = $('metadataScaleWarning');
      if (!warning) return;
      warning.classList.toggle('hidden', metadataWarningsForDomain(collectMetadataDomain()).length === 0);
    }
    function flattenMetadataRows(value, prefix = '', rows = [], depth = 0) {
      if (value && typeof value === 'object' && !Array.isArray(value) && depth < 4) {
        const entries = Object.entries(value);
        if (!entries.length && prefix) rows.push([prefix, '']);
        for (const [key, item] of entries) {
          flattenMetadataRows(item, prefix ? `${prefix}.${key}` : key, rows, depth + 1);
        }
      } else {
        rows.push([prefix, Array.isArray(value) || (value && typeof value === 'object') ? JSON.stringify(value) : String(value ?? '')]);
      }
      return rows;
    }
    function renderMetadataRawTable() {
      const rows = flattenMetadataRows(currentRawMetadataSummary());
      $('metadataRawTable').innerHTML = `<thead><tr><th>${escapeHtml(t('metadataField'))}</th><th>${escapeHtml(t('metadataValue'))}</th></tr></thead><tbody>` + rows.map(([key, value]) => {
        return `<tr><td>${escapeHtml(key)}</td><td>${escapeHtml(value)}</td></tr>`;
      }).join('') + '</tbody>';
    }
    function metadataDefaults() {
      return {...((state.settings && state.settings.metadata_defaults) || {}), ...metadataDefaultsFromLocalStorage()};
    }
    function renderMetadataDefaultsTable() {
      const defaults = metadataDefaults();
      const rows = Object.entries(defaults);
      if (!rows.length) {
        $('metadataDefaultsTable').innerHTML = `<tbody><tr><td>${escapeHtml(t('metadataNoDefaults'))}</td></tr></tbody>`;
        return;
      }
      $('metadataDefaultsTable').innerHTML = `<thead><tr><th>${escapeHtml(t('metadataField'))}</th><th>${escapeHtml(t('metadataValue'))}</th></tr></thead><tbody>` + rows.map(([key, value]) => {
        return `<tr><td>${escapeHtml(key)}</td><td>${escapeHtml(String(value ?? ''))}</td></tr>`;
      }).join('') + '</tbody>';
    }
    function setMetadataTab(tab) {
      const active = ['domain', 'raw', 'defaults'].includes(tab) ? tab : 'domain';
      document.querySelectorAll('#metadataTabs button').forEach(btn => btn.classList.toggle('active', btn.dataset.metadataTab === active));
      $('metadataDomainPanel').hidden = active !== 'domain';
      $('metadataRawPanel').hidden = active !== 'raw';
      $('metadataDefaultsPanel').hidden = active !== 'defaults';
      if (active === 'raw') renderMetadataRawTable();
      if (active === 'defaults') renderMetadataDefaultsTable();
    }
    function openMetadataDialog(target = null) {
      state.metadataTarget = target && target.type === 'batch'
        ? {type: 'batch', itemId: target.itemId}
        : {type: 'workspace', itemId: null};
      if (!activeMetadataUpload().upload_id) return;
      const payload = normalizeMetadataPayload(activeCuratedMetadata() || {domain: {}, raw_summary: currentRawMetadataSummary()});
      applyMetadataDomain(payload.domain);
      $('metadataDialog').dataset.defaultsApplied = '';
      renderMetadataRawTable();
      renderMetadataDefaultsTable();
      setMetadataTab('domain');
      $('metadataDialog').showModal();
    }
    function currentMetadataPayloadForSubmission() {
      if (!state.curatedMetadata) return null;
      return normalizeMetadataPayload({
        ...state.curatedMetadata,
        raw_summary: rawMetadataSummaryForUpload(state.upload || {}, (state.curatedMetadata && state.curatedMetadata.raw_summary) || {})
      });
    }
    async function saveMetadataFromDialog() {
      const domain = collectMetadataDomain();
      let defaultsApplied = (activeCuratedMetadata() && activeCuratedMetadata().session_defaults_applied) || {};
      try {
        defaultsApplied = $('metadataDialog').dataset.defaultsApplied ? JSON.parse($('metadataDialog').dataset.defaultsApplied) : defaultsApplied;
      } catch (_) {}
      const payload = normalizeMetadataPayload({
        domain,
        raw_summary: currentRawMetadataSummary(),
        warnings: metadataWarningsForDomain(domain),
        session_defaults_applied: defaultsApplied
      });
      setActiveCuratedMetadata(payload);
      try {
        if (state.metadataTarget && state.metadataTarget.type === 'batch') {
          await saveBatchItemMetadata(state.metadataTarget.itemId, payload);
        }
        $('metadataDialog').close();
      } catch (error) {
        window.alert(t('statusFailed', {error: error.message || t('unknownError')}));
      }
    }
    async function saveMetadataDefaults() {
      const domain = collectMetadataDomain();
      const keep = ['project', 'om_instrument', 'om_objective_magnification', 'scale_source', 'pixel_size_um', 'scale_confidence', 'review_status'];
      const defaults = {};
      for (const key of keep) {
        if (domain[key] !== undefined && domain[key] !== '') defaults[key] = domain[key];
      }
      try { localStorage.setItem(METADATA_STORAGE_KEY, JSON.stringify(defaults)); } catch (_) {}
      try {
        await saveSettingsObject({...currentAppSettings(), metadata_defaults: defaults}, 'settingsSaved');
      } catch (error) {
        setSettingsStatus('settingsSaveFailed', {error: error.message || t('unknownError')});
      }
      renderMetadataDefaultsTable();
    }
    function applyMetadataDefaults() {
      const defaults = metadataDefaults();
      if (!Object.keys(defaults).length) return;
      const domain = {...collectMetadataDomain(), ...defaults};
      applyMetadataDomain(domain);
      $('metadataDialog').dataset.defaultsApplied = JSON.stringify({applied_at: new Date().toISOString(), fields: Object.keys(defaults)});
    }
    async function clearMetadataDefaults() {
      try { localStorage.removeItem(METADATA_STORAGE_KEY); } catch (_) {}
      try {
        await saveSettingsObject({...currentAppSettings(), metadata_defaults: {}}, 'settingsSaved');
      } catch (error) {
        setSettingsStatus('settingsSaveFailed', {error: error.message || t('unknownError')});
      }
      renderMetadataDefaultsTable();
    }
    function resetPageForClearedImage() {
      state.upload = null;
      state.run = null;
      state.curatedMetadata = null;
      state.returnToBatchId = null;
      state.viewMode = 'original';
      state.sideLayer = 'none';
      state.zoom = 1;
      state.pan = {x: 0, y: 0};
      state.splitter = 0.5;
      state.images.clear();
      state.boundaryImages.clear();
      activePollRunId = null;
      $('fileInput').value = '';
      setUploadWarning(null);
      applyAugmentationToControls(storedAugmentationSettings(), {save: false});
      applyPresetToControls(currentAppSettings().preprocess, {save: false});
      resetClassVisibility();
      applyShowTilingDefault();
      updateRunControls(null);
      $('fixBtn').disabled = true;
      $('resultPanel').classList.add('hidden');
      $('backToBatchBtn').classList.add('hidden');
      $('textOutput').textContent = '';
      $('decisionRationale').textContent = '';
      $('metricsTable').innerHTML = '';
      $('metricsDenominatorNote').textContent = '';
      $('csvLink').removeAttribute('href');
      $('pdfLink').removeAttribute('href');
      $('runFilesZipLink').removeAttribute('href');
      $('runFilesBtn').disabled = true;
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
      state.returnToBatchId = null;
      updateRunControls(null);
      $('startBtn').disabled = true;
      setProgress(0);
      setStatus('statusWaiting');
      setUploadProgress('uploadProgressUploading', 1);
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
      state.curatedMetadata = null;
      activePollRunId = null;
      state.zoom = 1; state.pan = {x: 0, y: 0};
      state.sideLayer = 'none';
      state.images.clear();
      state.boundaryImages.clear();
      renderUploadCard(payload);
      updateRunControls(null);
      setUploadProgress('uploadProgressComplete', 100);
      setTimeout(() => {
        if (uploadProgressMessage && uploadProgressMessage.key === 'uploadProgressComplete') setUploadProgress(null);
      }, 900);
      setStatus('statusImageLoaded');
      setProgress(0);
      applyShowTilingDefault();
      updateViewControls();
      drawMain();
    }
    function currentBatchSettings() {
      return {preprocess: presetPayload(), augmentation: augmentationPayload()};
    }
    function batchIsActive(batch = state.batch) {
      return Boolean(batch && ACTIVE_RUN_STATUSES.has(String(batch.status || '').toLowerCase()));
    }
    function batchStatusLabel(status) {
      const key = {
        draft: 'batchStatusDraft',
        queued: 'batchStatusQueued',
        running: 'batchStatusRunning',
        canceling: 'batchStatusCanceling',
        canceled: 'batchStatusCanceled',
        complete: 'batchStatusComplete',
        partial: 'batchStatusPartial',
        failed: 'batchStatusFailed'
      }[String(status || 'draft').toLowerCase()] || 'batchStatusDraft';
      return t(key);
    }
    function batchSettingsForSummary() {
      if (state.batch && state.batch.status !== 'draft' && state.batch.settings) return state.batch.settings;
      return currentBatchSettings();
    }
    function preprocessTextFromSettings(settings) {
      const preset = normalizedPreprocessPreset((settings && settings.preprocess) || settings || {});
      if (!preset.preprocessing_enabled) return t('preprocessingSummaryDisabled');
      const items = [];
      if (preset.illumination_normalization) items.push(t('illuminationNormalization'));
      if (preset.denoise) items.push(t('denoise'));
      if (preset.contrast_correction) items.push(t('contrastCorrection'));
      const panoramaItem = panoramaScalingSummaryItem(preset);
      if (panoramaItem) items.push(panoramaItem);
      return t('preprocessingSummaryEnabled', {items: items.join(', ') || t('preprocessingSummaryNone')});
    }
    function augmentationTextFromSettings(settings) {
      const augmentation = normalizedAugmentationSettings((settings && settings.augmentation) || {});
      if (!augmentation.enabled) return t('augmentationSummaryDisabled');
      const items = [
        `${t('augBrightness')} ${formatSignedPercent(augmentation.color.brightness_pct)}`,
        `${t('augContrast')} ${formatSignedPercent(augmentation.color.contrast_pct)}`,
        `${t('augSaturation')} ${formatSignedPercent(augmentation.color.saturation_pct)}`
      ];
      if (Math.abs(augmentation.color.hue_degrees) > 0.001) items.push(`${t('augHue')} ${augmentation.color.hue_degrees}°`);
      if (Math.abs(augmentation.color.gamma - 1) > 0.001) items.push(`${t('augGamma')} ${augmentation.color.gamma.toFixed(2)}`);
      if (augmentation.acquisition.blur_radius > 0) items.push(`${t('augBlur')} ${augmentation.acquisition.blur_radius}`);
      if (augmentation.acquisition.gaussian_noise_std > 0) items.push(`${t('augNoise')} ${augmentation.acquisition.gaussian_noise_std}`);
      return t('augmentationSummaryEnabled', {items: items.join(', ')});
    }
    function renderBatch() {
      if (!$('batchGallery')) return;
      const batch = state.batch;
      const items = (batch && batch.items) || [];
      const active = batchIsActive(batch);
      $('batchSummary').textContent = batch
        ? t('batchItemsSummary', {count: items.length, status: batchStatusLabel(batch.status)})
        : t('batchNoBatch');
      const settings = batchSettingsForSummary();
      $('batchSettingsSummary').textContent = t('batchSettingsSummary', {
        preprocess: preprocessTextFromSettings(settings),
        augmentation: augmentationTextFromSettings(settings)
      });
      $('batchStatus').textContent = items.length ? '' : t('batchNoImages');
      $('newBatchBtn').disabled = active;
      $('addBatchImagesBtn').disabled = active;
      $('runBatchBtn').disabled = !batch || !items.length || active || String(batch.status || 'draft') !== 'draft';
      $('stopBatchBtn').classList.toggle('hidden', !active);
      $('stopBatchBtn').disabled = !active || String(batch.status || '').toLowerCase() === 'canceling';
      $('batchGallery').innerHTML = items.map(item => renderBatchItemCard(item, active)).join('');
      document.querySelectorAll('[data-batch-metadata]').forEach(btn => {
        btn.addEventListener('click', () => openMetadataDialog({type: 'batch', itemId: btn.dataset.batchMetadata}));
      });
      document.querySelectorAll('[data-batch-load]').forEach(btn => {
        btn.addEventListener('click', () => loadBatchRun(btn.dataset.batchLoad, batch && batch.batch_id));
      });
      document.querySelectorAll('[data-batch-remove]').forEach(btn => {
        btn.addEventListener('click', () => removeBatchItem(btn.dataset.batchRemove));
      });
    }
    function renderBatchItemCard(item, batchActive) {
      const display = item.display || (item.upload && item.upload.display) || {};
      const thumb = smallestPreviewUrl(display.original);
      const status = batchStatusLabel(item.status);
      const stage = t('batchProgressLabel', {stage: stageLabel(item.stage || item.status || 'draft'), progress: Math.round(item.progress || 0)});
      const loadButton = item.run_id ? `<button type="button" data-batch-load="${escapeHtml(item.run_id)}">${escapeHtml(t('batchLoad'))}</button>` : '';
      const metadataDisabled = batchActive || item.run_id ? ' disabled' : '';
      const removeDisabled = batchActive || item.run_id ? ' disabled' : '';
      return `<div class="batch-card">
        <div class="batch-thumb">${thumb ? `<img src="${escapeHtml(thumb)}" alt="${escapeHtml(item.original_name || '')}">` : `<span class="batch-thumb-placeholder">—</span>`}</div>
        <div class="batch-card-body">
          <div class="batch-card-title" title="${escapeHtml(item.original_name || '')}">${escapeHtml(item.original_name || '')}</div>
          <div class="batch-card-meta">${escapeHtml(item.width || '')} × ${escapeHtml(item.height || '')}</div>
          <div class="batch-card-status">${escapeHtml(status)}</div>
          <div class="batch-progress">
            <div class="progress"><div style="width:${Math.max(0, Math.min(100, Number(item.progress || 0)))}%"></div></div>
            <div class="batch-card-status">${escapeHtml(stage)}</div>
          </div>
          ${item.error ? `<div class="batch-card-error">${escapeHtml(item.error)}</div>` : ''}
          <div class="batch-card-actions">
            <button type="button" data-batch-metadata="${escapeHtml(item.item_id)}"${metadataDisabled}>${escapeHtml(t('batchEditMetadata'))}</button>
            <button type="button" class="danger" data-batch-remove="${escapeHtml(item.item_id)}"${removeDisabled}>${escapeHtml(t('batchRemoveImage'))}</button>
            ${loadButton}
          </div>
        </div>
      </div>`;
    }
    function batchIdFromLocation() {
      const match = window.location.pathname.match(/^\/batch\/([^/]+)$/);
      return match ? decodeURIComponent(match[1]) : null;
    }
    async function createBatch() {
      const response = await fetch('/api/batches', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({settings: currentBatchSettings()})
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'series create failed');
      state.batch = payload;
      renderBatch();
      showBatch(true, payload.batch_id);
      return payload;
    }
    async function ensureBatchDraft() {
      if (state.batch && state.batch.status === 'draft') return state.batch;
      return createBatch();
    }
    async function uploadBatchFile(file) {
      const body = new FormData();
      body.append('file', file);
      const response = await fetch('/api/uploads', {method: 'POST', body});
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'upload failed');
      return payload;
    }
    async function addBatchFiles(files) {
      const selected = Array.from(files || []);
      if (!selected.length) return;
      for (const file of selected) {
        if (!isSupportedUploadFile(file)) throw new Error(t('invalidImageFormat', {name: file.name || t('selectedImage')}));
      }
      let batch = await ensureBatchDraft();
      const uploadIds = [];
      $('batchStatus').textContent = t('batchAddingImages');
      for (let index = 0; index < selected.length; index++) {
        const file = selected[index];
        $('batchStatus').textContent = t('batchUploading', {done: index + 1, total: selected.length, name: file.name || t('selectedImage')});
        const upload = await uploadBatchFile(file);
        uploadIds.push(upload.upload_id);
      }
      const response = await fetch(`/api/batches/${encodeURIComponent(batch.batch_id)}/items`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({upload_ids: uploadIds})
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'series add failed');
      state.batch = payload;
      renderBatch();
      showBatch(true, payload.batch_id);
    }
    async function removeBatchItem(itemId) {
      if (!state.batch || !state.batch.batch_id || !itemId) return;
      const item = batchItemById(itemId);
      const name = (item && item.original_name) || itemId;
      if (!window.confirm(t('batchRemoveImageConfirm', {name}))) return;
      try {
        const response = await fetch(`/api/batches/${encodeURIComponent(state.batch.batch_id)}/items/${encodeURIComponent(itemId)}`, {method: 'DELETE'});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'remove failed');
        state.batch = payload;
        $('batchStatus').textContent = t('batchImageRemoved');
        renderBatch();
      } catch (error) {
        $('batchStatus').textContent = t('batchRemoveFailed', {error: error.message || t('unknownError')});
      }
    }
    async function loadBatch(batchId, options = {}) {
      if (!batchId) {
        state.batch = null;
        state.batchPollId = null;
        renderBatch();
        return null;
      }
      const response = await fetch(`/api/batches/${encodeURIComponent(batchId)}`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'series load failed');
      state.batch = payload;
      renderBatch();
      if (batchIsActive(payload) && options.startPolling && state.batchPollId !== payload.batch_id) {
        state.batchPollId = payload.batch_id;
        setTimeout(() => pollBatch(payload.batch_id), 700);
      }
      return payload;
    }
    async function saveBatchItemMetadata(itemId, curatedMetadata) {
      if (!state.batch || !itemId) return;
      const response = await fetch(`/api/batches/${encodeURIComponent(state.batch.batch_id)}/items/${encodeURIComponent(itemId)}/metadata`, {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({curated_metadata: curatedMetadata})
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'metadata save failed');
      state.batch = payload;
      $('batchStatus').textContent = t('batchMetadataSaved');
      renderBatch();
    }
    async function runBatch() {
      if (!state.batch) await createBatch();
      if (!state.batch || !state.batch.items.length) return;
      $('runBatchBtn').disabled = true;
      try {
        const response = await fetch(`/api/batches/${encodeURIComponent(state.batch.batch_id)}/run`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(currentBatchSettings())
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'series run failed');
        state.batch = payload;
        $('batchStatus').textContent = t('batchRunStarted');
        renderBatch();
        state.batchPollId = payload.batch_id;
        pollBatch(payload.batch_id);
      } catch (error) {
        $('batchStatus').textContent = t('batchRunFailed', {error: error.message || t('unknownError')});
        renderBatch();
      }
    }
    async function stopBatch() {
      if (!state.batch || !state.batch.batch_id) return;
      $('stopBatchBtn').disabled = true;
      try {
        const response = await fetch(`/api/batches/${encodeURIComponent(state.batch.batch_id)}/cancel`, {method: 'POST'});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'series cancel failed');
        state.batch = payload;
        renderBatch();
      } catch (error) {
        $('batchStatus').textContent = t('batchStopFailed', {error: error.message || t('unknownError')});
        renderBatch();
      }
    }
    async function pollBatch(batchId) {
      if (state.batchPollId !== batchId) return;
      try {
        const payload = await loadBatch(batchId, {startPolling: false});
        if (!payload) return;
        if (batchIsActive(payload)) {
          setTimeout(() => pollBatch(batchId), 900);
        } else {
          state.batchPollId = null;
          await refreshHistory();
        }
      } catch (error) {
        state.batchPollId = null;
        $('batchStatus').textContent = t('statusFailed', {error: error.message || t('unknownError')});
      }
    }
    async function loadBatchRun(runId, batchId) {
      if (!runId) return;
      state.returnToBatchId = batchId || (state.batch && state.batch.batch_id) || null;
      await loadRun(runId, {returnToBatchId: state.returnToBatchId});
    }
    async function applyUploadPreview({buttonId, targetView, statusKey}) {
      if (!state.upload) return;
      const button = buttonId ? $(buttonId) : null;
      if (button) button.disabled = true;
      setUploadWarning(null);
      startPreviewPreparationProgress(18, 96);
      try {
        const response = await fetch(`/api/uploads/${encodeURIComponent(state.upload.upload_id)}/preprocess`, {
          method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({...presetPayload(), augmentation: augmentationPayload()})
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
        applyShowTilingDefault();
        setViewMode(targetView);
        setSideLayer('none');
        setStatus(statusKey);
        setProgress(0);
        drawMain();
      } catch (error) {
        clearUploadProgress();
        setProgress(0);
        const message = error && error.message ? error.message : t('unknownError');
        setUploadWarning('uploadFailed', {error: message});
        setStatus('statusFailed', {error: message});
      } finally {
        if (button) button.disabled = false;
      }
    }
    $('applyAugmentationBtn').addEventListener('click', async () => {
      if (!state.upload) return;
      $('augmentationEnabled').checked = true;
      updateAugmentationValueLabels();
      updateAugmentationSummary();
      renderBatch();
      saveAugmentationSettings();
      await applyUploadPreview({
        buttonId: 'applyAugmentationBtn',
        targetView: 'augmented',
        statusKey: 'statusAugmentationUpdated'
      });
    });
    $('applyPreprocessBtn').addEventListener('click', async () => {
      if (!state.upload) return;
      $('preprocessingEnabled').checked = true;
      updatePreprocessSummary();
      savePreprocessPreset();
      await applyUploadPreview({
        buttonId: 'applyPreprocessBtn',
        targetView: 'preprocessed',
        statusKey: 'statusPreprocessUpdated'
      });
    });
    $('startBtn').addEventListener('click', async () => {
      if (!state.upload) return;
      saveAugmentationSettings();
      savePreprocessPreset();
      $('startBtn').disabled = true;
      state.returnToBatchId = null;
      setProgress(1);
      setStatus('statusProgress', {stage: stageLabel('queued'), progress: 1, eta: ''});
      try {
        const startPayload = {upload_id: state.upload.upload_id, ...presetPayload(), augmentation: augmentationPayload()};
        const curatedMetadata = currentMetadataPayloadForSubmission();
        if (curatedMetadata) startPayload.curated_metadata = curatedMetadata;
        const response = await fetch('/api/runs/start', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(startPayload)
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
        applyShowTilingDefault();
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
        setStatus('statusCanceled');
        await refreshHistory();
        return;
      }
      if (payload.status === 'canceling') {
        setStatus('statusCanceling');
      }
      setTimeout(() => pollRun(runId), 900);
    }
    function imageDimensionsText(file) {
      if (!file || !file.is_image || !file.width || !file.height) return '';
      return `${file.width}x${file.height}`;
    }
    function renderRunFiles(payload) {
      const files = payload.files || [];
      $('runFilesStatus').textContent = files.length ? '' : t('runFilesEmpty');
      $('runFilesSummary').textContent = t('runFilesSummary', {
        count: payload.file_count ?? files.length,
        size: formatBytes(payload.total_size_bytes || 0)
      });
      $('runFilesZipLink').href = (payload.downloads && payload.downloads.artifacts_zip) || (state.run && state.run.downloads && state.run.downloads.artifacts_zip) || '';
      $('runFilesTable').innerHTML = `<thead><tr><th>${escapeHtml(t('runFilesHeaderPath'))}</th><th>${escapeHtml(t('runFilesHeaderKind'))}</th><th class="numeric">${escapeHtml(t('runFilesHeaderSize'))}</th><th class="numeric">${escapeHtml(t('runFilesHeaderImageSize'))}</th></tr></thead><tbody>` + files.map(file => {
        const kind = file.is_image ? t('runFilesImageKind') : t('runFilesFileKind');
        return `<tr><td>${escapeHtml(file.path || file.name || '')}</td><td>${escapeHtml(kind)}</td><td class="numeric">${escapeHtml(formatBytes(file.size_bytes))}</td><td class="numeric">${escapeHtml(imageDimensionsText(file))}</td></tr>`;
      }).join('') + '</tbody>';
    }
    async function openRunFilesDialog() {
      if (!state.run || !state.run.run_id) return;
      $('runFilesStatus').textContent = t('runFilesLoading');
      $('runFilesSummary').textContent = '';
      $('runFilesTable').innerHTML = '';
      $('runFilesZipLink').href = (state.run.downloads && state.run.downloads.artifacts_zip) || `/api/runs/${encodeURIComponent(state.run.run_id)}/artifacts.zip`;
      $('runFilesDialog').showModal();
      try {
        const url = (state.run.downloads && state.run.downloads.files) || `/api/runs/${encodeURIComponent(state.run.run_id)}/files`;
        const response = await fetch(url);
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || response.statusText);
        renderRunFiles(payload);
      } catch (error) {
        $('runFilesStatus').textContent = t('runFilesLoadFailed', {error: error.message || t('unknownError')});
      }
    }
    function renderResults(run) {
      $('resultPanel').classList.remove('hidden');
      const returnBatchId = state.returnToBatchId || (run.batch && run.batch.batch_id) || null;
      $('backToBatchBtn').classList.toggle('hidden', !returnBatchId);
      $('textOutput').textContent = localizedRunText(run);
      $('decisionRationale').textContent = decisionRationale(run);
      $('csvLink').href = run.downloads.metrics_csv;
      $('pdfLink').href = run.downloads.pdf_report;
      $('runFilesZipLink').href = run.downloads.artifacts_zip;
      $('runFilesBtn').disabled = !(run.downloads && run.downloads.files);
      const rows = run.metrics || [];
      $('metricsTable').innerHTML = `<thead><tr><th>${escapeHtml(t('metricsHeaderMetric'))}</th><th>${escapeHtml(t('metricsHeaderValue'))}</th><th>${escapeHtml(t('metricsHeaderAreaPx'))}</th><th>${escapeHtml(t('metricsHeaderPhysicalArea'))}</th></tr></thead><tbody>` + rows.map(row => {
        const value = row.percent == null ? row.value : `${Number(row.percent).toFixed(1)}%`;
        const areaPx = row.area_px == null ? '' : String(row.area_px);
        const physicalArea = formatPhysicalArea(row);
        const level = Math.max(0, Math.min(2, Number(row.level || 0)));
        return `<tr class="metric-level-${level}" data-metric-key="${escapeHtml(row.key || '')}"><td class="metric-label" style="--metric-level:${level}">${escapeHtml(localizedMetricLabel(row))}</td><td>${escapeHtml(value)}</td><td>${escapeHtml(areaPx)}</td><td>${escapeHtml(physicalArea)}</td></tr>`;
      }).join('') + '</tbody>';
      $('metricsDenominatorNote').textContent = t('metricsDenominatorNote');
    }
    async function refreshHistory() {
      const [runsResponse, batchesResponse] = await Promise.all([fetch('/api/runs'), fetch('/api/batches')]);
      const payload = await runsResponse.json();
      const batchPayload = await batchesResponse.json();
      const runs = payload.runs || [];
      const batches = batchPayload.batches || [];
      state.historyRuns = runs;
      state.historyBatches = batches;
      renderHistoryViews();
    }
    function renderHistoryViews() {
      const runs = state.historyRuns || [];
      $('historyList').innerHTML = renderCompactHistory(runs);
      renderHistoryPage();
      attachHistoryActions();
    }
    function attachHistoryActions() {
      document.querySelectorAll('[data-load-run]').forEach(btn => btn.addEventListener('click', () => loadRun(btn.dataset.loadRun)));
      document.querySelectorAll('[data-delete-run]').forEach(btn => btn.addEventListener('click', () => removeRun(btn.dataset.deleteRun)));
      document.querySelectorAll('[data-preview-run]').forEach(btn => btn.addEventListener('click', () => openHistoryPreview(btn.dataset.previewUrl, btn.dataset.previewTitle)));
      document.querySelectorAll('[data-open-batch]').forEach(btn => btn.addEventListener('click', () => showBatch(true, btn.dataset.openBatch)));
    }
    function renderHistoryPage() {
      document.querySelectorAll('#historyModeButtons button').forEach(btn => btn.classList.toggle('active', btn.dataset.historyMode === state.historyMode));
      if (state.historyMode === 'batches') {
        $('historyPageList').innerHTML = renderBatchHistoryTable(state.historyBatches || []);
        return;
      }
      const runs = state.historyMode === 'single'
        ? (state.historyRuns || []).filter(run => !(run.batch && run.batch.batch_id))
        : (state.historyRuns || []);
      $('historyPageList').innerHTML = renderHistoryTable(runs);
    }
    function renderCompactHistory(runs) {
      return runs.map(run => historyRow(run)).join('') || `<p class="muted">${escapeHtml(t('historyNoRuns'))}</p>`;
    }
    function batchCountsText(batch) {
      const counts = (batch && batch.item_counts) || {};
      const parts = Object.entries(counts)
        .filter(([, value]) => Number(value || 0) > 0)
        .map(([status, value]) => `${batchStatusLabel(status)}: ${Number(value || 0)}`);
      return parts.join(' · ') || '—';
    }
    function renderBatchHistoryTable(batches) {
      if (!batches.length) return `<p class="muted">${escapeHtml(t('historyNoBatches'))}</p>`;
      const rows = batches.map(batch => {
        return `<tr>
          <td class="filename" title="${escapeHtml(batch.batch_id || '')}">${escapeHtml(batch.batch_id || '')}</td>
          <td>${escapeHtml(formatDate(batch.created_at))}</td>
          <td>${escapeHtml(batchStatusLabel(batch.status))}</td>
          <td class="numeric">${escapeHtml(String(batch.items_count || 0))}</td>
          <td class="numeric">${escapeHtml(Math.round(Number(batch.progress || 0)) + '%')}</td>
          <td>${escapeHtml(batchCountsText(batch))}</td>
          <td><div class="history-actions"><button data-open-batch="${escapeHtml(batch.batch_id || '')}">${escapeHtml(t('historyOpenBatch'))}</button></div></td>
        </tr>`;
      }).join('');
      return `<div class="history-table-wrap"><table class="history-table">
        <thead><tr>
          <th>${escapeHtml(t('historyBatchId'))}</th>
          <th>${escapeHtml(t('historyDate'))}</th>
          <th>${escapeHtml(t('historyBatchStatus'))}</th>
          <th class="numeric">${escapeHtml(t('historyBatchImages'))}</th>
          <th class="numeric">${escapeHtml(t('historyBatchProgress'))}</th>
          <th>${escapeHtml(t('historyBatchCounts'))}</th>
          <th>${escapeHtml(t('historyActions'))}</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table></div>`;
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
      const title = runFilename(run) || run.run_id || t('historyPreviewTitle');
      const date = formatDate(run.created_at);
      const summaryText = localizedRunText(run) || run.status || '';
      return `<div class="history-row">
        <div class="history-row-media">
          ${renderHistoryThumbnail(run)}
          <button class="history-row-load" data-load-run="${escapeHtml(run.run_id)}">${escapeHtml(t('historyLoad'))}</button>
        </div>
        <div class="history-row-text">
          <div class="history-row-title" title="${escapeHtml(title)}">${escapeHtml(title)}</div>
          <div class="history-row-run">${escapeHtml(run.run_id || '')}</div>
          <div class="muted">${escapeHtml(date)}</div>
          <div class="history-row-summary">${escapeHtml(summaryText)}</div>
        </div>
      </div>`;
    }
    $('closeHistoryPreviewBtn').addEventListener('click', () => $('historyPreviewDialog').close());
    $('historyPreviewDialog').addEventListener('click', event => {
      if (event.target === $('historyPreviewDialog')) $('historyPreviewDialog').close();
    });
    $('runFilesBtn').addEventListener('click', () => openRunFilesDialog());
    $('closeRunFilesBtn').addEventListener('click', () => $('runFilesDialog').close());
    $('runFilesDialog').addEventListener('click', event => {
      if (event.target === $('runFilesDialog')) $('runFilesDialog').close();
    });
    $('editPreprocessBtn').addEventListener('click', () => $('preprocessDialog').showModal());
    $('closePreprocessBtn').addEventListener('click', () => $('preprocessDialog').close());
    $('donePreprocessBtn').addEventListener('click', () => $('preprocessDialog').close());
    $('preprocessDialog').addEventListener('click', event => {
      if (event.target === $('preprocessDialog')) $('preprocessDialog').close();
    });
    $('editAugmentationBtn').addEventListener('click', () => $('augmentationDialog').showModal());
    $('closeAugmentationBtn').addEventListener('click', () => $('augmentationDialog').close());
    $('doneAugmentationBtn').addEventListener('click', () => {
      saveAugmentationSettings();
      $('augmentationDialog').close();
    });
    $('augmentationDialog').addEventListener('click', event => {
      if (event.target === $('augmentationDialog')) $('augmentationDialog').close();
    });
    $('metadataBtn').addEventListener('click', openMetadataDialog);
    $('closeMetadataBtn').addEventListener('click', () => $('metadataDialog').close());
    $('cancelMetadataBtn').addEventListener('click', () => $('metadataDialog').close());
    $('saveMetadataBtn').addEventListener('click', saveMetadataFromDialog);
    $('applyMetadataDefaultsBtn').addEventListener('click', applyMetadataDefaults);
    $('saveMetadataDefaultsBtn').addEventListener('click', saveMetadataDefaults);
    $('clearMetadataDefaultsBtn').addEventListener('click', clearMetadataDefaults);
    $('metadataDialog').addEventListener('click', event => {
      if (event.target === $('metadataDialog')) $('metadataDialog').close();
    });
    document.querySelectorAll('#metadataTabs button').forEach(btn => btn.addEventListener('click', () => setMetadataTab(btn.dataset.metadataTab)));
    metadataFields().forEach(field => field.addEventListener('input', updateMetadataScaleWarning));
    metadataFields().forEach(field => field.addEventListener('change', updateMetadataScaleWarning));
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
        $('resultPanel').classList.add('hidden');
        $('textOutput').textContent = '';
        $('decisionRationale').textContent = '';
        $('metricsTable').innerHTML = '';
        $('metricsDenominatorNote').textContent = '';
        setSideLayer('none');
        setViewMode('original');
        drawMain();
      }
      setStatus('statusRunRemoved', {runId});
      await refreshHistory();
    }
    async function loadRun(runId, options = {}) {
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
      state.curatedMetadata = normalizeMetadataPayload((payload.input && payload.input.curated_metadata) || null);
      state.returnToBatchId = options.returnToBatchId || (payload.batch && payload.batch.batch_id) || null;
      activePollRunId = runIsActive(payload) ? payload.run_id : null;
      state.zoom = 1;
      state.pan = {x: 0, y: 0};
      state.sideLayer = 'none';
      state.images.clear();
      state.boundaryImages.clear();
      if (upload) renderUploadCard(upload);
      else renderMetadataStatus();
      const runPreprocess = payload.preprocess || {};
      applyPresetToControls({...((runPreprocess && runPreprocess.preset) || {}), preprocessing_enabled: runPreprocess.enabled !== false});
      const runAugmentation = payload.augmentation || {};
      applyAugmentationToControls({...((runAugmentation && runAugmentation.settings) || {}), enabled: runAugmentation.enabled === true}, {save: true});
      applyShowTilingDefault();
      updateRunControls(payload);
      renderResults(payload);
      setSideLayer('none');
      setViewMode('final');
      setProgress(payload.status === 'complete' ? 100 : (payload.progress || 0));
      setStatus(upload ? 'statusRunLoaded' : 'statusRunLoadedNoUpload', {runId: payload.run_id});
      showWorkspace(true);
      resetWindowScroll();
      if (runIsActive(payload)) pollRun(payload.run_id);
    }
    const PAGE_SLUGS = {workspace: '/workspace', batch: '/batch', history: '/history', settings: '/settings'};
    function pageFromLocation() {
      if (window.location.pathname === PAGE_SLUGS.batch || window.location.pathname.startsWith(`${PAGE_SLUGS.batch}/`)) return 'batch';
      if (window.location.pathname === PAGE_SLUGS.history) return 'history';
      if (window.location.pathname === PAGE_SLUGS.settings) return 'settings';
      return 'workspace';
    }
    function resetWindowScroll() {
      requestAnimationFrame(() => window.scrollTo({top: 0, left: 0, behavior: 'auto'}));
    }
    function setPage(page, options = {}) {
      const nextPage = ['workspace', 'batch', 'history', 'settings'].includes(page) ? page : 'workspace';
      const batchId = options.batchId || (nextPage === 'batch' ? batchIdFromLocation() : null);
      const slug = nextPage === 'batch' && batchId ? `/batch/${encodeURIComponent(batchId)}` : PAGE_SLUGS[nextPage];
      document.body.dataset.page = nextPage;
      $('workspaceView').classList.toggle('hidden', nextPage !== 'workspace');
      $('batchView').classList.toggle('hidden', nextPage !== 'batch');
      $('historyView').classList.toggle('hidden', nextPage !== 'history');
      $('settingsView').classList.toggle('hidden', nextPage !== 'settings');
      $('workspaceTab').classList.toggle('active', nextPage === 'workspace');
      $('batchTab').classList.toggle('active', nextPage === 'batch');
      $('historyTab').classList.toggle('active', nextPage === 'history');
      $('settingsTab').classList.toggle('active', nextPage === 'settings');
      if (options.push && window.location.pathname !== slug) {
        window.history.pushState({page: nextPage}, '', slug);
      }
      if (nextPage === 'batch') {
        if (batchId && (!state.batch || state.batch.batch_id !== batchId)) {
          loadBatch(batchId, {startPolling: true}).catch(error => {
            $('batchStatus').textContent = t('statusFailed', {error: error.message || t('unknownError')});
            renderBatch();
          });
        } else {
          renderBatch();
          if (state.batch && batchIsActive(state.batch)) {
            if (state.batchPollId !== state.batch.batch_id) {
              state.batchPollId = state.batch.batch_id;
              setTimeout(() => pollBatch(state.batch.batch_id), 700);
            }
          }
        }
      }
      if (nextPage === 'history') refreshHistory();
      if (nextPage === 'settings') renderSettingsForm(currentAppSettings());
      resizeCanvas();
      if (options.resetScroll) resetWindowScroll();
    }
    function showWorkspace(push = false) { setPage('workspace', {push}); }
    function showBatch(push = false, batchId = null) { setPage('batch', {push, batchId}); }
    function showHistory(push = false) { setPage('history', {push}); }
    function showSettings(push = false) { setPage('settings', {push}); }
    $('workspaceTab').addEventListener('click', () => showWorkspace(true));
    $('batchTab').addEventListener('click', () => showBatch(true, state.batch && state.batch.batch_id));
    $('historyTab').addEventListener('click', () => showHistory(true));
    $('settingsTab').addEventListener('click', () => showSettings(true));
    document.querySelectorAll('#historyModeButtons button').forEach(btn => btn.addEventListener('click', () => {
      state.historyMode = btn.dataset.historyMode || 'all';
      renderHistoryPage();
      attachHistoryActions();
    }));
    $('newBatchBtn').addEventListener('click', () => {
      state.batch = null;
      state.batchPollId = null;
      renderBatch();
      showBatch(true);
    });
    $('addBatchImagesBtn').addEventListener('click', () => $('batchFileInput').click());
    $('batchFileInput').addEventListener('change', event => {
      addBatchFiles(event.target.files).catch(error => {
        $('batchStatus').textContent = t('batchUploadFailed', {error: error.message || t('unknownError')});
        renderBatch();
      }).finally(() => {
        $('batchFileInput').value = '';
      });
    });
    $('runBatchBtn').addEventListener('click', runBatch);
    $('stopBatchBtn').addEventListener('click', stopBatch);
    $('backToBatchBtn').addEventListener('click', () => {
      const batchId = state.returnToBatchId || (state.run && state.run.batch && state.run.batch.batch_id);
      showBatch(true, batchId);
    });
    $('saveSettingsBtn').addEventListener('click', saveSettingsFromPage);
    $('resetSettingsBtn').addEventListener('click', resetSettingsFromPage);
    ['settingsPanoramaScaling','settingsPanoramaScalingMode'].forEach(id => $(id).addEventListener('change', () => updatePanoramaScalingControls('settings')));
    window.addEventListener('popstate', () => setPage(pageFromLocation(), {push: false}));
    loadAppSettings();
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
    function completeRunForEditor() {
      return Boolean(state.run && state.run.run_id && state.run.status === 'complete');
    }
    function editorLayerAvailable(layer) {
      if (layer === 'artifact') return Boolean(state.upload || completeRunForEditor());
      return completeRunForEditor();
    }
    function preferredEditorLayer() {
      if (state.viewMode === 'sulfide' && editorLayerAvailable('sulfide')) return 'sulfide';
      if (state.viewMode === 'final' && editorLayerAvailable('final')) return 'final';
      if (completeRunForEditor() && editorLayerAvailable('final')) return 'final';
      return 'artifact';
    }
    function updateEditorLayerTabs() {
      document.querySelectorAll('#editLayerTabs button').forEach(btn => {
        const available = editorLayerAvailable(btn.dataset.layer);
        btn.disabled = !available;
        btn.classList.toggle('active', btn.dataset.layer === state.editor.layer);
      });
    }
    function updateFixRestartLabel() {
      const key = completeRunForEditor() ? 'fixAndRestart' : 'saveArtefacts';
      $('fixRestartBtn').dataset.i18n = key;
      $('fixRestartBtn').textContent = t(key);
    }
    async function ensureUploadPreparedForArtifactEditor() {
      if (!state.upload || !state.upload.upload_id) throw new Error(t('noImageLoaded'));
      startPreviewPreparationProgress(18, 96);
      const response = await fetch(`/api/uploads/${encodeURIComponent(state.upload.upload_id)}/preprocess`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({...presetPayload(), augmentation: augmentationPayload()})
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
      updateRunControls(state.run);
      updateViewControls();
      return payload;
    }
    async function openFixDialog() {
      if (!state.upload && !state.run) return;
      $('fixDialog').showModal();
      resizeCanvas();
      setEditorStatus('editorLoading');
      try {
        if (completeRunForEditor()) await refreshRunForEditor();
        else await ensureUploadPreparedForArtifactEditor();
        const desiredLayer = preferredEditorLayer();
        await switchEditorLayer(desiredLayer);
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
      if (!['artifact', 'sulfide', 'final'].includes(layer) || !editorLayerAvailable(layer)) return;
      state.editor.layer = layer;
      updateEditorLayerTabs();
      updateFixRestartLabel();
      $('classSelector').style.display = layer === 'final' ? 'block' : 'none';
      if (layer === 'sulfide') $('editClass').value = '1';
      $('editorHelpText').textContent = t(layer === 'artifact' ? 'editorArtifactHelp' : 'editorHelp');
      const masks = (state.run && state.run.masks) || {};
      const uploadArtifact = state.upload && state.upload.artifact_mask;
      const url = layer === 'final' ? masks.final : layer === 'sulfide' ? masks.sulfide : (masks.artifact || (uploadArtifact && uploadArtifact.mask_url));
      if (url) {
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
      } else if (layer === 'artifact') {
        const shape = editorSourceShape();
        state.editor.width = shape.width;
        state.editor.height = shape.height;
        state.editor.mask = new Uint8Array(shape.width * shape.height);
      } else {
        throw new Error(t('editorMissingMask'));
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
    function editorSourceShape() {
      if (completeRunForEditor() && state.run.image && state.run.image.width && state.run.image.height) {
        return {width: Number(state.run.image.width), height: Number(state.run.image.height)};
      }
      const preprocess = state.upload && state.upload.preprocess;
      if (preprocess && preprocess.width && preprocess.height) {
        return {width: Number(preprocess.width), height: Number(preprocess.height)};
      }
      if (state.upload && state.upload.width && state.upload.height) {
        return {width: Number(state.upload.width), height: Number(state.upload.height)};
      }
      return {width: 1, height: 1};
    }
    function editorBasePreview() {
      const source = completeRunForEditor() ? state.run : state.upload;
      const display = (source && source.display) || {};
      return bestPreview(display.preprocessed || display.augmented || display.original || []);
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
      if (!state.editor.mask) return;
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
      const artifactColor = artifactOverlayColor(175);
      for (let i = 0, p = 0; i < state.editor.mask.length; i++, p += 4) {
        const value = state.editor.mask[i];
        if (!value) continue;
        let color = value === 1 ? [30,185,85,145] : value === 2 ? [230,65,65,155] : [40,120,245,160];
        if (state.editor.layer === 'sulfide') color = [245,190,35,150];
        if (state.editor.layer === 'artifact') color = artifactColor;
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
      let artefacts = 0;
      for (const value of state.editor.mask) {
        if (state.editor.layer === 'artifact') {
          if (value > 0) artefacts += 1;
        } else if (state.editor.layer === 'sulfide') {
          if (value > 0) sulfide += 1;
        } else {
          if (value === 1) ordinary += 1;
          else if (value === 2) fine += 1;
          else if (value === 3) talc += 1;
        }
      }
      if (state.editor.layer === 'artifact') {
        const rows = [
          {label: t('statArtefacts'), px: artefacts, denom: total, denomLabel: t('statOfImage')},
          {label: t('statCleanArea'), px: total - artefacts, denom: total, denomLabel: t('statOfImage')},
        ];
        $('editorStats').innerHTML = '<tbody>' + rows.map(row => {
          const pct = row.px / Math.max(1, row.denom) * 100;
          return `<tr><td>${escapeHtml(row.label)}</td><td>${row.px.toLocaleString(localeCode())} px</td><td>${pct.toFixed(2)}% ${escapeHtml(row.denomLabel)}</td></tr>`;
        }).join('') + '</tbody>';
        return;
      }
      if (state.editor.layer === 'final') sulfide = ordinary + fine;
      const nonSulfide = total - sulfide;
      const sulfideDenom = Math.max(1, sulfide);
      const rows = [
        {label: t('statSulfide'), px: sulfide, denom: total, denomLabel: t('statOfImage')},
        {label: t('statNonSulfide'), px: nonSulfide, denom: total, denomLabel: t('statOfImage')},
        {separator: true},
        {label: t('statOrdinary'), px: ordinary, denom: state.editor.layer === 'final' ? sulfideDenom : total, denomLabel: state.editor.layer === 'final' ? t('statOfSulfides') : t('statOfImage')},
        {label: t('statFine'), px: fine, denom: state.editor.layer === 'final' ? sulfideDenom : total, denomLabel: state.editor.layer === 'final' ? t('statOfSulfides') : t('statOfImage')},
        {label: t('statTalc'), px: talc, denom: total, denomLabel: t('statOfImage')},
      ];
      $('editorStats').innerHTML = '<tbody>' + rows.map(row => {
        if (row.separator) return '<tr class="stat-separator"><td colspan="3"></td></tr>';
        const pct = row.px / Math.max(1, row.denom) * 100;
        return `<tr><td>${escapeHtml(row.label)}</td><td>${row.px.toLocaleString(localeCode())} px</td><td>${pct.toFixed(2)}% ${escapeHtml(row.denomLabel)}</td></tr>`;
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
      if (!state.editor.dirty) return;
      if (!completeRunForEditor()) {
        if (state.editor.layer !== 'artifact' || !state.upload) return;
        const response = await fetch(`/api/uploads/${encodeURIComponent(state.upload.upload_id)}/artifact-mask`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({mask_png: editorMaskDataUrl(), comment: $('editComment').value})
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'artifact save failed');
        state.upload = payload;
        state.editor.dirty = false;
        $('fixRestartBtn').disabled = true;
        renderUploadCard(payload);
        updateRunControls(state.run);
        setEditorStatus('editSavedArtefacts');
        drawMain();
        return;
      }
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
      updateRunControls(payload);
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
