from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ore_classifier.talc_blue_line_converter import read_image_rgb, sha256_file, utc_now_iso, write_mask


DEFAULT_SAM2_MODEL_ID = "facebook/sam2.1-hiera-tiny"
MAX_SAM2_REGION_FRACTION = 0.50
_SAM2_PREDICTOR_CACHE: dict[tuple[str, str], tuple[Any, Any, str]] = {}


class Sam2AssistFailure(RuntimeError):
    pass


class Sam2AssistUnavailable(Sam2AssistFailure):
    pass


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return module_name in sys.modules


def resolve_sam2_device(device: str | None, torch_module: Any) -> tuple[str, bool, bool]:
    cuda_available = False
    mps_available = False
    try:
        cuda_available = bool(torch_module.cuda.is_available())
    except Exception:  # noqa: BLE001 - runtime device probing is best-effort.
        cuda_available = False
    try:
        mps_available = bool(torch_module.backends.mps.is_available())
    except Exception:  # noqa: BLE001 - runtime device probing is best-effort.
        mps_available = False

    requested = str(device or "auto").strip().lower()
    if requested and requested != "auto":
        return requested, cuda_available, mps_available
    if cuda_available:
        return "cuda", cuda_available, mps_available
    if mps_available:
        return "mps", cuda_available, mps_available
    return "cpu", cuda_available, mps_available


def sam2_assist_status(
    model_id: str = DEFAULT_SAM2_MODEL_ID,
    device: str | None = None,
    *,
    check_load: bool = False,
    force_reload: bool = False,
) -> dict[str, Any]:
    torch_available = module_available("torch")
    sam2_available = module_available("sam2")
    resolved_device = device or "auto"
    cuda_available = False
    mps_available = False
    if torch_available:
        try:
            import torch

            resolved_device, cuda_available, mps_available = resolve_sam2_device(device, torch)
        except Exception:  # noqa: BLE001 - status should stay best-effort.
            resolved_device = "unknown"
    status = {
        "schema_version": "sam2-assist-status-v0.1",
        "available": bool(torch_available and sam2_available),
        "torch_available": torch_available,
        "sam2_available": sam2_available,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "model_id": model_id,
        "device": resolved_device,
        "load_status": "not_checked",
        "cached": bool(_SAM2_PREDICTOR_CACHE),
    }
    if check_load:
        status.update(sam2_load_status(model_id=model_id, device=device, force_reload=force_reload))
    return status


def load_sam2_predictor(
    model_id: str = DEFAULT_SAM2_MODEL_ID,
    device: str | None = None,
    *,
    force_reload: bool = False,
) -> tuple[Any, Any, str]:
    try:
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise Sam2AssistUnavailable("SAM2 is not installed; install the official facebookresearch/sam2 package") from exc
    resolved_device, _cuda_available, _mps_available = resolve_sam2_device(device, torch)
    cache_key = (model_id, resolved_device)
    if not force_reload and cache_key in _SAM2_PREDICTOR_CACHE:
        return _SAM2_PREDICTOR_CACHE[cache_key]

    try:
        predictor = SAM2ImagePredictor.from_pretrained(model_id, device=resolved_device)
    except Exception as exc:  # noqa: BLE001 - model downloads/checkpoints fail in environment-specific ways.
        raise Sam2AssistFailure(f"SAM2 model load failed for {model_id!r}: {exc}") from exc
    if hasattr(predictor, "model") and hasattr(predictor.model, "to"):
        predictor.model.to(resolved_device)
    if hasattr(predictor, "model") and hasattr(predictor.model, "eval"):
        predictor.model.eval()
    loaded = (predictor, torch, resolved_device)
    _SAM2_PREDICTOR_CACHE[cache_key] = loaded
    return loaded


def sam2_load_status(
    model_id: str = DEFAULT_SAM2_MODEL_ID,
    device: str | None = None,
    *,
    force_reload: bool = False,
) -> dict[str, Any]:
    missing = []
    if not module_available("torch"):
        missing.append("torch")
    if not module_available("sam2"):
        missing.append("sam2")
    if missing:
        return {"available": False, "load_status": "unavailable", "load_error": f"missing {', '.join(missing)}", "cached": False}
    try:
        predictor, _torch, resolved_device = load_sam2_predictor(
            model_id=model_id,
            device=device,
            force_reload=force_reload,
        )
    except Sam2AssistUnavailable as exc:
        return {"available": False, "load_status": "unavailable", "load_error": str(exc), "cached": False}
    except Sam2AssistFailure as exc:
        return {"available": False, "load_status": "failed", "load_error": str(exc), "cached": False}
    return {
        "available": True,
        "load_status": "loaded",
        "load_error": None,
        "device": resolved_device,
        "cached": True,
        "predictor_class": predictor.__class__.__name__,
    }


def _clip_point_xy(geometry: dict[str, Any], shape_hw: tuple[int, int]) -> list[int]:
    if geometry.get("type") != "point_xy":
        raise Sam2AssistFailure(f"unsupported point prompt geometry type: {geometry.get('type')}")
    try:
        x = int(round(float(geometry.get("x", geometry.get("point_x")))))
        y = int(round(float(geometry.get("y", geometry.get("point_y")))))
    except (TypeError, ValueError) as exc:
        raise Sam2AssistFailure("point prompt must include numeric x/y") from exc
    height, width = shape_hw
    return [max(0, min(width - 1, x)), max(0, min(height - 1, y))]


def _bbox_from_geometry(geometry: dict[str, Any]) -> list[int]:
    geometry_type = geometry.get("type")
    if geometry_type == "rectangle_xyxy":
        try:
            x1 = int(round(float(geometry["x1"])))
            y1 = int(round(float(geometry["y1"])))
            x2 = int(round(float(geometry["x2"])))
            y2 = int(round(float(geometry["y2"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise Sam2AssistFailure("rectangle prompt must include numeric x1/y1/x2/y2") from exc
    elif geometry_type == "polygon_xy":
        points = geometry.get("points")
        if not isinstance(points, list) or len(points) < 3:
            raise Sam2AssistFailure("polygon prompt must include at least three [x, y] points")
        xs = [int(round(float(point[0]))) for point in points]
        ys = [int(round(float(point[1]))) for point in points]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        raise Sam2AssistFailure(f"unsupported SAM2 prompt geometry type: {geometry_type}")
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    if right - left < 2 or bottom - top < 2:
        raise Sam2AssistFailure("SAM2 prompt box is too small")
    return [left, top, right, bottom]


def _clip_bbox(bbox: list[int], shape_hw: tuple[int, int]) -> list[int]:
    height, width = shape_hw
    x1 = max(0, min(width, int(bbox[0])))
    y1 = max(0, min(height, int(bbox[1])))
    x2 = max(0, min(width, int(bbox[2])))
    y2 = max(0, min(height, int(bbox[3])))
    if not (x1 < x2 and y1 < y2):
        raise Sam2AssistFailure(f"SAM2 prompt box is empty after clipping: {bbox}")
    return [x1, y1, x2, y2]


def _select_best_mask(masks: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, float]:
    if masks.size == 0:
        raise Sam2AssistFailure("SAM2 returned no masks")
    score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
    index = int(np.argmax(score_values)) if score_values.size else 0
    return np.asarray(masks[index] > 0, dtype=np.uint8) * 255, float(score_values[index]) if score_values.size else 0.0


def _postprocess_sam2_mask(
    mask: np.ndarray,
    prompt_geometry: dict[str, Any],
    shape_hw: tuple[int, int],
    *,
    max_fraction: float = MAX_SAM2_REGION_FRACTION,
) -> np.ndarray:
    binary = np.asarray(mask > 0, dtype=np.uint8) * 255
    prompt_type = str(prompt_geometry.get("type") or "")
    if prompt_type != "point_xy":
        bbox = _clip_bbox(_bbox_from_geometry(prompt_geometry), shape_hw)
        clipped = np.zeros_like(binary)
        x1, y1, x2, y2 = bbox
        clipped[y1:y2, x1:x2] = binary[y1:y2, x1:x2]
        binary = clipped

    mask_pixels = int(np.count_nonzero(binary))
    total_pixels = int(shape_hw[0] * shape_hw[1])
    if total_pixels > 0 and mask_pixels / total_pixels > max_fraction:
        percent = mask_pixels * 100.0 / total_pixels
        raise Sam2AssistFailure(
            f"SAM2 mask covers {percent:.1f}% of the image; draw a smaller SAM2 box or use brush/polygon instead"
        )
    return binary


def generate_sam2_region_mask(
    *,
    image_path: Path,
    prompt_geometry: dict[str, Any],
    out_dir: Path,
    model_id: str = DEFAULT_SAM2_MODEL_ID,
    device: str | None = None,
    output_name: str | None = None,
) -> dict[str, Any]:
    image_rgb = read_image_rgb(image_path)
    shape_hw = image_rgb.shape[:2]
    predictor, _torch, resolved_device = load_sam2_predictor(model_id=model_id, device=device)
    predictor.set_image(image_rgb)
    prompt_type = str(prompt_geometry.get("type") or "")
    if prompt_type == "point_xy":
        point_xy = _clip_point_xy(prompt_geometry, shape_hw)
        masks, scores, _logits = predictor.predict(
            point_coords=np.asarray([point_xy], dtype=np.float32),
            point_labels=np.asarray([1], dtype=np.int32),
            multimask_output=True,
        )
    else:
        bbox = _clip_bbox(_bbox_from_geometry(prompt_geometry), shape_hw)
        masks, scores, _logits = predictor.predict(box=np.asarray(bbox, dtype=np.float32), multimask_output=True)
    mask, score = _select_best_mask(np.asarray(masks), np.asarray(scores))
    mask = _postprocess_sam2_mask(mask, prompt_geometry, shape_hw)

    out_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_name or f"sam2_{prompt_type or 'prompt'}"
    mask_path = out_dir / f"{output_stem}_mask.png"
    summary_path = out_dir / f"{output_stem}_summary.json"
    write_mask(mask_path, mask)
    summary = {
        "schema_version": "sam2-region-assist-v0.1",
        "generated_at": utc_now_iso(),
        "image_path": str(image_path),
        "image_sha256": sha256_file(image_path),
        "model_id": model_id,
        "device": resolved_device,
        "prompt_geometry": prompt_geometry,
        "score": score,
        "mask_pixels": int(np.count_nonzero(mask)),
        "mask": {"path": str(mask_path)},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
