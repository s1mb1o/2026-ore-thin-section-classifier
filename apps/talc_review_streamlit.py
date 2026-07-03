from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

APP_PATH = Path(__file__).resolve()
PROJECT_ROOT = APP_PATH.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402


def patch_drawable_canvas_streamlit_compat() -> None:
    """Restore the legacy helper expected by streamlit-drawable-canvas."""
    try:
        import streamlit.elements.image as st_image
        from streamlit.elements.lib.image_utils import image_to_url
    except Exception:  # noqa: BLE001 - optional Streamlit internals changed.
        return
    if hasattr(st_image, "image_to_url"):
        return

    try:
        from streamlit.elements.lib.layout_utils import create_layout_config
    except Exception:  # noqa: BLE001 - Streamlit 1.50 exposes LayoutConfig instead.
        from streamlit.elements.lib.layout_utils import LayoutConfig

        def create_layout_config(width: int, allow_content_width: bool = True) -> Any:  # noqa: ARG001
            return LayoutConfig(width=width)

    def legacy_image_to_url(
        image: Any,
        width: int,
        clamp: bool,
        channels: str,
        output_format: str,
        image_id: str,
    ) -> str:
        layout_config = create_layout_config(width=width, allow_content_width=True)
        return image_to_url(image, layout_config, clamp, channels, output_format, image_id)

    st_image.image_to_url = legacy_image_to_url  # type: ignore[attr-defined]


patch_drawable_canvas_streamlit_compat()

MASK_SHAPE_EDITOR_DIR = PROJECT_ROOT / "apps/components/mask_shape_editor"
mask_shape_editor_component = components.declare_component("mask_shape_editor", path=str(MASK_SHAPE_EDITOR_DIR))

try:
    from streamlit_drawable_canvas import st_canvas  # type: ignore
except Exception:  # noqa: BLE001 - optional UI component.
    st_canvas = None

try:
    from ore_classifier.sam2_region_assist import DEFAULT_SAM2_MODEL_ID, generate_sam2_region_mask, sam2_assist_status
except Exception:  # noqa: BLE001 - optional SAM2 integration.
    DEFAULT_SAM2_MODEL_ID = "facebook/sam2.1-hiera-tiny"
    generate_sam2_region_mask = None
    sam2_assist_status = None

from ore_classifier.talc_blue_line_converter import (  # noqa: E402
    apply_edit_mask,
    make_overlay,
    polygon_mask,
    read_image_rgb,
    read_mask,
    rectangle_mask,
    save_reviewed_masks,
    write_mask,
)


ACTION_OPTIONS = {
    "Mark talc": "add_talc",
    "Remove talc": "erase_talc",
    "Mark uncertain": "uncertain",
    "Exclude artifact": "exclude_artifact",
}

ACTION_COLORS = {
    "add_talc": "#00ff00",
    "erase_talc": "#ff00ff",
    "uncertain": "#ffff00",
    "exclude_artifact": "#ff8800",
}

ACTION_FILL_RGBA = {
    "add_talc": "rgba(0, 255, 0, 0.45)",
    "erase_talc": "rgba(255, 0, 255, 0.45)",
    "uncertain": "rgba(255, 255, 0, 0.45)",
    "exclude_artifact": "rgba(255, 136, 0, 0.45)",
}

PREVIEW_MODES = ["Side by side", "QA overlay", "Original lines"]
CANVAS_BACKGROUNDS = ["Current mask", "QA overlay", "Original lines", "Original photo"]
EDITOR_MODES = ["Review canvas", "Upload mask", "Advanced", "Save review"]
CANVAS_TOOL_OPTIONS = {
    "Brush": "brush",
    "Erase": "erase",
    "Filled polygon": "polygon",
    "Filled box": "box",
    "SAM2 assist": "sam2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--conversion-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/talc_blue_line_conversion",
    )
    parser.add_argument("--max-display-width", type=int, default=1100)
    args, _unknown = parser.parse_known_args()
    return args


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    candidate = PROJECT_ROOT / path
    return candidate if candidate.exists() else path


def load_manifest(conversion_dir: Path) -> dict[str, Any]:
    manifest_path = conversion_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def sample_dir(sample: dict[str, Any]) -> Path:
    return resolve_path(sample["paths"]["final_talc_mask"]).parent


def load_masks(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    directory = sample_dir(sample)
    reviewed_talc = directory / "reviewed" / "reviewed_talc_mask.png"
    reviewed_ignore = directory / "reviewed" / "reviewed_ignore_mask.png"
    if reviewed_talc.exists():
        talc = read_mask(reviewed_talc)
    else:
        talc = read_mask(resolve_path(sample["paths"]["final_talc_mask"]))
    if reviewed_ignore.exists():
        ignore = read_mask(reviewed_ignore, talc.shape[:2])
    else:
        ignore = read_mask(resolve_path(sample["paths"]["ignore_mask"]), talc.shape[:2])
    return talc, ignore


def resize_preview(image_rgb: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    height, width = image_rgb.shape[:2]
    scale = min(1.0, max(320, int(max_width)) / max(1, width))
    if scale == 1.0:
        return image_rgb, scale
    preview = cv2.resize(image_rgb, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    return preview, scale


def image_data_url(image_rgb: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def make_original_lines_reference(image_rgb: np.ndarray, stroke_mask: np.ndarray) -> np.ndarray:
    return make_overlay(image_rgb, stroke_mask=stroke_mask)


def resize_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] == shape_hw:
        return np.where(mask > 0, 255, 0).astype(np.uint8)
    return cv2.resize(np.where(mask > 0, 255, 0).astype(np.uint8), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)


def canvas_color_masks(
    canvas_rgba: np.ndarray,
    original_shape_hw: tuple[int, int],
    background_rgb: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    rgb = canvas_rgba[:, :, :3].astype(np.uint8)
    if background_rgb is not None and background_rgb.shape[:2] == rgb.shape[:2]:
        diff = np.abs(rgb.astype(np.int16) - background_rgb[:, :, :3].astype(np.int16))
        changed = np.any(diff > 12, axis=2)
    else:
        changed = np.ones(rgb.shape[:2], dtype=bool)
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    masks = {
        "add_talc": changed & (green > 180) & (red < 110) & (blue < 140),
        "erase_talc": changed & (red > 180) & (blue > 160) & (green < 130),
        "uncertain": changed & (red > 180) & (green > 180) & (blue < 130),
        "exclude_artifact": changed & (red > 200) & (green > 90) & (green < 190) & (blue < 100),
    }
    out: dict[str, np.ndarray] = {}
    for action, mask in masks.items():
        out[action] = resize_mask(mask.astype(np.uint8) * 255, original_shape_hw)
    return out


def session_key(image_id: str, suffix: str) -> str:
    return f"talc_review_{image_id}_{suffix}"


def reset_invalid_choice(key: str, valid_options: list[str], default: str) -> None:
    if key in st.session_state and st.session_state[key] not in valid_options:
        st.session_state[key] = default


def segmented_choice(label: str, options: list[str], *, default: str, key: str) -> str:
    reset_invalid_choice(key, options, default)
    if key in st.session_state:
        return st.segmented_control(label, options, key=key)
    return st.segmented_control(label, options, default=default, key=key)


def set_flash(level: str, message: str) -> None:
    st.session_state["talc_review_flash"] = {"level": level, "message": message}


def show_flash() -> None:
    flash = st.session_state.pop("talc_review_flash", None)
    if not isinstance(flash, dict):
        return
    level = str(flash.get("level", "info"))
    message = str(flash.get("message", ""))
    if not message:
        return
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    else:
        st.info(message)


def set_current_masks(image_id: str, talc: np.ndarray, ignore: np.ndarray) -> None:
    st.session_state[session_key(image_id, "talc_mask")] = talc
    st.session_state[session_key(image_id, "ignore_mask")] = ignore


def current_masks(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    image_id = sample["image_id"]
    talc_key = session_key(image_id, "talc_mask")
    ignore_key = session_key(image_id, "ignore_mask")
    if talc_key not in st.session_state or ignore_key not in st.session_state:
        talc, ignore = load_masks(sample)
        set_current_masks(image_id, talc, ignore)
    return st.session_state[talc_key], st.session_state[ignore_key]


def append_edit(image_id: str, edit: dict[str, Any]) -> None:
    edits_key = session_key(image_id, "edits")
    st.session_state.setdefault(edits_key, [])
    st.session_state[edits_key].append(edit)


def current_edits(image_id: str) -> list[dict[str, Any]]:
    return list(st.session_state.get(session_key(image_id, "edits"), []))


def clear_edits(image_id: str) -> None:
    st.session_state.pop(session_key(image_id, "edits"), None)


def apply_mask_action(sample: dict[str, Any], edit_mask: np.ndarray, action: str, edit: dict[str, Any]) -> None:
    talc, ignore = current_masks(sample)
    talc, ignore = apply_edit_mask(talc, ignore, edit_mask, action)
    set_current_masks(sample["image_id"], talc, ignore)
    append_edit(sample["image_id"], edit)


def render_edit_metrics(sample: dict[str, Any]) -> None:
    talc, ignore = current_masks(sample)
    edits = current_edits(sample["image_id"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Current talc px", int(np.count_nonzero(talc)))
    col2.metric("Current ignore px", int(np.count_nonzero(ignore)))
    col3.metric("Unsaved edits", len(edits))


def normalize_points(points: Any, shape_hw: tuple[int, int]) -> list[dict[str, int]]:
    height, width = shape_hw
    normalized: list[dict[str, int]] = []
    if hasattr(points, "to_dict"):
        try:
            points = points.to_dict("records")
        except TypeError:
            points = points.to_dict()
    if not isinstance(points, list):
        return normalized
    for point in points:
        try:
            x_raw = point.get("x") if isinstance(point, dict) else point[0]
            y_raw = point.get("y") if isinstance(point, dict) else point[1]
            x = max(0, min(width - 1, int(round(float(x_raw)))))
            y = max(0, min(height - 1, int(round(float(y_raw)))))
        except (IndexError, KeyError, TypeError, ValueError):
            continue
        normalized.append({"x": x, "y": y})
    return normalized


def default_polygon_points(shape_hw: tuple[int, int]) -> list[dict[str, int]]:
    height, width = shape_hw
    left = max(0, width // 2 - width // 10)
    right = min(width - 1, width // 2 + width // 10)
    top = max(0, height // 2 - height // 10)
    bottom = min(height - 1, height // 2 + height // 10)
    return [{"x": left, "y": top}, {"x": right, "y": top}, {"x": right, "y": bottom}, {"x": left, "y": bottom}]


def mask_points(points: list[dict[str, int]]) -> list[list[int]]:
    return [[int(point["x"]), int(point["y"])] for point in points]


def default_box_geometry(shape_hw: tuple[int, int]) -> dict[str, int]:
    height, width = shape_hw
    return {
        "x1": max(0, width // 2 - width // 10),
        "y1": max(0, height // 2 - height // 10),
        "x2": min(width, width // 2 + width // 10),
        "y2": min(height, height // 2 + height // 10),
    }


def default_point_geometry(shape_hw: tuple[int, int]) -> dict[str, int]:
    height, width = shape_hw
    return {"x": width // 2, "y": height // 2}


def geometry_to_preview(geometry: dict[str, Any], shape_type: str, scale: float, preview_shape_hw: tuple[int, int]) -> dict[str, Any]:
    preview_height, preview_width = preview_shape_hw
    if shape_type == "polygon":
        points = normalize_points(geometry.get("points", []), (10**9, 10**9))
        return {
            "points": [
                {
                    "x": max(0, min(preview_width - 1, int(round(point["x"] * scale)))),
                    "y": max(0, min(preview_height - 1, int(round(point["y"] * scale)))),
                }
                for point in points
            ]
        }
    if shape_type == "point":
        return {
            "x": max(0, min(preview_width - 1, int(round(float(geometry.get("x", 0)) * scale)))),
            "y": max(0, min(preview_height - 1, int(round(float(geometry.get("y", 0)) * scale)))),
        }
    return {
        "x1": max(0, min(preview_width, int(round(float(geometry.get("x1", 0)) * scale)))),
        "y1": max(0, min(preview_height, int(round(float(geometry.get("y1", 0)) * scale)))),
        "x2": max(0, min(preview_width, int(round(float(geometry.get("x2", 0)) * scale)))),
        "y2": max(0, min(preview_height, int(round(float(geometry.get("y2", 0)) * scale)))),
    }


def geometry_to_original(geometry: dict[str, Any], shape_type: str, scale: float, shape_hw: tuple[int, int]) -> dict[str, Any] | None:
    if scale <= 0:
        return None
    height, width = shape_hw
    factor = 1.0 / scale
    if shape_type == "polygon":
        points = []
        for point in geometry.get("points", []):
            try:
                points.append(
                    {
                        "x": max(0, min(width - 1, int(round(float(point["x"]) * factor)))),
                        "y": max(0, min(height - 1, int(round(float(point["y"]) * factor)))),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        return {"points": points} if len(points) >= 3 else None
    if shape_type == "point":
        try:
            return {
                "x": max(0, min(width - 1, int(round(float(geometry["x"]) * factor)))),
                "y": max(0, min(height - 1, int(round(float(geometry["y"]) * factor)))),
            }
        except (KeyError, TypeError, ValueError):
            return None
    try:
        return {
            "x1": max(0, min(width, int(round(float(geometry["x1"]) * factor)))),
            "y1": max(0, min(height, int(round(float(geometry["y1"]) * factor)))),
            "x2": max(0, min(width, int(round(float(geometry["x2"]) * factor)))),
            "y2": max(0, min(height, int(round(float(geometry["y2"]) * factor)))),
        }
    except (KeyError, TypeError, ValueError):
        return None


def save_uploaded_mask(uploaded_file: Any, sample: dict[str, Any], action: str) -> None:
    image_id = sample["image_id"]
    talc, ignore = current_masks(sample)
    image = Image.open(uploaded_file).convert("L")
    mask = np.asarray(image)
    mask = resize_mask(mask, talc.shape[:2])
    apply_mask_action(
        sample,
        mask,
        action,
        {
            "edit_type": "uploaded_mask",
            "target_action": action,
            "source_name": uploaded_file.name,
        },
    )
    set_flash("success", f"Applied uploaded mask to {image_id}.")
    st.rerun()


def render_sam2_setup_controls(image_id: str, *, button_key_suffix: str) -> tuple[str, str | None]:
    model_id = st.text_input("SAM2 model", DEFAULT_SAM2_MODEL_ID, key=session_key(image_id, "sam2_model_id"))
    device_label = st.selectbox("SAM2 device", ["auto", "mps", "cuda", "cpu"], key=session_key(image_id, "sam2_device"))
    device = None if device_label == "auto" else device_label
    status_placeholder = st.empty()
    if sam2_assist_status is None:
        status_placeholder.info("SAM2 status helper is not available in this checkout.")
        return model_id, device
    status = sam2_assist_status(model_id=model_id, device=device)
    status_placeholder.json(status, expanded=False)
    force_reload = st.checkbox("Force SAM2 reload", value=False, key=session_key(image_id, "sam2_force_reload"))
    if st.button("Load/check SAM2", key=session_key(image_id, f"sam2_load_check_{button_key_suffix}")):
        status = sam2_assist_status(model_id=model_id, device=device, check_load=True, force_reload=force_reload)
        status_placeholder.json(status, expanded=False)
        if not status.get("available"):
            st.code("python3 -m pip install torch\npython3 -m pip install git+https://github.com/facebookresearch/sam2.git", language="bash")
    return model_id, device


def apply_sam2_prompt(
    sample: dict[str, Any],
    shape_hw: tuple[int, int],
    prompt_geometry: dict[str, Any],
    action: str,
    model_id: str,
    device: str | None,
    source: str,
) -> None:
    if generate_sam2_region_mask is None:
        st.info("SAM2 helper is not available in this checkout.")
        return
    try:
        image_id = sample["image_id"]
        result = generate_sam2_region_mask(
            image_path=resolve_path(sample["paths"]["source_image"]),
            prompt_geometry=prompt_geometry,
            out_dir=sample_dir(sample) / "reviewed" / "sam2_assist",
            model_id=model_id,
            device=device,
            output_name=f"{source}_{prompt_geometry['type']}_{len(current_edits(image_id)) + 1}",
        )
        mask_path = resolve_path(result["mask"]["path"])
        edit_mask = read_mask(mask_path, shape_hw)
        apply_mask_action(
            sample,
            edit_mask,
            action,
            {
                "edit_type": f"{source}_{prompt_geometry['type']}",
                "target_action": action,
                "prompt": prompt_geometry,
                "sam2_result": result,
            },
        )
        set_flash("success", "SAM2 edit applied.")
        st.rerun()
    except Exception as exc:  # noqa: BLE001 - Streamlit should surface runtime setup failures.
        st.error(str(exc))


def render_canvas_sam2_tool(
    sample: dict[str, Any],
    image_rgb: np.ndarray,
    preview: np.ndarray,
    scale: float,
    action: str,
) -> None:
    image_id = sample["image_id"]
    if generate_sam2_region_mask is None or sam2_assist_status is None:
        st.info("SAM2 helper is not available in this checkout.")
        return
    model_id, device = render_sam2_setup_controls(image_id, button_key_suffix="canvas")
    prompt_label = segmented_choice(
        "SAM2 prompt",
        ["Point", "Box"],
        default="Point",
        key=session_key(image_id, "canvas_sam2_prompt"),
    )
    shape_type = "point" if prompt_label == "Point" else "box"
    geometry_key = session_key(image_id, f"canvas_sam2_geometry_{shape_type}")
    if geometry_key not in st.session_state:
        st.session_state[geometry_key] = (
            default_point_geometry(image_rgb.shape[:2]) if shape_type == "point" else default_box_geometry(image_rgb.shape[:2])
        )

    initial_geometry = geometry_to_preview(st.session_state[geometry_key], shape_type, scale, preview.shape[:2])
    value = mask_shape_editor_component(
        backgroundImage=image_data_url(preview),
        width=int(preview.shape[1]),
        height=int(preview.shape[0]),
        shapeType=shape_type,
        fillColor="rgba(31, 111, 235, 0.22)",
        strokeColor=ACTION_COLORS[action],
        initialGeometry=initial_geometry,
        key=session_key(image_id, f"canvas_sam2_prompt_editor_{shape_type}"),
        default={"type": shape_type, "geometry": initial_geometry, "width": int(preview.shape[1]), "height": int(preview.shape[0])},
    )
    if isinstance(value, dict) and isinstance(value.get("geometry"), dict):
        original_geometry = geometry_to_original(value["geometry"], shape_type, scale, image_rgb.shape[:2])
        if original_geometry is not None:
            st.session_state[geometry_key] = original_geometry

    if st.button("Run SAM2", type="primary", key=session_key(image_id, "canvas_sam2_run")):
        geometry = st.session_state[geometry_key]
        if shape_type == "point":
            prompt_geometry = {"type": "point_xy", "x": int(geometry["x"]), "y": int(geometry["y"])}
        else:
            prompt_geometry = {
                "type": "rectangle_xyxy",
                "x1": int(geometry["x1"]),
                "y1": int(geometry["y1"]),
                "x2": int(geometry["x2"]),
                "y2": int(geometry["y2"]),
            }
        apply_sam2_prompt(sample, image_rgb.shape[:2], prompt_geometry, action, model_id, device, "canvas_sam2")


def render_canvas_shape_tool(
    sample: dict[str, Any],
    image_rgb: np.ndarray,
    preview: np.ndarray,
    scale: float,
    action: str,
    background_label: str,
    shape_type: str,
) -> None:
    image_id = sample["image_id"]
    geometry_key = session_key(image_id, f"canvas_shape_{shape_type}")
    if geometry_key not in st.session_state:
        if shape_type == "polygon":
            st.session_state[geometry_key] = {"points": default_polygon_points(image_rgb.shape[:2])}
        else:
            st.session_state[geometry_key] = default_box_geometry(image_rgb.shape[:2])

    initial_geometry = geometry_to_preview(st.session_state[geometry_key], shape_type, scale, preview.shape[:2])
    value = mask_shape_editor_component(
        backgroundImage=image_data_url(preview),
        width=int(preview.shape[1]),
        height=int(preview.shape[0]),
        shapeType=shape_type,
        fillColor=ACTION_FILL_RGBA[action],
        strokeColor=ACTION_COLORS[action],
        initialGeometry=initial_geometry,
        key=session_key(image_id, f"canvas_shape_editor_{shape_type}"),
        default={"type": shape_type, "geometry": initial_geometry, "width": int(preview.shape[1]), "height": int(preview.shape[0])},
    )
    if isinstance(value, dict) and isinstance(value.get("geometry"), dict):
        original_geometry = geometry_to_original(value["geometry"], shape_type, scale, image_rgb.shape[:2])
        if original_geometry is not None:
            st.session_state[geometry_key] = original_geometry

    shape_label = "polygon" if shape_type == "polygon" else "box"
    col_apply, col_reset = st.columns([1, 1])
    if col_reset.button("Reset shape", key=session_key(image_id, f"reset_canvas_{shape_type}")):
        if shape_type == "polygon":
            st.session_state[geometry_key] = {"points": default_polygon_points(image_rgb.shape[:2])}
        else:
            st.session_state[geometry_key] = default_box_geometry(image_rgb.shape[:2])
        st.rerun()
    if col_apply.button(f"Apply filled {shape_label}", type="primary", key=session_key(image_id, f"apply_canvas_{shape_type}")):
        geometry = st.session_state[geometry_key]
        if shape_type == "polygon":
            points = normalize_points(geometry.get("points", []), image_rgb.shape[:2])
            if len(points) < 3:
                st.warning("Polygon needs at least three points.")
                return
            edit_mask = polygon_mask(image_rgb.shape[:2], mask_points(points))
            saved_geometry: dict[str, Any] = {"points": mask_points(points)}
        else:
            edit_mask = rectangle_mask(
                image_rgb.shape[:2],
                int(geometry["x1"]),
                int(geometry["y1"]),
                int(geometry["x2"]),
                int(geometry["y2"]),
            )
            saved_geometry = {key: int(geometry[key]) for key in ["x1", "y1", "x2", "y2"]}
        apply_mask_action(
            sample,
            edit_mask,
            action,
            {
                "edit_type": f"canvas_filled_{shape_label}",
                "target_action": action,
                "background": background_label,
                "geometry": saved_geometry,
            },
        )
        set_flash("success", f"Filled {shape_label} applied.")
        st.rerun()


def render_canvas_editor(
    sample: dict[str, Any],
    image_rgb: np.ndarray,
    current_mask_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    original_lines_rgb: np.ndarray,
    max_display_width: int,
) -> None:
    render_edit_metrics(sample)
    background_key = session_key(sample["image_id"], "canvas_background")
    reset_invalid_choice(background_key, CANVAS_BACKGROUNDS, "Current mask")
    background_label = st.selectbox(
        "View while editing",
        CANVAS_BACKGROUNDS,
        key=background_key,
    )
    background_image = {
        "Current mask": current_mask_rgb,
        "Original lines": original_lines_rgb,
        "QA overlay": overlay_rgb,
        "Original photo": image_rgb,
    }[background_label]
    preview, scale = resize_preview(background_image, max_display_width)
    canvas_tool_labels = list(CANVAS_TOOL_OPTIONS.keys())
    tool_key = session_key(sample["image_id"], "canvas_tool")
    tool_label = segmented_choice(
        "Tool",
        canvas_tool_labels,
        default="Brush",
        key=tool_key,
    )
    tool_kind = CANVAS_TOOL_OPTIONS[tool_label]
    if tool_kind == "erase":
        action = "erase_talc"
    else:
        action_label = st.selectbox("Apply as", list(ACTION_OPTIONS.keys()), key=session_key(sample["image_id"], "canvas_action"))
        action = ACTION_OPTIONS[action_label]
    if tool_kind == "sam2":
        render_canvas_sam2_tool(sample, image_rgb, preview, scale, action)
        return
    if tool_kind in {"polygon", "box"}:
        render_canvas_shape_tool(sample, image_rgb, preview, scale, action, background_label, tool_kind)
        return
    if st_canvas is None:
        st.info("Canvas component is not installed.")
        return
    mode = "freedraw"
    stroke_color = ACTION_COLORS[action]
    stroke_width = st.slider("Stroke width", 2, 48, 10, key=session_key(sample["image_id"], "canvas_stroke_width"))
    fill_color = "rgba(0, 0, 0, 0)"
    drawing_key = session_key(sample["image_id"], "canvas_drawing")
    size_key = session_key(sample["image_id"], "canvas_size")
    drawing_tool_key = session_key(sample["image_id"], "canvas_drawing_tool")
    canvas_size = (int(preview.shape[1]), int(preview.shape[0]))
    if st.session_state.get(size_key) != canvas_size or st.session_state.get(drawing_tool_key) != tool_kind:
        st.session_state[size_key] = canvas_size
        st.session_state[drawing_tool_key] = tool_kind
        st.session_state.pop(drawing_key, None)
    canvas = st_canvas(
        fill_color=fill_color,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_image=Image.fromarray(preview),
        update_streamlit=True,
        height=preview.shape[0],
        width=preview.shape[1],
        drawing_mode=mode,
        initial_drawing=st.session_state.get(drawing_key),
        key=session_key(sample["image_id"], "canvas"),
    )
    if canvas.json_data is not None:
        st.session_state[drawing_key] = canvas.json_data
    col_apply, col_clear = st.columns([1, 1])
    apply_clicked = col_apply.button("Apply draft", type="primary")
    if col_clear.button("Clear draft"):
        st.session_state.pop(drawing_key, None)
        st.rerun()
    if apply_clicked:
        if canvas.image_data is None:
            st.warning("No canvas edit to apply.")
            return
        masks = canvas_color_masks(canvas.image_data.astype(np.uint8), image_rgb.shape[:2], background_rgb=preview)
        applied = 0
        for action_name, edit_mask in masks.items():
            if np.count_nonzero(edit_mask) == 0:
                continue
            talc, ignore = current_masks(sample)
            talc, ignore = apply_edit_mask(talc, ignore, edit_mask, action_name)
            set_current_masks(sample["image_id"], talc, ignore)
            applied += int(np.count_nonzero(edit_mask))
        if applied:
            append_edit(
                sample["image_id"],
                {
                    "edit_type": "canvas_mask_area",
                    "scale": scale,
                    "tool": tool_label,
                    "background": background_label,
                    "applied_pixels_in_preview": applied,
                },
            )
            st.session_state.pop(drawing_key, None)
            set_flash("success", "Draft applied.")
            st.rerun()
        else:
            st.warning("No colored canvas pixels were detected.")


def render_geometry_editor(
    sample: dict[str, Any],
    image_rgb: np.ndarray,
    current_mask_rgb: np.ndarray,
    original_lines_rgb: np.ndarray,
    max_display_width: int,
) -> None:
    image_id = sample["image_id"]
    render_edit_metrics(sample)
    background_label = st.selectbox(
        "Geometry background",
        ["Current mask", "Original lines", "Original photo"],
        key=session_key(image_id, "geometry_background"),
    )
    background_image = {
        "Current mask": current_mask_rgb,
        "Original lines": original_lines_rgb,
        "Original photo": image_rgb,
    }[background_label]
    shape_label = segmented_choice(
        "Geometry type",
        ["Polygon", "Box"],
        default="Polygon",
        key=session_key(image_id, "geometry_type"),
    )
    shape_type = "polygon" if shape_label == "Polygon" else "box"
    action_label = st.selectbox("Geometry action", list(ACTION_OPTIONS.keys()), key=session_key(image_id, "geometry_action"))
    action = ACTION_OPTIONS[action_label]
    geometry_key = session_key(image_id, f"geometry_{shape_type}")
    if geometry_key not in st.session_state:
        if shape_type == "polygon":
            st.session_state[geometry_key] = {"points": default_polygon_points(image_rgb.shape[:2])}
        else:
            st.session_state[geometry_key] = default_box_geometry(image_rgb.shape[:2])

    preview, scale = resize_preview(background_image, max_display_width)
    initial_geometry = geometry_to_preview(st.session_state[geometry_key], shape_type, scale, preview.shape[:2])
    value = mask_shape_editor_component(
        backgroundImage=image_data_url(preview),
        width=int(preview.shape[1]),
        height=int(preview.shape[0]),
        shapeType=shape_type,
        fillColor=ACTION_FILL_RGBA[action],
        strokeColor=ACTION_COLORS[action],
        initialGeometry=initial_geometry,
        key=session_key(image_id, f"mask_shape_editor_{shape_type}"),
        default={"type": shape_type, "geometry": initial_geometry, "width": int(preview.shape[1]), "height": int(preview.shape[0])},
    )
    if isinstance(value, dict) and isinstance(value.get("geometry"), dict):
        original_geometry = geometry_to_original(value["geometry"], shape_type, scale, image_rgb.shape[:2])
        if original_geometry is not None:
            st.session_state[geometry_key] = original_geometry

    if st.button("Apply geometry", type="primary", key=session_key(image_id, "apply_geometry")):
        geometry = st.session_state[geometry_key]
        if shape_type == "polygon":
            points = normalize_points(geometry.get("points", []), image_rgb.shape[:2])
            if len(points) < 3:
                st.warning("Polygon needs at least three points.")
                return
            edit_mask = polygon_mask(image_rgb.shape[:2], mask_points(points))
            edit_type = "geometry_polygon_drag"
            saved_geometry: dict[str, Any] = {"points": mask_points(points)}
        else:
            edit_mask = rectangle_mask(
                image_rgb.shape[:2],
                int(geometry["x1"]),
                int(geometry["y1"]),
                int(geometry["x2"]),
                int(geometry["y2"]),
            )
            edit_type = "geometry_box_drag"
            saved_geometry = {key: int(geometry[key]) for key in ["x1", "y1", "x2", "y2"]}
        apply_mask_action(
            sample,
            edit_mask,
            action,
            {
                "edit_type": edit_type,
                "target_action": action,
                "background": background_label,
                "geometry": saved_geometry,
            },
        )
        set_flash("success", "Geometry edit applied.")
        st.rerun()


def render_rectangle_editor(sample: dict[str, Any], shape_hw: tuple[int, int], *, show_metrics: bool = True) -> None:
    image_id = sample["image_id"]
    height, width = shape_hw
    if show_metrics:
        render_edit_metrics(sample)
    with st.form(session_key(image_id, "rect_form")):
        action_label = st.selectbox("Rectangle action", list(ACTION_OPTIONS.keys()))
        col1, col2, col3, col4 = st.columns(4)
        x1 = col1.number_input("x1", min_value=0, max_value=width, value=0)
        y1 = col2.number_input("y1", min_value=0, max_value=height, value=0)
        x2 = col3.number_input("x2", min_value=0, max_value=width, value=min(width, 256))
        y2 = col4.number_input("y2", min_value=0, max_value=height, value=min(height, 256))
        submitted = st.form_submit_button("Apply rectangle")
    if submitted:
        action = ACTION_OPTIONS[action_label]
        edit_mask = rectangle_mask(shape_hw, int(x1), int(y1), int(x2), int(y2))
        apply_mask_action(
            sample,
            edit_mask,
            action,
            {
                "edit_type": "rectangle_xyxy",
                "target_action": action,
                "geometry": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
            },
        )
        set_flash("success", "Rectangle edit applied.")
        st.rerun()


def render_polygon_editor(sample: dict[str, Any], shape_hw: tuple[int, int], *, show_metrics: bool = True) -> None:
    image_id = sample["image_id"]
    if show_metrics:
        render_edit_metrics(sample)
    points_key = session_key(image_id, "polygon_points")
    version_key = session_key(image_id, "polygon_points_version")
    st.session_state.setdefault(points_key, default_polygon_points(shape_hw))
    st.session_state.setdefault(version_key, 0)

    points = normalize_points(st.session_state[points_key], shape_hw)
    if len(points) < 3:
        points = default_polygon_points(shape_hw)
        st.session_state[points_key] = points
    edited_points = st.data_editor(
        points,
        num_rows="dynamic",
        width="stretch",
        hide_index=False,
        column_config={
            "x": st.column_config.NumberColumn("x", min_value=0, max_value=shape_hw[1] - 1, step=1),
            "y": st.column_config.NumberColumn("y", min_value=0, max_value=shape_hw[0] - 1, step=1),
        },
        key=f"{session_key(image_id, 'polygon_table')}_{st.session_state[version_key]}",
    )
    normalized = normalize_points(edited_points, shape_hw)
    if normalized:
        st.session_state[points_key] = normalized
        points = normalized

    col_add, col_delete = st.columns(2)
    insert_after = col_add.number_input(
        "Insert midpoint after index",
        min_value=0,
        max_value=max(0, len(points) - 1),
        value=0,
        step=1,
        key=session_key(image_id, "polygon_insert_after"),
    )
    if col_add.button("Add polygon point"):
        index = int(insert_after)
        next_point = points[(index + 1) % len(points)]
        current_point = points[index]
        points.insert(
            index + 1,
            {
                "x": int(round((current_point["x"] + next_point["x"]) / 2)),
                "y": int(round((current_point["y"] + next_point["y"]) / 2)),
            },
        )
        st.session_state[points_key] = points
        st.session_state[version_key] += 1
        st.rerun()

    delete_index = col_delete.number_input(
        "Delete point index",
        min_value=0,
        max_value=max(0, len(points) - 1),
        value=0,
        step=1,
        key=session_key(image_id, "polygon_delete_index"),
    )
    if col_delete.button("Delete polygon point", disabled=len(points) <= 3):
        points.pop(int(delete_index))
        st.session_state[points_key] = points
        st.session_state[version_key] += 1
        st.rerun()

    with st.form(session_key(image_id, "polygon_form")):
        action_label = st.selectbox("Polygon action", list(ACTION_OPTIONS.keys()))
        submitted = st.form_submit_button("Apply polygon")
    if submitted:
        if len(points) < 3:
            st.warning("Polygon needs at least three points.")
            return
        action = ACTION_OPTIONS[action_label]
        edit_mask = polygon_mask(shape_hw, mask_points(points))
        apply_mask_action(
            sample,
            edit_mask,
            action,
            {
                "edit_type": "polygon_xy",
                "target_action": action,
                "geometry": {"points": mask_points(points)},
            },
        )
        set_flash("success", "Polygon edit applied.")
        st.rerun()


def render_sam2_editor(sample: dict[str, Any], shape_hw: tuple[int, int], *, show_metrics: bool = True) -> None:
    if generate_sam2_region_mask is None or sam2_assist_status is None:
        st.info("SAM2 helper is not available in this checkout.")
        return
    if show_metrics:
        render_edit_metrics(sample)
    image_id = sample["image_id"]
    height, width = shape_hw
    model_id, device = render_sam2_setup_controls(image_id, button_key_suffix="editor")
    action_label = st.selectbox("SAM2 action", list(ACTION_OPTIONS.keys()), key=session_key(image_id, "sam2_action"))
    prompt_type = segmented_choice(
        "SAM2 prompt",
        ["Point", "Box"],
        default="Point",
        key=session_key(image_id, "sam2_prompt_type"),
    )
    if prompt_type == "Point":
        col1, col2 = st.columns(2)
        x = col1.number_input("point x", min_value=0, max_value=width - 1, value=width // 2)
        y = col2.number_input("point y", min_value=0, max_value=height - 1, value=height // 2)
        prompt_geometry = {"type": "point_xy", "x": int(x), "y": int(y)}
    else:
        col1, col2, col3, col4 = st.columns(4)
        x1 = col1.number_input("box x1", min_value=0, max_value=width, value=max(0, width // 2 - width // 10))
        y1 = col2.number_input("box y1", min_value=0, max_value=height, value=max(0, height // 2 - height // 10))
        x2 = col3.number_input("box x2", min_value=0, max_value=width, value=min(width, width // 2 + width // 10))
        y2 = col4.number_input("box y2", min_value=0, max_value=height, value=min(height, height // 2 + height // 10))
        prompt_geometry = {"type": "rectangle_xyxy", "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
    if st.button("Run SAM2"):
        apply_sam2_prompt(
            sample,
            shape_hw,
            prompt_geometry,
            ACTION_OPTIONS[action_label],
            model_id,
            device,
            "sam2_editor",
        )


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="Talc Region Review", layout="wide")
    st.title("Talc Region Review")
    show_flash()

    conversion_dir_input = st.sidebar.text_input("Conversion directory", str(args.conversion_dir))
    conversion_dir = resolve_path(conversion_dir_input)
    max_display_width = st.sidebar.slider("Display width", 480, 1800, int(args.max_display_width), 20)
    preview_mode = st.sidebar.selectbox("Top preview", PREVIEW_MODES)

    try:
        manifest = load_manifest(conversion_dir)
    except Exception as exc:  # noqa: BLE001 - show actionable UI error.
        st.error(str(exc))
        return
    samples = manifest.get("samples", [])
    if not samples:
        st.warning("No samples in manifest.")
        return

    labels = [
        f"{sample['image_id']} | {sample.get('status', 'unknown')} | overlap={sample.get('overlap_pixels', 0)}"
        for sample in samples
    ]
    selected_index = st.sidebar.selectbox("Sample", list(range(len(samples))), format_func=lambda i: labels[i])
    sample = samples[int(selected_index)]
    image_id = sample["image_id"]
    image_rgb = read_image_rgb(resolve_path(sample["paths"]["source_image"]))
    talc_mask, ignore_mask = current_masks(sample)
    sulfide_mask = read_mask(resolve_path(sample["paths"]["sulfide_mask"]), image_rgb.shape[:2])
    overlap_mask = read_mask(resolve_path(sample["paths"]["sulfide_overlap_mask"]), image_rgb.shape[:2])
    stroke_mask = read_mask(resolve_path(sample["paths"]["raw_blue_stroke"]), image_rgb.shape[:2])

    original_lines_rgb = make_original_lines_reference(image_rgb, stroke_mask)
    overlay_rgb = make_overlay(
        image_rgb,
        talc_mask=talc_mask,
        stroke_mask=stroke_mask,
        sulfide_mask=sulfide_mask,
        overlap_mask=overlap_mask,
        ignore_mask=ignore_mask,
    )
    current_mask_rgb = make_overlay(
        image_rgb,
        talc_mask=talc_mask,
        sulfide_mask=sulfide_mask,
        overlap_mask=overlap_mask,
        ignore_mask=ignore_mask,
    )

    left, right = st.columns([2, 1])
    with left:
        if preview_mode == "Side by side":
            line_preview, _ = resize_preview(original_lines_rgb, max(320, max_display_width // 2))
            qa_preview, _ = resize_preview(overlay_rgb, max(320, max_display_width // 2))
            line_col, qa_col = st.columns(2)
            line_col.image(line_preview, caption=f"{image_id} | original lines")
            qa_col.image(qa_preview, caption=f"{image_id} | QA overlay")
        elif preview_mode == "Original lines":
            preview, _scale = resize_preview(original_lines_rgb, max_display_width)
            st.image(preview, caption=f"{image_id} | original lines")
        else:
            preview, _scale = resize_preview(overlay_rgb, max_display_width)
            st.image(preview, caption=f"{image_id} | QA overlay")
    with right:
        st.metric("Talc px", int(np.count_nonzero(talc_mask)))
        st.metric("Ignore px", int(np.count_nonzero(ignore_mask)))
        st.metric("Overlap px", int(np.count_nonzero(overlap_mask)))
        st.metric("Original line px", int(np.count_nonzero(stroke_mask)))
        st.write(sample.get("status", "unknown"))
        if st.button("Reload base masks"):
            talc, ignore = load_masks(sample)
            set_current_masks(image_id, talc, ignore)
            clear_edits(image_id)
            st.session_state.pop(session_key(image_id, "canvas_drawing"), None)
            st.session_state.pop(session_key(image_id, "canvas_size"), None)
            st.session_state.pop(session_key(image_id, "canvas_drawing_tool"), None)
            st.session_state.pop(session_key(image_id, "canvas_shape_polygon"), None)
            st.session_state.pop(session_key(image_id, "canvas_shape_box"), None)
            set_flash("success", "Base masks reloaded and unsaved edits cleared.")
            st.rerun()

    editor_key = session_key(image_id, "editor_mode")
    editor_mode = segmented_choice(
        "Workspace",
        EDITOR_MODES,
        default="Review canvas",
        key=editor_key,
    )
    if editor_mode == "Review canvas":
        render_canvas_editor(sample, image_rgb, current_mask_rgb, overlay_rgb, original_lines_rgb, max_display_width)
    elif editor_mode == "Upload mask":
        render_edit_metrics(sample)
        uploaded = st.file_uploader("Mask file", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
        upload_action_label = st.selectbox("Upload action", list(ACTION_OPTIONS.keys()))
        if uploaded is not None and st.button("Apply mask file"):
            save_uploaded_mask(uploaded, sample, ACTION_OPTIONS[upload_action_label])
    elif editor_mode == "Advanced":
        render_edit_metrics(sample)
        with st.expander("Exact polygon coordinates"):
            render_polygon_editor(sample, image_rgb.shape[:2], show_metrics=False)
        with st.expander("Exact rectangle coordinates"):
            render_rectangle_editor(sample, image_rgb.shape[:2], show_metrics=False)
        with st.expander("Coordinate SAM2 prompt"):
            render_sam2_editor(sample, image_rgb.shape[:2], show_metrics=False)
    elif editor_mode == "Save review":
        render_edit_metrics(sample)
        if st.button("Save reviewed masks", type="primary"):
            review_summary = save_reviewed_masks(sample_dir(sample), talc_mask, ignore_mask, current_edits(image_id))
            st.success("Reviewed masks saved.")
            st.json(review_summary)
        reviewed_dir = sample_dir(sample) / "reviewed"
        st.write(str(reviewed_dir))
        if reviewed_dir.exists():
            for path in sorted(reviewed_dir.glob("*")):
                st.write(path.name)


if __name__ == "__main__":
    main()
