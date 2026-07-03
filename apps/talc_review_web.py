#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import mimetypes
import shutil
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ore_classifier.sam2_region_assist import (  # noqa: E402
    DEFAULT_SAM2_MODEL_ID,
    Sam2AssistFailure,
    generate_sam2_region_mask,
    sam2_assist_status,
)
from ore_classifier.talc_blue_line_converter import (  # noqa: E402
    TalcConversionConfig,
    convert_talc_annotation_folder,
    ensure_uint8_mask,
    make_overlay,
    mask_pixels,
    read_image_rgb,
    read_mask,
    utc_now_iso,
    write_image_rgb,
    write_mask,
)


DEFAULT_ANNOTATED_DIR = ROOT / "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования"
DEFAULT_WORKSPACE_DIR = ROOT / "outputs/talc_blue_line_conversion"
MAX_POST_BYTES = 150 * 1024 * 1024


class ApiError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class ReviewSample:
    sample_id: str
    image_name: str
    annotated_path: Path
    original_path: Path | None
    sample_dir: Path
    summary: dict[str, Any]
    status: str
    review_state: str


def json_response(payload: Any, status: int = HTTPStatus.OK) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def resolve_path(path: Path | str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


def sanitize_view_settings(payload: dict[str, Any]) -> dict[str, Any]:
    settings = payload.get("view_settings")
    if not isinstance(settings, dict):
        return {}
    sanitized: dict[str, Any] = {}
    threshold = settings.get("brightness_threshold_luma")
    if threshold is not None:
        try:
            sanitized["brightness_threshold_luma"] = max(0, min(255, int(threshold)))
        except (TypeError, ValueError):
            pass
    for key in ("brightness_threshold_formula", "background_mode"):
        value = settings.get(key)
        if isinstance(value, str) and value:
            sanitized[key] = value[:500]
    return sanitized


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def decode_mask_data_url(data_url: str, expected_shape_hw: tuple[int, int]) -> np.ndarray:
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
            f"mask dimensions {mask.shape[1]}x{mask.shape[0]} do not match sample "
            f"{expected_shape_hw[1]}x{expected_shape_hw[0]}",
        )
    return ensure_uint8_mask(mask)


class TalcReviewStore:
    def __init__(
        self,
        *,
        annotated_dir: Path | None,
        original_dir: Path | None,
        workspace_dir: Path,
        conversion_dir: Path | None,
        sulfide_mask_dir: Path | None,
        silicate_mask_dir: Path | None,
        reconvert: bool,
        limit: int | None,
        sam2_model_id: str,
        sam2_device: str | None,
    ) -> None:
        self.lock = threading.RLock()
        self.annotated_dir = resolve_path(annotated_dir) if annotated_dir else None
        self.original_dir = resolve_path(original_dir) if original_dir else None
        self.workspace_dir = resolve_path(conversion_dir or workspace_dir)
        self.sulfide_mask_dir = resolve_path(sulfide_mask_dir) if sulfide_mask_dir else None
        self.silicate_mask_dir = resolve_path(silicate_mask_dir) if silicate_mask_dir else None
        self.sam2_model_id = sam2_model_id
        self.sam2_device = sam2_device
        self.manifest: dict[str, Any] = {}
        self.samples: list[ReviewSample] = []
        self.samples_by_id: dict[str, ReviewSample] = {}
        self.artifacts: dict[str, Path] = {}
        self.allowed_roots: list[Path] = [ROOT.resolve(), self.workspace_dir.resolve()]
        self.load_or_convert(reconvert=reconvert, limit=limit)

    def load_or_convert(self, *, reconvert: bool, limit: int | None) -> None:
        manifest_path = self.workspace_dir / "manifest.json"
        if reconvert or not manifest_path.exists():
            if not self.annotated_dir:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    f"missing conversion manifest at {manifest_path}; pass --annotated-dir or --conversion-dir with an existing manifest",
                )
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
            self.manifest = convert_talc_annotation_folder(
                self.annotated_dir,
                self.workspace_dir,
                TalcConversionConfig(),
                sulfide_mask_dir=self.sulfide_mask_dir,
                silicate_mask_dir=self.silicate_mask_dir,
                limit=limit,
            )
        else:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        input_path = resolve_path(self.manifest.get("input_path", self.annotated_dir or DEFAULT_ANNOTATED_DIR))
        if self.annotated_dir is None:
            self.annotated_dir = input_path
        if self.original_dir is None:
            self.original_dir = self.annotated_dir.parent

        for root in [self.annotated_dir, self.original_dir, self.sulfide_mask_dir, self.silicate_mask_dir]:
            if root is not None:
                self.allowed_roots.append(root.resolve())
        self.refresh_samples()

    def refresh_samples(self) -> None:
        with self.lock:
            samples: list[ReviewSample] = []
            for summary in self.manifest.get("samples", []):
                sample_id = str(summary.get("image_id") or Path(summary.get("image_path", "")).stem)
                annotated_path = resolve_path(summary.get("image_path") or summary["paths"]["source_image"])
                source_copy = resolve_path(summary["paths"]["source_image"])
                sample_dir = source_copy.parent
                image_name = annotated_path.name
                original = self.original_dir / image_name if self.original_dir else None
                original_path = original.resolve() if original and original.exists() else None
                reviewed = (sample_dir / "reviewed/reviewed_talc_mask.png").exists()
                current = (sample_dir / "current_talc_mask.png").exists()
                if original_path is None:
                    status = "missing_original"
                else:
                    status = str(summary.get("status") or "unknown")
                if reviewed:
                    review_state = "reviewed"
                elif current:
                    review_state = "working"
                else:
                    review_state = "not_opened"
                samples.append(
                    ReviewSample(
                        sample_id=sample_id,
                        image_name=image_name,
                        annotated_path=annotated_path,
                        original_path=original_path,
                        sample_dir=sample_dir,
                        summary=summary,
                        status=status,
                        review_state=review_state,
                    )
                )
            samples.sort(key=lambda item: item.image_name.lower())
            self.samples = samples
            self.samples_by_id = {sample.sample_id: sample for sample in samples}

    def get_sample(self, sample_id: str) -> ReviewSample:
        with self.lock:
            sample = self.samples_by_id.get(sample_id)
        if sample is None:
            raise ApiError(HTTPStatus.NOT_FOUND, f"unknown sample: {sample_id}")
        return sample

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
        version = int(resolved.stat().st_mtime)
        filename = urllib.parse.quote(resolved.name)
        return f"/artifacts/{artifact_id}/{filename}?v={version}"

    def artifact_path(self, artifact_id: str) -> Path:
        path = self.artifacts.get(artifact_id)
        if path is None or not path.exists():
            raise ApiError(HTTPStatus.NOT_FOUND, "unknown artifact")
        return path

    def current_mask_path(self, sample: ReviewSample) -> Path:
        return sample.sample_dir / "current_talc_mask.png"

    def working_state_path(self, sample: ReviewSample) -> Path:
        return sample.sample_dir / "working_state.json"

    def ensure_current_mask(self, sample: ReviewSample) -> Path:
        with self.lock:
            current_path = self.current_mask_path(sample)
            expected_shape = (int(sample.summary["height"]), int(sample.summary["width"]))
            if current_path.exists():
                try:
                    current_mask = read_mask(current_path)
                    if current_mask.shape[:2] == expected_shape:
                        return current_path
                    recovery_reason = f"wrong_size_{current_mask.shape[1]}x{current_mask.shape[0]}"
                except Exception as exc:
                    recovery_reason = f"unreadable_{type(exc).__name__}"
            else:
                recovery_reason = "created"

            if recovery_reason != "created":
                recovered_path = sample.sample_dir / f"current_talc_mask.recovered.{int(time.time())}.png"
                try:
                    shutil.move(current_path, recovered_path)
                except OSError:
                    recovered_path = None
            else:
                recovered_path = None

            reviewed_path = sample.sample_dir / "reviewed/reviewed_talc_mask.png"
            final_path = resolve_path(sample.summary["paths"]["final_talc_mask"])
            source_path = reviewed_path if reviewed_path.exists() else final_path
            source_label = "reviewed" if source_path == reviewed_path else "autodetected"
            current_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, current_path)
            current_mask = read_mask(current_path)
            if current_mask.shape[:2] != expected_shape:
                current_mask = read_mask(current_path, expected_shape)
                write_mask(current_path, current_mask)
            state = {
                "schema_version": "talc-current-mask-state-v0.1",
                "sample_id": sample.sample_id,
                "created_at": utc_now_iso(),
                "updated_at": utc_now_iso(),
                "source": source_label if recovery_reason == "created" else f"{source_label}_recovered",
                "recovery_reason": None if recovery_reason == "created" else recovery_reason,
                "recovered_path": str(recovered_path) if recovered_path else None,
                "current_talc_mask": str(current_path),
                "current_talc_pixels": mask_pixels(current_mask),
                "edits": [],
            }
            self.working_state_path(sample).write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.refresh_samples()
            return current_path

    def reset_current_mask(self, sample_id: str) -> dict[str, Any]:
        sample = self.get_sample(sample_id)
        final_path = resolve_path(sample.summary["paths"]["final_talc_mask"])
        current_path = self.current_mask_path(sample)
        shutil.copy2(final_path, current_path)
        state = {
            "schema_version": "talc-current-mask-state-v0.1",
            "sample_id": sample.sample_id,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "source": "autodetected_reset",
            "current_talc_mask": str(current_path),
            "current_talc_pixels": mask_pixels(read_mask(current_path)),
            "edits": [],
        }
        self.working_state_path(sample).write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.refresh_samples()
        return self.sample_payload(sample_id)

    def manifest_payload(self) -> dict[str, Any]:
        with self.lock:
            samples = [self.sample_card(sample) for sample in self.samples]
        counts: dict[str, int] = {}
        for sample in samples:
            key = f"{sample['status']}:{sample['review_state']}"
            counts[key] = counts.get(key, 0) + 1
        return {
            "schema_version": "talc-review-web-manifest-v0.1",
            "generated_at": utc_now_iso(),
            "workspace_dir": str(self.workspace_dir),
            "annotated_dir": str(self.annotated_dir) if self.annotated_dir else None,
            "original_dir": str(self.original_dir) if self.original_dir else None,
            "sample_count": len(samples),
            "counts": counts,
            "samples": samples,
        }

    def sample_card(self, sample: ReviewSample) -> dict[str, Any]:
        return {
            "sample_id": sample.sample_id,
            "image_name": sample.image_name,
            "status": sample.status,
            "review_state": sample.review_state,
            "candidate_talc_pixels": int(sample.summary.get("candidate_talc_pixels") or 0),
            "final_talc_pixels": int(sample.summary.get("final_talc_pixels") or 0),
            "overlap_pixels": int(sample.summary.get("overlap_pixels") or 0),
            "has_original": sample.original_path is not None,
        }

    def sample_payload(self, sample_id: str) -> dict[str, Any]:
        sample = self.get_sample(sample_id)
        current_path = self.ensure_current_mask(sample)
        # Refresh after first-open state creation so review_state is current.
        sample = self.get_sample(sample_id)
        summary = sample.summary
        paths = summary.get("paths", {})
        current_mask = read_mask(current_path)
        final_mask = read_mask(resolve_path(paths["final_talc_mask"]), current_mask.shape[:2])
        ignore_path = resolve_path(paths["ignore_mask"]) if paths.get("ignore_mask") else None
        ignore_mask = read_mask(ignore_path, current_mask.shape[:2]) if ignore_path and ignore_path.exists() else np.zeros_like(current_mask)
        urls = {
            "original": self.artifact_url(sample.original_path),
            "annotated": self.artifact_url(sample.annotated_path),
            "source_copy": self.artifact_url(paths.get("source_image")),
            "qa_overlay": self.artifact_url(paths.get("qa_overlay")),
            "current_mask": self.artifact_url(current_path),
            "autodetected_mask": self.artifact_url(paths.get("final_talc_mask")),
            "candidate_mask": self.artifact_url(paths.get("candidate_talc_mask")),
            "filled_talc_region": self.artifact_url(paths.get("filled_talc_region")),
            "raw_blue_stroke": self.artifact_url(paths.get("raw_blue_stroke")),
            "closed_blue_stroke": self.artifact_url(paths.get("closed_blue_stroke")),
            "sulfide_mask": self.artifact_url(paths.get("sulfide_mask")),
            "sulfide_overlap": self.artifact_url(paths.get("sulfide_overlap_mask")),
            "ignore_mask": self.artifact_url(paths.get("ignore_mask")),
            "reviewed_talc_mask": self.artifact_url(sample.sample_dir / "reviewed/reviewed_talc_mask.png"),
            "reviewed_overlay": self.artifact_url(sample.sample_dir / "reviewed/reviewed_overlay.png"),
        }
        return {
            "schema_version": "talc-review-web-sample-v0.1",
            "sample": self.sample_card(sample),
            "image": {
                "width": int(summary["width"]),
                "height": int(summary["height"]),
                "name": sample.image_name,
                "annotated_path": str(sample.annotated_path),
                "original_path": str(sample.original_path) if sample.original_path else None,
                "sample_dir": str(sample.sample_dir),
            },
            "metrics": {
                "current_talc_pixels": mask_pixels(current_mask),
                "autodetected_talc_pixels": mask_pixels(final_mask),
                "ignore_pixels": mask_pixels(ignore_mask),
                "candidate_talc_pixels": int(summary.get("candidate_talc_pixels") or 0),
                "overlap_pixels": int(summary.get("overlap_pixels") or 0),
            },
            "urls": urls,
            "editable": sample.original_path is not None,
            "summary": summary,
        }

    def save_current_mask(self, sample_id: str, payload: dict[str, Any], *, reviewed: bool) -> dict[str, Any]:
        sample = self.get_sample(sample_id)
        expected_shape = (int(sample.summary["height"]), int(sample.summary["width"]))
        mask = decode_mask_data_url(payload.get("mask_png", ""), expected_shape)
        current_path = self.current_mask_path(sample)
        write_mask(current_path, mask)
        edits = payload.get("edits")
        if not isinstance(edits, list):
            edits = []
        view_settings = sanitize_view_settings(payload)
        state = {
            "schema_version": "talc-current-mask-state-v0.1",
            "sample_id": sample.sample_id,
            "created_at": self._existing_state_created_at(sample),
            "updated_at": utc_now_iso(),
            "source": "browser_review",
            "current_talc_mask": str(current_path),
            "current_talc_pixels": mask_pixels(mask),
            "edits": edits,
            "view_settings": view_settings,
        }
        self.working_state_path(sample).write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result = {
            "schema_version": "talc-review-web-save-v0.1",
            "sample_id": sample.sample_id,
            "saved_at": state["updated_at"],
            "current_talc_mask": str(current_path),
            "current_talc_pixels": mask_pixels(mask),
            "reviewed": False,
        }
        if reviewed:
            result["reviewed"] = True
            result["review_summary"] = self._write_reviewed_outputs(sample, mask, edits, payload)
        self.refresh_samples()
        result["sample"] = self.sample_card(self.get_sample(sample_id))
        return result

    def _existing_state_created_at(self, sample: ReviewSample) -> str:
        state_path = self.working_state_path(sample)
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                return str(state.get("created_at") or utc_now_iso())
            except json.JSONDecodeError:
                pass
        return utc_now_iso()

    def _write_reviewed_outputs(
        self,
        sample: ReviewSample,
        talc_mask: np.ndarray,
        edits: list[dict[str, Any]],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        paths = sample.summary.get("paths", {})
        ignore_path = resolve_path(paths["ignore_mask"]) if paths.get("ignore_mask") else None
        ignore_mask = read_mask(ignore_path, talc_mask.shape[:2]) if ignore_path and ignore_path.exists() else np.zeros_like(talc_mask)
        image_path = sample.original_path or sample.annotated_path
        image_rgb = read_image_rgb(image_path)
        if image_rgb.shape[:2] != talc_mask.shape[:2]:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"original image dimensions do not match mask for {sample.sample_id}",
            )
        reviewed_dir = sample.sample_dir / "reviewed"
        reviewed_dir.mkdir(parents=True, exist_ok=True)
        reviewed_talc_path = reviewed_dir / "reviewed_talc_mask.png"
        reviewed_ignore_path = reviewed_dir / "reviewed_ignore_mask.png"
        reviewed_overlay_path = reviewed_dir / "reviewed_overlay.png"
        patch_path = reviewed_dir / "review_patch.json"
        summary_path = reviewed_dir / "review_summary.json"
        write_mask(reviewed_talc_path, talc_mask)
        write_mask(reviewed_ignore_path, ignore_mask)
        write_image_rgb(reviewed_overlay_path, make_overlay(image_rgb, talc_mask=talc_mask, ignore_mask=ignore_mask))
        saved_at = utc_now_iso()
        patch = {
            "schema_version": "talc-review-web-patch-v0.1",
            "sample_id": sample.sample_id,
            "image_name": sample.image_name,
            "saved_at": saved_at,
            "reviewer": payload.get("reviewer") or None,
            "notes": payload.get("notes") or None,
            "view_settings": sanitize_view_settings(payload),
            "base_conversion_summary": str(sample.sample_dir / "conversion_summary.json"),
            "annotated_image_path": str(sample.annotated_path),
            "original_image_path": str(sample.original_path) if sample.original_path else None,
            "edits": edits,
        }
        patch_path.write_text(json.dumps(patch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        review_summary = {
            "schema_version": "talc-review-web-summary-v0.1",
            "sample_id": sample.sample_id,
            "image_name": sample.image_name,
            "saved_at": saved_at,
            "reviewed_talc_pixels": mask_pixels(talc_mask),
            "reviewed_ignore_pixels": mask_pixels(ignore_mask),
            "paths": {
                "reviewed_talc_mask": str(reviewed_talc_path),
                "reviewed_ignore_mask": str(reviewed_ignore_path),
                "reviewed_overlay": str(reviewed_overlay_path),
                "review_patch": str(patch_path),
            },
        }
        summary_path.write_text(json.dumps(review_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return review_summary

    def sam2_status(self, *, check_load: bool = False) -> dict[str, Any]:
        return sam2_assist_status(
            model_id=self.sam2_model_id,
            device=self.sam2_device,
            check_load=check_load,
        )

    def run_sam2(self, sample_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        sample = self.get_sample(sample_id)
        image_path = sample.original_path or sample.annotated_path
        if image_path is None:
            raise ApiError(HTTPStatus.BAD_REQUEST, "sample has no image for SAM2")
        prompt_geometry = payload.get("prompt_geometry")
        if not isinstance(prompt_geometry, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "prompt_geometry must be an object")
        out_dir = sample.sample_dir / "sam2_assist"
        try:
            summary = generate_sam2_region_mask(
                image_path=image_path,
                prompt_geometry=prompt_geometry,
                out_dir=out_dir,
                model_id=str(payload.get("model_id") or self.sam2_model_id),
                device=str(payload.get("device") or self.sam2_device or "auto"),
                output_name=f"sam2_{len(list(out_dir.glob('*_mask.png'))) + 1:03d}",
            )
        except Sam2AssistFailure as exc:
            return {
                "schema_version": "talc-review-web-sam2-v0.1",
                "available": False,
                "error": str(exc),
                "status": self.sam2_status(check_load=False),
            }
        mask_path = resolve_path(summary["mask"]["path"])
        return {
            "schema_version": "talc-review-web-sam2-v0.1",
            "available": True,
            "summary": summary,
            "mask_url": self.artifact_url(mask_path),
        }


class TalcReviewHandler(BaseHTTPRequestHandler):
    server: "TalcReviewHTTPServer"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), fmt % args))

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._handle_get()
        except ApiError as exc:
            self.send_json({"error": exc.message}, status=exc.status)
        except Exception as exc:  # noqa: BLE001 - keep server alive and report the fault.
            self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._handle_post()
        except ApiError as exc:
            self.send_json({"error": exc.message}, status=exc.status)
        except Exception as exc:  # noqa: BLE001 - keep server alive and report the fault.
            self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_get(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/":
            self.send_html(render_html_page())
            return
        if path == "/api/manifest":
            self.send_json(self.server.store.manifest_payload())
            return
        if path == "/api/sam2/status":
            query = urllib.parse.parse_qs(parsed.query)
            check_load = query.get("check_load", ["0"])[0] in {"1", "true", "yes"}
            self.send_json(self.server.store.sam2_status(check_load=check_load))
            return
        if path.startswith("/api/samples/"):
            sample_id = urllib.parse.unquote(path.removeprefix("/api/samples/"))
            self.send_json(self.server.store.sample_payload(sample_id))
            return
        if path.startswith("/artifacts/"):
            parts = path.split("/", 3)
            if len(parts) < 3:
                raise ApiError(HTTPStatus.NOT_FOUND, "bad artifact URL")
            self.send_artifact(self.server.store.artifact_path(parts[2]))
            return
        raise ApiError(HTTPStatus.NOT_FOUND, "not found")

    def _handle_post(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        payload = self.read_json_payload()
        if path.startswith("/api/samples/"):
            tail = path.removeprefix("/api/samples/")
            if "/" not in tail:
                raise ApiError(HTTPStatus.NOT_FOUND, "missing sample action")
            sample_id_raw, action = tail.rsplit("/", 1)
            sample_id = urllib.parse.unquote(sample_id_raw)
            if action == "autosave":
                self.send_json(self.server.store.save_current_mask(sample_id, payload, reviewed=False))
                return
            if action == "save":
                self.send_json(self.server.store.save_current_mask(sample_id, payload, reviewed=True))
                return
            if action == "reset":
                self.send_json(self.server.store.reset_current_mask(sample_id))
                return
            if action == "sam2":
                self.send_json(self.server.store.run_sam2(sample_id, payload))
                return
        raise ApiError(HTTPStatus.NOT_FOUND, "not found")

    def read_json_payload(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length") or "0")
        if length > MAX_POST_BYTES:
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
        body = json_response(payload, status=status)
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

    def send_artifact(self, path: Path) -> None:
        content_type, _ = mimetypes.guess_type(str(path))
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class TalcReviewHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], store: TalcReviewStore) -> None:
        super().__init__(server_address, TalcReviewHandler)
        self.store = store


def render_html_page() -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Talc mask review</title>
<style>{CSS}</style>
</head>
<body>
<div id="app" class="app-shell">
  <aside class="queue-pane">
    <div class="pane-title">Talc samples</div>
    <label class="visually-hidden" for="searchBox">Search filename</label>
    <input id="searchBox" class="text-input" placeholder="Search filename" aria-label="Search filename">
    <label class="visually-hidden" for="filterSelect">Sample filter</label>
    <select id="filterSelect" class="select-input" aria-label="Sample filter">
      <option value="all">All samples</option>
      <option value="needs">Needs review</option>
      <option value="overlap">Sulfide overlap</option>
      <option value="ok">Candidate OK</option>
      <option value="reviewed">Reviewed</option>
      <option value="missing">Missing original</option>
    </select>
    <div id="queueStats" class="queue-stats"></div>
    <div id="sampleList" class="sample-list"></div>
  </aside>
  <main class="work-pane">
    <div class="topbar">
      <div class="topbar-title">
        <div id="sampleTitle" class="sample-title">Loading...</div>
        <div id="sampleSubtitle" class="sample-subtitle"></div>
      </div>
      <div class="topbar-controls">
        <div class="toolbar">
          <button type="button" data-tool="brush" class="tool-button active" aria-pressed="true" aria-keyshortcuts="B" title="Brush (B): left mouse draws talc, right mouse erases">Brush</button>
          <button type="button" data-tool="fill" class="tool-button" aria-pressed="false" aria-keyshortcuts="F" title="Fill (F): click an area bounded by blue lines, sulfides, existing talc regions, or image edges">Fill</button>
          <button type="button" data-tool="rectangle" class="tool-button" aria-pressed="false" title="Rectangle: drag or click two corners, then edit handles">Rectangle</button>
          <button type="button" data-tool="polygon" class="tool-button" aria-pressed="false" title="Polygon: place points, close on the first point, right-click a point to remove">Polygon</button>
          <button type="button" data-tool="sam2" class="tool-button" aria-pressed="false" title="SAM2: draw a box or hold still over a point for preview">SAM2</button>
          <span class="toolbar-separator" aria-hidden="true"></span>
          <button type="button" id="undoBtn" class="icon-button" title="Undo last mask edit">Undo</button>
          <span class="toolbar-separator" aria-hidden="true"></span>
          <button type="button" id="zoomInBtn" class="small-button" title="Zoom in">Zoom In</button>
          <button type="button" id="zoomOutBtn" class="small-button" title="Zoom out">Zoom Out</button>
          <button type="button" id="fitBtn" class="small-button" title="Fit image to viewer">Fit</button>
          <span id="zoomValue" class="zoom-value" aria-live="polite">100%</span>
          <span class="toolbar-separator" aria-hidden="true"></span>
          <div class="tool-params" aria-label="Tool parameters">
            <div id="brushParams" class="tool-param-group">
              <label for="brushSize">Brush</label>
              <input id="brushSize" type="range" min="2" max="120" value="28" aria-label="Brush size">
              <span id="brushSizeValue">28 px</span>
            </div>
            <div id="sam2Params" class="tool-param-group hidden">
              <label class="visually-hidden" for="sam2PromptMode">SAM2 prompt mode</label>
              <select id="sam2PromptMode" class="select-input compact" aria-label="SAM2 prompt mode">
                <option value="rectangle_xyxy">SAM2 box</option>
                <option value="point_xy">SAM2 point</option>
              </select>
              <button type="button" id="sam2ApplyBtn" class="small-button" disabled>Apply SAM2</button>
              <button type="button" id="sam2StatusBtn" class="small-button" title="Load/check optional SAM2 assist">Load SAM2</button>
            </div>
          </div>
        </div>
        <div class="review-actions" aria-label="Review actions">
          <button type="button" id="saveBtn" class="primary-button">Save</button>
          <button type="button" id="saveNextBtn" class="primary-button">Save &amp; Next</button>
        </div>
      </div>
    </div>
    <div class="viewer-wrap" id="viewerWrap">
      <canvas id="viewerCanvas"></canvas>
      <div id="emptyState" class="empty-state">Loading sample...</div>
    </div>
    <div id="statusLine" class="status-line">Ready.</div>
  </main>
  <aside class="details-pane">
    <div class="pane-title">Talc mask</div>
    <label class="field-label" for="themeSelect">Theme</label>
    <select id="themeSelect" class="select-input" aria-label="Theme">
      <option value="system">System</option>
      <option value="light">Light</option>
      <option value="dark">Dark</option>
    </select>
    <label class="field-label" for="baseMode">Background</label>
    <select id="baseMode" class="select-input" aria-label="Background image">
      <option value="original">Original photo</option>
      <option value="annotated">MS Paint annotation</option>
      <option value="qa">Converter QA overlay</option>
      <option value="sulfide">Sulfide mask (sulfide/non-sulfide mask segmentation)</option>
      <option value="mask">Mask-only background</option>
    </select>
    <div class="brightness-filter">
      <div class="range-header">
        <label class="field-label inline" for="brightnessThreshold">Dark pixel preview threshold</label>
        <span id="brightnessThresholdValue" class="range-value">255 (off)</span>
      </div>
      <input id="brightnessThreshold" type="range" min="0" max="255" value="255" aria-label="Dark pixel preview brightness threshold">
      <div class="range-actions">
        <button type="button" id="brightnessThreshold90Btn" class="small-button">90</button>
        <button type="button" id="brightnessThresholdOffBtn" class="small-button">Off</button>
      </div>
      <div class="filter-hint">Luma = 0.299 R + 0.587 G + 0.114 B. Pixels brighter than the threshold are painted white; darker pixels stay visible.</div>
    </div>
    <div id="assetWarnings" class="asset-warnings hidden" role="status" aria-live="polite"></div>
    <div class="layers">
      <label><input type="checkbox" id="layerCurrent" checked> Editable talc mask overlay</label>
      <label><input type="checkbox" id="layerAuto"> Autodetected mask</label>
      <label><input type="checkbox" id="layerLines"> Original blue lines</label>
      <label><input type="checkbox" id="layerOverlap" checked> Sulfide overlap</label>
      <label><input type="checkbox" id="layerIgnore"> Ignore/uncertain</label>
    </div>
    <div class="guard-controls">
      <label><input type="checkbox" id="protectSulfides" checked> Protect sulfides while drawing</label>
      <button type="button" id="subtractSulfidesBtn" class="small-button full-width">Subtract sulfides from mask</button>
    </div>
    <div id="metricsBox" class="metrics"></div>
    <label class="field-label" for="reviewerInput">Reviewer</label>
    <input id="reviewerInput" class="text-input" placeholder="optional">
    <label class="field-label" for="notesInput">Notes</label>
    <textarea id="notesInput" class="notes-input" rows="4" placeholder="optional"></textarea>
    <button type="button" id="resetBtn" class="danger-button">Reset to autodetected</button>
    <details class="advanced-box">
      <summary>Interaction help</summary>
      <ul>
        <li>Brush: left mouse adds talc, right mouse erases.</li>
        <li>Fill: click an empty area bounded by blue lines, sulfides, existing talc regions, or the image edge.</li>
        <li>Polygon: click to place points, click the first point to close, drag points/edges to edit, right-click a polygon point to remove it, and right-click elsewhere to cancel the current polygon.</li>
        <li>Rectangle: drag a box or click one corner then click the opposite corner; right-click cancels the current rectangle. Completed rectangles can be resized by corners or edges.</li>
        <li>Press Delete to remove the selected completed polygon or rectangle.</li>
        <li>Shapes stay editable until another image is opened or the sample is saved.</li>
        <li>SAM2 point: hover without moving to preview, then press Apply SAM2. SAM2 box applies after drawing the box.</li>
      </ul>
    </details>
  </aside>
</div>
<script>{JS}</script>
</body>
</html>"""


CSS = r"""
:root {
  color-scheme: light;
  --bg: #f4f6f8;
  --panel: #ffffff;
  --line: #d7dde4;
  --text: #18212f;
  --muted: #617083;
  --accent: #1772d0;
  --accent-weak: #dcedff;
  --danger: #b42318;
  --status-error: #b42318;
  --control-bg: #ffffff;
  --control-band: #fbfcfe;
  --viewer-bg: #dfe5ec;
  --canvas-bg: #111827;
  --mask-only-bg: #0f172a;
  --empty-bg: rgba(255,255,255,0.76);
  --hover-line: #9db8d7;
  --tag-bg: #eef2f6;
  --tag-text: #334155;
  --tag-warn-bg: #fff3cd;
  --tag-warn-text: #775a00;
  --tag-ok-bg: #e8f5ee;
  --tag-ok-text: #146c43;
  --tag-reviewed-bg: #e7f0ff;
  --tag-reviewed-text: #174ea6;
  --danger-border: #f1b9b4;
  --mask: #05a3d8;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #101418;
  --panel: #171d24;
  --line: #303947;
  --text: #e8edf3;
  --muted: #a6b1bf;
  --accent: #42a5f5;
  --accent-weak: #17334d;
  --danger: #ff8a80;
  --status-error: #ff8a80;
  --control-bg: #111820;
  --control-band: #141b22;
  --viewer-bg: #0b0f14;
  --canvas-bg: #05080c;
  --mask-only-bg: #05080c;
  --empty-bg: rgba(17,24,32,0.82);
  --hover-line: #5f7794;
  --tag-bg: #27313d;
  --tag-text: #d4dce7;
  --tag-warn-bg: #4a3710;
  --tag-warn-text: #ffd166;
  --tag-ok-bg: #163625;
  --tag-ok-text: #8bd6a7;
  --tag-reviewed-bg: #162f52;
  --tag-reviewed-text: #9bc4ff;
  --danger-border: #7f342e;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #101418;
    --panel: #171d24;
    --line: #303947;
    --text: #e8edf3;
    --muted: #a6b1bf;
    --accent: #42a5f5;
    --accent-weak: #17334d;
    --danger: #ff8a80;
    --status-error: #ff8a80;
    --control-bg: #111820;
    --control-band: #141b22;
    --viewer-bg: #0b0f14;
    --canvas-bg: #05080c;
    --mask-only-bg: #05080c;
    --empty-bg: rgba(17,24,32,0.82);
    --hover-line: #5f7794;
    --tag-bg: #27313d;
    --tag-text: #d4dce7;
    --tag-warn-bg: #4a3710;
    --tag-warn-text: #ffd166;
    --tag-ok-bg: #163625;
    --tag-ok-text: #8bd6a7;
    --tag-reviewed-bg: #162f52;
    --tag-reviewed-text: #9bc4ff;
    --danger-border: #7f342e;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); }
button, input, select, textarea { font: inherit; }
.visually-hidden { position: absolute !important; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
.app-shell { display: grid; grid-template-columns: 280px minmax(0, 1fr) 300px; height: 100vh; overflow: hidden; }
.queue-pane, .details-pane { background: var(--panel); border-right: 1px solid var(--line); padding: 14px; overflow: auto; }
.details-pane { border-right: 0; border-left: 1px solid var(--line); }
.pane-title { font-weight: 700; font-size: 15px; margin-bottom: 10px; }
.text-input, .select-input, .notes-input { width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; background: var(--control-bg); color: var(--text); margin-bottom: 8px; }
.text-input::placeholder, .notes-input::placeholder { color: var(--muted); opacity: 0.82; }
.select-input.compact { width: auto; min-width: 132px; margin: 0; padding: 6px 8px; }
.queue-stats { font-size: 12px; color: var(--muted); margin: 4px 0 10px; line-height: 1.4; }
.sample-list { display: flex; flex-direction: column; gap: 6px; }
.sample-card { border: 1px solid var(--line); background: var(--control-bg); color: var(--text); border-radius: 8px; padding: 8px; cursor: pointer; text-align: left; }
.sample-card.active { border-color: var(--accent); background: var(--accent-weak); }
.sample-card:hover { border-color: var(--hover-line); }
.sample-name { font-size: 13px; font-weight: 650; overflow-wrap: anywhere; }
.sample-tags { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
.tag { display: inline-flex; align-items: center; border-radius: 999px; background: var(--tag-bg); color: var(--tag-text); padding: 2px 7px; font-size: 11px; }
.tag.warn { background: var(--tag-warn-bg); color: var(--tag-warn-text); }
.tag.ok { background: var(--tag-ok-bg); color: var(--tag-ok-text); }
.tag.reviewed { background: var(--tag-reviewed-bg); color: var(--tag-reviewed-text); }
.work-pane { min-width: 0; display: flex; flex-direction: column; overflow: hidden; }
.topbar { min-height: 56px; background: var(--panel); border-bottom: 1px solid var(--line); padding: 10px 12px; display: grid; grid-template-columns: minmax(170px, 250px) minmax(260px, 1fr) auto; align-items: start; gap: 12px; }
.topbar-title { min-width: 0; }
.sample-title { font-size: 17px; font-weight: 750; }
.sample-subtitle { font-size: 12px; color: var(--muted); margin-top: 2px; }
.topbar-controls { display: contents; }
.toolbar { grid-column: 2; display: flex; align-items: center; gap: 6px; flex-wrap: wrap; justify-content: flex-start; min-width: 0; }
.review-actions { grid-column: 3; display: flex; align-items: center; justify-content: flex-end; gap: 8px; min-width: max-content; }
.tool-button, .small-button, .icon-button, .primary-button, .danger-button { border: 1px solid var(--line); border-radius: 6px; background: var(--control-bg); color: var(--text); padding: 7px 10px; cursor: pointer; }
.tool-button:disabled, .small-button:disabled, .icon-button:disabled, .primary-button:disabled, .danger-button:disabled { opacity: 0.48; cursor: not-allowed; }
.tool-button.active { background: var(--accent); border-color: var(--accent); color: #ffffff; }
.tool-button[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: #ffffff; }
.icon-button { min-width: 54px; }
.primary-button, .danger-button { font-weight: 700; }
.details-pane .primary-button, .details-pane .danger-button { width: 100%; margin-top: 8px; }
.review-actions .primary-button { width: auto; margin-top: 0; min-width: 76px; white-space: nowrap; }
.primary-button { background: var(--accent); border-color: var(--accent); color: #ffffff; }
.danger-button { background: var(--control-bg); border-color: var(--danger-border); color: var(--danger); }
.toolbar-separator { align-self: stretch; width: 1px; min-height: 30px; background: var(--line); margin: 0 4px; }
.tool-params { display: flex; align-items: center; gap: 8px; min-height: 34px; }
.tool-param-group { display: flex; align-items: center; gap: 8px; }
.tool-param-group.hidden { display: none; }
.tool-param-group label { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); font-size: 13px; white-space: nowrap; }
.tool-param-group input[type="range"] { max-width: 130px; }
.zoom-value { min-width: 48px; color: var(--muted); font-size: 12px; font-weight: 700; text-align: center; align-self: center; }
.viewer-wrap { position: relative; flex: 1; overflow: auto; padding: 14px; background: var(--viewer-bg); }
#viewerCanvas { display: block; background: var(--canvas-bg); image-rendering: auto; box-shadow: 0 0 0 1px rgba(0,0,0,0.22); }
.empty-state { position: absolute; inset: 14px; display: grid; place-items: center; background: var(--empty-bg); color: var(--muted); font-weight: 650; }
.empty-state.hidden { display: none; }
.status-line { min-height: 32px; padding: 8px 12px; font-size: 13px; color: var(--muted); background: var(--panel); border-top: 1px solid var(--line); }
.field-label { display: block; color: var(--muted); font-size: 12px; font-weight: 700; margin: 12px 0 4px; }
.field-label.inline { margin: 0; }
.brightness-filter { border: 1px solid var(--line); border-radius: 8px; padding: 10px; margin-top: 10px; display: grid; gap: 8px; }
.brightness-filter input[type="range"] { width: 100%; }
.range-header, .range-actions { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.range-actions { justify-content: flex-start; }
.range-value { color: var(--muted); font-size: 12px; font-weight: 700; white-space: nowrap; }
.filter-hint { color: var(--muted); font-size: 12px; line-height: 1.35; }
.layers { display: grid; gap: 7px; margin-top: 10px; font-size: 13px; }
.guard-controls { border: 1px solid var(--line); border-radius: 8px; display: grid; gap: 8px; margin-top: 12px; padding: 10px; font-size: 13px; }
.guard-controls label { display: flex; align-items: center; gap: 7px; }
.small-button.full-width { width: 100%; }
.asset-warnings { border: 1px solid var(--tag-warn-bg); background: var(--tag-warn-bg); color: var(--tag-warn-text); border-radius: 8px; padding: 8px 10px; margin-top: 10px; font-size: 12px; line-height: 1.35; }
.asset-warnings.hidden { display: none; }
.asset-warnings ul { padding-left: 18px; margin: 6px 0 0; }
.metrics { border: 1px solid var(--line); border-radius: 8px; margin-top: 12px; overflow: hidden; }
.metric-row { display: flex; justify-content: space-between; gap: 8px; padding: 8px 10px; border-bottom: 1px solid var(--line); font-size: 13px; }
.metric-row:last-child { border-bottom: 0; }
.metric-row span:first-child { color: var(--muted); }
.advanced-box { margin-top: 12px; color: var(--muted); font-size: 12px; line-height: 1.45; }
.advanced-box ul { margin: 8px 0 0; padding-left: 18px; }
@media (max-width: 1320px) {
  .topbar { grid-template-columns: minmax(170px, 220px) minmax(240px, 1fr); }
  .review-actions { grid-column: 2; justify-self: end; }
}
@media (max-width: 1100px) {
  .app-shell { grid-template-columns: 240px minmax(0, 1fr); }
  .details-pane { grid-column: 1 / -1; height: 260px; border-left: 0; border-top: 1px solid var(--line); }
  .topbar { grid-template-columns: minmax(160px, 1fr); }
  .topbar-title, .toolbar, .review-actions { grid-column: 1; }
  .review-actions { justify-self: start; }
}
@media (max-width: 760px) {
  .app-shell { grid-template-columns: 1fr; grid-template-rows: minmax(150px, 26vh) minmax(420px, 1fr) minmax(220px, 30vh); overflow: auto; }
  .queue-pane { border-right: 0; border-bottom: 1px solid var(--line); }
  .details-pane { grid-column: 1; height: auto; min-height: 220px; }
  .work-pane { min-height: 420px; }
  .viewer-wrap { min-height: 300px; }
  .toolbar { gap: 5px; }
  .tool-button, .small-button, .icon-button, .primary-button, .danger-button { padding: 6px 8px; }
}
"""


JS = r"""
const MAX_SAM2_REGION_FRACTION = 0.50;
const MIN_ZOOM = 0.10;
const MAX_ZOOM = 4.00;
const ZOOM_STEP = 1.15;
const SAM2_POINT_HOVER_PREVIEW_DELAY_MS = 2000;
const BRIGHTNESS_THRESHOLD_STORAGE_KEY = 'talcBrightnessThreshold';
const BRIGHTNESS_THRESHOLD_FORMULA = 'luma = 0.299*R + 0.587*G + 0.114*B; luma <= threshold keeps the pixel, luma > threshold paints it white';

const state = {
  manifest: null,
  samples: [],
  sample: null,
  sampleId: null,
  tool: 'brush',
  imageW: 1,
  imageH: 1,
  zoom: 1,
  viewPan: {
    active: false,
    pointerId: null,
    startClientX: 0,
    startClientY: 0,
    scrollLeft: 0,
    scrollTop: 0
  },
  dirty: false,
  saveState: 'saved',
  lastSavedAt: null,
  assetErrors: [],
  images: {},
  staticTints: {},
  sulfideGuardLoaded: false,
  undoStack: [],
  edits: [],
  shapes: [],
  nextShapeId: 1,
  activeShapeId: null,
  shapeDrag: null,
  polygon: { points: [], dragIndex: null },
  rect: { active: false, x1: 0, y1: 0, x2: 0, y2: 0, handle: null, lastPoint: null, startPoint: null, dragMoved: false },
  drawing: false,
  lastPoint: null,
  hoverPoint: null,
  activeStrokeMode: null,
  activePointerButton: 0,
  activeEditBaseline: null,
  activeBaseEditBaseline: null,
  samBox: null,
  brightnessPreview: {
    source: null,
    threshold: null
  },
  fillBoundaryLoaded: false,
  sam2Preview: {
    timer: null,
    requestId: 0,
    pendingKey: null,
    loadingKey: null,
    promptKey: null,
    prompt: null,
    img: null,
    tint: null,
    result: null,
    stats: null
  }
};

const viewer = document.getElementById('viewerCanvas');
const ctx = viewer.getContext('2d', { willReadFrequently: true });
const maskCanvas = document.createElement('canvas');
const maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true });
const baseMaskCanvas = document.createElement('canvas');
const baseMaskCtx = baseMaskCanvas.getContext('2d', { willReadFrequently: true });
const sulfideGuardCanvas = document.createElement('canvas');
const sulfideGuardCtx = sulfideGuardCanvas.getContext('2d', { willReadFrequently: true });
const currentTintCanvas = document.createElement('canvas');
const currentTintCtx = currentTintCanvas.getContext('2d', { willReadFrequently: true });
const brightnessSourceCanvas = document.createElement('canvas');
const brightnessSourceCtx = brightnessSourceCanvas.getContext('2d', { willReadFrequently: true });
const brightnessPreviewCanvas = document.createElement('canvas');
const brightnessPreviewCtx = brightnessPreviewCanvas.getContext('2d', { willReadFrequently: true });
const fillBoundaryCanvas = document.createElement('canvas');
const fillBoundaryCtx = fillBoundaryCanvas.getContext('2d', { willReadFrequently: true });
const emptyState = document.getElementById('emptyState');

const els = {
  sampleList: document.getElementById('sampleList'),
  searchBox: document.getElementById('searchBox'),
  filterSelect: document.getElementById('filterSelect'),
  queueStats: document.getElementById('queueStats'),
  sampleTitle: document.getElementById('sampleTitle'),
  sampleSubtitle: document.getElementById('sampleSubtitle'),
  statusLine: document.getElementById('statusLine'),
  brushSize: document.getElementById('brushSize'),
  brushSizeValue: document.getElementById('brushSizeValue'),
  brushParams: document.getElementById('brushParams'),
  sam2Params: document.getElementById('sam2Params'),
  zoomInBtn: document.getElementById('zoomInBtn'),
  zoomOutBtn: document.getElementById('zoomOutBtn'),
  fitBtn: document.getElementById('fitBtn'),
  zoomValue: document.getElementById('zoomValue'),
  themeSelect: document.getElementById('themeSelect'),
  baseMode: document.getElementById('baseMode'),
  brightnessThreshold: document.getElementById('brightnessThreshold'),
  brightnessThresholdValue: document.getElementById('brightnessThresholdValue'),
  brightnessThreshold90Btn: document.getElementById('brightnessThreshold90Btn'),
  brightnessThresholdOffBtn: document.getElementById('brightnessThresholdOffBtn'),
  assetWarnings: document.getElementById('assetWarnings'),
  metricsBox: document.getElementById('metricsBox'),
  reviewerInput: document.getElementById('reviewerInput'),
  notesInput: document.getElementById('notesInput'),
  undoBtn: document.getElementById('undoBtn'),
  saveBtn: document.getElementById('saveBtn'),
  saveNextBtn: document.getElementById('saveNextBtn'),
  resetBtn: document.getElementById('resetBtn'),
  protectSulfides: document.getElementById('protectSulfides'),
  subtractSulfidesBtn: document.getElementById('subtractSulfidesBtn'),
  sam2PromptMode: document.getElementById('sam2PromptMode'),
  sam2ApplyBtn: document.getElementById('sam2ApplyBtn'),
  sam2StatusBtn: document.getElementById('sam2StatusBtn'),
  layers: {
    current: document.getElementById('layerCurrent'),
    auto: document.getElementById('layerAuto'),
    lines: document.getElementById('layerLines'),
    overlap: document.getElementById('layerOverlap'),
    ignore: document.getElementById('layerIgnore')
  }
};

const THEME_STORAGE_KEY = 'talcReviewTheme';

const STATUS_LABELS = {
  candidate_ok: 'Candidate OK',
  needs_manual_review: 'Needs manual review',
  sulfide_overlap_review_required: 'Sulfide overlap',
  missing_original: 'Missing original',
  unknown: 'Unknown status'
};

const REVIEW_STATE_LABELS = {
  reviewed: 'Reviewed',
  working: 'Working draft',
  not_opened: 'Not opened'
};

const SAVE_STATE_LABELS = {
  saved: 'Working mask saved',
  saving: 'Saving working mask...',
  unsaved: 'Unsaved local changes',
  error: 'Autosave failed',
  reviewed: 'Reviewed mask saved'
};

function cssVar(name, fallback) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

function applyTheme(theme, persist = true) {
  const normalized = ['system', 'light', 'dark'].includes(theme) ? theme : 'system';
  if (normalized === 'system') {
    document.documentElement.removeAttribute('data-theme');
  } else {
    document.documentElement.dataset.theme = normalized;
  }
  if (els.themeSelect) els.themeSelect.value = normalized;
  if (persist) localStorage.setItem(THEME_STORAGE_KEY, normalized);
}

function setStatus(message, isError = false) {
  els.statusLine.textContent = message;
  els.statusLine.style.color = isError ? 'var(--status-error)' : 'var(--muted)';
}

function statusLabel(status) {
  return STATUS_LABELS[status] || humanizeEnum(status);
}

function reviewStateLabel(reviewState) {
  return REVIEW_STATE_LABELS[reviewState] || humanizeEnum(reviewState);
}

function humanizeEnum(value) {
  return String(value || 'unknown')
    .replace(/_/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function sampleStatusKind(sample) {
  if (sample.status === 'missing_original' || sample.status === 'needs_manual_review' || sample.status === 'sulfide_overlap_review_required') return 'warn';
  return 'ok';
}

function reviewStateKind(reviewState) {
  if (reviewState === 'reviewed') return 'reviewed';
  if (reviewState === 'working') return 'ok';
  return '';
}

async function apiGet(url) {
  const response = await fetch(url, { cache: 'no-store' });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

async function apiPost(url, payload) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload || {})
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function loadImage(url, label = null) {
  return new Promise((resolve) => {
    if (!url) {
      resolve(null);
      return;
    }
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => {
      if (label) state.assetErrors.push(`${label} could not be loaded`);
      resolve(null);
    };
    img.src = url;
  });
}

function imagePointFromEvent(event) {
  const rect = viewer.getBoundingClientRect();
  const x = (event.clientX - rect.left) * (viewer.width / rect.width);
  const y = (event.clientY - rect.top) * (viewer.height / rect.height);
  return {
    x: Math.max(0, Math.min(state.imageW - 1, x)),
    y: Math.max(0, Math.min(state.imageH - 1, y))
  };
}

function applyZoom() {
  viewer.style.width = `${Math.max(120, Math.round(state.imageW * state.zoom))}px`;
  viewer.style.height = `${Math.max(120, Math.round(state.imageH * state.zoom))}px`;
  if (els.zoomValue) els.zoomValue.textContent = `${Math.round(state.zoom * 100)}%`;
}

function clampZoom(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) return 1;
  return Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, numeric));
}

function setZoom(value, anchor = null) {
  const wrap = document.getElementById('viewerWrap');
  let anchorImagePoint = null;
  let anchorOffset = null;
  if (anchor && viewer.clientWidth > 0 && viewer.clientHeight > 0) {
    const rect = viewer.getBoundingClientRect();
    const wrapRect = wrap.getBoundingClientRect();
    anchorImagePoint = {
      x: (anchor.clientX - rect.left) * (viewer.width / rect.width),
      y: (anchor.clientY - rect.top) * (viewer.height / rect.height)
    };
    anchorOffset = {
      x: anchor.clientX - wrapRect.left,
      y: anchor.clientY - wrapRect.top
    };
  }
  state.zoom = clampZoom(value);
  applyZoom();
  if (anchorImagePoint && anchorOffset) {
    wrap.scrollLeft = Math.max(0, anchorImagePoint.x * state.zoom - anchorOffset.x);
    wrap.scrollTop = Math.max(0, anchorImagePoint.y * state.zoom - anchorOffset.y);
  }
  draw();
}

function zoomBy(factor, anchor = null) {
  setZoom(state.zoom * factor, anchor);
}

function startViewPan(event) {
  const wrap = document.getElementById('viewerWrap');
  event.preventDefault();
  clearSam2Preview({ redraw: false });
  state.viewPan = {
    active: true,
    pointerId: event.pointerId,
    startClientX: event.clientX,
    startClientY: event.clientY,
    scrollLeft: wrap.scrollLeft,
    scrollTop: wrap.scrollTop
  };
  viewer.setPointerCapture(event.pointerId);
  viewer.style.cursor = 'grabbing';
  setStatus('Pan view: drag while holding the mouse wheel.');
}

function updateViewPan(event) {
  if (!state.viewPan.active || state.viewPan.pointerId !== event.pointerId) return false;
  event.preventDefault();
  const wrap = document.getElementById('viewerWrap');
  const dx = event.clientX - state.viewPan.startClientX;
  const dy = event.clientY - state.viewPan.startClientY;
  wrap.scrollLeft = state.viewPan.scrollLeft - dx;
  wrap.scrollTop = state.viewPan.scrollTop - dy;
  return true;
}

function finishViewPan(event = null) {
  if (!state.viewPan.active) return false;
  if (event && state.viewPan.pointerId !== event.pointerId) return false;
  try {
    if (event && viewer.hasPointerCapture(event.pointerId)) viewer.releasePointerCapture(event.pointerId);
  } catch (err) {
    // Pointer capture can already be released by the browser on cancel.
  }
  state.viewPan = {
    active: false,
    pointerId: null,
    startClientX: 0,
    startClientY: 0,
    scrollLeft: 0,
    scrollTop: 0
  };
  updateViewerCursor(state.hoverPoint);
  return true;
}

function fitToViewer() {
  const wrap = document.getElementById('viewerWrap');
  const maxW = Math.max(240, wrap.clientWidth - 36);
  const maxH = Math.max(180, wrap.clientHeight - 36);
  setZoom(Math.min(maxW / state.imageW, maxH / state.imageH, 1));
}

function updateToolParams() {
  els.brushParams.classList.toggle('hidden', state.tool !== 'brush');
  els.sam2Params.classList.toggle('hidden', state.tool !== 'sam2');
  updateSam2ApplyButton();
}

function selectTool(tool, options = {}) {
  const button = Array.from(document.querySelectorAll('.tool-button')).find((candidate) => candidate.dataset.tool === tool);
  if (!button) return false;
  document.querySelectorAll('.tool-button').forEach((other) => {
    other.classList.remove('active');
    other.setAttribute('aria-pressed', 'false');
  });
  button.classList.add('active');
  button.setAttribute('aria-pressed', 'true');
  state.tool = tool;
  if (state.tool !== 'brush' && state.tool !== 'sam2') state.hoverPoint = null;
  clearSam2Preview({ redraw: false });
  updateToolParams();
  updateViewerCursor();
  draw();
  if (options.status !== false) {
    const suffix = options.shortcut ? ` (${options.shortcut})` : '';
    setStatus(`Tool: ${button.textContent}${suffix}.`);
  }
  return true;
}

function sam2PointModeActive() {
  return state.tool === 'sam2' && els.sam2PromptMode.value === 'point_xy';
}

function sam2PromptKey(promptGeometry) {
  if (!promptGeometry) return '';
  return `${promptGeometry.type}:${promptGeometry.x}:${promptGeometry.y}`;
}

function clearSam2Preview(options = {}) {
  const redraw = options.redraw !== false;
  if (state.sam2Preview.timer) {
    clearTimeout(state.sam2Preview.timer);
    state.sam2Preview.timer = null;
  }
  state.sam2Preview.requestId += 1;
  state.sam2Preview.pendingKey = null;
  state.sam2Preview.loadingKey = null;
  state.sam2Preview.promptKey = null;
  state.sam2Preview.prompt = null;
  state.sam2Preview.img = null;
  state.sam2Preview.tint = null;
  state.sam2Preview.result = null;
  state.sam2Preview.stats = null;
  updateSam2ApplyButton();
  if (redraw) draw();
}

function updateSam2ApplyButton() {
  if (!els.sam2ApplyBtn) return;
  if (state.tool !== 'sam2') {
    els.sam2ApplyBtn.disabled = true;
    els.sam2ApplyBtn.textContent = 'Apply SAM2';
    els.sam2ApplyBtn.title = 'Switch to SAM2 point mode to use hover preview.';
    return;
  }
  if (els.sam2PromptMode.value !== 'point_xy') {
    els.sam2ApplyBtn.disabled = true;
    els.sam2ApplyBtn.textContent = 'Apply SAM2';
    els.sam2ApplyBtn.title = 'Draw a SAM2 box on the canvas to apply box prompts.';
    return;
  }
  if (state.sam2Preview.loadingKey) {
    els.sam2ApplyBtn.disabled = true;
    els.sam2ApplyBtn.textContent = 'Previewing...';
    els.sam2ApplyBtn.title = 'SAM2 point preview is running.';
    return;
  }
  if (state.sam2Preview.img) {
    els.sam2ApplyBtn.disabled = false;
    els.sam2ApplyBtn.textContent = 'Apply SAM2';
    els.sam2ApplyBtn.title = 'Apply the visible SAM2 point preview to the talc mask.';
    return;
  }
  if (state.hoverPoint) {
    els.sam2ApplyBtn.disabled = false;
    els.sam2ApplyBtn.textContent = 'Run & Apply';
    els.sam2ApplyBtn.title = 'Run SAM2 at the current hover point and apply the result.';
    return;
  }
  els.sam2ApplyBtn.disabled = true;
  els.sam2ApplyBtn.textContent = 'Apply SAM2';
  els.sam2ApplyBtn.title = 'Hover over the image to preview a SAM2 point prompt.';
}

function formatInt(value) {
  return Number(value || 0).toLocaleString('en-US');
}

function filteredSamples() {
  const query = els.searchBox.value.trim().toLowerCase();
  const filter = els.filterSelect.value;
  return state.samples.filter((sample) => {
    if (query && !sample.image_name.toLowerCase().includes(query) && !sample.sample_id.toLowerCase().includes(query)) return false;
    if (filter === 'needs') return sample.status === 'needs_manual_review';
    if (filter === 'overlap') return sample.overlap_pixels > 0 || sample.status === 'sulfide_overlap_review_required';
    if (filter === 'ok') return sample.status === 'candidate_ok';
    if (filter === 'reviewed') return sample.review_state === 'reviewed';
    if (filter === 'missing') return sample.status === 'missing_original';
    return true;
  });
}

function nextVisibleSampleId(currentId) {
  const visible = filteredSamples();
  if (visible.length === 0) return null;
  const index = visible.findIndex((sample) => sample.sample_id === currentId);
  if (index < 0) return visible[0].sample_id;
  if (visible.length === 1) return null;
  return visible[(index + 1) % visible.length].sample_id;
}

function hasDraftGeometry() {
  return state.polygon.points.length > 0 || state.rect.active || state.drawing || Boolean(state.shapeDrag);
}

function canLeaveCurrentSample(targetSampleId) {
  if (!state.sampleId || targetSampleId === state.sampleId) return true;
  if (state.saveState === 'saving') {
    setStatus('Please wait: the working mask is still autosaving.', true);
    return false;
  }
  if (!state.dirty && !hasDraftGeometry()) return true;
  const reason = state.dirty
    ? 'The current mask has local changes that were not autosaved.'
    : 'The current image has an unfinished polygon/rectangle draft.';
  return window.confirm(`${reason}\n\nSwitch to another sample and discard only the unfinished local state?`);
}

function renderQueue() {
  const visible = filteredSamples();
  const needsCount = state.samples.filter((sample) => sample.status === 'needs_manual_review' || sample.status === 'sulfide_overlap_review_required').length;
  const reviewedCount = state.samples.filter((sample) => sample.review_state === 'reviewed').length;
  els.queueStats.textContent = `${visible.length} shown / ${state.samples.length} total · ${needsCount} need review · ${reviewedCount} reviewed`;
  els.sampleList.innerHTML = '';
  for (const sample of visible) {
    const button = document.createElement('button');
    button.className = 'sample-card' + (sample.sample_id === state.sampleId ? ' active' : '');
    button.innerHTML = `
      <div class="sample-name">${escapeHtml(sample.image_name)}</div>
      <div class="sample-tags">
        ${tagHtml(statusLabel(sample.status), sampleStatusKind(sample))}
        ${tagHtml(reviewStateLabel(sample.review_state), reviewStateKind(sample.review_state))}
        ${sample.overlap_pixels > 0 ? tagHtml('Has sulfide overlap', 'warn') : ''}
      </div>`;
    button.addEventListener('click', () => loadSample(sample.sample_id));
    els.sampleList.appendChild(button);
  }
}

function tagHtml(text, kind) {
  return `<span class="tag ${kind || ''}">${escapeHtml(text)}</span>`;
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
}

function formatPct(value, denominator = state.imageW * state.imageH) {
  if (!Number.isFinite(value) || !Number.isFinite(denominator) || denominator <= 0) return '0.00%';
  return `${((value / denominator) * 100).toFixed(2)}%`;
}

function pxWithPct(value, denominator = state.imageW * state.imageH) {
  return `${formatInt(value)} (${formatPct(value, denominator)})`;
}

function renderAssetWarnings() {
  if (!els.assetWarnings) return;
  if (!state.assetErrors.length) {
    els.assetWarnings.classList.add('hidden');
    els.assetWarnings.innerHTML = '';
    return;
  }
  const rows = [...new Set(state.assetErrors)].map((item) => `<li>${escapeHtml(item)}</li>`).join('');
  els.assetWarnings.classList.remove('hidden');
  els.assetWarnings.innerHTML = `<strong>Layer warning</strong><ul>${rows}</ul>`;
}

function markLocalDirty() {
  state.dirty = true;
  state.saveState = 'unsaved';
  updateMetrics();
}

function makeMaskCanvasFromImage(img) {
  if (!img) return null;
  const c = document.createElement('canvas');
  c.width = state.imageW;
  c.height = state.imageH;
  const cctx = c.getContext('2d', { willReadFrequently: true });
  cctx.drawImage(img, 0, 0, state.imageW, state.imageH);
  return c;
}

function buildTintFromImage(img, rgba) {
  const source = makeMaskCanvasFromImage(img);
  if (!source) return null;
  return buildTintFromCanvas(source, rgba);
}

function buildTintFromCanvas(sourceCanvas, rgba) {
  const out = document.createElement('canvas');
  out.width = state.imageW;
  out.height = state.imageH;
  const outCtx = out.getContext('2d', { willReadFrequently: true });
  const srcCtx = sourceCanvas.getContext('2d', { willReadFrequently: true });
  const src = srcCtx.getImageData(0, 0, state.imageW, state.imageH).data;
  const imageData = outCtx.createImageData(state.imageW, state.imageH);
  const dst = imageData.data;
  for (let i = 0; i < src.length; i += 4) {
    if (src[i] > 0 || src[i + 1] > 0 || src[i + 2] > 0) {
      dst[i] = rgba[0];
      dst[i + 1] = rgba[1];
      dst[i + 2] = rgba[2];
      dst[i + 3] = rgba[3];
    }
  }
  outCtx.putImageData(imageData, 0, 0);
  return out;
}

function refreshCurrentTint() {
  currentTintCanvas.width = state.imageW;
  currentTintCanvas.height = state.imageH;
  const tint = buildTintFromCanvas(maskCanvas, [0, 163, 216, 112]);
  currentTintCtx.clearRect(0, 0, state.imageW, state.imageH);
  if (tint) currentTintCtx.drawImage(tint, 0, 0);
}

function countCurrentMaskPixels() {
  const data = maskCtx.getImageData(0, 0, state.imageW, state.imageH).data;
  let count = 0;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i] > 0 || data[i + 1] > 0 || data[i + 2] > 0) count += 1;
  }
  return count;
}

function captureMaskData() {
  return maskCtx.getImageData(0, 0, state.imageW, state.imageH);
}

function captureBaseMaskData() {
  return baseMaskCtx.getImageData(0, 0, state.imageW, state.imageH);
}

function cloneShapes() {
  return state.shapes.map((shape) => {
    if (shape.type === 'polygon') {
      return { id: shape.id, type: 'polygon', points: shape.points.map((p) => ({ x: p.x, y: p.y })) };
    }
    return { id: shape.id, type: 'rectangle', x1: shape.x1, y1: shape.y1, x2: shape.x2, y2: shape.y2 };
  });
}

function restoreShapes(shapes) {
  state.shapes = (shapes || []).map((shape) => {
    if (shape.type === 'polygon') {
      return { id: shape.id, type: 'polygon', points: shape.points.map((p) => ({ x: p.x, y: p.y })) };
    }
    return { id: shape.id, type: 'rectangle', x1: shape.x1, y1: shape.y1, x2: shape.x2, y2: shape.y2 };
  });
  state.nextShapeId = Math.max(1, ...state.shapes.map((shape) => shape.id + 1));
}

function shapeById(shapeId) {
  return state.shapes.find((shape) => shape.id === shapeId) || null;
}

function hasSulfideGuard() {
  return state.sulfideGuardLoaded && sulfideGuardCanvas.width === state.imageW && sulfideGuardCanvas.height === state.imageH;
}

function countCurrentSulfideOverlapPixels() {
  if (!hasSulfideGuard()) return 0;
  const mask = maskCtx.getImageData(0, 0, state.imageW, state.imageH).data;
  const guard = sulfideGuardCtx.getImageData(0, 0, state.imageW, state.imageH).data;
  let count = 0;
  for (let i = 0; i < mask.length; i += 4) {
    const maskActive = mask[i] > 0 || mask[i + 1] > 0 || mask[i + 2] > 0;
    const guardActive = guard[i] > 0 || guard[i + 1] > 0 || guard[i + 2] > 0;
    if (maskActive && guardActive) count += 1;
  }
  return count;
}

function removeSulfidePixelsFromCanvas(targetCtx, baselineData = null) {
  if (!hasSulfideGuard()) return 0;
  const maskData = targetCtx.getImageData(0, 0, state.imageW, state.imageH);
  const mask = maskData.data;
  const guard = sulfideGuardCtx.getImageData(0, 0, state.imageW, state.imageH).data;
  const baseline = baselineData ? baselineData.data : null;
  let removed = 0;
  for (let i = 0; i < mask.length; i += 4) {
    const maskActive = mask[i] > 0 || mask[i + 1] > 0 || mask[i + 2] > 0;
    const guardActive = guard[i] > 0 || guard[i + 1] > 0 || guard[i + 2] > 0;
    const baselineActive = baseline && (baseline[i] > 0 || baseline[i + 1] > 0 || baseline[i + 2] > 0);
    if (maskActive && guardActive && !baselineActive) {
      mask[i] = 0;
      mask[i + 1] = 0;
      mask[i + 2] = 0;
      mask[i + 3] = 255;
      removed += 1;
    }
  }
  if (removed > 0) targetCtx.putImageData(maskData, 0, 0);
  return removed;
}

function removeSulfidePixelsFromMask(baselineData = null) {
  return removeSulfidePixelsFromCanvas(maskCtx, baselineData);
}

function enforceSulfideProtection(kind, baselineData = null, record = true) {
  if (!els.protectSulfides.checked) return 0;
  const removed = removeSulfidePixelsFromMask(baselineData);
  if (removed > 0 && record) {
    state.edits.push({ type: 'protect_sulfides', tool: kind, removed_pixels: removed, at: new Date().toISOString() });
  }
  return removed;
}

function rasterizeShape(targetCtx, shape) {
  targetCtx.save();
  targetCtx.fillStyle = '#fff';
  if (shape.type === 'polygon') {
    if (shape.points.length < 3) {
      targetCtx.restore();
      return;
    }
    targetCtx.beginPath();
    targetCtx.moveTo(shape.points[0].x, shape.points[0].y);
    for (const point of shape.points.slice(1)) targetCtx.lineTo(point.x, point.y);
    targetCtx.closePath();
    targetCtx.fill();
  } else if (shape.type === 'rectangle') {
    const r = normalizedRect(shape);
    targetCtx.fillRect(r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1);
  }
  targetCtx.restore();
}

function rebuildMaskFromBase(options = {}) {
  const reason = options.reason || 'shape';
  const recordProtection = Boolean(options.recordProtection);
  const baselineData = captureBaseMaskData();
  maskCtx.clearRect(0, 0, state.imageW, state.imageH);
  maskCtx.drawImage(baseMaskCanvas, 0, 0, state.imageW, state.imageH);
  for (const shape of state.shapes) rasterizeShape(maskCtx, shape);
  const protectedPixels = enforceSulfideProtection(reason, baselineData, recordProtection);
  refreshCurrentTint();
  updateMetrics();
  draw();
  return protectedPixels;
}

function flattenShapesToBase(record = false) {
  if (state.shapes.length === 0) return false;
  const baselineData = captureBaseMaskData();
  for (const shape of state.shapes) rasterizeShape(baseMaskCtx, shape);
  if (els.protectSulfides.checked) {
    const protectedPixels = removeSulfidePixelsFromCanvas(baseMaskCtx, baselineData);
    if (protectedPixels > 0 && record) {
      state.edits.push({ type: 'protect_sulfides', tool: 'flatten_shapes', removed_pixels: protectedPixels, at: new Date().toISOString() });
    }
  }
  state.shapes = [];
  state.activeShapeId = null;
  state.shapeDrag = null;
  rebuildMaskFromBase({ recordProtection: false, reason: 'flatten_shapes' });
  if (record) state.edits.push({ type: 'flatten_shapes', at: new Date().toISOString() });
  return true;
}

function updateMetrics() {
  if (!state.sample) return;
  const metrics = state.sample.metrics;
  const currentPixels = countCurrentMaskPixels();
  const currentSulfidePixels = countCurrentSulfideOverlapPixels();
  const totalPixels = state.imageW * state.imageH;
  const saveLabel = SAVE_STATE_LABELS[state.saveState] || SAVE_STATE_LABELS.saved;
  const reviewLabel = reviewStateLabel(state.sample.sample.review_state);
  els.metricsBox.innerHTML = `
    <div class="metric-row"><span>Current talc area</span><strong>${pxWithPct(currentPixels, totalPixels)}</strong></div>
    <div class="metric-row"><span>Autodetected talc</span><strong>${pxWithPct(metrics.autodetected_talc_pixels, totalPixels)}</strong></div>
    <div class="metric-row"><span>Blue-line candidate</span><strong>${pxWithPct(metrics.candidate_talc_pixels, totalPixels)}</strong></div>
    <div class="metric-row"><span>Original candidate on sulfide</span><strong>${pxWithPct(metrics.overlap_pixels, totalPixels)}</strong></div>
    <div class="metric-row"><span>Current talc on sulfide</span><strong>${pxWithPct(currentSulfidePixels, totalPixels)}</strong></div>
    <div class="metric-row"><span>Working mask</span><strong>${escapeHtml(saveLabel)}</strong></div>
    <div class="metric-row"><span>Review state</span><strong>${escapeHtml(reviewLabel)}</strong></div>`;
}

function clampByte(value, fallback = 255) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.max(0, Math.min(255, Math.round(numeric)));
}

function currentBrightnessThreshold() {
  return clampByte(els.brightnessThreshold ? els.brightnessThreshold.value : 255, 255);
}

function brightnessThresholdLabel(value) {
  return value >= 255 ? '255 (off)' : `${value}`;
}

function resetBrightnessPreviewCache() {
  state.brightnessPreview.source = null;
  state.brightnessPreview.threshold = null;
}

function updateBrightnessThresholdUi(persist = true) {
  const threshold = currentBrightnessThreshold();
  if (els.brightnessThreshold) els.brightnessThreshold.value = String(threshold);
  if (els.brightnessThresholdValue) els.brightnessThresholdValue.textContent = brightnessThresholdLabel(threshold);
  if (persist) localStorage.setItem(BRIGHTNESS_THRESHOLD_STORAGE_KEY, String(threshold));
}

function setBrightnessThreshold(value, persist = true) {
  if (!els.brightnessThreshold) return;
  els.brightnessThreshold.value = String(clampByte(value, 255));
  resetBrightnessPreviewCache();
  updateBrightnessThresholdUi(persist);
  drawWithAvailabilityStatus();
}

function viewSettingsPayload() {
  return {
    brightness_threshold_luma: currentBrightnessThreshold(),
    brightness_threshold_formula: BRIGHTNESS_THRESHOLD_FORMULA,
    background_mode: els.baseMode ? els.baseMode.value : null
  };
}

function brightnessFilteredBackground(base) {
  const threshold = currentBrightnessThreshold();
  if (!base || threshold >= 255) return base;
  if (
    state.brightnessPreview.source === base
    && state.brightnessPreview.threshold === threshold
    && brightnessPreviewCanvas.width === state.imageW
    && brightnessPreviewCanvas.height === state.imageH
  ) {
    return brightnessPreviewCanvas;
  }

  brightnessSourceCanvas.width = state.imageW;
  brightnessSourceCanvas.height = state.imageH;
  brightnessPreviewCanvas.width = state.imageW;
  brightnessPreviewCanvas.height = state.imageH;

  if (threshold <= 0) {
    brightnessPreviewCtx.fillStyle = '#ffffff';
    brightnessPreviewCtx.fillRect(0, 0, state.imageW, state.imageH);
  } else {
    brightnessSourceCtx.clearRect(0, 0, state.imageW, state.imageH);
    brightnessSourceCtx.drawImage(base, 0, 0, state.imageW, state.imageH);
    const imageData = brightnessSourceCtx.getImageData(0, 0, state.imageW, state.imageH);
    const data = imageData.data;
    for (let i = 0; i < data.length; i += 4) {
      const luma = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      if (luma > threshold) {
        data[i] = 255;
        data[i + 1] = 255;
        data[i + 2] = 255;
        data[i + 3] = 255;
      }
    }
    brightnessPreviewCtx.putImageData(imageData, 0, 0);
  }

  state.brightnessPreview.source = base;
  state.brightnessPreview.threshold = threshold;
  return brightnessPreviewCanvas;
}

function draw() {
  ctx.clearRect(0, 0, viewer.width, viewer.height);
  if (!state.sample) return;
  const baseMode = els.baseMode.value;
  let base = state.images.original || state.images.annotated;
  if (baseMode === 'annotated') base = state.images.annotated || base;
  if (baseMode === 'qa') base = state.images.qa || base;
  if (baseMode === 'mask') {
    ctx.fillStyle = cssVar('--mask-only-bg', '#0f172a');
    ctx.fillRect(0, 0, state.imageW, state.imageH);
  } else if (baseMode === 'sulfide') {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, state.imageW, state.imageH);
    if (state.images.sulfideMask) ctx.drawImage(state.images.sulfideMask, 0, 0, state.imageW, state.imageH);
  } else if (base) {
    ctx.drawImage(brightnessFilteredBackground(base), 0, 0, state.imageW, state.imageH);
  }
  if (els.layers.auto.checked && state.staticTints.auto) ctx.drawImage(state.staticTints.auto, 0, 0);
  if (els.layers.lines.checked && state.staticTints.lines) ctx.drawImage(state.staticTints.lines, 0, 0);
  if (els.layers.overlap.checked && state.staticTints.overlap) ctx.drawImage(state.staticTints.overlap, 0, 0);
  if (els.layers.ignore.checked && state.staticTints.ignore) ctx.drawImage(state.staticTints.ignore, 0, 0);
  if (els.layers.current.checked) ctx.drawImage(currentTintCanvas, 0, 0);
  drawSam2ResultPreview();
  drawShapeGuides();
  drawPolygonDraft();
  drawRectDraft();
  drawSamBoxDraft();
  drawSam2PromptPreview();
  drawBrushCursor();
}

function describeUnavailableBackground() {
  if (!state.sample) return null;
  const baseMode = els.baseMode.value;
  if (baseMode === 'original' && !state.images.original) return 'Original photo is not available for this sample.';
  if (baseMode === 'annotated' && !state.images.annotated) return 'MS Paint annotation image is not available for this sample.';
  if (baseMode === 'qa' && !state.images.qa) return 'Converter QA overlay is not available for this sample.';
  if (baseMode === 'sulfide' && !state.images.sulfideMask) return 'Sulfide mask is not available for this sample.';
  return null;
}

function selectedUnavailableLayerMessages() {
  if (!state.sample) return [];
  const messages = [];
  if (els.layers.auto.checked && !state.staticTints.auto) messages.push('Autodetected mask layer is not available.');
  if (els.layers.lines.checked && !state.staticTints.lines) messages.push('Original blue lines layer is not available.');
  if (els.layers.overlap.checked && !state.staticTints.overlap) messages.push('Sulfide overlap layer is not available.');
  if (els.layers.ignore.checked && !state.staticTints.ignore) messages.push('Ignore/uncertain layer is not available.');
  if (els.layers.current.checked && !currentTintCanvas.width) messages.push('Editable talc mask layer is not available.');
  return messages;
}

function drawWithAvailabilityStatus() {
  draw();
  const missing = describeUnavailableBackground();
  const layerMissing = selectedUnavailableLayerMessages();
  if (missing || layerMissing.length) setStatus(missing || layerMissing[0], true);
}

function drawShapeGuides() {
  for (const shape of state.shapes) {
    if (shape.type === 'polygon') drawPolygonGuide(shape);
    if (shape.type === 'rectangle') drawRectangleGuide(shape);
  }
}

function drawPolygonGuide(shape) {
  if (shape.points.length < 3) return;
  const active = shape.id === state.activeShapeId;
  ctx.save();
  ctx.lineWidth = Math.max(active ? 3 : 2, (active ? 3 : 2) / state.zoom);
  ctx.strokeStyle = active ? '#ffe066' : 'rgba(255, 224, 102, 0.75)';
  ctx.fillStyle = active ? '#ffe066' : 'rgba(255, 224, 102, 0.8)';
  ctx.beginPath();
  ctx.moveTo(shape.points[0].x, shape.points[0].y);
  for (const point of shape.points.slice(1)) ctx.lineTo(point.x, point.y);
  ctx.closePath();
  ctx.stroke();
  for (const point of shape.points) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, Math.max(active ? 5 : 4, (active ? 5 : 4) / state.zoom), 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawRectangleGuide(shape) {
  const active = shape.id === state.activeShapeId;
  const r = normalizedRect(shape);
  ctx.save();
  ctx.strokeStyle = active ? '#ffe066' : 'rgba(255, 224, 102, 0.75)';
  ctx.fillStyle = active ? '#ffe066' : 'rgba(255, 224, 102, 0.8)';
  ctx.lineWidth = Math.max(active ? 3 : 2, (active ? 3 : 2) / state.zoom);
  ctx.strokeRect(r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1);
  for (const handle of rectHandles(r)) {
    const size = active ? 8 / state.zoom : 6 / state.zoom;
    ctx.fillRect(handle.x - size / 2, handle.y - size / 2, size, size);
  }
  ctx.restore();
}

function drawPolygonDraft() {
  const points = state.polygon.points;
  if (points.length === 0) return;
  ctx.save();
  ctx.lineWidth = Math.max(2, 2 / state.zoom);
  ctx.strokeStyle = '#ffe066';
  ctx.fillStyle = '#ffe066';
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (const point of points.slice(1)) ctx.lineTo(point.x, point.y);
  ctx.stroke();
  points.forEach((point, index) => {
    ctx.beginPath();
    ctx.arc(point.x, point.y, Math.max(index === 0 ? 7 : 5, (index === 0 ? 7 : 5) / state.zoom), 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();
}

function drawRectDraft() {
  if (!state.rect.active) return;
  const r = normalizedRect(state.rect);
  ctx.save();
  ctx.strokeStyle = '#ffe066';
  ctx.fillStyle = 'rgba(255, 224, 102, 0.12)';
  ctx.lineWidth = Math.max(2, 2 / state.zoom);
  ctx.fillRect(r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1);
  ctx.strokeRect(r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1);
  for (const handle of rectHandles(r)) {
    ctx.fillRect(handle.x - 4 / state.zoom, handle.y - 4 / state.zoom, 8 / state.zoom, 8 / state.zoom);
  }
  ctx.restore();
}

function drawSamBoxDraft() {
  if (!state.samBox) return;
  const r = normalizedRect(state.samBox);
  ctx.save();
  ctx.strokeStyle = '#ff6b35';
  ctx.lineWidth = Math.max(2, 2 / state.zoom);
  ctx.setLineDash([Math.max(8, 8 / state.zoom), Math.max(6, 6 / state.zoom)]);
  ctx.fillStyle = 'rgba(255, 107, 53, 0.08)';
  ctx.fillRect(r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1);
  ctx.strokeRect(r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1);
  ctx.restore();
}

function drawSam2ResultPreview() {
  if (state.tool !== 'sam2' || !state.sam2Preview.tint) return;
  ctx.save();
  ctx.drawImage(state.sam2Preview.tint, 0, 0);
  if (state.sam2Preview.prompt && state.sam2Preview.prompt.type === 'point_xy') {
    const x = state.sam2Preview.prompt.x;
    const y = state.sam2Preview.prompt.y;
    const radius = Math.max(10, 10 / state.zoom);
    ctx.strokeStyle = '#ff6b35';
    ctx.lineWidth = Math.max(2, 2 / state.zoom);
    ctx.setLineDash([Math.max(5, 5 / state.zoom), Math.max(4, 4 / state.zoom)]);
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(x - radius * 0.55, y);
    ctx.lineTo(x + radius * 0.55, y);
    ctx.moveTo(x, y - radius * 0.55);
    ctx.lineTo(x, y + radius * 0.55);
    ctx.stroke();
  }
  ctx.restore();
}

function drawSam2PromptPreview() {
  if (state.tool !== 'sam2' || !state.hoverPoint || !state.sample || !state.sample.editable || state.samBox) return;
  ctx.save();
  ctx.strokeStyle = '#ff6b35';
  ctx.fillStyle = 'rgba(255, 107, 53, 0.07)';
  ctx.lineWidth = Math.max(2, 2 / state.zoom);
  ctx.setLineDash([Math.max(8, 8 / state.zoom), Math.max(6, 6 / state.zoom)]);
  if (els.sam2PromptMode.value === 'rectangle_xyxy') {
    const side = Math.min(
      Math.max(96 / state.zoom, Number(els.brushSize.value) * 3),
      Math.max(24, Math.min(state.imageW, state.imageH) / 3)
    );
    const half = side / 2;
    const r = {
      x1: Math.max(0, state.hoverPoint.x - half),
      y1: Math.max(0, state.hoverPoint.y - half),
      x2: Math.min(state.imageW - 1, state.hoverPoint.x + half),
      y2: Math.min(state.imageH - 1, state.hoverPoint.y + half)
    };
    ctx.fillRect(r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1);
    ctx.strokeRect(r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1);
  } else {
    const radius = Math.min(
      Math.max(32 / state.zoom, Number(els.brushSize.value)),
      Math.max(16, Math.min(state.imageW, state.imageH) / 6)
    );
    ctx.beginPath();
    ctx.arc(state.hoverPoint.x, state.hoverPoint.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(state.hoverPoint.x - radius * 0.35, state.hoverPoint.y);
    ctx.lineTo(state.hoverPoint.x + radius * 0.35, state.hoverPoint.y);
    ctx.moveTo(state.hoverPoint.x, state.hoverPoint.y - radius * 0.35);
    ctx.lineTo(state.hoverPoint.x, state.hoverPoint.y + radius * 0.35);
    ctx.stroke();
  }
  ctx.restore();
}

function drawBrushCursor() {
  if (state.tool !== 'brush' || !state.hoverPoint || !state.sample || !state.sample.editable) return;
  const radius = Number(els.brushSize.value) / 2;
  if (!Number.isFinite(radius) || radius <= 0) return;
  ctx.save();
  ctx.beginPath();
  ctx.arc(state.hoverPoint.x, state.hoverPoint.y, radius, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(0, 163, 216, 0.10)';
  ctx.fill();
  ctx.lineWidth = Math.max(3, 3 / state.zoom);
  ctx.strokeStyle = 'rgba(15, 23, 42, 0.88)';
  ctx.stroke();
  ctx.lineWidth = Math.max(1.5, 1.5 / state.zoom);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.96)';
  ctx.stroke();
  ctx.restore();
}

function cursorForRectHandle(handle) {
  if (handle === 'move') return 'move';
  if (handle === 'n' || handle === 's') return 'ns-resize';
  if (handle === 'e' || handle === 'w') return 'ew-resize';
  if (handle === 'nw' || handle === 'se') return 'nwse-resize';
  if (handle === 'ne' || handle === 'sw') return 'nesw-resize';
  return 'crosshair';
}

function canvasCursorForPoint(point = null) {
  if (!state.sample || !state.sample.editable) return 'default';
  if (state.viewPan.active) return 'grabbing';
  if (state.drawing || state.shapeDrag) return 'grabbing';
  if (state.tool === 'brush') return 'crosshair';
  if (state.tool === 'fill') return 'cell';
  if (state.tool === 'sam2') return 'crosshair';
  if (state.tool === 'rectangle') {
    if (state.rect.active) {
      const draftHandle = point ? hitRectHandle(point) : null;
      return cursorForRectHandle(draftHandle || 'crosshair');
    }
    const hit = point ? hitRectangleShape(point) : null;
    return hit ? cursorForRectHandle(hit.handle) : 'crosshair';
  }
  if (state.tool === 'polygon') {
    if (point && state.polygon.points.length > 0 && nearestPolygonPoint(point) !== null) return 'pointer';
    if (point && hitPolygonShapePoint(point)) return 'grab';
    if (point && hitPolygonShapeSegment(point)) return 'copy';
    if (point && hitPolygonShapeBody(point)) return 'move';
    return 'crosshair';
  }
  return 'default';
}

function updateViewerCursor(point = null) {
  viewer.style.cursor = canvasCursorForPoint(point || state.hoverPoint);
}

function pushUndo() {
  try {
    state.undoStack.push({
      mask: maskCanvas.toDataURL('image/png'),
      base: baseMaskCanvas.toDataURL('image/png'),
      shapes: cloneShapes(),
      activeShapeId: state.activeShapeId,
      polygonPoints: state.polygon.points.map((p) => ({ x: p.x, y: p.y })),
      rect: { ...state.rect }
    });
    if (state.undoStack.length > 20) state.undoStack.shift();
  } catch (err) {
    console.warn('undo snapshot failed', err);
  }
}

async function undo() {
  const snapshot = state.undoStack.pop();
  if (!snapshot) {
    setStatus('Nothing to undo.');
    return;
  }
  markLocalDirty();
  const img = await loadImage(snapshot.base || snapshot);
  if (!img) return;
  baseMaskCtx.clearRect(0, 0, state.imageW, state.imageH);
  baseMaskCtx.drawImage(img, 0, 0, state.imageW, state.imageH);
  if (snapshot.shapes) {
    restoreShapes(snapshot.shapes);
    state.activeShapeId = snapshot.activeShapeId || null;
    state.polygon.points = (snapshot.polygonPoints || []).map((p) => ({ x: p.x, y: p.y }));
    state.rect = snapshot.rect ? { ...snapshot.rect } : { active: false, x1: 0, y1: 0, x2: 0, y2: 0, handle: null, lastPoint: null, startPoint: null, dragMoved: false };
  } else {
    state.shapes = [];
    state.activeShapeId = null;
    state.polygon.points = [];
    state.rect.active = false;
  }
  state.shapeDrag = null;
  maskCtx.clearRect(0, 0, state.imageW, state.imageH);
  if (snapshot.mask) {
    const maskImg = await loadImage(snapshot.mask);
    if (maskImg) maskCtx.drawImage(maskImg, 0, 0, state.imageW, state.imageH);
  } else {
    maskCtx.drawImage(img, 0, 0, state.imageW, state.imageH);
  }
  if (snapshot.shapes) rebuildMaskFromBase({ recordProtection: false, reason: 'undo' });
  state.edits.push({ type: 'undo', at: new Date().toISOString() });
  refreshCurrentTint();
  updateMetrics();
  draw();
  await autosave('undo');
}

function isMaskDataActive(data, pixelIndex, threshold = 16) {
  const i = pixelIndex * 4;
  return data[i] > threshold || data[i + 1] > threshold || data[i + 2] > threshold;
}

function prepareFillBoundaries(rawLines, closedLines) {
  fillBoundaryCanvas.width = state.imageW;
  fillBoundaryCanvas.height = state.imageH;
  fillBoundaryCtx.clearRect(0, 0, state.imageW, state.imageH);
  state.fillBoundaryLoaded = Boolean(rawLines || closedLines);
  if (rawLines) fillBoundaryCtx.drawImage(rawLines, 0, 0, state.imageW, state.imageH);
  if (closedLines) fillBoundaryCtx.drawImage(closedLines, 0, 0, state.imageW, state.imageH);
}

async function fillAtPoint(point) {
  if (!state.sampleId || !state.sample || !state.sample.editable) return;
  const x = Math.max(0, Math.min(state.imageW - 1, Math.floor(point.x)));
  const y = Math.max(0, Math.min(state.imageH - 1, Math.floor(point.y)));
  const width = state.imageW;
  const height = state.imageH;
  const total = width * height;
  const start = y * width + x;
  const currentData = maskCtx.getImageData(0, 0, width, height).data;
  const boundaryData = state.fillBoundaryLoaded ? fillBoundaryCtx.getImageData(0, 0, width, height).data : null;
  const sulfideBoundaryData = hasSulfideGuard() ? sulfideGuardCtx.getImageData(0, 0, width, height).data : null;
  const boundaryLabels = [];
  if (state.fillBoundaryLoaded) boundaryLabels.push('blue_lines');
  boundaryLabels.push('current_talc_regions');
  if (sulfideBoundaryData) boundaryLabels.push('sulfide_pixels');
  boundaryLabels.push('screen_edges');
  const blocked = (pixel) => (
    isMaskDataActive(currentData, pixel, 0)
    || (boundaryData && isMaskDataActive(boundaryData, pixel, 16))
    || (sulfideBoundaryData && isMaskDataActive(sulfideBoundaryData, pixel, 0))
  );
  if (blocked(start)) {
    setStatus('Fill point is on a blue line, sulfide pixel, or existing talc region; click inside an empty bounded area.', true);
    return;
  }

  const baselineData = captureBaseMaskData();
  const baseData = baseMaskCtx.getImageData(0, 0, width, height);
  const base = baseData.data;
  const visited = new Uint8Array(total);
  const queue = new Int32Array(total);
  let head = 0;
  let tail = 0;
  let filled = 0;
  visited[start] = 1;
  queue[tail] = start;
  tail += 1;

  while (head < tail) {
    const pixel = queue[head];
    head += 1;
    if (blocked(pixel)) continue;
    const offset = pixel * 4;
    base[offset] = 255;
    base[offset + 1] = 255;
    base[offset + 2] = 255;
    base[offset + 3] = 255;
    filled += 1;
    const px = pixel % width;
    const py = Math.floor(pixel / width);
    const neighbors = [
      px > 0 ? pixel - 1 : -1,
      px < width - 1 ? pixel + 1 : -1,
      py > 0 ? pixel - width : -1,
      py < height - 1 ? pixel + width : -1
    ];
    for (const next of neighbors) {
      if (next < 0 || visited[next]) continue;
      if (blocked(next)) continue;
      visited[next] = 1;
      queue[tail] = next;
      tail += 1;
    }
  }

  if (filled === 0) {
    setStatus('Nothing to fill at this point.', true);
    return;
  }

  pushUndo();
  baseMaskCtx.putImageData(baseData, 0, 0);
  let protectedPixels = 0;
  if (els.protectSulfides.checked) {
    protectedPixels = removeSulfidePixelsFromCanvas(baseMaskCtx, baselineData);
  }
  state.edits.push({
    type: 'fill',
    x,
    y,
    filled_pixels: filled,
    protected_pixels: protectedPixels,
    boundaries: boundaryLabels,
    at: new Date().toISOString()
  });
  markLocalDirty();
  rebuildMaskFromBase({ recordProtection: false, reason: 'fill' });
  const message = protectedPixels > 0
    ? `Autosaved fill; added ${formatInt(filled)} px and protected ${formatInt(protectedPixels)} sulfide px.`
    : `Autosaved fill; added ${formatInt(filled)} px.`;
  await autosave('fill', message);
}

function drawMaskLine(from, to, mode, targetCtx = maskCtx) {
  targetCtx.save();
  targetCtx.globalCompositeOperation = 'source-over';
  targetCtx.strokeStyle = mode === 'eraser' ? '#000' : '#fff';
  targetCtx.fillStyle = targetCtx.strokeStyle;
  targetCtx.lineWidth = Number(els.brushSize.value);
  targetCtx.lineCap = 'round';
  targetCtx.lineJoin = 'round';
  targetCtx.beginPath();
  targetCtx.moveTo(from.x, from.y);
  targetCtx.lineTo(to.x, to.y);
  targetCtx.stroke();
  targetCtx.beginPath();
  targetCtx.arc(to.x, to.y, Number(els.brushSize.value) / 2, 0, Math.PI * 2);
  targetCtx.fill();
  targetCtx.restore();
}

function strokeModeForPointer(event) {
  if (state.tool !== 'brush') return null;
  if (event.button === 0) return 'brush';
  if (event.button === 2) return 'eraser';
  return null;
}

function commitShapeChange(kind, edit) {
  if (edit) state.edits.push(edit);
  markLocalDirty();
  const protectedPixels = rebuildMaskFromBase({ recordProtection: true, reason: kind });
  const message = protectedPixels > 0
    ? `Autosaved ${kind}; protected ${formatInt(protectedPixels)} sulfide px.`
    : undefined;
  autosave(kind, message).catch((err) => setStatus(`Autosave failed: ${err.message}`, true));
}

function addPolygonShape(points) {
  if (points.length < 3) return false;
  pushUndo();
  const shape = {
    id: state.nextShapeId++,
    type: 'polygon',
    points: points.map((p) => ({ x: p.x, y: p.y }))
  };
  state.shapes.push(shape);
  state.activeShapeId = shape.id;
  state.polygon.points = [];
  commitShapeChange('polygon', {
    type: 'polygon_add_talc',
    shape_id: shape.id,
    points: shape.points.map((p) => [Math.round(p.x), Math.round(p.y)]),
    at: new Date().toISOString()
  });
  setStatus('Polygon closed and autosaved.');
  return true;
}

function addRectangleShape(rect) {
  const r = normalizedRect(rect);
  if (r.x2 - r.x1 < 2 || r.y2 - r.y1 < 2) return false;
  pushUndo();
  const shape = {
    id: state.nextShapeId++,
    type: 'rectangle',
    x1: r.x1,
    y1: r.y1,
    x2: r.x2,
    y2: r.y2
  };
  state.shapes.push(shape);
  state.activeShapeId = shape.id;
  state.rect.active = false;
  commitShapeChange('rectangle', {
    type: 'rectangle_add_talc',
    shape_id: shape.id,
    x1: Math.round(r.x1),
    y1: Math.round(r.y1),
    x2: Math.round(r.x2),
    y2: Math.round(r.y2),
    at: new Date().toISOString()
  });
  setStatus('Rectangle drawn and autosaved.');
  return true;
}

function afterMaskEdit(kind, baselineData = null, options = {}) {
  let protectedPixels = 0;
  markLocalDirty();
  if (options.baseBaselineData) {
    if (els.protectSulfides.checked) {
      protectedPixels = removeSulfidePixelsFromCanvas(baseMaskCtx, options.baseBaselineData);
      if (protectedPixels > 0) {
        state.edits.push({ type: 'protect_sulfides', tool: kind, removed_pixels: protectedPixels, at: new Date().toISOString() });
      }
    }
    rebuildMaskFromBase({ recordProtection: false, reason: kind });
  } else {
    protectedPixels = enforceSulfideProtection(kind, baselineData);
    refreshCurrentTint();
    updateMetrics();
    draw();
  }
  const message = protectedPixels > 0
    ? `Autosaved ${kind}; protected ${formatInt(protectedPixels)} sulfide px.`
    : undefined;
  autosave(kind, message).catch((err) => setStatus(`Autosave failed: ${err.message}`, true));
}

async function autosave(reason, message) {
  if (!state.sampleId) return;
  state.saveState = 'saving';
  updateMetrics();
  try {
    await apiPost(`/api/samples/${encodeURIComponent(state.sampleId)}/autosave`, {
      mask_png: maskCanvas.toDataURL('image/png'),
      edits: state.edits,
      reason,
      view_settings: viewSettingsPayload()
    });
    state.dirty = false;
    state.saveState = 'saved';
    state.lastSavedAt = new Date();
    updateMetrics();
    setStatus(message || `Autosaved ${reason}.`);
  } catch (err) {
    state.dirty = true;
    state.saveState = 'error';
    updateMetrics();
    throw err;
  }
}

async function subtractSulfidesFromMask() {
  if (!state.sampleId) return;
  if (!hasSulfideGuard()) {
    setStatus('No sulfide mask is available for this sample.', true);
    return;
  }
  if (countCurrentSulfideOverlapPixels() === 0) {
    updateMetrics();
    draw();
    setStatus('No talc pixels overlap the sulfide mask.');
    return;
  }
  pushUndo();
  const flattened = flattenShapesToBase(false);
  const removed = removeSulfidePixelsFromCanvas(baseMaskCtx);
  if (removed === 0) {
    state.undoStack.pop();
    rebuildMaskFromBase({ recordProtection: false, reason: 'subtract_sulfides' });
    updateMetrics();
    draw();
    setStatus('No talc pixels overlap the sulfide mask.');
    return;
  }
  if (flattened) state.edits.push({ type: 'flatten_shapes', reason: 'subtract_sulfides', at: new Date().toISOString() });
  state.edits.push({ type: 'subtract_sulfides', removed_pixels: removed, at: new Date().toISOString() });
  markLocalDirty();
  rebuildMaskFromBase({ recordProtection: false, reason: 'subtract_sulfides' });
  await autosave('subtract sulfides', `Autosaved sulfide subtraction; removed ${formatInt(removed)} px.`);
}

async function saveReview(moveNext) {
  if (!state.sampleId) return;
  const currentSampleId = state.sampleId;
  const nextInVisibleQueue = moveNext ? nextVisibleSampleId(currentSampleId) : null;
  setStatus('Saving reviewed mask...');
  flattenShapesToBase(true);
  const result = await apiPost(`/api/samples/${encodeURIComponent(state.sampleId)}/save`, {
    mask_png: maskCanvas.toDataURL('image/png'),
    edits: state.edits,
    reviewer: els.reviewerInput.value.trim(),
    notes: els.notesInput.value.trim(),
    view_settings: viewSettingsPayload()
  });
  state.dirty = false;
  state.saveState = 'reviewed';
  state.lastSavedAt = new Date();
  updateMetrics();
  setStatus(`Saved ${result.sample_id}.`);
  await loadManifest(false);
  if (moveNext) {
    const nextVisibleAfterSave = filteredSamples()[0]?.sample_id || null;
    const nextId = nextInVisibleQueue && state.samples.some((sample) => sample.sample_id === nextInVisibleQueue)
      ? nextInVisibleQueue
      : nextVisibleAfterSave;
    if (nextId) {
      await loadSample(nextId, { force: true });
    } else {
      setStatus('Saved. No next sample is visible in the current filter.');
    }
  } else {
    await loadSample(state.sampleId, { force: true });
  }
}

async function resetCurrent() {
  if (!state.sampleId) return;
  if (!window.confirm('Reset current talc mask to the autodetected mask for this sample?')) return;
  await apiPost(`/api/samples/${encodeURIComponent(state.sampleId)}/reset`, {});
  await loadSample(state.sampleId);
  setStatus('Current mask reset to autodetected.');
}

function nearestPolygonPoint(point) {
  const tolerance = Math.max(10 / state.zoom, 5);
  let best = null;
  let bestDist = Infinity;
  state.polygon.points.forEach((candidate, index) => {
    const dist = Math.hypot(candidate.x - point.x, candidate.y - point.y);
    if (dist < bestDist && dist <= tolerance) {
      best = index;
      bestDist = dist;
    }
  });
  return best;
}

function nearestPolygonSegment(point) {
  if (state.polygon.points.length < 2) return null;
  const tolerance = Math.max(10 / state.zoom, 5);
  let best = null;
  let bestDist = Infinity;
  const points = state.polygon.points;
  for (let i = 0; i < points.length; i += 1) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    if (i === points.length - 1 && points.length < 3) continue;
    const dist = pointToSegmentDistance(point, a, b);
    if (dist < bestDist && dist <= tolerance) {
      best = i + 1;
      bestDist = dist;
    }
  }
  return best;
}

function pointToSegmentDistance(p, a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  if (dx === 0 && dy === 0) return Math.hypot(p.x - a.x, p.y - a.y);
  const t = Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy)));
  return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

function removePolygonPoint(index) {
  if (index === null) return false;
  state.polygon.points.splice(index, 1);
  state.polygon.dragIndex = null;
  draw();
  setStatus('Polygon point removed.');
  return true;
}

function removePolygonShapePoint(hit) {
  if (!hit || !hit.shape || hit.index === null) return false;
  if (hit.shape.points.length <= 3) {
    setStatus('Polygon needs at least 3 points.', true);
    return true;
  }
  pushUndo();
  hit.shape.points.splice(hit.index, 1);
  state.activeShapeId = hit.shape.id;
  state.shapeDrag = null;
  commitShapeChange('polygon', {
    type: 'polygon_point_remove',
    shape_id: hit.shape.id,
    point_index: hit.index,
    points: hit.shape.points.map((p) => [Math.round(p.x), Math.round(p.y)]),
    at: new Date().toISOString()
  });
  setStatus('Polygon point removed and autosaved.');
  return true;
}

function deleteSelectedShape() {
  if (!state.sample || !state.sample.editable) return false;
  if (state.drawing || state.shapeDrag || state.polygon.points.length > 0 || state.rect.active) return false;
  const shape = shapeById(state.activeShapeId);
  if (!shape) return false;
  const shapeIndex = state.shapes.findIndex((candidate) => candidate.id === shape.id);
  if (shapeIndex < 0) return false;
  pushUndo();
  const [removed] = state.shapes.splice(shapeIndex, 1);
  state.activeShapeId = null;
  state.shapeDrag = null;
  const edit = removed.type === 'polygon'
    ? {
        type: 'polygon_shape_delete',
        shape_id: removed.id,
        points: removed.points.map((p) => [Math.round(p.x), Math.round(p.y)]),
        at: new Date().toISOString()
      }
    : {
        type: 'rectangle_shape_delete',
        shape_id: removed.id,
        ...Object.fromEntries(Object.entries(normalizedRect(removed)).map(([key, value]) => [key, Math.round(value)])),
        at: new Date().toISOString()
      };
  commitShapeChange(removed.type, edit);
  setStatus(`${removed.type === 'polygon' ? 'Polygon' : 'Rectangle'} deleted and autosaved.`);
  return true;
}

function hitPolygonShapePoint(point) {
  const tolerance = Math.max(10 / state.zoom, 5);
  let best = null;
  let bestDist = Infinity;
  for (const shape of state.shapes) {
    if (shape.type !== 'polygon') continue;
    shape.points.forEach((candidate, index) => {
      const dist = Math.hypot(candidate.x - point.x, candidate.y - point.y);
      if (dist < bestDist && dist <= tolerance) {
        best = { shape, index };
        bestDist = dist;
      }
    });
  }
  return best;
}

function hitPolygonShapeSegment(point) {
  const tolerance = Math.max(10 / state.zoom, 5);
  let best = null;
  let bestDist = Infinity;
  for (const shape of state.shapes) {
    if (shape.type !== 'polygon' || shape.points.length < 3) continue;
    for (let i = 0; i < shape.points.length; i += 1) {
      const a = shape.points[i];
      const b = shape.points[(i + 1) % shape.points.length];
      const dist = pointToSegmentDistance(point, a, b);
      if (dist < bestDist && dist <= tolerance) {
        best = { shape, insertAt: i + 1 };
        bestDist = dist;
      }
    }
  }
  return best;
}

function pointInPolygon(point, points) {
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i, i += 1) {
    const pi = points[i];
    const pj = points[j];
    const intersects = ((pi.y > point.y) !== (pj.y > point.y))
      && (point.x < ((pj.x - pi.x) * (point.y - pi.y)) / ((pj.y - pi.y) || 1e-9) + pi.x);
    if (intersects) inside = !inside;
  }
  return inside;
}

function hitPolygonShapeBody(point) {
  for (const shape of state.shapes) {
    if (shape.type === 'polygon' && pointInPolygon(point, shape.points)) return shape;
  }
  return null;
}

function normalizedRect(rect) {
  return {
    x1: Math.min(rect.x1, rect.x2),
    y1: Math.min(rect.y1, rect.y2),
    x2: Math.max(rect.x1, rect.x2),
    y2: Math.max(rect.y1, rect.y2)
  };
}

function rectHandles(r) {
  const cx = (r.x1 + r.x2) / 2;
  const cy = (r.y1 + r.y2) / 2;
  return [
    { name: 'nw', x: r.x1, y: r.y1 }, { name: 'n', x: cx, y: r.y1 }, { name: 'ne', x: r.x2, y: r.y1 },
    { name: 'e', x: r.x2, y: cy }, { name: 'se', x: r.x2, y: r.y2 }, { name: 's', x: cx, y: r.y2 },
    { name: 'sw', x: r.x1, y: r.y2 }, { name: 'w', x: r.x1, y: cy }
  ];
}

function hitRectHandle(point) {
  if (!state.rect.active) return null;
  const tolerance = Math.max(10 / state.zoom, 5);
  const r = normalizedRect(state.rect);
  for (const handle of rectHandles(r)) {
    if (Math.hypot(point.x - handle.x, point.y - handle.y) <= tolerance) return handle.name;
  }
  if (point.x >= r.x1 && point.x <= r.x2 && point.y >= r.y1 && point.y <= r.y2) return 'move';
  return null;
}

function hitRectangleShape(point) {
  let body = null;
  for (const shape of state.shapes) {
    if (shape.type !== 'rectangle') continue;
    const r = normalizedRect(shape);
    for (const handle of rectHandles(r)) {
      if (Math.hypot(point.x - handle.x, point.y - handle.y) <= Math.max(10 / state.zoom, 5)) {
        return { shape, handle: handle.name };
      }
    }
    if (point.x >= r.x1 && point.x <= r.x2 && point.y >= r.y1 && point.y <= r.y2) body = { shape, handle: 'move' };
  }
  return body;
}

function updateRectHandle(rect, handle, point, previous) {
  const dx = previous ? point.x - previous.x : 0;
  const dy = previous ? point.y - previous.y : 0;
  if (handle === 'move') {
    rect.x1 += dx; rect.x2 += dx; rect.y1 += dy; rect.y2 += dy;
    return;
  }
  if (handle.includes('w')) rect.x1 = point.x;
  if (handle.includes('e')) rect.x2 = point.x;
  if (handle.includes('n')) rect.y1 = point.y;
  if (handle.includes('s')) rect.y2 = point.y;
}

function updateShapeDrag(point) {
  const drag = state.shapeDrag;
  if (!drag) return;
  const shape = shapeById(drag.shapeId);
  if (!shape) return;
  if (shape.type === 'polygon') {
    if (drag.mode === 'point') {
      shape.points[drag.index] = point;
    } else if (drag.mode === 'move') {
      const dx = point.x - drag.lastPoint.x;
      const dy = point.y - drag.lastPoint.y;
      shape.points.forEach((p) => {
        p.x = Math.max(0, Math.min(state.imageW - 1, p.x + dx));
        p.y = Math.max(0, Math.min(state.imageH - 1, p.y + dy));
      });
      drag.lastPoint = point;
    }
  } else if (shape.type === 'rectangle') {
    updateRectHandle(shape, drag.handle, point, drag.lastPoint);
    drag.lastPoint = point;
  }
  drag.changed = true;
  rebuildMaskFromBase({ recordProtection: false, reason: 'shape_preview' });
}

function finishShapeDrag() {
  const drag = state.shapeDrag;
  if (!drag) return;
  state.shapeDrag = null;
  if (!drag.changed) return;
  const shape = shapeById(drag.shapeId);
  if (!shape) return;
  const edit = shape.type === 'polygon'
    ? {
        type: 'polygon_shape_edit',
        shape_id: shape.id,
        points: shape.points.map((p) => [Math.round(p.x), Math.round(p.y)]),
        at: new Date().toISOString()
      }
    : {
        type: 'rectangle_shape_edit',
        shape_id: shape.id,
        ...Object.fromEntries(Object.entries(normalizedRect(shape)).map(([key, value]) => [key, Math.round(value)])),
        at: new Date().toISOString()
      };
  commitShapeChange(shape.type, edit);
}

async function fetchSam2Mask(promptGeometry, runningMessage) {
  if (!state.sampleId) return null;
  setStatus(runningMessage || 'Running SAM2 assist...');
  const result = await apiPost(`/api/samples/${encodeURIComponent(state.sampleId)}/sam2`, { prompt_geometry: promptGeometry });
  if (!result.available) {
    setStatus(`SAM2 unavailable: ${result.error}`, true);
    return null;
  }
  const img = await loadImage(result.mask_url);
  if (!img) {
    setStatus('SAM2 returned a mask, but the browser could not load it.', true);
    return null;
  }
  const stats = positiveMaskStats(img);
  if (stats.positivePixels === 0) {
    setStatus('SAM2 returned an empty mask.', true);
    return null;
  }
  if (stats.fraction > MAX_SAM2_REGION_FRACTION) {
    setStatus(`SAM2 mask covers ${Math.round(stats.fraction * 100)}% of the image; draw a smaller SAM2 box or use brush/polygon.`, true);
    return null;
  }
  return { promptGeometry, result, img, stats };
}

async function runSam2(promptGeometry) {
  const maskResult = await fetchSam2Mask(promptGeometry, 'Running SAM2 assist...');
  if (!maskResult) return;
  applySam2MaskResult(maskResult);
}

function applySam2MaskResult(maskResult) {
  const baselineData = captureMaskData();
  const baseBaselineData = captureBaseMaskData();
  pushUndo();
  const mergedPixels = mergePositiveMaskImage(maskResult.img, baseMaskCtx);
  if (mergedPixels === 0) {
    state.undoStack.pop();
    setStatus('SAM2 returned no positive mask pixels.', true);
    return;
  }
  clearSam2Preview({ redraw: false });
  state.edits.push({
    type: 'sam2_add_talc',
    prompt_geometry: maskResult.promptGeometry,
    score: maskResult.result.summary.score,
    mask_pixels: mergedPixels,
    mask_fraction: maskResult.stats.fraction,
    at: new Date().toISOString()
  });
  afterMaskEdit('sam2', baselineData, { baseBaselineData });
}

function scheduleSam2PointHoverPreview(point) {
  if (!sam2PointModeActive() || !state.sample || !state.sample.editable || state.samBox) {
    clearSam2Preview();
    return;
  }
  const prompt = { type: 'point_xy', x: Math.round(point.x), y: Math.round(point.y) };
  const key = sam2PromptKey(prompt);
  if (state.sam2Preview.promptKey === key || state.sam2Preview.pendingKey === key || state.sam2Preview.loadingKey === key) {
    updateSam2ApplyButton();
    return;
  }
  clearSam2Preview({ redraw: false });
  state.sam2Preview.pendingKey = key;
  state.sam2Preview.prompt = prompt;
  state.sam2Preview.timer = setTimeout(() => {
    requestSam2PointHoverPreview(prompt, key).catch((err) => setStatus(`SAM2 preview failed: ${err.message}`, true));
  }, SAM2_POINT_HOVER_PREVIEW_DELAY_MS);
  updateSam2ApplyButton();
}

async function requestSam2PointHoverPreview(promptGeometry, key) {
  const requestId = state.sam2Preview.requestId + 1;
  state.sam2Preview.requestId = requestId;
  state.sam2Preview.timer = null;
  state.sam2Preview.pendingKey = null;
  state.sam2Preview.loadingKey = key;
  updateSam2ApplyButton();
  const maskResult = await fetchSam2Mask(promptGeometry, 'Running SAM2 point preview...');
  if (requestId !== state.sam2Preview.requestId) return;
  state.sam2Preview.loadingKey = null;
  if (!maskResult) {
    updateSam2ApplyButton();
    return;
  }
  state.sam2Preview.promptKey = key;
  state.sam2Preview.prompt = promptGeometry;
  state.sam2Preview.img = maskResult.img;
  state.sam2Preview.tint = buildTintFromImage(maskResult.img, [255, 107, 53, 105]);
  state.sam2Preview.result = maskResult.result;
  state.sam2Preview.stats = maskResult.stats;
  setStatus('SAM2 point preview ready; press Apply SAM2 to add it.');
  updateSam2ApplyButton();
  draw();
}

async function applySam2PointPreviewOrRun() {
  if (!sam2PointModeActive()) {
    setStatus('Switch SAM2 to point mode to use preview apply.', true);
    return;
  }
  if (state.sam2Preview.img) {
    applySam2MaskResult({
      promptGeometry: state.sam2Preview.prompt,
      result: state.sam2Preview.result,
      img: state.sam2Preview.img,
      stats: state.sam2Preview.stats
    });
    return;
  }
  if (!state.hoverPoint) {
    setStatus('Hover over the image first to choose a SAM2 point.', true);
    return;
  }
  const prompt = { type: 'point_xy', x: Math.round(state.hoverPoint.x), y: Math.round(state.hoverPoint.y) };
  await runSam2(prompt);
}

function positiveMaskStats(img) {
  const temp = document.createElement('canvas');
  temp.width = state.imageW;
  temp.height = state.imageH;
  const tempCtx = temp.getContext('2d', { willReadFrequently: true });
  tempCtx.drawImage(img, 0, 0, state.imageW, state.imageH);
  const src = tempCtx.getImageData(0, 0, state.imageW, state.imageH).data;
  let positivePixels = 0;
  for (let i = 0; i < src.length; i += 4) {
    if (isPositiveMaskPixel(src, i)) positivePixels += 1;
  }
  const totalPixels = state.imageW * state.imageH;
  return {
    positivePixels,
    totalPixels,
    fraction: totalPixels > 0 ? positivePixels / totalPixels : 0
  };
}

function isPositiveMaskPixel(src, i) {
  return src[i + 3] >= 16 && Math.max(src[i], src[i + 1], src[i + 2]) >= 128;
}

function mergePositiveMaskImage(img, targetCtx = maskCtx) {
  const temp = document.createElement('canvas');
  temp.width = state.imageW;
  temp.height = state.imageH;
  const tempCtx = temp.getContext('2d', { willReadFrequently: true });
  tempCtx.drawImage(img, 0, 0, state.imageW, state.imageH);
  const src = tempCtx.getImageData(0, 0, state.imageW, state.imageH).data;
  const dstData = targetCtx.getImageData(0, 0, state.imageW, state.imageH);
  const dst = dstData.data;
  let positivePixels = 0;
  for (let i = 0; i < src.length; i += 4) {
    if (isPositiveMaskPixel(src, i)) {
      dst[i] = 255;
      dst[i + 1] = 255;
      dst[i + 2] = 255;
      dst[i + 3] = 255;
      positivePixels += 1;
    }
  }
  targetCtx.putImageData(dstData, 0, 0);
  return positivePixels;
}

viewer.addEventListener('contextmenu', (event) => {
  if (state.tool === 'polygon' || state.tool === 'brush' || state.tool === 'fill' || state.tool === 'rectangle' || state.tool === 'sam2') event.preventDefault();
});

viewer.addEventListener('auxclick', (event) => {
  if (event.button === 1) event.preventDefault();
});

viewer.addEventListener('wheel', (event) => {
  if (!state.sample) return;
  event.preventDefault();
  zoomBy(event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP, event);
}, { passive: false });

viewer.addEventListener('pointerdown', async (event) => {
  if (!state.sample) return;
  if (event.button === 1) {
    startViewPan(event);
    return;
  }
  if (!state.sample.editable) return;
  const point = imagePointFromEvent(event);
  updateViewerCursor(point);
  if (state.tool === 'brush' || state.tool === 'sam2') state.hoverPoint = point;
  if (state.tool === 'polygon' && event.button === 2) {
    event.preventDefault();
    const draftPoint = nearestPolygonPoint(point);
    if (draftPoint !== null && removePolygonPoint(draftPoint)) return;
    const shapePoint = hitPolygonShapePoint(point);
    if (shapePoint && removePolygonShapePoint(shapePoint)) return;
    state.polygon.points = [];
    state.polygon.dragIndex = null;
    state.activeShapeId = null;
    draw();
    setStatus('Polygon cancelled.');
    return;
  }
  if (state.tool === 'rectangle' && event.button === 2) {
    event.preventDefault();
    state.rect.active = false;
    state.rect.handle = null;
    state.rect.startPoint = null;
    state.rect.dragMoved = false;
    state.activeShapeId = null;
    draw();
    setStatus('Rectangle cancelled.');
    return;
  }
  const strokeMode = strokeModeForPointer(event);
  if (strokeMode) {
    event.preventDefault();
    viewer.setPointerCapture(event.pointerId);
    updateViewerCursor(point);
    state.activeEditBaseline = captureMaskData();
    state.activeBaseEditBaseline = captureBaseMaskData();
    pushUndo();
    state.drawing = true;
    state.activeStrokeMode = strokeMode;
    state.activePointerButton = event.button;
    state.lastPoint = point;
    drawMaskLine(point, point, strokeMode, baseMaskCtx);
    rebuildMaskFromBase({ recordProtection: false, reason: `${strokeMode}_preview` });
    return;
  }
  if (event.button !== 0) return;
  if (state.tool === 'fill') {
    event.preventDefault();
    fillAtPoint(point).catch((err) => setStatus(`Fill failed: ${err.message}`, true));
    return;
  }
  viewer.setPointerCapture(event.pointerId);
  if (state.tool === 'polygon') {
    const index = nearestPolygonPoint(point);
    if (state.polygon.points.length >= 3 && index === 0) {
      addPolygonShape(state.polygon.points);
      return;
    }
    if (event.altKey && removePolygonPoint(index)) return;
    if (state.polygon.points.length > 0 && index !== null) {
      state.polygon.dragIndex = index;
      return;
    }
    if (state.polygon.points.length > 0) {
      const insertAt = nearestPolygonSegment(point);
      if (insertAt !== null) {
        state.polygon.points.splice(insertAt, 0, point);
        setStatus('Polygon point inserted.');
      } else {
        state.polygon.points.push(point);
        setStatus('Polygon point added.');
      }
      draw();
      return;
    }
    const shapePoint = hitPolygonShapePoint(point);
    if (shapePoint) {
      pushUndo();
      state.activeShapeId = shapePoint.shape.id;
      state.shapeDrag = { shapeId: shapePoint.shape.id, mode: 'point', index: shapePoint.index, changed: false };
      return;
    }
    const shapeSegment = hitPolygonShapeSegment(point);
    if (shapeSegment) {
      pushUndo();
      const insertAt = shapeSegment.insertAt % shapeSegment.shape.points.length;
      shapeSegment.shape.points.splice(insertAt, 0, point);
      state.activeShapeId = shapeSegment.shape.id;
      state.shapeDrag = { shapeId: shapeSegment.shape.id, mode: 'point', index: insertAt, changed: true };
      rebuildMaskFromBase({ recordProtection: false, reason: 'polygon_preview' });
      return;
    }
    const shapeBody = hitPolygonShapeBody(point);
    if (shapeBody) {
      pushUndo();
      state.activeShapeId = shapeBody.id;
      state.shapeDrag = { shapeId: shapeBody.id, mode: 'move', lastPoint: point, changed: false };
      return;
    }
    state.polygon.points.push(point);
    setStatus('Polygon point added.');
    draw();
    return;
  }
  if (state.tool === 'rectangle') {
    if (state.rect.active) {
      state.rect.x2 = point.x;
      state.rect.y2 = point.y;
      const added = addRectangleShape(state.rect);
      state.rect.active = false;
      state.rect.handle = null;
      state.rect.startPoint = null;
      state.rect.dragMoved = false;
      if (!added) {
        draw();
        setStatus('Rectangle is too small; click an opposite corner farther away or right-click to cancel.', true);
      }
      return;
    }
    const shapeHit = hitRectangleShape(point);
    if (shapeHit) {
      pushUndo();
      state.activeShapeId = shapeHit.shape.id;
      state.shapeDrag = { shapeId: shapeHit.shape.id, handle: shapeHit.handle, lastPoint: point, changed: false };
      return;
    }
    state.rect.handle = 'draw';
    state.rect.lastPoint = point;
    state.rect.startPoint = point;
    state.rect.dragMoved = false;
    state.rect.active = true;
    state.rect.x1 = point.x;
    state.rect.y1 = point.y;
    state.rect.x2 = point.x;
    state.rect.y2 = point.y;
    state.activeShapeId = null;
    draw();
    return;
  }
  if (state.tool === 'sam2') {
    if (els.sam2PromptMode.value === 'rectangle_xyxy') {
      clearSam2Preview({ redraw: false });
      state.samBox = { active: true, x1: point.x, y1: point.y, x2: point.x, y2: point.y };
      draw();
    } else {
      scheduleSam2PointHoverPreview(point);
      draw();
      setStatus('Hold still for SAM2 point preview, then press Apply SAM2.');
    }
  }
});

viewer.addEventListener('pointermove', (event) => {
  if (state.viewPan.active) {
    updateViewPan(event);
    return;
  }
  if (!state.sample || !state.sample.editable) return;
  const point = imagePointFromEvent(event);
  updateViewerCursor(point);
  if (state.tool === 'brush' || state.tool === 'sam2') {
    state.hoverPoint = point;
    if (sam2PointModeActive()) scheduleSam2PointHoverPreview(point);
    else if (state.tool !== 'sam2') updateSam2ApplyButton();
  }
  if (state.shapeDrag) {
    updateShapeDrag(point);
    return;
  }
  if (state.drawing && state.lastPoint) {
    drawMaskLine(state.lastPoint, point, state.activeStrokeMode || state.tool, baseMaskCtx);
    state.lastPoint = point;
    rebuildMaskFromBase({ recordProtection: false, reason: `${state.activeStrokeMode || state.tool}_preview` });
    return;
  }
  if (state.tool === 'polygon' && state.polygon.dragIndex !== null) {
    state.polygon.points[state.polygon.dragIndex] = point;
    draw();
    return;
  }
  if (state.tool === 'rectangle' && state.rect.active) {
    state.rect.x2 = point.x;
    state.rect.y2 = point.y;
    if (state.rect.startPoint && event.buttons === 1) {
      const moved = Math.hypot(point.x - state.rect.startPoint.x, point.y - state.rect.startPoint.y);
      if (moved >= Math.max(3 / state.zoom, 2)) state.rect.dragMoved = true;
    }
    draw();
    return;
  }
  if (state.tool === 'sam2' && state.samBox) {
    state.samBox.x2 = point.x;
    state.samBox.y2 = point.y;
    draw();
    return;
  }
  if (state.tool === 'brush') {
    draw();
  } else if (state.tool === 'sam2') {
    updateSam2ApplyButton();
    draw();
  }
});

viewer.addEventListener('pointerleave', () => {
  if (state.viewPan.active) return;
  clearSam2Preview({ redraw: false });
  if (!state.hoverPoint) {
    updateSam2ApplyButton();
    return;
  }
  state.hoverPoint = null;
  updateViewerCursor(null);
  updateSam2ApplyButton();
  draw();
});

viewer.addEventListener('pointerup', (event) => {
  if (state.viewPan.active) {
    finishViewPan(event);
    return;
  }
  if (state.shapeDrag) {
    finishShapeDrag();
  }
  if (state.drawing) {
    const editType = state.activeStrokeMode || state.tool;
    state.drawing = false;
    state.lastPoint = null;
    state.edits.push({
      type: editType,
      source_tool: state.tool,
      mouse_button: state.activePointerButton,
      brush_size: Number(els.brushSize.value),
      at: new Date().toISOString()
    });
    afterMaskEdit(editType, state.activeEditBaseline, { baseBaselineData: state.activeBaseEditBaseline });
    state.activeEditBaseline = null;
    state.activeBaseEditBaseline = null;
    state.activeStrokeMode = null;
    state.activePointerButton = 0;
  }
  if (state.tool === 'polygon') state.polygon.dragIndex = null;
  if (state.tool === 'rectangle' && state.rect.active && state.rect.handle === 'draw' && state.rect.dragMoved) {
    const added = addRectangleShape(state.rect);
    state.rect.active = false;
    state.rect.handle = null;
    state.rect.startPoint = null;
    state.rect.dragMoved = false;
    if (!added) draw();
  } else if (state.tool === 'rectangle' && state.rect.active && state.rect.handle === 'draw') {
    state.rect.lastPoint = null;
    setStatus('Rectangle first corner set; click the opposite corner to finish or right-click to cancel.');
    draw();
  }
  if (state.tool === 'sam2' && state.samBox) {
    const r = normalizedRect(state.samBox);
    const prompt = { type: 'rectangle_xyxy', x1: Math.round(r.x1), y1: Math.round(r.y1), x2: Math.round(r.x2), y2: Math.round(r.y2) };
    state.samBox = null;
    if (prompt.x2 - prompt.x1 > 4 && prompt.y2 - prompt.y1 > 4) {
      runSam2(prompt).catch((err) => setStatus(`SAM2 failed: ${err.message}`, true));
    } else {
      draw();
    }
  }
  updateViewerCursor(state.hoverPoint);
});

viewer.addEventListener('pointercancel', (event) => {
  finishViewPan(event);
});

document.querySelectorAll('.tool-button').forEach((button) => {
  button.addEventListener('click', () => {
    selectTool(button.dataset.tool);
  });
});

els.searchBox.addEventListener('input', renderQueue);
els.filterSelect.addEventListener('change', renderQueue);
els.brushSize.addEventListener('input', () => {
  els.brushSizeValue.textContent = `${els.brushSize.value} px`;
  if (state.hoverPoint) draw();
});
els.brightnessThreshold.addEventListener('input', () => {
  resetBrightnessPreviewCache();
  updateBrightnessThresholdUi(true);
  drawWithAvailabilityStatus();
});
els.brightnessThreshold90Btn.addEventListener('click', () => setBrightnessThreshold(90));
els.brightnessThresholdOffBtn.addEventListener('click', () => setBrightnessThreshold(255));
els.sam2PromptMode.addEventListener('change', () => {
  clearSam2Preview({ redraw: false });
  updateSam2ApplyButton();
  if (state.tool === 'sam2') draw();
});
els.zoomInBtn.addEventListener('click', () => zoomBy(ZOOM_STEP));
els.zoomOutBtn.addEventListener('click', () => zoomBy(1 / ZOOM_STEP));
els.themeSelect.addEventListener('change', () => applyTheme(els.themeSelect.value));
els.subtractSulfidesBtn.addEventListener('click', () => {
  subtractSulfidesFromMask().catch((err) => setStatus(`Sulfide subtraction failed: ${err.message}`, true));
});
els.fitBtn.addEventListener('click', fitToViewer);
els.undoBtn.addEventListener('click', () => undo().catch((err) => setStatus(`Undo failed: ${err.message}`, true)));
els.saveBtn.addEventListener('click', () => saveReview(false).catch((err) => setStatus(`Save failed: ${err.message}`, true)));
els.saveNextBtn.addEventListener('click', () => saveReview(true).catch((err) => setStatus(`Save failed: ${err.message}`, true)));
els.resetBtn.addEventListener('click', () => resetCurrent().catch((err) => setStatus(`Reset failed: ${err.message}`, true)));
els.sam2ApplyBtn.addEventListener('click', () => {
  applySam2PointPreviewOrRun().catch((err) => setStatus(`SAM2 apply failed: ${err.message}`, true));
});
els.sam2StatusBtn.addEventListener('click', async () => {
  try {
    const status = await apiGet('/api/sam2/status?check_load=1');
    setStatus(status.available ? `SAM2 loaded on ${status.device}.` : `SAM2 unavailable: ${status.load_error || 'missing optional dependency'}`, !status.available);
  } catch (err) {
    setStatus(`SAM2 status failed: ${err.message}`, true);
  }
});
els.baseMode.addEventListener('change', drawWithAvailabilityStatus);
Object.values(els.layers).forEach((layer) => layer.addEventListener('change', drawWithAvailabilityStatus));
if (window.matchMedia) {
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (els.themeSelect.value === 'system') draw();
  });
}
function isTextEditingTarget(target) {
  if (!target) return false;
  const tag = target.tagName;
  return target.isContentEditable || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
}

window.addEventListener('keydown', (event) => {
  const key = event.key.toLowerCase();
  const shortcutAllowed = !isTextEditingTarget(event.target) && !event.metaKey && !event.ctrlKey && !event.altKey;
  if (shortcutAllowed && (key === 'b' || key === 'f')) {
    const tool = key === 'b' ? 'brush' : 'fill';
    if (selectTool(tool, { shortcut: key.toUpperCase() })) {
      event.preventDefault();
      return;
    }
  }
  if ((event.key === 'Delete' || event.key === 'Backspace') && !isTextEditingTarget(event.target)) {
    if (deleteSelectedShape()) {
      event.preventDefault();
      return;
    }
  }
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'z') {
    event.preventDefault();
    undo().catch((err) => setStatus(`Undo failed: ${err.message}`, true));
  }
});

async function loadManifest(loadFirst) {
  state.manifest = await apiGet('/api/manifest');
  state.samples = state.manifest.samples || [];
  renderQueue();
  if (loadFirst && state.samples.length > 0) await loadSample(state.samples[0].sample_id);
}

async function loadSample(sampleId, options = {}) {
  if (!options.force && !canLeaveCurrentSample(sampleId)) return;
  setStatus('Loading sample...');
  emptyState.classList.remove('hidden');
  state.assetErrors = [];
  renderAssetWarnings();
  state.sample = await apiGet(`/api/samples/${encodeURIComponent(sampleId)}`);
  state.sampleId = sampleId;
  state.imageW = Number(state.sample.image.width);
  state.imageH = Number(state.sample.image.height);
  viewer.width = state.imageW;
  viewer.height = state.imageH;
  maskCanvas.width = state.imageW;
  maskCanvas.height = state.imageH;
  baseMaskCanvas.width = state.imageW;
  baseMaskCanvas.height = state.imageH;
  currentTintCanvas.width = state.imageW;
  currentTintCanvas.height = state.imageH;

  const urls = state.sample.urls;
  const [
    original, annotated, qa, currentMask, autoMask, rawLines, closedLines, overlapMask, ignoreMask, sulfideMask
  ] = await Promise.all([
    loadImage(urls.original, urls.original ? 'Original photo' : null),
    loadImage(urls.annotated || urls.source_copy, (urls.annotated || urls.source_copy) ? 'MS Paint annotation' : null),
    loadImage(urls.qa_overlay, urls.qa_overlay ? 'Converter QA overlay' : null),
    loadImage(urls.current_mask, urls.current_mask ? 'Current working mask' : null),
    loadImage(urls.autodetected_mask, urls.autodetected_mask ? 'Autodetected mask' : null),
    loadImage(urls.raw_blue_stroke, urls.raw_blue_stroke ? 'Original blue lines' : null),
    loadImage(urls.closed_blue_stroke, urls.closed_blue_stroke ? 'Closed blue line boundary' : null),
    loadImage(urls.sulfide_overlap, urls.sulfide_overlap ? 'Sulfide overlap mask' : null),
    loadImage(urls.ignore_mask, urls.ignore_mask ? 'Ignore/uncertain mask' : null),
    loadImage(urls.sulfide_mask, urls.sulfide_mask ? 'Sulfide mask' : null)
  ]);
  if (!currentMask) {
    state.assetErrors.push('Current working mask is unavailable; edits are disabled until it loads');
    state.sample.editable = false;
  }
  if (!original && !annotated) state.assetErrors.push('No display image is available for this sample');
  renderAssetWarnings();
  state.images = { original, annotated, qa, sulfideMask };
  prepareFillBoundaries(rawLines, closedLines);
  resetBrightnessPreviewCache();
  baseMaskCtx.clearRect(0, 0, state.imageW, state.imageH);
  maskCtx.clearRect(0, 0, state.imageW, state.imageH);
  if (currentMask) {
    baseMaskCtx.drawImage(currentMask, 0, 0, state.imageW, state.imageH);
    maskCtx.drawImage(currentMask, 0, 0, state.imageW, state.imageH);
  }
  sulfideGuardCanvas.width = state.imageW;
  sulfideGuardCanvas.height = state.imageH;
  sulfideGuardCtx.clearRect(0, 0, state.imageW, state.imageH);
  state.sulfideGuardLoaded = Boolean(sulfideMask);
  if (sulfideMask) sulfideGuardCtx.drawImage(sulfideMask, 0, 0, state.imageW, state.imageH);
  state.staticTints = {
    auto: buildTintFromImage(autoMask, [47, 120, 255, 90]),
    lines: buildTintFromImage(rawLines, [20, 40, 255, 170]),
    overlap: buildTintFromImage(overlapMask, [255, 85, 30, 140]),
    ignore: buildTintFromImage(ignoreMask, [255, 214, 10, 110])
  };
  refreshCurrentTint();
  state.undoStack = [];
  state.edits = [];
  state.shapes = [];
  state.nextShapeId = 1;
  state.activeShapeId = null;
  state.shapeDrag = null;
  state.dirty = false;
  state.saveState = state.sample.sample.review_state === 'reviewed' ? 'reviewed' : 'saved';
  state.lastSavedAt = null;
  state.polygon.points = [];
  state.polygon.dragIndex = null;
  state.rect.active = false;
  state.rect.handle = null;
  state.rect.lastPoint = null;
  state.rect.startPoint = null;
  state.rect.dragMoved = false;
  state.drawing = false;
  state.lastPoint = null;
  state.hoverPoint = null;
  state.activeStrokeMode = null;
  state.activePointerButton = 0;
  state.activeEditBaseline = null;
  state.activeBaseEditBaseline = null;
  state.samBox = null;
  clearSam2Preview({ redraw: false });

  els.sampleTitle.textContent = state.sample.image.name;
  els.sampleSubtitle.textContent = `${statusLabel(state.sample.sample.status)} · ${reviewStateLabel(state.sample.sample.review_state)} · ${state.imageW} x ${state.imageH}`;
  emptyState.classList.add('hidden');
  fitToViewer();
  updateMetrics();
  draw();
  renderQueue();
  updateViewerCursor();
  setStatus(state.sample.editable ? 'Editing current talc mask.' : 'Original image is missing; editing disabled for this sample.', !state.sample.editable);
}

applyTheme(localStorage.getItem(THEME_STORAGE_KEY) || 'system', false);
setBrightnessThreshold(localStorage.getItem(BRIGHTNESS_THRESHOLD_STORAGE_KEY) || 255, false);
updateToolParams();

loadManifest(true).catch((err) => {
  emptyState.textContent = `Failed to start: ${err.message}`;
  setStatus(err.message, true);
});
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local talc mask review app with browser canvas editing.")
    parser.add_argument("--annotated-dir", type=Path, default=None, help="Folder with MS Paint talc annotations.")
    parser.add_argument("--original-dir", type=Path, default=None, help="Folder with clean originals; defaults to parent of annotated dir.")
    parser.add_argument("--workspace-dir", type=Path, default=DEFAULT_WORKSPACE_DIR, help="Conversion/review workspace.")
    parser.add_argument("--conversion-dir", type=Path, default=None, help="Prepared conversion workspace containing manifest.json.")
    parser.add_argument("--sulfide-mask-dir", type=Path, default=None, help="Optional sulfide masks by image stem for conversion.")
    parser.add_argument("--silicate-mask-dir", type=Path, default=None, help="Optional silicate support masks by image stem for conversion.")
    parser.add_argument("--reconvert", action="store_true", help="Regenerate conversion workspace before starting.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of converted samples for debugging.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=0, help="Bind port; 0 asks the OS for a free port.")
    parser.add_argument("--sam2-model-id", default=DEFAULT_SAM2_MODEL_ID, help="SAM2 model id for optional assist.")
    parser.add_argument("--sam2-device", default="auto", help="SAM2 device: auto, cpu, mps, cuda.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    annotated_dir = args.annotated_dir
    if annotated_dir is None and args.conversion_dir is None:
        annotated_dir = DEFAULT_ANNOTATED_DIR
    original_dir = args.original_dir
    if original_dir is None and annotated_dir is not None:
        original_dir = annotated_dir.parent
    try:
        store = TalcReviewStore(
            annotated_dir=annotated_dir,
            original_dir=original_dir,
            workspace_dir=args.workspace_dir,
            conversion_dir=args.conversion_dir,
            sulfide_mask_dir=args.sulfide_mask_dir,
            silicate_mask_dir=args.silicate_mask_dir,
            reconvert=args.reconvert,
            limit=args.limit,
            sam2_model_id=args.sam2_model_id,
            sam2_device=args.sam2_device,
        )
    except ApiError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 2
    server = TalcReviewHTTPServer((args.host, args.port), store)
    host, port = server.server_address[:2]
    print(f"Talc review app: http://{host}:{port}/", flush=True)
    print(f"Workspace: {store.workspace_dir}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
