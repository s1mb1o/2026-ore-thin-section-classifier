#!/usr/bin/env python3
"""Grain review / labeling app (path B, stage 2).

A local browser app to human-classify sulfide grains as ordinary vs fine
intergrowth (or 'uncertain'). It shows a paginated grid of grain crops produced
by `scripts/build_grain_dataset.py`, pre-labelled by the heuristic, and persists
human corrections to `annotations.json` in the dataset directory, keyed by
`grain_uid`. Those annotations feed `train_grain_classifier.py` /
`aggregate_grade_from_grains.py`.

Stdlib only (http.server), no framework — same architecture as
`apps/talc_review_web.py`. Keyboard in grid mode: O=ordinary, F=fine,
U=uncertain, arrows to move. Tinder mode: left=fine, right=ordinary, up=postpone,
down=uncertain. Run:

    python3 apps/grain_review_web.py --dataset-dir outputs/grain_dataset_v0 --port 0
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import threading
from collections import deque
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

GRAIN_CLASSES = ["ordinary_intergrowth", "fine_intergrowth", "uncertain"]
MAX_POST_BYTES = 8 * 1024 * 1024
SORT_VALUES = {"manifest", "review_value"}
SMALL_AREA_QUANTILE = 0.20

# Full morphology feature set carried per grain in grains_manifest.csv, surfaced
# in the labeling UI so the annotator can decide ordinary vs fine from the same
# numbers the v2 pipeline reports.
FEATURE_FIELDS = [
    "area_px",
    "footprint_area_px",
    "dark_inside_area_px",
    "dark_inside_ratio",
    "solidity",
    "compactness",
    "boundary_complexity",
    "bbox_w",
    "bbox_h",
]
BBOX_FIELDS = ("bbox_x", "bbox_y", "bbox_w", "bbox_h")

# Heuristic "fine" thresholds — mirror ComponentRuleConfig defaults
# (src/ore_classifier/component_analysis.py). A grain is pre-labelled fine if ANY
# of these trip. Shown to the annotator as the reason behind the pre-label.
FINE_DARK_INSIDE_RATIO = 0.18
FINE_SOLIDITY_MAX = 0.62
FINE_COMPACTNESS_MAX = 0.12


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ApiError(RuntimeError):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class GrainReviewStore:
    """Shared, thread-safe state: the grain manifest + the annotations file."""

    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir.resolve()
        self.crops_root = (self.dataset_dir / "crops").resolve()
        self.manifest_path = self.dataset_dir / "grains_manifest.csv"
        self.annotations_path = self.dataset_dir / "annotations.json"
        self.summary_path = self.dataset_dir / "dataset_summary.json"
        if not self.manifest_path.exists():
            raise SystemExit(f"grains_manifest.csv not found in {self.dataset_dir}")
        self.lock = threading.RLock()
        self.dataset_summary = self._load_dataset_summary()
        self.grains: list[dict[str, str]] = self._load_manifest()
        self.small_area_threshold_px = self._area_quantile(SMALL_AREA_QUANTILE)
        self.index_by_uid = {g["grain_uid"]: i for i, g in enumerate(self.grains)}
        self.labels: dict[str, dict[str, Any]] = self._load_annotations()

    def _load_manifest(self) -> list[dict[str, str]]:
        with self.manifest_path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def _load_dataset_summary(self) -> dict[str, Any]:
        if not self.summary_path.exists():
            return {}
        payload = json.loads(self.summary_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}

    def _area_quantile(self, fraction: float) -> float:
        areas = sorted(area for g in self.grains if (area := _num(g.get("area_px"))) is not None and area > 0)
        if not areas:
            return 0.0
        index = round((len(areas) - 1) * max(0.0, min(1.0, fraction)))
        return float(areas[index])

    def _load_annotations(self) -> dict[str, dict[str, Any]]:
        if not self.annotations_path.exists():
            return {}
        payload = json.loads(self.annotations_path.read_text(encoding="utf-8"))
        labels = payload.get("labels", {}) if isinstance(payload, dict) else {}
        return {k: v for k, v in labels.items() if isinstance(v, dict)}

    def _save_annotations(self) -> None:
        payload = {
            "schema_version": "grain-annotations-v0.1",
            "updated_at": utc_now_iso(),
            "dataset_dir": str(self.dataset_dir),
            "labels": self.labels,
        }
        tmp = self.annotations_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.annotations_path)

    def stats(self) -> dict[str, Any]:
        counts = {cls: 0 for cls in GRAIN_CLASSES}
        for entry in self.labels.values():
            label = entry.get("label")
            if label in counts:
                counts[label] += 1
        return {"total": len(self.grains), "labeled": len(self.labels), "counts": counts}

    def page(
        self, *, offset: int, limit: int, grade: str, view: str, sort: str = "manifest", focus: str = ""
    ) -> dict[str, Any]:
        if sort not in SORT_VALUES:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"bad sort {sort}")
        with self.lock:
            filtered = self._filter(grade=grade, view=view)
            if sort == "review_value":
                filtered = sorted(filtered, key=self._review_sort_key)
            focus_found = False
            if focus:
                for pos, index in enumerate(filtered):
                    if self.grains[index].get("grain_uid") == focus:
                        offset = (pos // limit) * limit
                        focus_found = True
                        break
            window = filtered[offset : offset + limit]
            items = [self._item_payload(i) for i in window]
            return {
                "items": items,
                "offset": offset,
                "limit": limit,
                "filtered_total": len(filtered),
                "sort": sort,
                "focus": focus,
                "focus_found": focus_found,
                "stats": self.stats(),
            }

    def _filter(self, *, grade: str, view: str) -> list[int]:
        result = []
        for i, g in enumerate(self.grains):
            if grade not in ("", "all") and g.get("grade_label") != grade:
                continue
            labeled = g["grain_uid"] in self.labels
            if view == "unlabeled" and labeled:
                continue
            if view == "labeled" and not labeled:
                continue
            result.append(i)
        return result

    def _fine_signal_state(
        self, g: dict[str, str]
    ) -> tuple[dict[str, float | None], dict[str, bool], int, bool]:
        features = {key: _num(g.get(key)) for key in FEATURE_FIELDS}
        dir_ = features.get("dark_inside_ratio")
        sol = features.get("solidity")
        cmp_ = features.get("compactness")
        fine_signals = {
            "dark_inside_ratio": dir_ is not None and dir_ >= FINE_DARK_INSIDE_RATIO,
            "solidity": sol is not None and sol <= FINE_SOLIDITY_MAX,
            "compactness": cmp_ is not None and cmp_ <= FINE_COMPACTNESS_MAX,
        }
        fine_vote_count = sum(1 for matched in fine_signals.values() if matched)
        boundary_only_fine = (not fine_signals["dark_inside_ratio"]) and (
            fine_signals["solidity"] or fine_signals["compactness"]
        )
        return features, fine_signals, fine_vote_count, boundary_only_fine

    def _review_sort_key(self, index: int) -> tuple[int, int]:
        g = self.grains[index]
        features, fine_signals, fine_vote_count, boundary_only_fine = self._fine_signal_state(g)
        small_fine_context = self._small_fine_context(g, features, fine_signals)
        value = self._review_value_payload(
            g, features, fine_signals, fine_vote_count, boundary_only_fine, small_fine_context
        )
        return (-int(value["score"]), index)

    def _small_fine_context(
        self, g: dict[str, str], features: dict[str, float | None], fine_signals: dict[str, bool]
    ) -> dict[str, Any]:
        area = features.get("area_px") or 0.0
        ordinary_shape = not fine_signals["solidity"] and not fine_signals["compactness"]
        matched = (
            g.get("grade_label") == "fine_intergrowth"
            and self.small_area_threshold_px > 0
            and area > 0
            and area <= self.small_area_threshold_px
            and ordinary_shape
        )
        return {
            "matched": matched,
            "area_px": area,
            "threshold_px": self.small_area_threshold_px,
            "ordinary_shape": ordinary_shape,
            "quantile": SMALL_AREA_QUANTILE,
        }

    def _review_value_payload(
        self,
        g: dict[str, str],
        features: dict[str, float | None],
        fine_signals: dict[str, bool],
        fine_vote_count: int,
        boundary_only_fine: bool,
        small_fine_context: dict[str, Any],
    ) -> dict[str, Any]:
        uid = g["grain_uid"]
        label = self.labels.get(uid, {}).get("label")
        status_component = 40.0 if label is None else (25.0 if label == "uncertain" else 0.0)
        total_votes = max(len(fine_signals), 1)
        fine_ratio = fine_vote_count / total_votes
        ambiguity_component = 25.0 * max(0.0, 1.0 - abs(fine_ratio - 0.5) / 0.5)

        def closeness(value: float | None, threshold: float, span: float) -> float:
            if value is None:
                return 0.0
            return max(0.0, 1.0 - abs(value - threshold) / span)

        threshold_values = [
            closeness(features.get("dark_inside_ratio"), FINE_DARK_INSIDE_RATIO, FINE_DARK_INSIDE_RATIO),
            closeness(features.get("solidity"), FINE_SOLIDITY_MAX, 0.25),
            closeness(features.get("compactness"), FINE_COMPACTNESS_MAX, FINE_COMPACTNESS_MAX),
        ]
        threshold_component = 20.0 * (sum(threshold_values) / len(threshold_values))
        boundary_component = 15.0 if boundary_only_fine else 0.0
        small_context_component = 20.0 if small_fine_context.get("matched") else 0.0
        area = max(0.0, features.get("area_px") or 0.0)
        impact_component = 10.0 * min(1.0, math.log1p(area) / math.log1p(5000.0))
        score = min(
            100.0,
            status_component
            + ambiguity_component
            + threshold_component
            + boundary_component
            + small_context_component
            + impact_component,
        )

        reasons = []
        if label is None:
            reasons.append("без метки")
        elif label == "uncertain":
            reasons.append("не уверен")
        if ambiguity_component >= 10.0:
            reasons.append("спорные признаки")
        if boundary_only_fine:
            reasons.append("тонкое только по границе")
        if small_fine_context.get("matched"):
            reasons.append("мелкое в тонких")
        if threshold_component >= 10.0:
            reasons.append("близко к порогам")
        if impact_component >= 7.0:
            reasons.append("заметная площадь")
        if not reasons:
            reasons.append("низкий приоритет")
        return {
            "score": int(round(score)),
            "status": int(round(status_component)),
            "ambiguity": int(round(ambiguity_component)),
            "threshold": int(round(threshold_component)),
            "boundary": int(round(boundary_component)),
            "small_context": int(round(small_context_component)),
            "impact": int(round(impact_component)),
            "reasons": reasons[:4],
        }

    def _item_payload(self, index: int) -> dict[str, Any]:
        g = self.grains[index]
        uid = g["grain_uid"]
        features, fine_signals, fine_vote_count, boundary_only_fine = self._fine_signal_state(g)
        small_fine_context = self._small_fine_context(g, features, fine_signals)
        bbox_values = {key: _num(g.get(key)) for key in BBOX_FIELDS}
        bbox = {
            "x": bbox_values["bbox_x"],
            "y": bbox_values["bbox_y"],
            "w": bbox_values["bbox_w"],
            "h": bbox_values["bbox_h"],
        }
        source_path = str(g.get("source_dataset_path", "") or "")
        dir_ = features.get("dark_inside_ratio")
        sol = features.get("solidity")
        cmp_ = features.get("compactness")
        total_votes = len(fine_signals)
        fine_score = round(100 * fine_vote_count / total_votes) if total_votes else 0
        ordinary_score = 100 - fine_score
        reasons: list[str] = []
        if fine_signals["dark_inside_ratio"]:
            reasons.append(f"тёмное внутри {dir_:.2f} ≥ {FINE_DARK_INSIDE_RATIO}")
        if fine_signals["solidity"]:
            reasons.append(f"выпуклость {sol:.2f} ≤ {FINE_SOLIDITY_MAX}")
        if fine_signals["compactness"]:
            reasons.append(f"компактность {cmp_:.3f} ≤ {FINE_COMPACTNESS_MAX}")
        heuristic_rows = [
            {
                "label": "Тёмное внутри",
                "value": dir_,
                "digits": 2,
                "rule": f"≥ {FINE_DARK_INSIDE_RATIO}",
                "matched": fine_signals["dark_inside_ratio"],
            },
            {
                "label": "Выпуклость",
                "value": sol,
                "digits": 2,
                "rule": f"≤ {FINE_SOLIDITY_MAX}",
                "matched": fine_signals["solidity"],
            },
            {
                "label": "Компактность",
                "value": cmp_,
                "digits": 3,
                "rule": f"≤ {FINE_COMPACTNESS_MAX}",
                "matched": fine_signals["compactness"],
            },
        ]
        # Ambiguous case: "fine" is driven ONLY by the boundary (low solidity/
        # compactness) while the interior shows no replacement (dark_inside_ratio
        # < threshold) — a massive homogeneous grain with a merely ragged contour,
        # which is likely NOT труднообогатимое. Flag it for the annotator.
        review_value = self._review_value_payload(
            g, features, fine_signals, fine_vote_count, boundary_only_fine, small_fine_context
        )
        return {
            "grain_uid": uid,
            "crop_url": "/crops/" + quote(g["crop_path"].split("crops/", 1)[-1]),
            "source_url": "/source/" + quote(uid) if source_path else None,
            "contour_url": "/contours/" + quote(uid) if g.get("run_id") and g.get("component_id") else None,
            "image_rel_path": g.get("image_rel_path", ""),
            "source_name": Path(source_path).name if source_path else "",
            "bbox": bbox,
            "grade_label": g.get("grade_label", ""),
            "heuristic_label": g.get("heuristic_label", ""),
            "features": features,
            "fine_signals": fine_signals,
            "fine_reasons": reasons,
            "heuristic_rows": heuristic_rows,
            "heuristic_scores": {
                "ordinary": ordinary_score,
                "fine": fine_score,
                "fine_votes": fine_vote_count,
                "total_votes": total_votes,
            },
            "boundary_only_fine": boundary_only_fine,
            "small_fine_context": small_fine_context,
            "review_value": review_value,
            "label": self.labels.get(uid, {}).get("label"),
        }

    def annotate(self, grain_uid: str, label: str | None) -> dict[str, Any]:
        if grain_uid not in self.index_by_uid:
            raise ApiError(HTTPStatus.NOT_FOUND, f"unknown grain_uid {grain_uid}")
        if label is not None and label not in GRAIN_CLASSES:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"bad label {label}")
        with self.lock:
            if label is None:
                self.labels.pop(grain_uid, None)
            else:
                self.labels[grain_uid] = {"label": label, "at": utc_now_iso()}
            self._save_annotations()
            return self.stats()

    def crop_file(self, rel: str) -> Path:
        candidate = (self.crops_root / rel).resolve()
        # is_relative_to enforces a real path-boundary; a bare startswith would let
        # a sibling dir sharing the 'crops' name prefix (crops_backup/, crops2/…) escape.
        if not candidate.is_relative_to(self.crops_root) or not candidate.is_file():
            raise ApiError(HTTPStatus.NOT_FOUND, "crop not found")
        return candidate

    def source_file(self, grain_uid: str) -> Path:
        if grain_uid not in self.index_by_uid:
            raise ApiError(HTTPStatus.NOT_FOUND, "source image not found")
        row = self.grains[self.index_by_uid[grain_uid]]
        raw_path = str(row.get("source_dataset_path", "") or "")
        if not raw_path:
            raise ApiError(HTTPStatus.NOT_FOUND, "source image not found")
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = self.dataset_dir / candidate
        candidate = candidate.resolve()
        # The only accepted key is a grain UID already present in the manifest;
        # the request never accepts arbitrary filesystem paths from the browser.
        if not candidate.is_file():
            raise ApiError(HTTPStatus.NOT_FOUND, "source image not found")
        return candidate

    def contour_png(self, grain_uid: str) -> bytes:
        try:
            from PIL import Image, ImageChops, ImageFilter
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise ApiError(HTTPStatus.NOT_FOUND, "Pillow is required for contour overlays") from exc
        if grain_uid not in self.index_by_uid:
            raise ApiError(HTTPStatus.NOT_FOUND, "contour not found")
        row = self.grains[self.index_by_uid[grain_uid]]
        crop_path = self.crop_file(str(row.get("crop_path", "")).split("crops/", 1)[-1])
        with Image.open(crop_path) as crop_image:
            crop_size = crop_image.size
        mask_path = self._sulfide_mask_path(row)
        if mask_path is None:
            raise ApiError(HTTPStatus.NOT_FOUND, "contour source mask not found")
        try:
            x = int(float(row["bbox_x"]))
            y = int(float(row["bbox_y"]))
            w = int(float(row["bbox_w"]))
            h = int(float(row["bbox_h"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiError(HTTPStatus.NOT_FOUND, "contour bbox not found") from exc
        pad = int(self.dataset_summary.get("params", {}).get("crop_pad_px", 10) or 10)
        with Image.open(mask_path) as mask_image:
            mask = mask_image.convert("L")
            left = max(0, x - pad)
            top = max(0, y - pad)
            right = min(mask.width, x + w + pad)
            bottom = min(mask.height, y + h + pad)
            if right <= left or bottom <= top:
                raise ApiError(HTTPStatus.NOT_FOUND, "contour crop is empty")
            mask_crop = mask.crop((left, top, right, bottom))
        component = self._selected_component_mask(mask_crop, row, left=left, top=top)
        if component.size != crop_size:
            component = component.resize(crop_size, Image.Resampling.NEAREST)
        eroded = component.filter(ImageFilter.MinFilter(3))
        boundary = ImageChops.subtract(component, eroded)
        boundary = boundary.filter(ImageFilter.MaxFilter(3))
        overlay = Image.new("RGBA", component.size, (0, 255, 191, 0))
        overlay.putalpha(boundary)
        out = io.BytesIO()
        overlay.save(out, format="PNG")
        return out.getvalue()

    def _sulfide_mask_path(self, row: dict[str, str]) -> Path | None:
        batch_raw = str(self.dataset_summary.get("batch_dir", "") or "")
        if not batch_raw:
            return None
        batch_dir = Path(batch_raw)
        if not batch_dir.is_absolute():
            batch_dir = (Path.cwd() / batch_dir).resolve()
        run_id = str(row.get("run_id", "") or "")
        grade = str(row.get("grade_label", "") or "")
        candidates = [
            batch_dir / "runs" / grade / run_id / "binary_sulfide" / "sulfide_mask.png",
        ]
        candidates.extend(batch_dir.glob(f"runs/*/{run_id}/binary_sulfide/sulfide_mask.png"))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def _selected_component_mask(self, mask_crop: Any, row: dict[str, str], *, left: int, top: int) -> Any:
        from PIL import Image

        width, height = mask_crop.size
        px = mask_crop.load()
        try:
            cx = int(round(float(row.get("centroid_x", "")) - left))
            cy = int(round(float(row.get("centroid_y", "")) - top))
        except (TypeError, ValueError):
            cx, cy = width // 2, height // 2
        start = self._nearest_mask_pixel(px, width, height, cx, cy)
        component = Image.new("L", (width, height), 0)
        if start is None:
            return component
        out = component.load()
        queue: deque[tuple[int, int]] = deque([start])
        seen = {start}
        while queue:
            x, y = queue.popleft()
            out[x, y] = 255
            for nx in (x - 1, x, x + 1):
                for ny in (y - 1, y, y + 1):
                    if nx == x and ny == y:
                        continue
                    if nx < 0 or ny < 0 or nx >= width or ny >= height or (nx, ny) in seen:
                        continue
                    seen.add((nx, ny))
                    if px[nx, ny] > 0:
                        queue.append((nx, ny))
        return component

    @staticmethod
    def _nearest_mask_pixel(px: Any, width: int, height: int, cx: int, cy: int) -> tuple[int, int] | None:
        if 0 <= cx < width and 0 <= cy < height and px[cx, cy] > 0:
            return (cx, cy)
        best: tuple[int, int] | None = None
        best_dist: int | None = None
        for y in range(height):
            for x in range(width):
                if px[x, y] <= 0:
                    continue
                dist = (x - cx) ** 2 + (y - cy) ** 2
                if best_dist is None or dist < best_dist:
                    best = (x, y)
                    best_dist = dist
        return best


class GrainReviewHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], store: GrainReviewStore) -> None:
        super().__init__(server_address, GrainReviewHandler)
        self.store = store


class GrainReviewHandler(BaseHTTPRequestHandler):
    server: "GrainReviewHTTPServer"

    def log_message(self, *args: Any) -> None:  # keep the console quiet
        return

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._handle_get()
        except ApiError as exc:
            self.send_json({"error": exc.message}, exc.status)
        except Exception as exc:  # noqa: BLE001
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._handle_post()
        except ApiError as exc:
            self.send_json({"error": exc.message}, exc.status)
        except Exception as exc:  # noqa: BLE001
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_get(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/" or path == "/tinder" or path.startswith("/tinder/"):
            self.send_html(render_page())
            return
        if path == "/api/page":
            q = parse_qs(parsed.query)
            payload = self.server.store.page(
                offset=max(0, int((q.get("offset", ["0"])[0]) or 0)),
                limit=min(200, max(1, int((q.get("limit", ["60"])[0]) or 60))),
                grade=(q.get("grade", ["all"])[0]),
                view=(q.get("view", ["all"])[0]),
                sort=(q.get("sort", ["manifest"])[0]),
                focus=(q.get("focus", [""])[0]),
            )
            self.send_json(payload)
            return
        if path.startswith("/crops/"):
            rel = unquote(path[len("/crops/") :])
            self.send_file(self.server.store.crop_file(rel))
            return
        if path.startswith("/source/"):
            grain_uid = unquote(path[len("/source/") :])
            self.send_file(self.server.store.source_file(grain_uid))
            return
        if path.startswith("/contours/"):
            grain_uid = unquote(path[len("/contours/") :])
            self.send_png(self.server.store.contour_png(grain_uid))
            return
        raise ApiError(HTTPStatus.NOT_FOUND, "not found")

    def _handle_post(self) -> None:
        parsed = urlparse(self.path)
        payload = self.read_json_payload()
        if parsed.path == "/api/annotate":
            stats = self.server.store.annotate(str(payload.get("grain_uid", "")), payload.get("label"))
            self.send_json({"ok": True, "stats": stats})
            return
        raise ApiError(HTTPStatus.NOT_FOUND, "not found")

    def read_json_payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0 or length > MAX_POST_BYTES:
            raise ApiError(HTTPStatus.BAD_REQUEST, "empty or oversized body")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"invalid json: {exc}") from exc
        if not isinstance(data, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "body must be an object")
        return data

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
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

    def send_file(self, path: Path) -> None:
        import mimetypes

        data = path.read_bytes()
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_png(self, data: bytes) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def render_page() -> str:
    return """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Разметка зёрен — ordinary / fine</title>
<style>
:root{color-scheme:dark;--bg:#12151b;--panel:#1b1f28;--line:#2b313d;--text:#e7ecf3;--muted:#93a0b4;
--ord:#1fa25a;--fine:#d83f45;--unc:#8a93a5;--accent:#3aa0a4}
*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--text)}
header{position:sticky;top:0;background:var(--panel);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;z-index:5}
header h1{font-size:15px;margin:0 12px 0 0;font-weight:650}
select,button{background:#232833;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:6px 10px;font-size:13px}
button{cursor:pointer}
.prog{color:var(--muted);font-size:13px}
.wrap{display:flex;align-items:flex-start}
main{flex:1;min-width:0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;padding:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;display:flex;flex-direction:column;cursor:pointer}
.card.sel{outline:2px solid var(--accent)}
.card img{width:100%;height:120px;object-fit:contain;background:#0c0e12}
.meta{font-size:10px;color:var(--muted);padding:4px 6px 0;display:flex;justify-content:space-between;align-items:flex-start;gap:4px}
.meta span:first-child{display:flex;gap:3px;flex-wrap:wrap;min-width:0}
.chips{display:flex;gap:3px;padding:3px 6px 4px;flex-wrap:wrap}
.mc{font-size:10px;padding:1px 4px;border-radius:5px;background:#232833;color:var(--muted);font-variant-numeric:tabular-nums}
.mc.fs{background:rgba(216,63,69,.22);color:#f2969a}
.badge{font-size:10px;padding:1px 5px;border-radius:6px}
.b-ord{background:rgba(31,162,90,.2);color:#7fe0a8}.b-fine{background:rgba(216,63,69,.2);color:#f2969a}
.b-warn{background:rgba(230,160,30,.22);color:#f0c264}
.b-value{background:rgba(58,160,164,.2);color:#87dadd}
.b-small{background:rgba(175,116,255,.22);color:#c7a8ff}
.warn{background:rgba(230,160,30,.15);border:1px solid rgba(230,160,30,.45);color:#f0c264;padding:6px 8px;border-radius:8px;font-size:12px;margin:8px 0;line-height:1.35}
.card.warn-edge{outline:1px solid rgba(230,160,30,.5)}
.row{display:flex;margin-top:auto}.row button{flex:1;border-radius:0;border:0;border-top:1px solid var(--line);font-size:12px;padding:5px 0}
.row .a-ord.on{background:var(--ord);color:#04170c}.row .a-fine.on{background:var(--fine);color:#1c0405}.row .a-unc.on{background:var(--unc);color:#10131a}
aside{width:320px;flex:none;position:sticky;top:52px;height:calc(100vh - 52px);overflow:auto;border-left:1px solid var(--line);padding:14px;background:var(--panel)}
aside h2{font-size:13px;margin:0 0 8px;color:var(--muted);font-weight:600}
aside img{width:100%;max-height:300px;object-fit:contain;background:#0c0e12;border-radius:8px}
.crop-viewer{position:relative;width:100%;background:#0c0e12;border-radius:8px;overflow:hidden}
.crop-viewer .crop-base{display:block;width:100%;max-height:300px;object-fit:contain;background:#0c0e12}
.crop-viewer.hide-background .crop-base{opacity:0}
.crop-viewer .contour-overlay{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;pointer-events:none;display:none;background:transparent;border:0}
.crop-viewer.show-contour .contour-overlay{display:block}
.crop-toggles{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:7px 0 4px}
.crop-toggle{display:flex;align-items:center;gap:6px;margin:0;color:var(--muted);font-size:12px}
.crop-toggle input{accent-color:var(--accent)}
.verdict{font-size:13px;margin:10px 0;line-height:1.4}
.ftab{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
.ftab td{padding:3px 4px;border-bottom:1px solid var(--line)}
.ftab td:last-child{text-align:right}
.ftab tr.fs td{color:#f2969a}
.ftab.heur{margin:8px 0}.ftab.heur th{padding:4px;border-bottom:1px solid var(--line);color:var(--muted);font-weight:600;text-align:left}
.ftab.heur th:last-child,.ftab.heur td:last-child{text-align:right}
.ftab.heur .vote-fine{color:#f2969a}.ftab.heur .vote-ordinary{color:#7fe0a8}
.dbtns{display:flex;gap:6px;margin-top:12px}
.dbtns button{flex:1}
.dbtns .on.a-ord{background:var(--ord);color:#04170c;border-color:var(--ord)}
.dbtns .on.a-fine{background:var(--fine);color:#1c0405;border-color:var(--fine)}
.dbtns .on.a-unc{background:var(--unc);color:#10131a;border-color:var(--unc)}
.dhint{color:var(--muted);font-size:13px}
.hint{color:var(--muted);font-size:12px;padding:0 16px 20px}
.pager{display:flex;gap:8px;align-items:center;padding:0 16px 20px}
.hidden{display:none!important}
.tinder{padding:0 14px 20px}
.tinder-stage{display:grid;grid-template-columns:minmax(340px,1.35fr) minmax(320px,.85fr);gap:14px;align-items:start}
.tpanel{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;min-width:0}
.tpanel h2{font-size:13px;line-height:1.25;margin:0;padding:10px 12px;border-bottom:1px solid var(--line);font-weight:650;color:var(--text)}
.tviewer{position:relative;height:min(62vh,620px);min-height:340px;background:#0c0e12}
.tviewer img,.tviewer svg{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
.tviewer svg{pointer-events:none}
.grain-box{fill:rgba(0,255,191,.10);stroke:#00ffbf;stroke-width:10px;vector-effect:non-scaling-stroke;stroke-linejoin:round}
.zoom-panel{display:flex;flex-direction:column}
.zoom-panel img{width:100%;height:300px;object-fit:contain;background:#0c0e12;border-bottom:1px solid var(--line)}
.tinder-body{padding:12px}
.tinder-verdict{font-size:13px;line-height:1.4;margin-bottom:10px}
.swipe-actions{display:grid;grid-template-columns:1fr 1fr 1fr;grid-template-areas:". up ." "left down right";gap:8px;margin-top:12px}
.swipe-actions button{min-height:58px;border-radius:8px;padding:7px 8px;display:flex;flex-direction:column;align-items:center;justify-content:center;line-height:1.15}
.swipe-actions b{font-size:13px}.swipe-actions span{font-size:10px;color:var(--muted);margin-top:3px}
.swipe-left{grid-area:left;border-color:rgba(216,63,69,.65);background:rgba(216,63,69,.13)}
.swipe-right{grid-area:right;border-color:rgba(31,162,90,.65);background:rgba(31,162,90,.13)}
.swipe-up{grid-area:up;border-color:rgba(58,160,164,.65);background:rgba(58,160,164,.13)}
.swipe-down{grid-area:down;border-color:rgba(138,147,165,.65);background:rgba(138,147,165,.13)}
@media(max-width:900px){.tinder-stage{grid-template-columns:1fr}.tviewer{height:420px}aside{display:none}.wrap{display:block}}
</style></head><body>
<header>
<h1>Разметка зёрен</h1>
<label>Режим <select id="mode"><option value="grid">сетка</option><option value="tinder">Tinder mode</option></select></label>
<label>Сорт <select id="grade"><option value="all">все</option><option value="ordinary_intergrowth">рядовая</option><option value="fine_intergrowth">труднообог.</option><option value="talcose">оталькован.</option></select></label>
<label>Показ <select id="view"><option value="all">все</option><option value="unlabeled">без метки</option><option value="labeled">размечены</option></select></label>
<label>Сортировка <select id="sort"><option value="manifest">по списку</option><option value="review_value">ценные для проверки</option></select></label>
<span class="prog" id="prog"></span>
<span style="flex:1"></span>
<span class="prog" id="keyhint">O=рядовое · F=тонкое · U=неясно · ←/→ навигация</span>
</header>
<div class="pager"><button id="prev">← стр.</button><span class="prog" id="pageinfo"></span><button id="next">стр. →</button></div>
<section class="tinder hidden" id="tinder"></section>
<div class="wrap" id="gridWrap">
<main>
<div class="grid" id="grid"></div>
<div class="hint">Клик по карточке — выделить; справа — полный отчёт по зерну (те же признаки, что в v2-пайплайне) и подсказка эвристики. Красным подсвечены признаки, по которым срабатывает «тонкое». Кнопки/клавиши присваивают класс, автосохранение в annotations.json.</div>
</main>
<aside id="detail"><div class="dhint">Выберите зерно, чтобы увидеть отчёт по признакам.</div></aside>
</div>
<script>
const state={offset:0,limit:60,grade:'all',view:'all',sort:'manifest',mode:'grid',items:[],sel:0,filteredTotal:0,showContour:false,showBackground:true};
const $=s=>document.querySelector(s);
const CLASS_RU={ordinary_intergrowth:'рядовое',fine_intergrowth:'тонкое',uncertain:'неясно'};
function initialRoute(){
  const parts=window.location.pathname.split('/').filter(Boolean);
  if(parts[0]==='tinder'&&parts[1])return {mode:'tinder',focusUid:decodeURIComponent(parts.slice(1).join('/'))};
  return {mode:'grid',focusUid:null};
}
const route=initialRoute();state.mode=route.mode;state.focusUid=route.focusUid;
function fmt(x,d){const n=parseFloat(x);return (x===null||isNaN(n))?'—':n.toFixed(d===undefined?2:d);}
function statsText(s){return `размечено ${s.labeled}/${s.total} · рядовых ${s.counts.ordinary_intergrowth} · тонких ${s.counts.fine_intergrowth} · неясно ${s.counts.uncertain}`;}
function selectedItem(){return state.items[state.sel];}
async function load(){
  const q=new URLSearchParams({offset:state.offset,limit:state.limit,grade:state.grade,view:state.view,sort:state.sort});
  if(state.focusUid)q.set('focus',state.focusUid);
  const r=await fetch('/api/page?'+q);const d=await r.json();
  const focusUid=state.focusUid;state.focusUid=null;
  state.items=d.items;state.offset=d.offset;state.sel=0;state.filteredTotal=d.filtered_total;
  if(focusUid){
    const focusIndex=state.items.findIndex(it=>it.grain_uid===focusUid);
    if(focusIndex>=0)state.sel=focusIndex;
  }
  $('#prog').textContent=statsText(d.stats);
  $('#pageinfo').textContent=`${d.filtered_total?d.offset+1:0}–${Math.min(d.offset+d.limit,d.filtered_total)} из ${d.filtered_total}`;
  render();
}
function chip(name,val,isFine,d){return `<span class="mc ${isFine?'fs':''}">${name} ${fmt(val,d)}</span>`;}
function render(){
  const gridMode=state.mode==='grid';
  $('#gridWrap').classList.toggle('hidden',!gridMode);
  $('#tinder').classList.toggle('hidden',gridMode);
  $('#keyhint').textContent=gridMode?'O=рядовое · F=тонкое · U=неясно · ←/→ навигация':'← тонкое · → рядовое · ↑ отложить · ↓ не уверен';
  if(!gridMode){renderTinder();updateRoute();return;}
  const g=$('#grid');g.innerHTML='';
  state.items.forEach((it,i)=>{
    const c=document.createElement('div');c.className='card'+(i===state.sel?' sel':'')+(it.boundary_only_fine?' warn-edge':'');
    const f=it.features,sg=it.fine_signals;
    const badge=it.heuristic_label==='fine_intergrowth'?'<span class="badge b-fine">эвр: тонкое</span>':'<span class="badge b-ord">эвр: рядовое</span>';
    const warn=it.boundary_only_fine?'<span class="badge b-warn" title="«тонкое» только из-за рваной границы, замещение низкое">⚠ край</span>':'';
    const small=it.small_fine_context&&it.small_fine_context.matched?`<span class="badge b-small" title="Мелкое зерно из тонких: площадь ${fmt(it.small_fine_context.area_px,0)} ≤ ${fmt(it.small_fine_context.threshold_px,0)} px, форма рядовая">мелк</span>`:'';
    const value=it.review_value?`<span class="badge b-value" title="${it.review_value.reasons.join(' · ')}">ценн ${it.review_value.score}</span>`:'';
    c.innerHTML=`<img loading="lazy" src="${it.crop_url}">
    <div class="meta"><span>${badge}${warn}${small}${value}</span><span>a=${fmt(f.area_px,0)}</span></div>
    <div class="chips">${chip('d',f.dark_inside_ratio,sg.dark_inside_ratio,2)}${chip('s',f.solidity,sg.solidity,2)}${chip('c',f.compactness,sg.compactness,3)}</div>
    <div class="row">
      <button class="a-fine ${it.label==='fine_intergrowth'?'on':''}" data-l="fine_intergrowth">тонкое</button>
      <button class="a-unc ${it.label==='uncertain'?'on':''}" data-l="uncertain">?</button>
      <button class="a-ord ${it.label==='ordinary_intergrowth'?'on':''}" data-l="ordinary_intergrowth">рядовое</button>
    </div>`;
    c.onclick=(e)=>{if(e.target.tagName!=='BUTTON'){state.sel=i;highlight();}};
    c.querySelectorAll('button').forEach(b=>b.onclick=()=>{state.sel=i;assign(b.dataset.l);});
    g.appendChild(c);
  });
  renderDetail();
  updateRoute();
}
function updateRoute(){
  const it=selectedItem();
  const path=state.mode==='tinder'&&it?`/tinder/${encodeURIComponent(it.grain_uid)}`:'/';
  if(window.location.pathname!==path)history.replaceState({mode:state.mode,grain_uid:it?it.grain_uid:null},'',path);
}
function highlight(){document.querySelectorAll('.card').forEach((c,i)=>c.classList.toggle('sel',i===state.sel));renderDetail();updateRoute();}
function frow(label,val,d,isFine){return `<tr class="${isFine?'fs':''}"><td>${label}</td><td>${fmt(val,d)}</td></tr>`;}
function heuristicScoreText(it){
  const s=it.heuristic_scores||{ordinary:0,fine:0,fine_votes:0,total_votes:3};
  return `рядовое ${s.ordinary}% · тонкое ${s.fine}%`;
}
function reviewValueText(it){
  const v=it.review_value;if(!v)return '';
  const reasons=(v.reasons||[]).join(' · ');
  return `Ценность проверки: ${v.score}/100${reasons?' · '+reasons:''}`;
}
function smallFineText(it){
  const s=it.small_fine_context;
  if(!s||!s.matched)return '';
  return `Мелкое зерно в тонких: площадь ${fmt(s.area_px,0)} px ≤ ${fmt(s.threshold_px,0)} px, форма по solidity/compactness выглядит рядовой`;
}
function heuristicTable(it){
  const rows=it.heuristic_rows||[];
  const body=rows.map(r=>{
    const vote=r.matched?'тонкое':'рядовое';
    const cls=r.matched?'vote-fine':'vote-ordinary';
    return `<tr class="${r.matched?'fs':''}"><td>${r.label}</td><td>${fmt(r.value,r.digits)}</td><td>${r.rule}</td><td class="${cls}">${vote}</td></tr>`;
  }).join('');
  return `<table class="ftab heur"><thead><tr><th>Признак</th><th>Значение</th><th>Порог тонкого</th><th>Голос</th></tr></thead><tbody>${body}</tbody></table>`;
}
function cropViewer(it){
  const disabled=it.contour_url?'':' disabled';
  const contourChecked=state.showContour?' checked':'';
  const backgroundChecked=state.showBackground?' checked':'';
  const classes=[
    state.showContour&&it.contour_url?'show-contour':'',
    state.showBackground?'':'hide-background',
  ].filter(Boolean).join(' ');
  const classAttr=classes?` ${classes}`:'';
  const overlay=it.contour_url?`<img class="contour-overlay" src="${it.contour_url}" alt="">`:'';
  return `<div class="crop-viewer${classAttr}"><img class="crop-base" src="${it.crop_url}" alt="grain crop">${overlay}</div>
  <div class="crop-toggles">
    <label class="crop-toggle"><input type="checkbox" class="contour-toggle"${contourChecked}${disabled}> контур зерна</label>
    <label class="crop-toggle"><input type="checkbox" class="background-toggle"${backgroundChecked}> background</label>
  </div>`;
}
function bindCropToggles(root){
  const contour=root.querySelector('.contour-toggle');
  if(contour)contour.onchange=()=>{state.showContour=contour.checked;render();};
  const background=root.querySelector('.background-toggle');
  if(background)background.onchange=()=>{state.showBackground=background.checked;render();};
}
function renderDetail(){
  const a=$('#detail');const it=selectedItem();
  if(!it){a.innerHTML='<div class="dhint">Выберите зерно, чтобы увидеть отчёт по признакам.</div>';return;}
  const f=it.features,sg=it.fine_signals;
  const verdict=it.heuristic_label==='fine_intergrowth'
    ? '<b style="color:var(--fine)">тонкое</b>'
    : '<b style="color:var(--ord)">рядовое</b>';
  const cur=it.label?`ваша метка: <b>${CLASS_RU[it.label]}</b>`:'ещё не размечено';
  const warnBox=it.boundary_only_fine
    ? '<div class="warn">⚠ «тонкое» только из-за <b>границы</b> (замещение низкое, dark_inside_ratio &lt; 0.18). Если это массивное однородное зерно с рваным краем — вероятно <b>рядовое</b>, а не труднообогатимое.</div>'
    : '';
  a.innerHTML=`<h2>Отчёт по зерну · ${it.grade_label}</h2>
  ${cropViewer(it)}
  <div class="verdict">Эвристика: ${verdict}${heuristicTable(it)}<span class="dhint">Счёт эвристики: ${heuristicScoreText(it)}</span><br><span class="dhint">${reviewValueText(it)}</span>${smallFineText(it)?'<br><span class="dhint">'+smallFineText(it)+'</span>':''}<br><span class="dhint">${cur}</span></div>
  ${warnBox}
  <table class="ftab">
    ${frow('Доля тёмного (замещение)',f.dark_inside_ratio,2,sg.dark_inside_ratio)}
    ${frow('Выпуклость (solidity)',f.solidity,2,sg.solidity)}
    ${frow('Компактность',f.compactness,3,sg.compactness)}
    ${frow('Сложность границы',f.boundary_complexity,2,false)}
    ${frow('Площадь, px',f.area_px,0,false)}
    ${frow('Площадь контура, px',f.footprint_area_px,0,false)}
    ${frow('Тёмное внутри, px',f.dark_inside_area_px,0,false)}
    <tr><td>BBox</td><td>${fmt(f.bbox_w,0)} × ${fmt(f.bbox_h,0)}</td></tr>
  </table>
  <div class="dbtns">
    <button class="a-fine ${it.label==='fine_intergrowth'?'on':''}" data-l="fine_intergrowth">тонкое (F)</button>
    <button class="a-unc ${it.label==='uncertain'?'on':''}" data-l="uncertain">? (U)</button>
    <button class="a-ord ${it.label==='ordinary_intergrowth'?'on':''}" data-l="ordinary_intergrowth">рядовое (O)</button>
  </div>`;
  bindCropToggles(a);
  a.querySelectorAll('.dbtns button').forEach(b=>b.onclick=()=>assign(b.dataset.l));
}
function detailTable(it){
  const f=it.features,sg=it.fine_signals;
  return `<table class="ftab">
    ${frow('Доля тёмного (замещение)',f.dark_inside_ratio,2,sg.dark_inside_ratio)}
    ${frow('Выпуклость (solidity)',f.solidity,2,sg.solidity)}
    ${frow('Компактность',f.compactness,3,sg.compactness)}
    ${frow('Сложность границы',f.boundary_complexity,2,false)}
    ${frow('Площадь, px',f.area_px,0,false)}
    ${frow('Площадь контура, px',f.footprint_area_px,0,false)}
    ${frow('Тёмное внутри, px',f.dark_inside_area_px,0,false)}
    <tr><td>BBox</td><td>${fmt(f.bbox_w,0)} × ${fmt(f.bbox_h,0)}</td></tr>
  </table>`;
}
function renderTinder(){
  const t=$('#tinder');const it=selectedItem();
  if(!it){t.innerHTML='<div class="tpanel"><h2>Нет зёрен в текущем фильтре</h2><div class="tinder-body dhint">Измените фильтр или страницу.</div></div>';return;}
  const f=it.features;
  const verdict=it.heuristic_label==='fine_intergrowth'
    ? '<b style="color:var(--fine)">тонкое</b>'
    : '<b style="color:var(--ord)">рядовое</b>';
  const cur=it.label?`текущая метка: <b>${CLASS_RU[it.label]}</b>`:'ещё не размечено';
  const title=it.source_name||it.image_rel_path||it.grain_uid;
  t.innerHTML=`<div class="tinder-stage">
    <section class="tpanel">
      <h2>Оригинал · ${title}</h2>
      <div class="tviewer" id="sourceViewer"><img id="sourceImg" alt="original image"><svg id="sourceOverlay" preserveAspectRatio="xMidYMid meet"><rect class="grain-box" id="grainBox"></rect></svg></div>
    </section>
    <section class="tpanel zoom-panel">
      <h2>Зерно · ${it.grain_uid}</h2>
      ${cropViewer(it)}
      <div class="tinder-body">
        <div class="tinder-verdict">Эвристика: ${verdict}${heuristicTable(it)}<span class="dhint">Счёт эвристики: ${heuristicScoreText(it)}</span><br><span class="dhint">${reviewValueText(it)}</span>${smallFineText(it)?'<br><span class="dhint">'+smallFineText(it)+'</span>':''}<br><span class="dhint">${cur}</span></div>
        ${it.boundary_only_fine?'<div class="warn">«тонкое» только из-за границы; если зерно однородное, лучше отправить вправо как рядовое.</div>':''}
        ${detailTable(it)}
        <div class="swipe-actions">
          <button class="swipe-up" id="swipeUp"><b>↑ Отложить</b><span>без метки, следующий</span></button>
          <button class="swipe-left" id="swipeLeft"><b>← Тонкое</b><span>труднообогатительное, плохо</span></button>
          <button class="swipe-down" id="swipeDown"><b>↓ Не уверен</b><span>не могу принять решение</span></button>
          <button class="swipe-right" id="swipeRight"><b>Рядовое →</b><span>рядовая руда, хорошо</span></button>
        </div>
      </div>
    </section>
  </div>`;
  const img=$('#sourceImg'),svg=$('#sourceOverlay'),rect=$('#grainBox');
  if(it.source_url){
    img.onload=()=>positionGrainBox(img,svg,rect,it);
    img.src=it.source_url;
    if(img.complete)positionGrainBox(img,svg,rect,it);
  }else{
    $('#sourceViewer').innerHTML='<div class="dhint" style="padding:18px">В манифесте нет пути к исходному изображению.</div>';
  }
  $('#swipeLeft').onclick=()=>assign('fine_intergrowth',{force:true});
  $('#swipeRight').onclick=()=>assign('ordinary_intergrowth',{force:true});
  $('#swipeDown').onclick=()=>assign('uncertain',{force:true});
  $('#swipeUp').onclick=()=>postpone();
  bindCropToggles(t);
}
function positionGrainBox(img,svg,rect,it){
  const b=it.bbox||{};const vals=[b.x,b.y,b.w,b.h].map(parseFloat);
  if(vals.some(v=>isNaN(v))||!img.naturalWidth||!img.naturalHeight){rect.setAttribute('display','none');return;}
  svg.setAttribute('viewBox',`0 0 ${img.naturalWidth} ${img.naturalHeight}`);
  rect.removeAttribute('display');
  rect.setAttribute('x',vals[0]);rect.setAttribute('y',vals[1]);rect.setAttribute('width',Math.max(vals[2],1));rect.setAttribute('height',Math.max(vals[3],1));
}
function advanceSelection(){
  if(state.sel<state.items.length-1){state.sel++;render();return;}
  if(state.offset+state.limit<state.filteredTotal){state.offset+=state.limit;load();return;}
  render();
}
function postpone(){if(!selectedItem())return;advanceSelection();}
async function assign(label,opts={}){
  const it=selectedItem();if(!it)return;
  const newLabel=opts.force?label:(it.label===label?null:label);
  await fetch('/api/annotate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({grain_uid:it.grain_uid,label:newLabel})});
  it.label=newLabel;
  if(newLabel&&opts.advance!==false)advanceSelection();else render();
  load_stats_only();
}
async function load_stats_only(){const r=await fetch('/api/page?'+new URLSearchParams({offset:0,limit:1,grade:state.grade,view:state.view,sort:state.sort}));const d=await r.json();$('#prog').textContent=statsText(d.stats);}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='SELECT')return;
  if(state.mode==='tinder'){
    if(e.key==='ArrowRight'){assign('ordinary_intergrowth',{force:true});}
    else if(e.key==='ArrowLeft'){assign('fine_intergrowth',{force:true});}
    else if(e.key==='ArrowUp'){postpone();}
    else if(e.key==='ArrowDown'){assign('uncertain',{force:true});}
    else if(e.key.toLowerCase()==='o'){assign('ordinary_intergrowth',{force:true});}
    else if(e.key.toLowerCase()==='f'){assign('fine_intergrowth',{force:true});}
    else if(e.key.toLowerCase()==='u'){assign('uncertain',{force:true});}
    return;
  }
  if(e.key==='ArrowRight'){state.sel=Math.min(state.sel+1,state.items.length-1);highlight();}
  else if(e.key==='ArrowLeft'){state.sel=Math.max(state.sel-1,0);highlight();}
  else if(e.key.toLowerCase()==='o'){assign('ordinary_intergrowth');}
  else if(e.key.toLowerCase()==='f'){assign('fine_intergrowth');}
  else if(e.key.toLowerCase()==='u'){assign('uncertain');}
});
$('#mode').onchange=e=>{state.mode=e.target.value;render();};
$('#grade').onchange=e=>{state.grade=e.target.value;state.offset=0;load();};
$('#view').onchange=e=>{state.view=e.target.value;state.offset=0;load();};
$('#sort').onchange=e=>{state.sort=e.target.value;state.offset=0;load();};
$('#prev').onclick=()=>{state.offset=Math.max(0,state.offset-state.limit);load();};
$('#next').onclick=()=>{state.offset+=state.limit;load();};
$('#mode').value=state.mode;
load();
</script></body></html>"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grain review / labeling app.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Output dir of build_grain_dataset.py (grains_manifest.csv + crops/).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 asks the OS for a free port.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    store = GrainReviewStore(args.dataset_dir)
    server = GrainReviewHTTPServer((args.host, args.port), store)
    host, port = server.server_address[0], server.server_address[1]
    stats = store.stats()
    print(f"Grain review: http://{host}:{port}/  ({stats['labeled']}/{stats['total']} labelled)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
