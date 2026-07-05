from __future__ import annotations

import base64
import csv
import http.client
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from http import HTTPStatus
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import apps.ore_pipeline_web as ore_pipeline_web  # noqa: E402
from apps.ore_pipeline_web import (  # noqa: E402
    ApiError,
    OrePipelineHTTPServer,
    OrePipelineStore,
    RunCancelled,
    build_openapi_document,
    build_pdf_report_pages,
    compute_talc_cluster_mask,
    gpu_status_payload,
    load_font,
    normalize_talc_clusterization_payload,
    render_html_page,
    summary_from_final_edit,
    text_width,
    wrap_text_lines,
)


def mask_data_url(mask: np.ndarray) -> str:
    handle = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8), mode="L").save(handle, format="PNG")
    return "data:image/png;base64," + base64.b64encode(handle.getvalue()).decode("ascii")


class OrePipelineWebTest(unittest.TestCase):
    def setUp(self) -> None:
        # Each test gets its own isolated workspace. A previous test's async run
        # job runs on a daemon thread that can outlive the test; a shared fixed
        # workspace let that thread litter run artifacts into the next test's
        # store, causing order-dependent flakiness. A unique per-test directory
        # keeps every store's runs/uploads/batches private.
        self.root = Path(tempfile.mkdtemp(prefix="test_ore_pipeline_web_"))
        self.image_path = self.root / "sample.png"
        rgb = np.full((120, 160, 3), (54, 62, 52), dtype=np.uint8)
        cv2.circle(rgb, (44, 46), 18, (226, 225, 212), -1)
        cv2.rectangle(rgb, (95, 35), (135, 82), (215, 214, 205), -1)
        cv2.rectangle(rgb, (12, 82), (52, 112), (66, 108, 70), -1)
        Image.fromarray(rgb, mode="RGB").save(self.image_path)
        self.image_path_2 = self.root / "sample_2.png"
        rgb_2 = np.full((100, 140, 3), (48, 57, 50), dtype=np.uint8)
        cv2.circle(rgb_2, (72, 52), 22, (228, 226, 216), -1)
        cv2.rectangle(rgb_2, (15, 18), (55, 70), (210, 209, 199), -1)
        Image.fromarray(rgb_2, mode="RGB").save(self.image_path_2)
        self.store = OrePipelineStore(
            workspace_dir=self.root / "workspace",
            backend="heuristic",
            checkpoint=None,
            processing_max_side=256,
            panorama_max_side=128,
            preview_max_sides=(128, 256),
            talc_backend="heuristic",
        )

    def tearDown(self) -> None:
        # The workspace is unique per test, so a leaked async worker from this
        # test can only touch its own (soon-discarded) directory; ignore_errors
        # keeps a live write from failing the cleanup.
        shutil.rmtree(self.root, ignore_errors=True)

    def assertInvalidStoreId(self, callback) -> None:  # noqa: N802 - unittest helper style.
        with self.assertRaises(ApiError) as raised:
            callback()
        self.assertEqual(raised.exception.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("invalid", raised.exception.message)

    def test_default_settings_disable_preprocessing_gate(self) -> None:
        settings = self.store.app_settings()
        self.assertFalse(settings["preprocess"]["preprocessing_enabled"])
        self.assertTrue(settings["preprocess"]["illumination_normalization"])
        self.assertTrue(settings["preprocess"]["denoise"])
        self.assertTrue(settings["preprocess"]["contrast_correction"])
        self.assertTrue(settings["preprocess"]["panorama_scaling"])

    def test_describe_decode_bomb_flags_bombs_but_not_real_images(self) -> None:
        describe = ore_pipeline_web.describe_decode_bomb
        # The demonstrated bomb: tiny file, thousands of megapixels.
        self.assertIsNotNone(describe(46000, 46000, "RGB", 6_170_050))
        # Largest real dataset image (26 MP, 13 MB) and a highly-compressed real one.
        self.assertIsNone(describe(6240, 4160, "RGB", 13_000_000))
        self.assertIsNone(describe(6240, 3895, "RGB", 1_090_000))
        # Tiny solid image has a high ratio but a trivial decoded footprint.
        self.assertIsNone(describe(100, 100, "RGB", 1_000))
        # Largest real panorama (dataset/Панорамы/16.jpg, ~574 MP) must NOT be flagged.
        self.assertIsNone(describe(27025, 21227, "RGB", 221_610_000))
        # Megapixel-limit boundary (cap is 1000 MP).
        self.assertIsNone(describe(30000, 30000, "RGB", 120_000_000))  # ~900 MP
        self.assertIsNotNone(describe(33000, 33000, "RGB", 120_000_000))  # ~1089 MP
        # Sub-cap dimensions but an extreme expansion ratio on a large raster.
        self.assertIsNotNone(describe(14000, 14000, "RGB", 1_000_000))
        # Malformed header values are ignored (no false positive / crash).
        self.assertIsNone(describe(0, 0, "RGB", 10))
        self.assertIsNone(describe("bad", 100, "RGB", 10))

    def test_detect_decode_bomb_setting_defaults_on_and_round_trips(self) -> None:
        self.assertTrue(self.store.app_settings()["detect_decode_bomb"])
        self.assertTrue(self.store.detect_decode_bomb_enabled())
        saved = self.store.save_app_settings({"detect_decode_bomb": False})
        self.assertFalse(saved["detect_decode_bomb"])
        self.assertFalse(self.store.app_settings()["detect_decode_bomb"])
        self.assertFalse(self.store.detect_decode_bomb_enabled())
        # Public settings expose the flag too (used by the UI form).
        self.assertFalse(self.store.public_app_settings()["detect_decode_bomb"])

    def _header_only_png_bytes(self, width: int, height: int) -> bytes:
        """A PNG whose IHDR declares width x height but carries no full raster.

        Image.open(...).size/.mode read only the header, so the decode-bomb check
        (which runs before any full decode) can be exercised without allocating a
        multi-gigabyte raster in the test.
        """
        import struct
        import zlib

        def chunk(tag: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

        ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
        return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\x00")) + chunk(b"IEND", b"")

    def test_upload_rejects_decode_bomb_when_enabled(self) -> None:
        bomb = self._header_only_png_bytes(46000, 46000)
        with self.assertRaises(ApiError) as ctx:
            self.store.register_upload_from_bytes(bomb, "bomb.png")
        self.assertEqual(ctx.exception.code, "decode_bomb")
        self.assertEqual(ctx.exception.status, int(HTTPStatus.REQUEST_ENTITY_TOO_LARGE))

    def test_upload_skips_decode_bomb_check_when_disabled(self) -> None:
        self.store.save_app_settings({"detect_decode_bomb": False})
        bomb = self._header_only_png_bytes(46000, 46000)
        # With the guard off the upload proceeds past the dimension check to decoding.
        # Stub the decode so the test does not actually allocate the multi-GB raster
        # (which is exactly the DoS this guard exists to prevent); we only need to
        # prove the guard was bypassed. Dimensions still come from the real header.
        with mock.patch.object(ore_pipeline_web, "load_image_pil", return_value=Image.new("RGB", (128, 128))):
            result = self.store.register_upload_from_bytes(bomb, "bomb.png")
        self.assertEqual(result["width"], 46000)
        self.assertEqual(result["height"], 46000)

    def test_settings_page_exposes_decode_bomb_toggle(self) -> None:
        html = render_html_page()
        self.assertIn('id="settingsDetectDecodeBomb"', html)
        self.assertIn("settingsDetectDecodeBomb", html)
        self.assertIn("uploadDecodeBomb", html)

    def test_run_and_upload_ids_reject_path_traversal(self) -> None:
        escaped_run_dir = self.store.workspace_dir / "escaped_run"
        escaped_run_dir.mkdir(parents=True)
        escaped_run = {
            "schema_version": "ore-pipeline-run-v0.1",
            "run_id": "escaped_run",
            "status": "prepared",
            "display": {},
            "summary": {},
            "input": {"upload_id": "escaped_upload"},
            "image": {"width": 1, "height": 1},
        }
        (escaped_run_dir / "run.json").write_text(json.dumps(escaped_run), encoding="utf-8")
        escaped_upload_dir = self.store.workspace_dir / "escaped_upload"
        escaped_upload_dir.mkdir(parents=True)
        (escaped_upload_dir / "upload.json").write_text(
            json.dumps({"schema_version": "ore-pipeline-upload-v0.1", "upload_id": "escaped_upload", "display": {}}),
            encoding="utf-8",
        )

        bad_run_id = "../escaped_run"
        self.assertInvalidStoreId(lambda: self.store.run_payload(bad_run_id))
        self.assertInvalidStoreId(lambda: self.store.cancel_run(bad_run_id))
        self.assertInvalidStoreId(lambda: self.store.metrics_csv_path(bad_run_id))
        self.assertInvalidStoreId(lambda: self.store.start_prepared_run(bad_run_id, run_async=False))
        self.assertInvalidStoreId(
            lambda: self.store.prepare_run_from_apply(
                bad_run_id,
                {"preprocessing_enabled": True},
                changed_step="preprocess",
            )
        )
        self.assertInvalidStoreId(lambda: self.store.create_edit_run(bad_run_id, {"edit_layer": "final", "mask_png": ""}))

        bad_upload_id = "../escaped_upload"
        self.assertInvalidStoreId(lambda: self.store.upload_payload(bad_upload_id))
        self.assertInvalidStoreId(lambda: self.store.prepare_upload(bad_upload_id, {"preprocessing_enabled": True}))
        self.assertInvalidStoreId(lambda: self.store.save_upload_artifact_mask(bad_upload_id, {"mask_png": ""}))
        self.assertInvalidStoreId(lambda: self.store.start_run(bad_upload_id, {"preprocessing_enabled": True}, run_async=False))

        persisted_run = json.loads((escaped_run_dir / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted_run["status"], "prepared")

    def test_http_run_and_upload_ids_reject_encoded_path_traversal(self) -> None:
        (self.store.workspace_dir / "escaped_run").mkdir(parents=True)
        (self.store.workspace_dir / "escaped_run" / "run.json").write_text(
            json.dumps({"schema_version": "ore-pipeline-run-v0.1", "run_id": "escaped_run", "status": "prepared"}),
            encoding="utf-8",
        )
        (self.store.workspace_dir / "escaped_upload").mkdir(parents=True)
        (self.store.workspace_dir / "escaped_upload" / "upload.json").write_text(
            json.dumps({"schema_version": "ore-pipeline-upload-v0.1", "upload_id": "escaped_upload", "display": {}}),
            encoding="utf-8",
        )
        server = OrePipelineHTTPServer(("127.0.0.1", 0), self.store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address[:2]
        try:
            for path in ("/api/runs/..%2Fescaped_run", "/api/uploads/..%2Fescaped_upload"):
                connection = http.client.HTTPConnection(host, port, timeout=5)
                connection.request("GET", path)
                response = connection.getresponse()
                payload = json.loads(response.read().decode("utf-8"))
                connection.close()
                self.assertEqual(response.status, HTTPStatus.BAD_REQUEST)
                self.assertIn("invalid", payload["error"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_pdf_report_wraps_long_russian_conclusion_text(self) -> None:
        page = Image.new("RGB", (1240, 1754), "white")
        draw = ImageDraw.Draw(page)
        font = load_font(27)
        text = (
            "Руда классифицирована как оталькованная: содержание талька — 41.5%, "
            "преобладание обычных срастаний — 100.0%."
        )

        lines = wrap_text_lines(draw, text, font, max_width=500)

        self.assertGreater(len(lines), 1)
        self.assertTrue(all(text_width(draw, line, font) <= 500 for line in lines))

    def test_pdf_report_builds_summary_table_and_visual_sections(self) -> None:
        upload = self.store.register_upload_from_path(self.image_path)
        preprocessed = self.store.prepare_upload(
            upload["upload_id"],
            {
                "illumination_normalization": True,
                "denoise": True,
                "contrast_correction": True,
                "panorama_scaling": True,
            },
        )
        run = self.store.start_run(upload["upload_id"], preprocessed["preprocess"]["preset"], run_async=False)

        pages = build_pdf_report_pages(run, self.store.runs_dir / run["run_id"])
        pdf_path = self.store.pdf_report_path(run["run_id"])
        persisted_run = json.loads((self.store.runs_dir / run["run_id"] / "run.json").read_text(encoding="utf-8"))

        self.assertEqual(len(pages), 5)
        self.assertTrue(all(page.size == (1240, 1754) for page in pages))
        self.assertTrue(pdf_path.read_bytes().startswith(b"%PDF"))
        self.assertGreater(pdf_path.stat().st_size, 50_000)
        self.assertGreaterEqual(run["elapsed_seconds"], 0)
        self.assertEqual(run["elapsed_seconds"], persisted_run["elapsed_seconds"])

    def test_talc_cluster_mask_uses_local_density_window(self) -> None:
        talc = np.zeros((7, 7), dtype=np.uint8)
        talc[3, 3] = 255
        analyzed = np.ones((7, 7), dtype=np.uint8) * 255

        cluster, stats = compute_talc_cluster_mask(
            talc,
            analyzed,
            {"radius_px": 1, "min_local_talc_percent": 10, "opacity_percent": 55},
        )

        self.assertEqual(int((cluster > 0).sum()), 9)
        self.assertEqual(stats["area_px"], 9)
        self.assertEqual(stats["source_talc_area_px"], 1)
        self.assertAlmostEqual(stats["fraction"], 9 / 49)
        self.assertEqual(stats["radius_px"], 1)
        self.assertEqual(stats["opacity_percent"], 55)

        no_cluster, no_stats = compute_talc_cluster_mask(
            talc,
            analyzed,
            {"radius_px": 1, "min_local_talc_percent": 20, "opacity_percent": 55},
        )
        self.assertEqual(int((no_cluster > 0).sum()), 0)
        self.assertEqual(no_stats["area_px"], 0)

        sulfide_exclusion = np.zeros((7, 7), dtype=np.uint8)
        sulfide_exclusion[2, 2] = 255
        clipped_cluster, clipped_stats = compute_talc_cluster_mask(
            talc,
            analyzed,
            {"radius_px": 1, "min_local_talc_percent": 10, "opacity_percent": 55},
            exclude_mask=sulfide_exclusion,
        )
        self.assertEqual(int((clipped_cluster > 0).sum()), 8)
        self.assertEqual(int(clipped_cluster[2, 2]), 0)
        self.assertEqual(clipped_stats["excluded_area_px"], 1)
        self.assertEqual(clipped_stats["analysis_area_px"], 48)
        self.assertAlmostEqual(clipped_stats["fraction"], 8 / 48)

    def test_talc_cluster_outputs_do_not_overlap_sulfides(self) -> None:
        run_dir = self.store.runs_dir / "run_talc_cluster_exclusion"
        (run_dir / "input").mkdir(parents=True)
        image = Image.new("RGB", (7, 7), (38, 48, 42))
        image.save(run_dir / "input/original_for_analysis.png")
        image.save(run_dir / "input/preprocessed.png")
        sulfide_mask = np.zeros((7, 7), dtype=np.uint8)
        sulfide_mask[2, 2] = 255
        talc_mask = np.zeros((7, 7), dtype=np.uint8)
        talc_mask[3, 3] = 255
        analyzed_mask = np.ones((7, 7), dtype=np.uint8) * 255
        final_mask = np.zeros((7, 7), dtype=np.uint8)
        final_mask[2, 2] = 1
        final_mask[3, 3] = 3

        self.store._write_run_outputs(
            run_dir=run_dir,
            summary=summary_from_final_edit(sulfide_mask, final_mask, analyzed_mask),
            components=[],
            sulfide_mask=sulfide_mask,
            talc_mask=talc_mask,
            analyzed_mask=analyzed_mask,
            final_mask=final_mask,
            talc_clusterization={"radius_px": 1, "min_local_talc_percent": 10, "opacity_percent": 55},
        )

        persisted_sulfide = np.asarray(Image.open(run_dir / "masks/sulfide_mask.png").convert("L"))
        cluster = np.asarray(Image.open(run_dir / "masks/talc_cluster_mask.png").convert("L")).copy()
        self.assertFalse(np.any((cluster > 0) & (persisted_sulfide > 0)))
        persisted_summary = json.loads((run_dir / "reports/ore_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted_summary["talc_cluster_area_px"], 8)

        cluster[2, 2] = 255
        Image.fromarray(cluster, mode="L").save(run_dir / "masks/talc_cluster_mask.png")
        self.store._build_display_layers(run_dir)
        repaired = np.asarray(Image.open(run_dir / "masks/talc_cluster_mask.png").convert("L"))
        self.assertFalse(np.any((repaired > 0) & (persisted_sulfide > 0)))
        display_manifest_path = run_dir / "display/display.json"
        display_manifest = json.loads(display_manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(display_manifest["talc_cluster_color_rgb"], [64, 220, 255])

        display_manifest["talc_cluster_color_rgb"] = [236, 72, 153]
        self.store._write_json(display_manifest_path, display_manifest)
        self.store._write_json(
            run_dir / "run.json",
            {
                "run_id": run_dir.name,
                "status": "complete",
                "progress": 100,
                "backend": "heuristic",
                "checkpoint": None,
                "preprocess": {"enabled": True, "preset": {}},
                "input": {
                    "upload_id": "upload_for_stale_cluster_color",
                    "original_artifact_path": str(run_dir / "input/original_for_analysis.png"),
                    "preset": {},
                },
                "image": {"width": 7, "height": 7, "name": "sample.png"},
                "summary": persisted_summary,
                "metrics": [],
                "display": display_manifest["layers"],
                "masks": {
                    "sulfide": str(run_dir / "masks/sulfide_mask.png"),
                    "final": str(run_dir / "masks/final_mask.png"),
                    "talc_cluster": str(run_dir / "masks/talc_cluster_mask.png"),
                },
                "reports": {"summary_json": str(run_dir / "reports/ore_summary.json")},
            },
        )
        payload = self.store.run_payload(run_dir.name)
        refreshed_manifest = json.loads(display_manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(refreshed_manifest["talc_cluster_color_rgb"], [64, 220, 255])
        self.assertIn("talc_cluster_overlay", payload["display"])

    def test_talc_clusterization_normalization_accepts_ui_aliases(self) -> None:
        normalized = normalize_talc_clusterization_payload(
            {"radiusPx": 32, "minDensityPercent": 6.5, "opacityPercent": 35},
            {"radius_px": 64, "min_local_talc_percent": 4, "opacity_percent": 45},
        )

        self.assertEqual(normalized["schema_version"], "ore-pipeline-talc-clusterization-v0.1")
        self.assertEqual(normalized["radius_px"], 32)
        self.assertEqual(normalized["min_local_talc_percent"], 6.5)
        self.assertEqual(normalized["opacity_percent"], 35)

    def test_async_start_returns_payload_while_worker_updates_run_json(self) -> None:
        upload = self.store.register_upload_from_path(self.image_path)
        preset = {
            "illumination_normalization": True,
            "denoise": True,
            "contrast_correction": True,
            "panorama_scaling": True,
        }

        for _ in range(3):
            run = self.store.start_run(upload["upload_id"], preset, run_async=True)
            self.assertIn(run["status"], {"queued", "running", "complete"})
            self.assertIn("run_id", run)

            for _ in range(40):
                run = self.store.run_payload(run["run_id"])
                if run["status"] in {"complete", "failed", "canceled"}:
                    break
                time.sleep(0.05)

            self.assertEqual(run["status"], "complete")

    def test_repeated_same_file_uploads_get_unique_ids(self) -> None:
        first = self.store.register_upload_from_path(self.image_path)
        second = self.store.register_upload_from_path(self.image_path)

        self.assertNotEqual(first["upload_id"], second["upload_id"])
        self.assertEqual(first["sha1"], second["sha1"])
        self.assertTrue((self.store.uploads_dir / first["upload_id"] / "upload.json").exists())
        self.assertTrue((self.store.uploads_dir / second["upload_id"] / "upload.json").exists())

    def test_path_upload_hashes_once_and_reuses_metadata_sha1(self) -> None:
        with mock.patch("apps.ore_pipeline_web.file_sha1", wraps=ore_pipeline_web.file_sha1) as hashed:
            upload = self.store.register_upload_from_path(self.image_path)

        self.assertEqual(hashed.call_count, 1)
        self.assertEqual(upload["sha1"], ore_pipeline_web.file_sha1(self.image_path))
        self.assertEqual(upload["raw_metadata"]["sha1"], upload["sha1"])

    def test_large_upload_defers_full_size_preprocessing_artifact(self) -> None:
        upload = self.store.register_upload_from_path(self.image_path)
        with mock.patch("apps.ore_pipeline_web.FULL_SIZE_PREPROCESS_MAX_PIXELS", 1000):
            preprocessed = self.store.prepare_upload(
                upload["upload_id"],
                {
                    "preprocessing_enabled": True,
                    "illumination_normalization": True,
                    "denoise": True,
                    "contrast_correction": True,
                    "panorama_scaling": True,
                },
            )
            run = self.store.start_run(upload["upload_id"], preprocessed["preprocess"]["preset"], run_async=False)

        self.assertTrue(preprocessed["preprocess"]["full_size_processing_deferred"])
        self.assertNotIn("preprocessed_full_path", preprocessed["preprocess"])
        with Image.open(preprocessed["preprocess"]["preprocessed_path"]) as image:
            self.assertEqual(image.size, (128, 96))
        self.assertIn("preprocessed", preprocessed["display"])

        self.assertEqual(run["status"], "complete")
        self.assertNotIn("preprocessed_full_path", run["input"])

    def test_upload_preprocess_run_and_download_artifacts(self) -> None:
        upload = self.store.register_upload_from_path(self.image_path)
        self.assertEqual(upload["width"], 160)
        self.assertEqual(upload["raw_metadata"]["width"], 160)
        self.assertEqual(upload["raw_metadata"]["height"], 120)
        self.assertEqual(upload["file_size_bytes"], self.image_path.stat().st_size)
        self.assertTrue(upload["sha1"])
        self.assertIn("original", upload["display"])

        preprocessed = self.store.prepare_upload(
            upload["upload_id"],
            {
                "preprocessing_enabled": True,
                "illumination_normalization": True,
                "denoise": True,
                "contrast_correction": True,
                "panorama_scaling": True,
            },
        )
        self.assertTrue(preprocessed["preprocess"]["source_scaled_for_processing"])
        self.assertEqual(preprocessed["preprocess"]["target_max_side"], 128)
        self.assertEqual(preprocessed["preprocess"]["panorama_scaling"]["mode"], "max_side")
        self.assertEqual(preprocessed["preprocess"]["panorama_scaling"]["max_side_px"], 128)
        self.assertEqual(preprocessed["preprocess"]["full_width"], 160)
        self.assertEqual(preprocessed["preprocess"]["full_height"], 120)
        self.assertEqual(preprocessed["preprocess"]["source_width"], 160)
        self.assertEqual(preprocessed["preprocess"]["source_height"], 120)
        with Image.open(preprocessed["preprocess"]["preprocessed_full_path"]) as image:
            self.assertEqual(image.size, (160, 120))
        with Image.open(preprocessed["preprocess"]["preprocessed_path"]) as image:
            self.assertEqual(image.size, (128, 96))
        self.assertIn("preprocessed", preprocessed["display"])
        self.assertTrue(preprocessed["tiling"]["enabled"])
        self.assertEqual(preprocessed["tiling"]["source_width"], 160)
        self.assertEqual(preprocessed["tiling"]["analysis_width"], preprocessed["preprocess"]["width"])
        self.assertGreaterEqual(preprocessed["tiling"]["tile_count"], 1)

        run = self.store.start_run(upload["upload_id"], preprocessed["preprocess"]["preset"], run_async=False)
        self.assertEqual(run["status"], "complete")
        self.assertEqual(run["progress"], 100)
        self.assertTrue(run["tiling"]["enabled"])
        self.assertEqual(run["tiling"]["tile_count"], preprocessed["tiling"]["tile_count"])
        self.assertIn("Руда классифицирована", run["text_output"])
        self.assertTrue(Path(self.store.metrics_csv_path(run["run_id"])).exists())
        self.assertTrue(Path(self.store.pdf_report_path(run["run_id"])).exists())
        self.assertIn("preprocessed_full_path", run["input"])
        with Image.open(run["input"]["preprocessed_full_path"]) as image:
            self.assertEqual(image.size, (160, 120))
        with Image.open(self.store.runs_dir / run["run_id"] / "input/preprocessed.png") as image:
            self.assertEqual(image.size, (128, 96))
        self.assertIn("sulfide", run["masks"])
        self.assertIn("sulfide_component_labels", run["masks"])
        self.assertTrue(run["sulfide_grains"]["label_map"].startswith("/artifacts/"))
        self.assertGreater(len(run["sulfide_grains"]["items"]), 0)
        first_grain = run["sulfide_grains"]["items"][0]
        self.assertIn(first_grain["type"], {"ordinary_intergrowth", "fine_intergrowth"})
        self.assertGreater(first_grain["component_id"], 0)
        self.assertGreater(first_grain["area_px"], 0)
        self.assertGreaterEqual(first_grain["share_percent"], 0.0)
        with Image.open(self.store.runs_dir / run["run_id"] / "masks/sulfide_component_labels_rgb.png") as image:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (128, 96))
        self.assertIn("ordinary_overlay", run["display"])
        self.assertIn("talc_cluster_overlay", run["display"])
        self.assertIn("talc_cluster", run["masks"])
        self.assertIn("talc_cluster_area_px", run["summary"])
        self.assertIn("talc_cluster_fraction", run["summary"])
        self.assertIn("talc_clusterization", run)
        self.assertTrue((self.store.runs_dir / run["run_id"] / "masks/talc_cluster_mask.png").exists())
        self.assertEqual(run["runtime"]["backend"], "heuristic")
        self.assertIsNone(run["runtime"]["checkpoints"]["binary_sulfide"])
        self.assertEqual(run["runtime"]["models"]["binary_sulfide"]["backend"], "heuristic")
        self.assertEqual(run["runtime"]["grain_backend"], "heuristic")
        self.assertEqual(run["runtime"]["models"]["grain_classification"]["backend"], "ore_grain_heuristics")
        self.assertTrue((self.store.runs_dir / run["run_id"] / "reports/runtime.json").exists())

        history = self.store.list_runs()["runs"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["run_id"], run["run_id"])
        self.assertEqual(history[0]["summary"]["ore_class"], run["summary"]["ore_class"])
        metric_keys = [row["key"] for row in history[0]["metrics"]]
        self.assertEqual(metric_keys[:4], ["analyzed_fraction", "sulfide_fraction", "ordinary_sulfide_fraction", "fine_sulfide_fraction"])
        self.assertIn("other_fraction", metric_keys)
        self.assertIn("talc_cluster_fraction", metric_keys)
        self.assertIn("artifact_fraction_image", metric_keys)
        self.assertEqual(history[0]["metrics"][0]["level"], 0)
        self.assertEqual(next(row for row in history[0]["metrics"] if row["key"] == "sulfide_fraction")["level"], 1)
        self.assertEqual(next(row for row in history[0]["metrics"] if row["key"] == "ordinary_sulfide_fraction")["parent_key"], "sulfide_fraction")
        self.assertTrue(history[0]["thumbnail"]["thumbnail_url"])
        self.assertTrue(history[0]["thumbnail"]["preview_url"])
        thumbnail_id = history[0]["thumbnail"]["thumbnail_url"].split("/")[2]
        self.assertTrue(self.store.artifact_path(thumbnail_id).exists())

    def test_run_configuration_runtime_overrides_do_not_mutate_store_defaults(self) -> None:
        talc_checkpoint = self.root / "talc.pt"
        talc_checkpoint.write_bytes(b"fake talc")
        self.store.talc_backend = "ml"
        self.store.talc_checkpoint = talc_checkpoint.resolve()
        upload = self.store.register_upload_from_path(self.image_path)

        run = self.store.start_run(
            upload["upload_id"],
            {"preprocessing_enabled": True, "panorama_scaling": False},
            runtime_settings={"backend": "heuristic", "talc_backend": "heuristic", "grain_backend": "heuristic"},
            run_async=False,
        )

        self.assertEqual(run["status"], "complete")
        self.assertEqual(self.store.talc_backend, "ml")
        self.assertEqual(self.store.talc_checkpoint, talc_checkpoint.resolve())
        self.assertEqual(run["runtime"]["backend"], "heuristic")
        self.assertEqual(run["runtime"]["talc_backend"], "heuristic")
        self.assertIsNone(run["runtime"]["checkpoints"]["talc"])
        self.assertEqual(run["runtime"]["models"]["talc"]["backend"], "heuristic_candidate")
        self.assertEqual(run["runtime"]["grain_backend"], "heuristic")

    def test_run_files_payload_and_zip_endpoint_include_image_dimensions(self) -> None:
        upload = self.store.register_upload_from_path(self.image_path)
        preprocessed = self.store.prepare_upload(
            upload["upload_id"],
            {
                "preprocessing_enabled": True,
                "illumination_normalization": True,
                "denoise": True,
                "contrast_correction": True,
                "panorama_scaling": True,
            },
        )
        run = self.store.start_run(upload["upload_id"], preprocessed["preprocess"]["preset"], run_async=False)
        files_payload = self.store.run_files_payload(run["run_id"])
        files_by_path = {file["path"]: file for file in files_payload["files"]}
        self.assertIn("run.json", files_by_path)
        self.assertIn("input/preprocessed.png", files_by_path)
        self.assertTrue(files_by_path["input/preprocessed.png"]["is_image"])
        self.assertEqual(files_by_path["input/preprocessed.png"]["width"], 128)
        self.assertEqual(files_by_path["input/preprocessed.png"]["height"], 96)
        self.assertTrue(files_by_path["input/preprocessed.png"]["view_url"].startswith("/artifacts/"))
        self.assertIn("reports/ore_summary.json", files_by_path)
        self.assertIn("reports/runtime.json", files_by_path)
        self.assertIn("masks/sulfide_component_labels_rgb.png", files_by_path)
        self.assertFalse(files_by_path["reports/ore_summary.json"]["is_image"])
        self.assertNotIn("reports/run_artifacts.zip", files_by_path)

        zip_path = self.store.run_zip_path(run["run_id"])
        self.assertTrue(zip_path.exists())
        with zipfile.ZipFile(zip_path) as archive:
            names = set(archive.namelist())
        self.assertIn("run.json", names)
        self.assertIn("input/preprocessed.png", names)
        self.assertIn("reports/ore_summary.json", names)
        self.assertIn("reports/runtime.json", names)
        self.assertNotIn("reports/run_artifacts.zip", names)

        server = OrePipelineHTTPServer(("127.0.0.1", 0), self.store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address[:2]
        try:
            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request("GET", f"/api/runs/{run['run_id']}/files")
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()
            self.assertEqual(response.status, 200)
            self.assertEqual(payload["run_id"], run["run_id"])
            http_files = {file["path"]: file for file in payload["files"]}
            self.assertEqual(http_files["input/preprocessed.png"]["width"], 128)
            self.assertTrue(http_files["run.json"]["view_url"].startswith("/artifacts/"))

            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request("GET", http_files["run.json"]["view_url"])
            response = connection.getresponse()
            body = response.read()
            content_type = response.getheader("Content-Type") or ""
            connection.close()
            self.assertEqual(response.status, 200)
            self.assertIn("application/json", content_type)
            self.assertIn(b'"run_id"', body)

            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request("GET", f"/api/runs/{run['run_id']}/artifacts.zip")
            response = connection.getresponse()
            body = response.read()
            disposition = response.getheader("Content-Disposition") or ""
            content_type = response.getheader("Content-Type") or ""
            connection.close()
            self.assertEqual(response.status, 200)
            self.assertIn("application/zip", content_type)
            self.assertIn("attachment", disposition)
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                self.assertIn("run.json", archive.namelist())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_file_downloads_stream_without_reading_entire_file(self) -> None:
        artifact_body = b"streamed artifact body" * 4096
        artifact_path = self.root / "streamed.bin"
        artifact_path.write_bytes(artifact_body)
        self.store.artifacts["stream-test"] = artifact_path
        upload = self.store.register_upload_from_path(self.image_path)
        preprocessed = self.store.prepare_upload(upload["upload_id"], {"preprocessing_enabled": True, "panorama_scaling": True})
        run = self.store.start_run(upload["upload_id"], preprocessed["preprocess"]["preset"], run_async=False)
        zip_path = self.store.run_zip_path(run["run_id"])

        server = OrePipelineHTTPServer(("127.0.0.1", 0), self.store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address[:2]
        try:
            with mock.patch("pathlib.Path.read_bytes", side_effect=AssertionError("send_file must stream")):
                connection = http.client.HTTPConnection(host, port, timeout=5)
                connection.request("GET", "/artifacts/stream-test/streamed.bin")
                response = connection.getresponse()
                body = response.read()
                content_length = response.getheader("Content-Length")
                connection.close()
                self.assertEqual(response.status, 200)
                self.assertEqual(content_length, str(len(artifact_body)))
                self.assertEqual(body, artifact_body)

                connection = http.client.HTTPConnection(host, port, timeout=5)
                connection.request("GET", "/artifacts/stream-test/streamed.bin", headers={"Range": "bytes=10-29"})
                response = connection.getresponse()
                ranged_body = response.read()
                content_length = response.getheader("Content-Length")
                content_range = response.getheader("Content-Range")
                accept_ranges = response.getheader("Accept-Ranges")
                connection.close()
                self.assertEqual(response.status, HTTPStatus.PARTIAL_CONTENT)
                self.assertEqual(content_length, "20")
                self.assertEqual(content_range, f"bytes 10-29/{len(artifact_body)}")
                self.assertEqual(accept_ranges, "bytes")
                self.assertEqual(ranged_body, artifact_body[10:30])

                connection = http.client.HTTPConnection(host, port, timeout=5)
                connection.request("GET", "/artifacts/stream-test/streamed.bin", headers={"Range": f"bytes={len(artifact_body)}-"})
                response = connection.getresponse()
                ranged_body = response.read()
                content_range = response.getheader("Content-Range")
                connection.close()
                self.assertEqual(response.status, HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.assertEqual(content_range, f"bytes */{len(artifact_body)}")
                self.assertEqual(ranged_body, b"")

                connection = http.client.HTTPConnection(host, port, timeout=5)
                connection.request("GET", f"/api/runs/{run['run_id']}/artifacts.zip")
                response = connection.getresponse()
                zip_body = response.read()
                content_length = response.getheader("Content-Length")
                connection.close()
                self.assertEqual(response.status, 200)
                self.assertEqual(content_length, str(zip_path.stat().st_size))
                with zipfile.ZipFile(io.BytesIO(zip_body)) as archive:
                    self.assertIn("run.json", archive.namelist())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_runtime_provenance_enriches_ml_checkpoint_summary(self) -> None:
        checkpoint = self.root / "ml_best.pt"
        checkpoint.write_bytes(b"fake")
        talc_checkpoint = self.root / "talc_best.pt"
        talc_checkpoint.write_bytes(b"fake talc")
        run_dir = self.store.runs_dir / "runtime_probe"
        (run_dir / "ml_pipeline/binary_sulfide").mkdir(parents=True)
        (run_dir / "ml_pipeline/talc_model").mkdir(parents=True)
        (run_dir / "ml_pipeline").mkdir(parents=True, exist_ok=True)
        (run_dir / "ml_pipeline/binary_sulfide/summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "binary-sulfide-inference-v0.2",
                    "checkpoint": str(checkpoint),
                    "checkpoint_meta": {
                        "model": "segformer_b2",
                        "epoch": 17,
                        "best_iou_sulfide": 0.97,
                        "state_dict_compatibility": "segformer_transformers_namespace_remap",
                    },
                    "device": "mps",
                    "tile_size": 1024,
                    "stride": 768,
                    "threshold": 0.5,
                    "tiles": 8,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "ml_pipeline/talc_model/summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "binary-talc-inference-v0.1",
                    "checkpoint": str(talc_checkpoint),
                    "checkpoint_meta": {"model": "segformer_b0", "epoch": 3},
                    "device": "mps",
                    "tile_size": 1024,
                    "stride": 768,
                    "threshold": 0.42,
                    "tiles": 8,
                    "talc_fraction_non_sulfide": 0.12,
                    "talc_fraction_analyzed": 0.09,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "ml_pipeline/pipeline_summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "ore-pipeline-run-v0.2",
                    "image": "input/preprocessed.png",
                    "talc_source": "ml_model",
                    "talc_checkpoint": str(talc_checkpoint),
                    "talc_threshold": 0.42,
                    "rule_config": {"ordinary_component_max_area_px": 1000},
                }
            ),
            encoding="utf-8",
        )
        metadata = {
            "run_id": "runtime_probe",
            "backend": "ml",
            "checkpoint": str(checkpoint),
            "completed_at": "2026-07-03T00:00:00+00:00",
            "reports": {},
        }

        runtime = self.store._finalize_runtime_provenance(metadata, run_dir)

        self.assertEqual(runtime["backend"], "ml")
        self.assertEqual(runtime["checkpoints"]["binary_sulfide"], str(checkpoint.resolve()))
        self.assertEqual(runtime["models"]["binary_sulfide"]["checkpoint_meta"]["model"], "segformer_b2")
        self.assertEqual(runtime["models"]["binary_sulfide"]["device"], "mps")
        self.assertEqual(runtime["checkpoints"]["talc"], str(talc_checkpoint.resolve()))
        self.assertEqual(runtime["models"]["talc"]["backend"], "ml_model")
        self.assertEqual(runtime["models"]["talc"]["checkpoint_meta"]["model"], "segformer_b0")
        self.assertEqual(runtime["models"]["talc"]["threshold"], 0.42)
        self.assertEqual(runtime["models"]["final_segmentation"]["backend"], "component_rules")
        self.assertEqual(metadata["reports"]["runtime_json"], str(run_dir / "reports/runtime.json"))
        self.assertTrue((run_dir / "reports/runtime.json").exists())

    def test_panorama_scaling_uses_explicit_bound_or_factor(self) -> None:
        upload = self.store.register_upload_from_path(self.image_path)

        bounded = self.store.prepare_upload(
            upload["upload_id"],
            {
                "preprocessing_enabled": True,
                "panorama_scaling": True,
                "panorama_scaling_mode": "max_side",
                "panorama_max_side_px": 96,
            },
        )
        self.assertEqual(bounded["preprocess"]["target_max_side"], 96)
        self.assertEqual(bounded["preprocess"]["width"], 96)
        self.assertEqual(bounded["preprocess"]["height"], 72)
        self.assertEqual(bounded["preprocess"]["full_width"], 160)
        self.assertEqual(bounded["preprocess"]["full_height"], 120)
        with Image.open(bounded["preprocess"]["preprocessed_full_path"]) as image:
            self.assertEqual(image.size, (160, 120))
        with Image.open(bounded["preprocess"]["preprocessed_path"]) as image:
            self.assertEqual(image.size, (96, 72))
        self.assertEqual(bounded["preprocess"]["panorama_scaling"]["mode"], "max_side")
        self.assertEqual(bounded["preprocess"]["panorama_scaling"]["max_side_px"], 96)

        factored = self.store.prepare_upload(
            upload["upload_id"],
            {
                "preprocessing_enabled": True,
                "panorama_scaling": True,
                "panorama_scaling_mode": "scale_factor",
                "panorama_scale_factor": 0.5,
            },
        )
        self.assertEqual(factored["preprocess"]["target_max_side"], 80)
        self.assertEqual(factored["preprocess"]["width"], 80)
        self.assertEqual(factored["preprocess"]["height"], 60)
        self.assertEqual(factored["preprocess"]["full_width"], 160)
        self.assertEqual(factored["preprocess"]["full_height"], 120)
        self.assertEqual(factored["preprocess"]["panorama_scaling"]["mode"], "scale_factor")
        self.assertEqual(factored["preprocess"]["panorama_scaling"]["scale_factor"], 0.5)

        off = self.store.prepare_upload(
            upload["upload_id"],
            {
                "preprocessing_enabled": True,
                "panorama_scaling": False,
                "panorama_scaling_mode": "scale_factor",
                "panorama_scale_factor": 0.5,
            },
        )
        self.assertEqual(off["preprocess"]["target_max_side"], 256)
        self.assertFalse(off["preprocess"]["source_scaled_for_processing"])
        self.assertEqual(off["preprocess"]["width"], 160)
        self.assertEqual(off["preprocess"]["height"], 120)
        self.assertEqual(off["preprocess"]["full_width"], 160)
        self.assertEqual(off["preprocess"]["full_height"], 120)
        self.assertEqual(off["preprocess"]["panorama_scaling"]["mode"], "off")
        self.assertFalse(off["preprocess"]["panorama_scaling"]["enabled"])

    def test_start_run_can_skip_preprocessing_and_hide_preprocessed_layer(self) -> None:
        upload = self.store.register_upload_from_path(self.image_path)

        run = self.store.start_run(
            upload["upload_id"],
            {
                "preprocessing_enabled": False,
                "illumination_normalization": True,
                "denoise": True,
                "contrast_correction": True,
                "panorama_scaling": True,
            },
            run_async=False,
        )

        self.assertEqual(run["status"], "complete")
        self.assertFalse(run["preprocess"]["enabled"])
        self.assertFalse(run["preprocess"]["preset"]["preprocessing_enabled"])
        self.assertNotIn("preprocessed", run["display"])
        self.assertIn("original", run["display"])
        self.assertIn("non_sulfide_base", run["display"])
        self.assertIn("sulfide_overlay", run["display"])

    def test_upload_artifact_mask_is_used_by_next_run(self) -> None:
        upload = self.store.register_upload_from_path(self.image_path)
        prepared = self.store.prepare_upload(
            upload["upload_id"],
            {"preprocessing_enabled": True, "panorama_scaling": False},
        )
        height = int(prepared["preprocess"]["height"])
        width = int(prepared["preprocess"]["width"])
        artifact = np.zeros((height, width), dtype=np.uint8)
        artifact[12:42, 18:58] = 255

        saved = self.store.save_upload_artifact_mask(
            upload["upload_id"],
            {"mask_png": mask_data_url(artifact), "comment": "polishing scratch"},
        )
        self.assertIn("artifact_mask", saved)
        self.assertTrue(saved["artifact_mask"]["mask_url"])

        run = self.store.start_run(
            upload["upload_id"],
            {"preprocessing_enabled": True, "panorama_scaling": False},
            run_async=False,
        )
        self.assertEqual(run["status"], "complete")
        self.assertIn("artifact", run["masks"])
        run_dir = self.store.runs_dir / run["run_id"]
        analyzed = np.asarray(Image.open(run_dir / "masks/analyzed_mask.png").convert("L"))
        sulfide = np.asarray(Image.open(run_dir / "masks/sulfide_mask.png").convert("L"))
        final = np.asarray(Image.open(run_dir / "masks/final_mask.png").convert("L"))
        saved_artifact = np.asarray(Image.open(run_dir / "masks/artifact_mask.png").convert("L"))
        self.assertGreater(int(saved_artifact[12:42, 18:58].sum()), 0)
        self.assertEqual(int(analyzed[12:42, 18:58].sum()), 0)
        self.assertEqual(int(sulfide[12:42, 18:58].sum()), 0)
        self.assertEqual(int(final[12:42, 18:58].sum()), 0)

    def test_augmentation_runs_before_preprocessing_and_adds_debug_layer(self) -> None:
        upload = self.store.register_upload_from_path(self.image_path)
        augmentation = {
            "enabled": True,
            "color": {
                "brightness_pct": 28,
                "contrast_pct": 20,
                "saturation_pct": 10,
                "hue_degrees": 0,
                "gamma": 1.0,
            },
            "acquisition": {"blur_radius": 0, "gaussian_noise_std": 0},
            "surface_artifacts": {
                "scratch_count": 8,
                "scratch_intensity_pct": 24,
                "polishing_haze_pct": 10,
                "pit_count": 16,
                "pit_intensity_pct": 20,
            },
            "runtime": {"random_seed": 7},
        }

        prepared = self.store.prepare_upload(
            upload["upload_id"],
            {
                "preprocessing_enabled": True,
                "illumination_normalization": False,
                "denoise": False,
                "contrast_correction": False,
                "panorama_scaling": False,
            },
            augmentation,
        )

        self.assertTrue(prepared["augmentation"]["enabled"])
        self.assertIn("augmented", prepared["display"])
        self.assertIn("preprocessed", prepared["display"])
        self.assertEqual(prepared["augmentation"]["width"], 160)
        self.assertEqual(prepared["augmentation"]["height"], 120)
        self.assertEqual(prepared["augmentation"]["source_width"], 160)
        self.assertEqual(prepared["augmentation"]["source_height"], 120)
        with Image.open(prepared["augmentation"]["augmented_path"]) as image:
            self.assertEqual(image.size, (160, 120))
        self.assertEqual(prepared["preprocess"]["full_width"], 160)
        self.assertEqual(prepared["preprocess"]["full_height"], 120)
        with Image.open(prepared["preprocess"]["preprocessed_full_path"]) as image:
            self.assertEqual(image.size, (160, 120))

        run = self.store.start_run(
            upload["upload_id"],
            {
                "preprocessing_enabled": False,
                "illumination_normalization": True,
                "denoise": True,
                "contrast_correction": True,
                "panorama_scaling": False,
            },
            run_async=False,
            augmentation_settings=augmentation,
        )

        self.assertTrue(run["augmentation"]["enabled"])
        self.assertIn("augmented", run["display"])
        self.assertNotIn("preprocessed", run["display"])
        self.assertIn("augmented_path", run["input"])
        run_dir = self.store.runs_dir / run["run_id"]
        original = np.asarray(Image.open(run_dir / "input/original_for_analysis.png").convert("RGB"))
        augmented = np.asarray(Image.open(run_dir / "input/augmented.png").convert("RGB"))
        with Image.open(run_dir / "input/augmented.png") as image:
            augmented_full_size = image.size
        preprocessed = np.asarray(Image.open(run_dir / "input/preprocessed.png").convert("RGB"))
        self.assertEqual(augmented_full_size, (160, 120))
        self.assertFalse(np.array_equal(original, augmented))
        np.testing.assert_array_equal(augmented, preprocessed)
        self.assertEqual(run["augmentation"]["settings"]["runtime"]["coordinate_mode"], "original")
        self.assertEqual(run["augmentation"]["settings"]["surface_artifacts"]["scratch_count"], 8)
        self.assertEqual(run["augmentation"]["settings"]["surface_artifacts"]["pit_count"], 16)

    def test_apply_after_completed_run_creates_prepared_run_then_start_continues_it(self) -> None:
        upload = self.store.register_upload_from_path(self.image_path)
        run = self.store.start_run(
            upload["upload_id"],
            {"preprocessing_enabled": True, "panorama_scaling": False},
            run_async=False,
        )
        self.assertEqual(run["status"], "complete")
        augmentation = {
            "enabled": True,
            "color": {"brightness_pct": 12, "contrast_pct": 7, "saturation_pct": 3, "hue_degrees": 0, "gamma": 1.0},
            "acquisition": {"blur_radius": 0, "gaussian_noise_std": 0},
            "surface_artifacts": {"scratch_count": 2, "scratch_intensity_pct": 10, "polishing_haze_pct": 0, "pit_count": 0, "pit_intensity_pct": 0},
            "runtime": {"random_seed": 11},
        }

        prepared = self.store.prepare_run_from_apply(
            run["run_id"],
            {"preprocessing_enabled": True, "panorama_scaling": False},
            augmentation_settings=augmentation,
            changed_step="augmentation",
        )

        self.assertNotEqual(prepared["run_id"], run["run_id"])
        self.assertEqual(prepared["status"], "prepared")
        self.assertEqual(prepared["progress"], 0)
        self.assertEqual(prepared["derivation"]["parent_run_id"], run["run_id"])
        self.assertEqual(prepared["derivation"]["changed_step"], "augmentation")
        self.assertTrue(prepared["augmentation"]["enabled"])
        self.assertIn("augmented", prepared["display"])
        self.assertIn("preprocessed", prepared["display"])
        self.assertNotIn("sulfide", prepared["masks"])
        self.assertEqual(prepared["summary"], {})
        self.assertEqual(prepared["metrics"], [])
        run_dir = self.store.runs_dir / prepared["run_id"]
        self.assertTrue((run_dir / "input/augmented.png").exists())
        self.assertTrue((run_dir / "input/preprocessed.png").exists())
        self.assertFalse((run_dir / "masks/sulfide_mask.png").exists())

        completed = self.store.start_prepared_run(prepared["run_id"], run_async=False)
        self.assertEqual(completed["run_id"], prepared["run_id"])
        self.assertEqual(completed["status"], "complete")
        self.assertIn("sulfide", completed["masks"])
        self.assertIn("ordinary_overlay", completed["display"])

    def test_apply_on_prepared_run_updates_same_run_before_start(self) -> None:
        upload = self.store.register_upload_from_path(self.image_path)
        run = self.store.start_run(
            upload["upload_id"],
            {"preprocessing_enabled": True, "panorama_scaling": False},
            run_async=False,
        )
        prepared = self.store.prepare_run_from_apply(
            run["run_id"],
            {"preprocessing_enabled": True, "panorama_scaling": False},
            augmentation_settings={"enabled": False},
            changed_step="preprocess",
        )
        updated = self.store.prepare_run_from_apply(
            prepared["run_id"],
            {"preprocessing_enabled": False, "panorama_scaling": False},
            augmentation_settings={"enabled": False},
            changed_step="preprocess",
        )

        self.assertEqual(updated["run_id"], prepared["run_id"])
        self.assertEqual(updated["status"], "prepared")
        self.assertFalse(updated["preprocess"]["enabled"])
        self.assertEqual(updated["derivation"]["parent_run_id"], run["run_id"])
        self.assertNotIn("preprocessed", updated["display"])
        self.assertEqual(len(self.store.list_runs()["runs"]), 2)

    def test_curated_metadata_is_saved_with_run_and_inherited_by_edit_run(self) -> None:
        upload = self.store.register_upload_from_path(self.image_path)
        curated = {
            "schema_version": "ore-pipeline-curated-metadata-v0.1",
            "domain": {
                "sample_id": "sample-17",
                "project": "hackathon",
                "pixel_size_um": "0.42",
                "scale_source": "calibration_slide",
                "scale_confidence": "calibrated",
            },
            "raw_summary": {
                "original_name": upload["original_name"],
                "width": upload["width"],
                "height": upload["height"],
            },
            "client_note": "preserve unknown fields",
        }

        run = self.store.start_run(
            upload["upload_id"],
            {"panorama_scaling": False},
            run_async=False,
            curated_metadata=curated,
        )

        self.assertEqual(run["input"]["curated_metadata"]["domain"]["sample_id"], "sample-17")
        self.assertEqual(run["input"]["curated_metadata"]["extra"]["client_note"], "preserve unknown fields")
        metadata_path = Path(run["input"]["curated_metadata_json"])
        self.assertTrue(metadata_path.exists())
        self.assertEqual(json.loads(metadata_path.read_text(encoding="utf-8"))["domain"]["project"], "hackathon")

        final_mask = np.asarray(Image.open(self.store.runs_dir / run["run_id"] / "masks/final_mask.png").convert("L")).copy()
        final_mask[0:8, 0:8] = 3
        edited = self.store.create_edit_run(
            run["run_id"],
            {
                "edit_layer": "final",
                "mask_png": mask_data_url(final_mask),
                "comment": "metadata inheritance check",
            },
        )

        self.assertEqual(edited["input"]["curated_metadata"]["domain"]["sample_id"], "sample-17")
        self.assertTrue(Path(edited["input"]["curated_metadata_json"]).exists())

    def test_calibrated_microns_per_pixel_adds_physical_area_to_metrics_and_csv(self) -> None:
        upload = self.store.register_upload_from_path(self.image_path)
        run = self.store.start_run(
            upload["upload_id"],
            {"preprocessing_enabled": False, "panorama_scaling": False},
            run_async=False,
            curated_metadata={
                "domain": {
                    "microns_per_pixel": "0.5",
                    "scale_source": "calibration_slide",
                    "scale_confidence": "calibrated",
                }
            },
        )

        self.assertEqual(run["scale"]["source_field"], "microns_per_pixel")
        self.assertAlmostEqual(run["scale"]["area_um2_per_analysis_pixel"], 0.25)
        sulfide_metric = next(row for row in run["metrics"] if row["key"] == "sulfide_fraction")
        expected_area_um2 = run["summary"]["sulfide_area_px"] * 0.25
        self.assertEqual(sulfide_metric["area_px"], run["summary"]["sulfide_area_px"])
        self.assertAlmostEqual(sulfide_metric["area_um2"], expected_area_um2)
        self.assertAlmostEqual(sulfide_metric["area_mm2"], expected_area_um2 / 1_000_000.0)

        csv_text = self.store.metrics_csv_path(run["run_id"]).read_text(encoding="utf-8")
        csv_rows = list(csv.DictReader(io.StringIO(csv_text)))
        sulfide_csv = next(row for row in csv_rows if row["key"] == "sulfide_fraction")
        self.assertEqual(int(sulfide_csv["area_px"]), run["summary"]["sulfide_area_px"])
        self.assertAlmostEqual(float(sulfide_csv["area_um2"]), expected_area_um2, places=6)
        self.assertEqual(sulfide_csv["scale_source"], "calibration_slide")
        self.assertEqual(sulfide_csv["scale_confidence"], "calibrated")

    def test_curated_metadata_api_rejects_non_object_payload(self) -> None:
        upload = self.store.register_upload_from_path(self.image_path)
        server = OrePipelineHTTPServer(("127.0.0.1", 0), self.store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address[:2]
        try:
            connection = http.client.HTTPConnection(host, port, timeout=5)
            body = json.dumps({"upload_id": upload["upload_id"], "curated_metadata": "bad"}).encode("utf-8")
            connection.request(
                "POST",
                "/api/runs/start",
                body=body,
                headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            )
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()
            self.assertEqual(response.status, 400)
            self.assertIn("curated_metadata must be an object", payload["error"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_slug_page_routes(self) -> None:
        server = OrePipelineHTTPServer(("127.0.0.1", 0), self.store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address[:2]
        try:
            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request("GET", "/")
            response = connection.getresponse()
            response.read()
            connection.close()
            self.assertEqual(response.status, 302)
            self.assertEqual(response.getheader("Location"), "/workspace")

            for page in ("/workspace", "/batch", "/batch/batch_demo", "/history", "/history_series", "/settings", "/status", "/api"):
                connection = http.client.HTTPConnection(host, port, timeout=5)
                connection.request("GET", page)
                response = connection.getresponse()
                body = response.read().decode("utf-8")
                connection.close()
                self.assertEqual(response.status, 200)
                self.assertIn("Классификатор рудного шлифа", body)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_status_payload_and_api_report_system_health(self) -> None:
        self.assertEqual(ore_pipeline_web.DEFAULT_TALC_BACKEND, "ml")
        self.assertEqual(self.store.app_settings()["runtime"]["talc_backend"], "heuristic")
        upload = self.store.register_upload_from_path(self.image_path)
        run = self.store.start_run(
            upload["upload_id"],
            {"preprocessing_enabled": True, "panorama_scaling": False},
            run_async=False,
        )
        payload = self.store.status_payload()

        self.assertEqual(payload["schema_version"], "ore-pipeline-status-v0.1")
        self.assertEqual(payload["app"]["version"], ore_pipeline_web.APP_VERSION)
        self.assertIn(payload["health"]["overall"], {"ok", "warning", "error"})
        self.assertGreaterEqual(payload["cpu"]["logical_cpus"], 1)
        self.assertIn("available", payload["gpu"])
        self.assertIn("total_bytes", payload["ram"])
        self.assertIn("free_bytes", payload["flash"])
        self.assertGreaterEqual(payload["history"]["runs_total"], 1)
        self.assertGreaterEqual(payload["history"]["history_size_bytes"], 0)
        self.assertEqual(payload["history"]["run_status_counts"].get(run["status"]), 1)
        self.assertEqual(payload["app"]["backend"], "heuristic")
        self.assertEqual(payload["app"]["talc_backend"], "heuristic")
        self.assertEqual(payload["app"]["models"]["binary_sulfide"]["backend"], "heuristic")
        self.assertEqual(payload["app"]["models"]["talc"]["backend"], "heuristic_candidate")
        self.assertIn("logs", payload)
        self.assertGreaterEqual(len(payload["logs"]["system"]), 1)

        server = OrePipelineHTTPServer(("127.0.0.1", 0), self.store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address[:2]
        try:
            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request("GET", "/workspace")
            response = connection.getresponse()
            response.read()
            connection.close()

            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request("GET", "/api/status")
            response = connection.getresponse()
            api_payload = json.loads(response.read().decode("utf-8"))
            connection.close()
            self.assertEqual(response.status, 200)
            self.assertEqual(api_payload["schema_version"], "ore-pipeline-status-v0.1")
            self.assertEqual(api_payload["history"]["runs_total"], payload["history"]["runs_total"])
            self.assertTrue(any(entry.get("path") == "/workspace" for entry in api_payload["logs"]["access"]))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_status_payload_caches_expensive_computation_but_refreshes_access_log(self) -> None:
        server = OrePipelineHTTPServer(("127.0.0.1", 0), self.store)
        try:
            with mock.patch.object(
                self.store, "status_payload", wraps=self.store.status_payload
            ) as spy:
                first = server.status_payload()
                second = server.status_payload()
                # Second call within the TTL must reuse the cached store computation.
                self.assertEqual(spy.call_count, 1)

                # The access log is still injected fresh: a new request logged between
                # the two calls appears in the later payload despite the cache hit.
                server.record_access_event(
                    client="1.2.3.4", method="GET", path="/workspace", status=200, size=10
                )
                third = server.status_payload()
                self.assertEqual(spy.call_count, 1)
                self.assertTrue(
                    any(entry.get("path") == "/workspace" for entry in third["logs"]["access"])
                )
                self.assertFalse(
                    any(entry.get("path") == "/workspace" for entry in first["logs"]["access"])
                )

                # Expiring the cache forces exactly one more recompute.
                server._status_cache_monotonic -= ore_pipeline_web.STATUS_CACHE_TTL_SECONDS + 1.0
                server.status_payload()
                self.assertEqual(spy.call_count, 2)

            # Each request gets its own logs dict, so the shared cache cannot be mutated.
            self.assertIsNot(first["logs"], second["logs"])
        finally:
            server.server_close()

    def test_status_payload_reports_foreground_operations(self) -> None:
        operation_id = self.store.begin_foreground_operation(
            "preprocess",
            "preparing upload",
            path="/api/uploads/upload_demo/preprocess",
            upload_id="upload_demo",
        )
        try:
            payload = self.store.status_payload()
            operations = payload["history"]["active_operations"]

            self.assertEqual(len(operations), 1)
            self.assertEqual(operations[0]["operation_id"], operation_id)
            self.assertEqual(operations[0]["kind"], "preprocess")
            self.assertEqual(operations[0]["label"], "preparing upload")
            self.assertEqual(operations[0]["upload_id"], "upload_demo")
            self.assertGreaterEqual(operations[0]["elapsed_seconds"], 0)
            self.assertTrue(any(check["key"] == "active_jobs" for check in payload["health"]["checks"]))
            self.assertIn(operation_id, self.store._active_runtime_jobs())
        finally:
            self.store.finish_foreground_operation(operation_id)

        payload = self.store.status_payload()
        self.assertEqual(payload["history"]["active_operations"], [])
        self.assertNotIn(operation_id, self.store._active_runtime_jobs())

    def test_gpu_status_tolerates_nvidia_smi_na_fields(self) -> None:
        completed = mock.Mock(stdout="0, NVIDIA GB10, 0, [N/A], [N/A], [N/A]\n")
        with (
            mock.patch("apps.ore_pipeline_web.shutil.which", return_value="/usr/bin/nvidia-smi"),
            mock.patch("apps.ore_pipeline_web.subprocess.run", return_value=completed),
        ):
            payload = gpu_status_payload()

        self.assertTrue(payload["available"])
        device = payload["devices"][0]
        self.assertEqual(device["name"], "NVIDIA GB10")
        self.assertEqual(device["utilization_percent"], 0.0)
        self.assertIsNone(device["memory_total_bytes"])
        self.assertIsNone(device["memory_used_bytes"])
        self.assertIsNone(device["memory_used_percent"])
        self.assertIsNone(device["temperature_c"])

    def test_gpu_status_reports_apple_silicon_metal_gpu_without_nvidia_smi(self) -> None:
        ore_pipeline_web._apple_gpu_devices.cache_clear()
        profiler_payload = {
            "SPDisplaysDataType": [
                {
                    "_name": "Apple M2 Max",
                    "sppci_model": "Apple M2 Max",
                    "sppci_device_type": "spdisplays_gpu",
                    "sppci_cores": "38",
                    "spdisplays_mtlgpufamilysupport": "spdisplays_metal4",
                    "spdisplays_ndrvs": [{"_name": "Color LCD"}, {"_name": "External"}],
                }
            ]
        }
        completed = mock.Mock(stdout=json.dumps(profiler_payload))
        try:
            with (
                mock.patch("apps.ore_pipeline_web.sys.platform", "darwin"),
                mock.patch("apps.ore_pipeline_web.shutil.which", side_effect=lambda name: "/usr/sbin/system_profiler" if name == "system_profiler" else None),
                mock.patch("apps.ore_pipeline_web.subprocess.run", return_value=completed),
                mock.patch("apps.ore_pipeline_web._torch_mps_available", return_value=True),
            ):
                payload = gpu_status_payload()
        finally:
            ore_pipeline_web._apple_gpu_devices.cache_clear()

        self.assertTrue(payload["available"])
        self.assertEqual(payload["source"], "system_profiler")
        device = payload["devices"][0]
        self.assertEqual(device["name"], "Apple M2 Max")
        self.assertEqual(device["backend"], "metal")
        self.assertTrue(device["mps_available"])
        self.assertEqual(device["cores"], 38)
        self.assertEqual(device["displays"], 2)
        self.assertIsNone(device["utilization_percent"])

    def test_batch_creates_sequential_runs_and_persists_item_metadata(self) -> None:
        upload_a = self.store.register_upload_from_path(self.image_path)
        upload_b = self.store.register_upload_from_path(self.image_path_2)
        batch = self.store.create_batch(
            {
                "settings": {
                    "preprocess": {"preprocessing_enabled": True, "panorama_scaling": False},
                    "augmentation": {"enabled": False},
                },
                "upload_ids": [upload_a["upload_id"], upload_b["upload_id"]],
            }
        )
        self.assertEqual(batch["status"], "draft")
        self.assertEqual(len(batch["items"]), 2)
        self.assertIn("display", batch["items"][0])

        item_b = batch["items"][1]
        batch = self.store.update_batch_item_metadata(
            batch["batch_id"],
            item_b["item_id"],
            {
                "curated_metadata": {
                    "domain": {"sample_id": "batch-sample-2", "project": "v2-batch"},
                    "raw_summary": {"original_name": upload_b["original_name"]},
                }
            },
        )
        self.assertEqual(batch["items"][1]["curated_metadata"]["domain"]["sample_id"], "batch-sample-2")

        completed = self.store.run_batch(
            batch["batch_id"],
            {
                "preprocess": {"preprocessing_enabled": True, "panorama_scaling": False},
                "augmentation": {"enabled": False},
            },
            run_async=False,
        )

        self.assertEqual(completed["status"], "complete")
        self.assertEqual(completed["progress"], 100)
        self.assertEqual(completed["item_counts"]["complete"], 2)
        self.assertTrue(completed["downloads"]["results_csv"].endswith("/results.csv"))
        for item in completed["items"]:
            self.assertEqual(item["status"], "complete")
            self.assertTrue(item["run_id"])
            run = self.store.run_payload(item["run_id"])
            self.assertEqual(run["batch"]["batch_id"], completed["batch_id"])
            self.assertEqual(run["batch"]["item_id"], item["item_id"])
        second_run = self.store.run_payload(completed["items"][1]["run_id"])
        self.assertEqual(second_run["input"]["curated_metadata"]["domain"]["sample_id"], "batch-sample-2")
        csv_path = self.store.batch_results_csv_path(completed["batch_id"])
        self.assertTrue(csv_path.exists())
        csv_text = csv_path.read_text(encoding="utf-8")
        self.assertIn("batch-sample-2", json.dumps(second_run["input"]["curated_metadata"], ensure_ascii=False))
        self.assertIn("ore_class", csv_text)
        history_runs = self.store.list_runs()["runs"]
        batch_history_runs = [run for run in history_runs if (run.get("batch") or {}).get("batch_id") == completed["batch_id"]]
        self.assertEqual(len(batch_history_runs), 2)
        batches = self.store.list_batches()["batches"]
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["batch_id"], completed["batch_id"])
        self.assertEqual(batches[0]["items_count"], 2)
        self.assertEqual(batches[0]["item_counts"]["complete"], 2)

    def test_batch_draft_item_can_be_removed_before_run(self) -> None:
        upload_a = self.store.register_upload_from_path(self.image_path)
        upload_b = self.store.register_upload_from_path(self.image_path_2)
        batch = self.store.create_batch({"upload_ids": [upload_a["upload_id"], upload_b["upload_id"]]})
        removed_item_id = batch["items"][0]["item_id"]

        batch = self.store.remove_batch_item(batch["batch_id"], removed_item_id)

        self.assertEqual(batch["status"], "draft")
        self.assertEqual(len(batch["items"]), 1)
        self.assertNotIn(removed_item_id, [item["item_id"] for item in batch["items"]])
        self.assertEqual([item["index"] for item in batch["items"]], [1])
        self.assertEqual(batch["item_counts"]["draft"], 1)

    def test_delete_batch_endpoint_removes_series_and_child_runs(self) -> None:
        upload_a = self.store.register_upload_from_path(self.image_path)
        upload_b = self.store.register_upload_from_path(self.image_path_2)
        batch = self.store.create_batch({"upload_ids": [upload_a["upload_id"], upload_b["upload_id"]]})
        completed = self.store.run_batch(
            batch["batch_id"],
            {
                "preprocess": {"preprocessing_enabled": True, "panorama_scaling": False},
                "augmentation": {"enabled": False},
            },
            run_async=False,
        )
        child_run_ids = [item["run_id"] for item in completed["items"]]
        batch_dir = self.store.batches_dir / completed["batch_id"]
        child_run_dirs = [self.store.runs_dir / run_id for run_id in child_run_ids]
        self.assertTrue(batch_dir.exists())
        for run_dir in child_run_dirs:
            self.assertTrue(run_dir.exists())

        server = OrePipelineHTTPServer(("127.0.0.1", 0), self.store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address[:2]
        try:
            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request("DELETE", f"/api/batches/{completed['batch_id']}")
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()
            self.assertEqual(response.status, 200)
            self.assertEqual(payload["removed_batch_id"], completed["batch_id"])
            self.assertEqual(sorted(payload["removed_run_ids"]), sorted(child_run_ids))
            self.assertFalse(batch_dir.exists())
            for run_dir in child_run_dirs:
                self.assertFalse(run_dir.exists())
            self.assertEqual(self.store.list_batches()["batches"], [])
            self.assertEqual(self.store.list_runs()["runs"], [])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_delete_history_endpoint_removes_runs_and_series_but_keeps_uploads_and_settings(self) -> None:
        upload_a = self.store.register_upload_from_path(self.image_path)
        upload_b = self.store.register_upload_from_path(self.image_path_2)
        standalone = self.store.start_run(upload_a["upload_id"], {"panorama_scaling": False}, run_async=False)
        batch = self.store.create_batch({"upload_ids": [upload_b["upload_id"]]})
        completed_batch = self.store.run_batch(
            batch["batch_id"],
            {
                "preprocess": {"preprocessing_enabled": True, "panorama_scaling": False},
                "augmentation": {"enabled": False},
            },
            run_async=False,
        )
        child_run_ids = [item["run_id"] for item in completed_batch["items"]]
        all_run_ids = [standalone["run_id"], *child_run_ids]
        self.store.save_app_settings({"theme": "dark"})

        for run_id in all_run_ids:
            self.assertTrue((self.store.runs_dir / run_id / "run.json").exists())
        self.assertTrue((self.store.batches_dir / completed_batch["batch_id"] / "batch_summary.json").exists())
        self.assertTrue((self.store.uploads_dir / upload_a["upload_id"] / "upload.json").exists())
        self.assertTrue(self.store.settings_path.exists())

        server = OrePipelineHTTPServer(("127.0.0.1", 0), self.store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address[:2]
        try:
            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request("DELETE", "/api/history")
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()
            self.assertEqual(response.status, 200)
            self.assertEqual(payload["schema_version"], "ore-pipeline-history-delete-v0.1")
            self.assertEqual(sorted(payload["removed_run_ids"]), sorted(all_run_ids))
            self.assertEqual(payload["removed_batch_ids"], [completed_batch["batch_id"]])
            self.assertEqual(self.store.list_runs()["runs"], [])
            self.assertEqual(self.store.list_batches()["batches"], [])
            for run_id in all_run_ids:
                self.assertFalse((self.store.runs_dir / run_id).exists())
            self.assertFalse((self.store.batches_dir / completed_batch["batch_id"]).exists())
            self.assertTrue((self.store.uploads_dir / upload_a["upload_id"] / "upload.json").exists())
            self.assertTrue((self.store.uploads_dir / upload_b["upload_id"] / "upload.json").exists())
            self.assertTrue(self.store.settings_path.exists())
            self.assertEqual(self.store.app_settings()["theme"], "dark")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_delete_history_rejects_active_jobs(self) -> None:
        upload = self.store.register_upload_from_path(self.image_path)
        run = self.store.start_run(upload["upload_id"], {"panorama_scaling": False}, run_async=False)
        with self.store.lock:
            self.store.jobs[run["run_id"]] = {"status": "running", "progress": 50}

        with self.assertRaises(ApiError):
            self.store.delete_history()

        self.assertTrue((self.store.runs_dir / run["run_id"] / "run.json").exists())

    def test_app_settings_are_persisted_and_exposed_by_api(self) -> None:
        fake_checkpoint = self.root / "fake_checkpoint.pt"
        fake_checkpoint.write_bytes(b"fake checkpoint")
        fake_talc_checkpoint = self.root / "fake_talc_checkpoint.pt"
        fake_talc_checkpoint.write_bytes(b"fake talc checkpoint")
        fake_grade_checkpoint = self.root / "fake_grade_checkpoint.pt"
        fake_grade_checkpoint.write_bytes(b"fake grade checkpoint")
        fake_grade_checkpoint = self.root / "fake_grade_checkpoint.pt"
        fake_grade_checkpoint.write_bytes(b"fake grade checkpoint")
        settings = self.store.save_app_settings(
            {
                "language": "en",
                "theme": "dark",
                "show_tiling": True,
                "runtime": {
                    "backend": "ml",
                    "checkpoint": str(fake_checkpoint),
                    "talc_backend": "ml",
                    "talc_checkpoint": str(fake_talc_checkpoint),
                    "talc_threshold": 0.42,
                    "grain_backend": "ml",
                    "grade_checkpoint": str(fake_grade_checkpoint),
                },
                "preprocess": {
                    "preprocessing_enabled": False,
                    "illumination_normalization": True,
                    "denoise": False,
                    "contrast_correction": True,
                    "panorama_scaling": False,
                    "panorama_scaling_mode": "scale_factor",
                    "panorama_max_side_px": 4096,
                    "panorama_scale_factor": 0.25,
                },
                "talc_clusterization": {
                    "radius_px": 96,
                    "min_local_talc_percent": 7.5,
                    "opacity_percent": 60,
                },
                "metadata_defaults": {
                    "project": "system-default-project",
                    "om_instrument": "scope-1",
                    "sample_id": "ignored-sample-specific",
                },
            }
        )
        self.assertEqual(settings["language"], "en")
        self.assertEqual(settings["theme"], "dark")
        self.assertTrue(settings["show_tiling"])
        self.assertEqual(settings["runtime"]["backend"], "ml")
        self.assertEqual(settings["runtime"]["checkpoint"], str(fake_checkpoint.resolve()))
        self.assertEqual(settings["runtime"]["talc_backend"], "ml")
        self.assertEqual(settings["runtime"]["talc_checkpoint"], str(fake_talc_checkpoint.resolve()))
        self.assertEqual(settings["runtime"]["talc_threshold"], 0.42)
        self.assertEqual(settings["runtime"]["grain_backend"], "ml")
        self.assertEqual(settings["runtime"]["grade_checkpoint"], str(fake_grade_checkpoint.resolve()))
        self.assertEqual(self.store.backend, "ml")
        self.assertEqual(self.store.checkpoint, fake_checkpoint.resolve())
        self.assertEqual(self.store.talc_backend, "ml")
        self.assertEqual(self.store.talc_checkpoint, fake_talc_checkpoint.resolve())
        self.assertEqual(self.store.talc_threshold, 0.42)
        self.assertEqual(self.store.grain_backend, "ml")
        self.assertEqual(self.store.grade_checkpoint, fake_grade_checkpoint.resolve())
        self.assertFalse(settings["preprocess"]["preprocessing_enabled"])
        self.assertEqual(settings["preprocess"]["panorama_scaling_mode"], "scale_factor")
        self.assertEqual(settings["preprocess"]["panorama_max_side_px"], 4096)
        self.assertEqual(settings["preprocess"]["panorama_scale_factor"], 0.25)
        self.assertEqual(settings["metadata_defaults"]["project"], "system-default-project")
        self.assertEqual(settings["talc_clusterization"]["radius_px"], 96)
        self.assertEqual(settings["talc_clusterization"]["min_local_talc_percent"], 7.5)
        self.assertEqual(settings["talc_clusterization"]["opacity_percent"], 60)
        self.assertNotIn("sample_id", settings["metadata_defaults"])

        restarted = OrePipelineStore(
            workspace_dir=self.root / "workspace",
            backend="heuristic",
            checkpoint=None,
            processing_max_side=256,
            panorama_max_side=128,
            preview_max_sides=(128, 256),
        )
        self.assertEqual(restarted.backend, "ml")
        self.assertEqual(restarted.checkpoint, fake_checkpoint.resolve())
        self.assertEqual(restarted.talc_backend, "ml")
        self.assertEqual(restarted.talc_checkpoint, fake_talc_checkpoint.resolve())
        self.assertEqual(restarted.talc_threshold, 0.42)
        self.assertEqual(restarted.grain_backend, "ml")
        self.assertEqual(restarted.grade_checkpoint, fake_grade_checkpoint.resolve())
        self.assertEqual(restarted.app_settings()["runtime"]["backend"], "ml")
        self.assertEqual(restarted.app_settings()["runtime"]["talc_backend"], "ml")
        self.assertEqual(restarted.app_settings()["runtime"]["grain_backend"], "ml")
        self.assertEqual(restarted.app_settings()["metadata_defaults"]["om_instrument"], "scope-1")
        self.assertEqual(restarted.app_settings()["preprocess"]["panorama_scaling_mode"], "scale_factor")
        self.assertEqual(restarted.app_settings()["preprocess"]["panorama_scale_factor"], 0.25)
        self.assertEqual(restarted.app_settings()["talc_clusterization"]["radius_px"], 96)

        server = OrePipelineHTTPServer(("127.0.0.1", 0), restarted)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address[:2]
        try:
            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request("GET", "/api/settings")
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()
            self.assertEqual(response.status, 200)
            self.assertEqual(payload["language"], "en")
            self.assertEqual(payload["runtime"]["backend"], "ml")
            self.assertEqual(payload["runtime"]["talc_backend"], "ml")
            self.assertEqual(payload["runtime"]["grain_backend"], "ml")

            connection = http.client.HTTPConnection(host, port, timeout=5)
            body = json.dumps({"theme": "neon"}).encode("utf-8")
            connection.request("PUT", "/api/settings", body=body, headers={"Content-Type": "application/json", "Content-Length": str(len(body))})
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()
            self.assertEqual(response.status, 400)
            self.assertIn("settings.theme", payload["error"])

            connection = http.client.HTTPConnection(host, port, timeout=5)
            body = json.dumps({"runtime": {"backend": "ml", "checkpoint": str(self.root / "missing.pt")}}).encode("utf-8")
            connection.request("PUT", "/api/settings", body=body, headers={"Content-Type": "application/json", "Content-Length": str(len(body))})
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()
            self.assertEqual(response.status, 400)
            self.assertIn("settings.runtime.checkpoint", payload["error"])

            connection = http.client.HTTPConnection(host, port, timeout=5)
            body = json.dumps({"runtime": {"talc_backend": "ml", "talc_checkpoint": str(self.root / "missing_talc.pt")}}).encode("utf-8")
            connection.request("PUT", "/api/settings", body=body, headers={"Content-Type": "application/json", "Content-Length": str(len(body))})
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()
            self.assertEqual(response.status, 400)
            self.assertIn("settings.runtime.talc_checkpoint", payload["error"])

            connection = http.client.HTTPConnection(host, port, timeout=5)
            body = json.dumps({"runtime": {"grain_backend": "ml", "grade_checkpoint": str(self.root / "missing_grade.pt")}}).encode("utf-8")
            connection.request("PUT", "/api/settings", body=body, headers={"Content-Type": "application/json", "Content-Length": str(len(body))})
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()
            self.assertEqual(response.status, 400)
            self.assertIn("settings.runtime.grade_checkpoint", payload["error"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_password_auth_settings_are_hashed_public_and_enforced(self) -> None:
        settings = self.store.save_app_settings({"auth": {"password": "secret-pass"}})
        auth = settings["auth"]
        self.assertTrue(auth["password_enabled"])
        self.assertEqual(auth["algorithm"], "pbkdf2_sha256")
        self.assertNotIn("secret-pass", json.dumps(auth))
        self.assertNotIn("password_hash", self.store.public_app_settings()["auth"])
        self.assertEqual(self.store.public_app_settings()["auth"], {"password_enabled": True})
        self.assertTrue(self.store.authenticate_password("secret-pass"))
        self.assertFalse(self.store.authenticate_password("wrong-pass"))

        server = OrePipelineHTTPServer(("127.0.0.1", 0), self.store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address[:2]
        try:
            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request("GET", "/workspace")
            response = connection.getresponse()
            response.read()
            connection.close()
            self.assertEqual(response.status, HTTPStatus.FOUND)
            self.assertTrue((response.getheader("Location") or "").startswith("/login?next=/workspace"))

            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request("GET", "/api/settings")
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()
            self.assertEqual(response.status, HTTPStatus.UNAUTHORIZED)
            self.assertIn("authentication", payload["error"])

            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request("GET", "/api/auth/status")
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()
            self.assertEqual(response.status, HTTPStatus.OK)
            self.assertTrue(payload["password_enabled"])
            self.assertFalse(payload["authenticated"])

            connection = http.client.HTTPConnection(host, port, timeout=5)
            bad_body = json.dumps({"password": "wrong-pass"}).encode("utf-8")
            connection.request("POST", "/api/auth/login", body=bad_body, headers={"Content-Type": "application/json", "Content-Length": str(len(bad_body))})
            response = connection.getresponse()
            response.read()
            connection.close()
            self.assertEqual(response.status, HTTPStatus.UNAUTHORIZED)

            connection = http.client.HTTPConnection(host, port, timeout=5)
            good_body = json.dumps({"password": "secret-pass"}).encode("utf-8")
            connection.request("POST", "/api/auth/login", body=good_body, headers={"Content-Type": "application/json", "Content-Length": str(len(good_body))})
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            set_cookie = response.getheader("Set-Cookie") or ""
            connection.close()
            self.assertEqual(response.status, HTTPStatus.OK)
            self.assertTrue(payload["authenticated"])
            self.assertIn("HttpOnly", set_cookie)
            cookie_header = set_cookie.split(";", 1)[0]

            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request("GET", "/api/settings", headers={"Cookie": cookie_header})
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()
            self.assertEqual(response.status, HTTPStatus.OK)
            self.assertEqual(payload["auth"], {"password_enabled": True})
            self.assertNotIn("password_hash", json.dumps(payload))

            connection = http.client.HTTPConnection(host, port, timeout=5)
            clear_body = json.dumps({"auth": {"clear_password": True}}).encode("utf-8")
            connection.request(
                "PUT",
                "/api/settings",
                body=clear_body,
                headers={"Content-Type": "application/json", "Content-Length": str(len(clear_body)), "Cookie": cookie_header},
            )
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            cleared_cookie = response.getheader("Set-Cookie") or ""
            connection.close()
            self.assertEqual(response.status, HTTPStatus.OK)
            self.assertEqual(payload["auth"], {"password_enabled": False})
            self.assertIn("Max-Age=0", cleared_cookie)

            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request("GET", "/workspace")
            response = connection.getresponse()
            body = response.read().decode("utf-8")
            connection.close()
            self.assertEqual(response.status, HTTPStatus.OK)
            self.assertIn("Классификатор рудного шлифа", body)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_setting_first_password_over_http_returns_authenticated_session(self) -> None:
        server = OrePipelineHTTPServer(("127.0.0.1", 0), self.store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address[:2]
        try:
            connection = http.client.HTTPConnection(host, port, timeout=5)
            body = json.dumps({"auth": {"password": "new-pass"}}).encode("utf-8")
            connection.request("PUT", "/api/settings", body=body, headers={"Content-Type": "application/json", "Content-Length": str(len(body))})
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            set_cookie = response.getheader("Set-Cookie") or ""
            connection.close()
            self.assertEqual(response.status, HTTPStatus.OK)
            self.assertEqual(payload["auth"], {"password_enabled": True})
            self.assertNotIn("password_hash", json.dumps(payload))
            self.assertIn("HttpOnly", set_cookie)
            cookie_header = set_cookie.split(";", 1)[0]

            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request("GET", "/api/settings")
            response = connection.getresponse()
            response.read()
            connection.close()
            self.assertEqual(response.status, HTTPStatus.UNAUTHORIZED)

            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request("GET", "/api/settings", headers={"Cookie": cookie_header})
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()
            self.assertEqual(response.status, HTTPStatus.OK)
            self.assertEqual(payload["auth"], {"password_enabled": True})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_runtime_test_endpoint_checks_heuristic_and_ml_probe(self) -> None:
        fake_checkpoint = self.root / "fake_checkpoint.pt"
        fake_checkpoint.write_bytes(b"fake checkpoint")
        fake_talc_checkpoint = self.root / "fake_talc_checkpoint.pt"
        fake_talc_checkpoint.write_bytes(b"fake talc checkpoint")
        fake_grade_checkpoint = self.root / "fake_grade_checkpoint.pt"
        fake_grade_checkpoint.write_bytes(b"fake grade checkpoint")
        server = OrePipelineHTTPServer(("127.0.0.1", 0), self.store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address[:2]

        def request_runtime_test(body: dict) -> tuple[int, dict]:
            raw = json.dumps(body).encode("utf-8")
            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request(
                "POST",
                "/api/runtime/test",
                body=raw,
                headers={"Content-Type": "application/json", "Content-Length": str(len(raw))},
            )
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()
            return response.status, payload

        try:
            status, payload = request_runtime_test({"runtime": {"backend": "heuristic"}})
            self.assertEqual(status, 200)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["backend"], "heuristic")
            self.assertEqual(payload["status"], "ok")

            with mock.patch("apps.ore_pipeline_web.subprocess.run") as run_probe:
                run_probe.return_value = subprocess.CompletedProcess(
                    args=["python"],
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "device": "cpu",
                            "torch": "test",
                            "transformers": "test",
                            "checkpoint_meta": {"model": "resunet", "epoch": 1},
                            "parameter_count": 123,
                            "seconds": 0.02,
                        }
                    )
                    + "\n",
                    stderr="",
                )
                status, payload = request_runtime_test(
                    {"runtime": {"backend": "ml", "checkpoint": str(fake_checkpoint), "talc_backend": "heuristic"}}
                )
            self.assertEqual(status, 200)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["backend"], "ml")
            self.assertEqual(payload["details"]["checkpoint_meta"]["model"], "resunet")
            self.assertEqual(payload["models"]["binary_sulfide"]["details"]["checkpoint_meta"]["model"], "resunet")
            self.assertEqual(payload["models"]["talc"]["backend"], "auto_candidate")
            self.assertEqual(self.store.backend, "heuristic")

            with mock.patch("apps.ore_pipeline_web.subprocess.run") as run_probe:
                run_probe.return_value = subprocess.CompletedProcess(
                    args=["python"],
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "device": "cpu",
                            "torch": "test",
                            "transformers": "test",
                            "checkpoint_meta": {"model": "segformer_b0", "epoch": 2},
                            "parameter_count": 456,
                            "seconds": 0.03,
                        }
                    )
                    + "\n",
                    stderr="",
                )
                status, payload = request_runtime_test(
                    {
                        "runtime": {
                            "backend": "heuristic",
                            "talc_backend": "ml",
                            "talc_checkpoint": str(fake_talc_checkpoint),
                            "talc_threshold": 0.42,
                        }
                    }
                )
            self.assertEqual(status, 200)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["backend"], "heuristic")
            self.assertEqual(payload["models"]["talc"]["details"]["checkpoint_meta"]["model"], "segformer_b0")
            self.assertEqual(payload["talc_threshold"], 0.42)

            with mock.patch("apps.ore_pipeline_web.subprocess.run") as run_probe:
                run_probe.return_value = subprocess.CompletedProcess(
                    args=["python"],
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "device": "cpu",
                            "torch": "test",
                            "checkpoint_meta": {"model": "efficientnet_b3", "classes": ["ordinary_intergrowth", "fine_intergrowth"]},
                            "parameter_count": 789,
                            "seconds": 0.04,
                        }
                    )
                    + "\n",
                    stderr="",
                )
                status, payload = request_runtime_test(
                    {
                        "runtime": {
                            "backend": "heuristic",
                            "talc_backend": "heuristic",
                            "grain_backend": "ml",
                            "grade_checkpoint": str(fake_grade_checkpoint),
                        }
                    }
                )
            self.assertEqual(status, 200)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["grain_backend"], "ml")
            self.assertEqual(payload["models"]["grain_classification"]["details"]["checkpoint_meta"]["model"], "efficientnet_b3")

            with mock.patch("apps.ore_pipeline_web.subprocess.run") as run_probe:
                run_probe.return_value = subprocess.CompletedProcess(args=["python"], returncode=1, stdout="", stderr="loader failed")
                status, payload = request_runtime_test(
                    {"runtime": {"backend": "ml", "checkpoint": str(fake_checkpoint), "talc_backend": "heuristic"}}
                )
            self.assertEqual(status, 200)
            self.assertFalse(payload["ok"])
            self.assertIn("loader failed", payload["message"])

            status, payload = request_runtime_test(
                {"runtime": {"backend": "ml", "checkpoint": str(self.root / "missing.pt"), "talc_backend": "heuristic"}}
            )
            self.assertEqual(status, 400)
            self.assertIn("settings.runtime.checkpoint", payload["error"])

            status, payload = request_runtime_test(
                {"runtime": {"backend": "heuristic", "talc_backend": "ml", "talc_checkpoint": str(self.root / "missing_talc.pt")}}
            )
            self.assertEqual(status, 400)
            self.assertIn("settings.runtime.talc_checkpoint", payload["error"])

            status, payload = request_runtime_test(
                {"runtime": {"backend": "heuristic", "talc_backend": "heuristic", "grain_backend": "ml", "grade_checkpoint": str(self.root / "missing_grade.pt")}}
            )
            self.assertEqual(status, 400)
            self.assertIn("settings.runtime.grade_checkpoint", payload["error"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_delete_run_endpoint_removes_history_artifact(self) -> None:
        upload = self.store.register_upload_from_path(self.image_path)
        run = self.store.start_run(upload["upload_id"], {"panorama_scaling": False}, run_async=False)
        run_dir = self.store.runs_dir / run["run_id"]
        self.assertTrue(run_dir.exists())

        server = OrePipelineHTTPServer(("127.0.0.1", 0), self.store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address[:2]
        try:
            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request("DELETE", f"/api/runs/{run['run_id']}")
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()
            self.assertEqual(response.status, 200)
            self.assertEqual(payload["removed_run_id"], run["run_id"])
            self.assertFalse(run_dir.exists())
            self.assertEqual(self.store.list_runs()["runs"], [])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_cancel_run_marks_active_job_and_raises_cancel_signal(self) -> None:
        upload = self.store.register_upload_from_path(self.image_path)
        prepared = self.store.prepare_upload(upload["upload_id"], {"panorama_scaling": False})
        run_id = "run_cancel_test"
        run_dir = self.store.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        self.store._initialize_run_from_upload(run_id, run_dir, prepared, prepared["preprocess"]["preset"])
        with self.store.lock:
            self.store.jobs[run_id] = {
                "progress": 25,
                "status": "running",
                "stage": "sulfide/non-sulfide segmentation",
                "started_at": time.time(),
                "eta_seconds": 12,
                "cancel_requested": False,
            }

        server = OrePipelineHTTPServer(("127.0.0.1", 0), self.store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address[:2]
        try:
            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request("POST", f"/api/runs/{run_id}/cancel")
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()
            self.assertEqual(response.status, 200)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.assertEqual(payload["status"], "canceling")
        self.assertTrue(payload["cancel_requested"])
        self.assertIsNone(payload["eta_seconds"])
        self.assertGreaterEqual(payload["elapsed_seconds"], 0)
        self.assertEqual(self.store._read_run(run_id)["status"], "canceling")
        with self.assertRaises(RunCancelled):
            self.store._check_cancelled(run_id)

    def test_ml_tile_progress_updates_run_payload(self) -> None:
        upload = self.store.register_upload_from_path(self.image_path)
        prepared = self.store.prepare_upload(upload["upload_id"], {"panorama_scaling": False})
        run_id = "run_tile_progress_test"
        run_dir = self.store.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        self.store._initialize_run_from_upload(run_id, run_dir, prepared, prepared["preprocess"]["preset"])
        with self.store.lock:
            self.store.jobs[run_id] = {
                "progress": 18,
                "status": "running",
                "stage": "running ML tiled inference",
                "started_at": time.time() - 2,
                "eta_seconds": None,
                "cancel_requested": False,
            }
        progress_path = run_dir / "ml_pipeline/binary_sulfide/progress.json"
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(
            json.dumps(
                {
                    "schema_version": "binary-sulfide-inference-progress-v0.1",
                    "stage": "running",
                    "tiles_processed": 3,
                    "tiles_total": 8,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        self.store._update_ml_tile_progress(run_id, progress_path)
        payload = self.store.run_payload(run_id)

        self.assertEqual(payload["tile_progress"]["tiles_processed"], 3)
        self.assertEqual(payload["tile_progress"]["tiles_total"], 8)
        self.assertIn("3/8 tiles", payload["stage"])
        self.assertGreaterEqual(payload["elapsed_seconds"], 2)
        self.assertGreater(payload["progress"], 18)
        self.assertLess(payload["progress"], 76)

    def test_ml_backend_passes_talc_checkpoint_to_pipeline(self) -> None:
        binary_checkpoint = self.root / "binary.pt"
        binary_checkpoint.write_bytes(b"binary")
        talc_checkpoint = self.root / "talc.pt"
        talc_checkpoint.write_bytes(b"talc")
        upload = self.store.register_upload_from_path(self.image_path)
        prepared = self.store.prepare_upload(upload["upload_id"], {"panorama_scaling": False})
        run_id = "run_talc_command_test"
        run_dir = self.store.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        self.store._initialize_run_from_upload(run_id, run_dir, prepared, prepared["preprocess"]["preset"])
        runtime = self.store._initial_runtime_from_settings(
            {
                "backend": "ml",
                "checkpoint": str(binary_checkpoint),
                "talc_backend": "ml",
                "talc_checkpoint": str(talc_checkpoint),
                "talc_threshold": 0.42,
                "grain_backend": "heuristic",
            }
        )
        with self.store.lock:
            self.store.jobs[run_id] = {
                "progress": 18,
                "status": "running",
                "stage": "running ML tiled inference",
                "started_at": time.time(),
                "eta_seconds": None,
                "cancel_requested": True,
            }

        class FakeProcess:
            returncode = None

            def poll(self):
                return None

            def terminate(self):
                return None

            def wait(self, timeout=None):
                self.returncode = -15
                return self.returncode

            def kill(self):
                return None

        with mock.patch("apps.ore_pipeline_web.subprocess.Popen", return_value=FakeProcess()) as popen:
            with self.assertRaises(RunCancelled):
                self.store._run_ml_backend(run_id, run_dir, checkpoint=binary_checkpoint, runtime=runtime)

        cmd = popen.call_args.args[0]
        self.assertIn("--talc-checkpoint", cmd)
        self.assertEqual(cmd[cmd.index("--talc-checkpoint") + 1], str(talc_checkpoint.resolve()))
        self.assertIn("--talc-threshold", cmd)
        self.assertEqual(cmd[cmd.index("--talc-threshold") + 1], "0.42")
        self.assertNotIn("--auto-talc-candidate", cmd)

    def test_list_runs_overlays_active_job_progress(self) -> None:
        upload = self.store.register_upload_from_path(self.image_path)
        prepared = self.store.prepare_upload(upload["upload_id"], {"panorama_scaling": False})
        run_id = "run_history_progress_test"
        run_dir = self.store.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        self.store._initialize_run_from_upload(run_id, run_dir, prepared, prepared["preprocess"]["preset"])
        with self.store.lock:
            self.store.jobs[run_id] = {
                "progress": 43,
                "status": "running",
                "stage": "running ML tiled inference (5/12 tiles)",
                "started_at": time.time() - 9,
                "eta_seconds": 30,
                "cancel_requested": False,
                "tile_progress": {
                    "schema_version": "ore-pipeline-tile-progress-v0.1",
                    "stage": "running",
                    "tiles_processed": 5,
                    "tiles_total": 12,
                    "progress_fraction": 5 / 12,
                },
            }

        history = self.store.list_runs()["runs"]

        self.assertEqual(history[0]["run_id"], run_id)
        self.assertEqual(history[0]["status"], "running")
        self.assertEqual(history[0]["progress"], 43)
        self.assertEqual(history[0]["stage"], "running ML tiled inference (5/12 tiles)")
        self.assertEqual(history[0]["eta_seconds"], 30)
        self.assertGreaterEqual(history[0]["elapsed_seconds"], 8)
        self.assertEqual(history[0]["tile_progress"]["tiles_processed"], 5)
        self.assertEqual(history[0]["tile_progress"]["tiles_total"], 12)

    def test_run_payload_rehydrates_artifact_urls_after_store_restart(self) -> None:
        upload = self.store.register_upload_from_path(self.image_path)
        run = self.store.start_run(upload["upload_id"], {"panorama_scaling": False}, run_async=False)
        sulfide_url = run["masks"]["sulfide"]
        artifact_id = sulfide_url.split("/")[2]

        self.store.artifacts.clear()
        refreshed = self.store.run_payload(run["run_id"])

        self.assertEqual(refreshed["masks"]["sulfide"], sulfide_url)
        self.assertTrue(self.store.artifact_path(artifact_id).exists())

    def test_sulfide_edit_creates_new_immutable_run_and_recalculates_final(self) -> None:
        upload = self.store.register_upload_from_path(self.image_path)
        run = self.store.start_run(upload["upload_id"], {"panorama_scaling": False}, run_async=False)
        sulfide_path = self.store.runs_dir / run["run_id"] / "masks/sulfide_mask.png"
        mask = np.asarray(Image.open(sulfide_path).convert("L")).copy()
        mask[:, : mask.shape[1] // 2] = 0

        edited = self.store.create_edit_run(
            run["run_id"],
            {
                "edit_layer": "sulfide",
                "mask_png": mask_data_url(mask),
                "comment": "remove left half false positive",
            },
        )

        self.assertNotEqual(edited["run_id"], run["run_id"])
        self.assertEqual(edited["derivation"]["parent_run_id"], run["run_id"])
        self.assertEqual(edited["derivation"]["operation"], "recalculate_from_sulfide_edit")
        self.assertEqual(edited["derivation"]["comment"], "remove left half false positive")
        self.assertTrue((self.store.runs_dir / edited["run_id"] / "edit_comment.txt").exists())
        self.assertLessEqual(
            edited["summary"]["sulfide_area_px"],
            run["summary"]["sulfide_area_px"],
        )

    def test_final_edit_recalculates_metrics_without_replacing_sulfide_mask(self) -> None:
        upload = self.store.register_upload_from_path(self.image_path)
        run = self.store.start_run(upload["upload_id"], {"panorama_scaling": False}, run_async=False)
        run_dir = self.store.runs_dir / run["run_id"]
        final_mask = np.asarray(Image.open(run_dir / "masks/final_mask.png").convert("L")).copy()
        final_mask[:] = 0
        final_mask[10:50, 10:70] = 3

        edited = self.store.create_edit_run(
            run["run_id"],
            {
                "edit_layer": "final",
                "mask_png": mask_data_url(final_mask),
                "comment": "mark talc rectangle",
            },
        )

        self.assertEqual(edited["derivation"]["operation"], "recalculate_metrics_from_final_edit")
        parent_sulfide = np.asarray(Image.open(run_dir / "masks/sulfide_mask.png").convert("L"))
        edited_sulfide = np.asarray(Image.open(self.store.runs_dir / edited["run_id"] / "masks/sulfide_mask.png").convert("L"))
        np.testing.assert_array_equal(parent_sulfide, edited_sulfide)
        self.assertGreater(edited["summary"]["talc_area_px"], 0)

    def test_artifact_edit_creates_derived_run_and_excludes_pixels(self) -> None:
        upload = self.store.register_upload_from_path(self.image_path)
        run = self.store.start_run(upload["upload_id"], {"panorama_scaling": False}, run_async=False)
        run_dir = self.store.runs_dir / run["run_id"]
        final_shape = np.asarray(Image.open(run_dir / "masks/final_mask.png").convert("L")).shape
        artifact = np.zeros(final_shape, dtype=np.uint8)
        artifact[15:55, 20:80] = 255

        edited = self.store.create_edit_run(
            run["run_id"],
            {
                "edit_layer": "artifact",
                "mask_png": mask_data_url(artifact),
                "comment": "exclude polishing artifact",
            },
        )

        self.assertNotEqual(edited["run_id"], run["run_id"])
        self.assertEqual(edited["derivation"]["operation"], "recalculate_from_artifact_edit")
        self.assertEqual(edited["derivation"]["comment"], "exclude polishing artifact")
        self.assertIn("artifact", edited["masks"])
        edited_dir = self.store.runs_dir / edited["run_id"]
        analyzed = np.asarray(Image.open(edited_dir / "masks/analyzed_mask.png").convert("L"))
        sulfide = np.asarray(Image.open(edited_dir / "masks/sulfide_mask.png").convert("L"))
        talc = np.asarray(Image.open(edited_dir / "masks/talc_mask.png").convert("L"))
        final = np.asarray(Image.open(edited_dir / "masks/final_mask.png").convert("L"))
        self.assertEqual(int(analyzed[15:55, 20:80].sum()), 0)
        self.assertEqual(int(sulfide[15:55, 20:80].sum()), 0)
        self.assertEqual(int(talc[15:55, 20:80].sum()), 0)
        self.assertEqual(int(final[15:55, 20:80].sum()), 0)
        parent_analyzed = np.asarray(Image.open(run_dir / "masks/analyzed_mask.png").convert("L"))
        self.assertGreater(int(parent_analyzed[15:55, 20:80].sum()), 0)

    def test_page_exposes_required_controls(self) -> None:
        html = render_html_page()
        self.assertIn("Перетащите изображение сюда", html)
        self.assertIn("PNG, JPEG, TIFF, RAW", html)
        self.assertIn('id="selectedUpload"', html)
        self.assertIn('id="selectedThumb"', html)
        self.assertIn('id="clearUploadBtn"', html)
        self.assertIn('id="uploadWarning"', html)
        self.assertIn('id="uploadProgressWrap"', html)
        self.assertIn('id="uploadProgressBar"', html)
        self.assertIn('id="metadataBtn"', html)
        self.assertIn('id="configurationBtn"', html)
        self.assertIn('id="metadataDialog"', html)
        self.assertIn('id="configurationDialog"', html)
        self.assertIn('id="configTalcClusterRadius"', html)
        self.assertIn('id="configTalcClusterMinLocal"', html)
        self.assertIn('id="configTalcClusterOpacity"', html)
        self.assertIn('id="settingsTalcClusterRadius"', html)
        self.assertIn('id="settingsTalcClusterMinLocal"', html)
        self.assertIn('id="settingsTalcClusterOpacity"', html)
        self.assertIn("settingsTalcClusterDefaults", html)
        self.assertIn("talcClusterization", html)
        self.assertIn("function normalizedTalcClusterization", html)
        self.assertIn("function currentRunConfiguration", html)
        self.assertIn("function runConfigurationSignature", html)
        self.assertIn("function clearUploadDependentArtifacts(upload)", html)
        self.assertIn("function invalidateConfigurationDependentState()", html)
        self.assertIn("delete cleaned.augmentation;", html)
        self.assertIn("delete cleaned.preprocess;", html)
        self.assertIn("delete cleaned.tiling;", html)
        self.assertIn("state.run = null;", html)
        self.assertIn("state.runFilesPayload = null;", html)
        self.assertIn("function openConfigurationDialog", html)
        self.assertIn("...configurationPayload()", html)
        self.assertIn('id="runFilesBtn"', html)
        self.assertIn('id="runFilesDialog"', html)
        self.assertIn('id="runFilesTable"', html)
        self.assertIn('id="runFilesZipLink"', html)
        self.assertIn('id="runFilePreviewDialog"', html)
        self.assertIn('id="runFilePreviewBody"', html)
        self.assertIn('id="runFilePreviewDownloadLink"', html)
        self.assertIn('id="sulfideGrainsTable"', html)
        self.assertIn('id="sulfideGrainsNote"', html)
        self.assertIn("sulfideGrainsTitle", html)
        self.assertIn("function renderSulfideGrains(run)", html)
        self.assertIn("function sortedSulfideGrains(items)", html)
        self.assertIn("function renderSulfideGrainHeader(key, labelKey", html)
        self.assertIn("function compareGrainSortValues(left, right, key, direction)", html)
        self.assertIn("data-grain-sort", html)
        self.assertIn("sulfideGrainsSortBy", html)
        self.assertIn("aria-sort", html)
        self.assertIn("function drawSelectedGrainOverlay", html)
        self.assertIn("data-grain-id", html)
        self.assertIn('id="runTechDetails"', html)
        self.assertIn('id="runTechNote"', html)
        self.assertIn("runTechTitle", html)
        self.assertIn(".run-tech-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }", html)
        self.assertIn("@media (max-width: 820px) { .run-tech-grid { grid-template-columns: 1fr; } }", html)
        self.assertIn("runTechTileProgress", html)
        self.assertIn("runTechSulfidePath", html)
        self.assertIn("runTechNonSulfideTalc", html)
        self.assertIn("function renderRunTechnicalDetails(run)", html)
        self.assertIn("function formatRuntimeModel", html)
        self.assertIn("modelFreeStage", html)
        self.assertIn("Просмотреть файлы", html)
        self.assertIn("View files", html)
        self.assertIn("Download ZIP", html)
        self.assertIn("runFilesHeaderAction", html)
        self.assertIn("runFilesViewAction", html)
        self.assertIn("runFilesPreviewAction", html)
        self.assertIn("runFilesDownloadAction", html)
        self.assertIn("runFilePreviewTitle", html)
        self.assertIn("runFilePreviewRowsLimited", html)
        self.assertIn("run-files-sort", html)
        self.assertIn("runFilesSortHeader('path'", html)
        self.assertIn("runFilesSortHeader('kind'", html)
        self.assertIn("runFilesSortHeader('size'", html)
        self.assertIn("runFilesSortHeader('image'", html)
        self.assertIn("data-run-files-sort", html)
        self.assertIn("data-run-file-preview", html)
        self.assertIn("function compareRunFiles", html)
        self.assertIn("function runFilePreviewKind(file)", html)
        self.assertIn("function renderCsvPreview(text)", html)
        self.assertIn("function openRunFilePreview(file)", html)
        self.assertIn("state.runFilesSort", html)
        self.assertIn("download=\"${escapeHtml(file.name || '')}\"", html)
        self.assertIn("/files", html)
        self.assertIn("/artifacts.zip", html)
        self.assertIn("function openRunFilesDialog()", html)
        self.assertIn("function renderRunFiles(payload)", html)
        self.assertIn('id="augmentationEnabled"', html)
        self.assertIn('id="editAugmentationBtn"', html)
        self.assertIn('id="applyAugmentationBtn"', html)
        self.assertIn('id="augmentationDialog"', html)
        self.assertIn('id="augmentationSummary"', html)
        self.assertIn('id="augBrightness"', html)
        self.assertIn('id="augContrast"', html)
        self.assertIn('id="augSaturation"', html)
        self.assertIn('id="augHue"', html)
        self.assertIn('id="augGamma"', html)
        self.assertIn('id="augBlur"', html)
        self.assertIn('id="augNoise"', html)
        self.assertIn('id="augSeed"', html)
        self.assertIn('id="augScratchCount"', html)
        self.assertIn('id="augScratchIntensity"', html)
        self.assertIn('id="augPolishingHaze"', html)
        self.assertIn('id="augPitCount"', html)
        self.assertIn('id="augPitIntensity"', html)
        self.assertIn("Color and tone", html)
        self.assertIn("Acquisition noise", html)
        self.assertIn("Grinding/polishing artifacts", html)
        self.assertIn("Артефакты шлифовки/полировки", html)
        self.assertIn("one deterministic, geometry-preserving augmented image before preprocessing", html)
        self.assertIn("const AUGMENTATION_STORAGE_KEY = 'orePipelineAugmentationSettings'", html)
        self.assertIn("const DEFAULT_AUGMENTATION_SETTINGS", html)
        self.assertIn("surface_artifacts", html)
        self.assertIn("function augmentationPayload()", html)
        self.assertIn("function storedAugmentationSettings()", html)
        self.assertIn("function saveAugmentationSettings()", html)
        self.assertIn("function applyAugmentationToControls(settings", html)
        self.assertIn("function updateAugmentationSummary()", html)
        self.assertIn("statusAugmentationUpdated", html)
        self.assertIn("statusAugmentationPreparedRun", html)
        self.assertIn("statusPreprocessPreparedRun", html)
        self.assertIn("statusConfigurationCleared", html)
        self.assertIn("buttonId: 'applyAugmentationBtn'", html)
        self.assertIn("augmentation: augmentationPayload()", html)
        self.assertIn("function runIsPrepared(run)", html)
        self.assertIn("function runCanBePreparedFromApply(run)", html)
        self.assertIn("function clearResultsPanel()", html)
        self.assertIn("function clearRunResultsForStart(preparedRun = null)", html)
        self.assertIn("let preparedRun = runIsPrepared(state.run) ? state.run : null;", html)
        self.assertLess(html.index("clearRunResultsForStart(preparedRun);"), html.index("const response = await fetch(startUrl"))
        # Start must always apply the current controls: a prepared run whose Apply-time
        # settings drifted is re-prepared in place before starting (see spec
        # ore-pipeline-apply-prepared-run-v0.1.md).
        self.assertIn("function preparedRunSettingsAreStale(run)", html)
        self.assertIn("preparedRunSettingsAreStale(preparedRun)", html)
        self.assertIn("/prepare", html)
        self.assertIn("/start", html)
        self.assertIn("changed_step: changedStep", html)
        self.assertLess(html.index("const previousSignature = runConfigurationSignature();"), html.index("invalidateConfigurationDependentState();"))
        self.assertIn("stagePrepared", html)
        self.assertIn('data-mode="augmented"', html)
        self.assertIn('data-side-layer="augmented"', html)
        self.assertLess(html.index('data-mode="original"'), html.index('data-mode="augmented"'))
        self.assertLess(html.index('data-mode="augmented"'), html.index('data-mode="preprocessed"'))
        self.assertNotIn('id="metadataStatus"', html)
        self.assertNotIn("Metadata is available after image upload.", html)
        self.assertNotIn("Метаданные доступны после загрузки изображения.", html)
        self.assertIn("Редактировать метаданные", html)
        self.assertIn("Edit Metadata", html)
        self.assertIn("Session specific", html)
        self.assertIn("Sample specific", html)
        self.assertIn("Scale value, µm/px", html)
        self.assertIn("Scale value is set without a calibrated scale source.", html)
        self.assertIn("Exclude this image from training/validation sets", html)
        self.assertLess(html.index('data-i18n="metadataProject"'), html.index('data-i18n="metadataSampleSpecific"'))
        self.assertLess(html.index('data-i18n="metadataInstrument"'), html.index('data-i18n="metadataSampleSpecific"'))
        self.assertLess(html.index('data-i18n="metadataSampleSpecific"'), html.index('data-i18n="metadataSampleId"'))
        self.assertIn("orePipelineMetadataDefaults", html)
        self.assertIn("function openMetadataDialog", html)
        self.assertIn("function currentMetadataPayloadForSubmission()", html)
        self.assertIn("curated_metadata", html)
        self.assertIn("metadataScaleWarning", html)
        self.assertIn("pixel_size_without_calibrated_scale", html)
        self.assertIn("const keep = ['project', 'om_instrument', 'om_objective_magnification', 'scale_source', 'pixel_size_um', 'scale_confidence', 'review_status'];", html)
        self.assertNotIn("const keep = ['project', 'source_role'", html)
        self.assertIn("SUPPORTED_UPLOAD_EXTENSIONS", html)
        self.assertIn("function uploadFileWithProgress(file)", html)
        self.assertIn("XMLHttpRequest", html)
        self.assertIn("startPreviewPreparationProgress", html)
        self.assertIn("uploadProgressPreparing", html)
        preview_progress_section = html.split("function startPreviewPreparationProgress", 1)[1].split("function uploadFileWithProgress", 1)[0]
        self.assertIn("setUploadProgress('uploadProgressPreparing', progress);", preview_progress_section)
        self.assertNotIn("setProgress(", preview_progress_section)
        self.assertNotIn("setStatus(", preview_progress_section)
        upload_progress_section = html.split("xhr.upload.addEventListener('progress'", 1)[1].split("xhr.upload.addEventListener('load'", 1)[0]
        self.assertIn("setUploadProgress('uploadProgressUploading', progress);", upload_progress_section)
        self.assertNotIn("setProgress(", upload_progress_section)
        self.assertNotIn("setStatus(", upload_progress_section)
        self.assertIn("function handleSelectedFile(file)", html)
        self.assertIn("function isSupportedUploadFile(file)", html)
        self.assertIn("invalidImageFormat", html)
        self.assertIn("Unsupported file format", html)
        self.assertIn('id="appVersionBadge"', html)
        self.assertIn('class="app-version-badge"', html)
        self.assertIn('data-i18n-aria-label="versionLabel"', html)
        self.assertIn(">v2</span>", html)
        self.assertIn('id="languageSelect"', html)
        self.assertIn('id="batchTab"', html)
        self.assertIn('id="batchView"', html)
        self.assertIn('id="batchFileInput"', html)
        self.assertIn('id="addBatchImagesBtn"', html)
        self.assertIn('id="runBatchBtn"', html)
        self.assertIn('id="stopBatchBtn"', html)
        self.assertIn('id="batchGallery"', html)
        self.assertLess(html.index('data-i18n="batchGallery"'), html.index('id="addBatchImagesBtn"'))
        self.assertLess(html.index('id="addBatchImagesBtn"'), html.index('id="batchGallery"'))
        self.assertIn('id="backToBatchBtn"', html)
        self.assertIn("function renderBatch()", html)
        self.assertIn("function runBatch()", html)
        self.assertIn("function removeBatchItem(itemId)", html)
        self.assertIn("function loadBatchRun(runId, batchId)", html)
        self.assertIn("/api/batches", html)
        self.assertIn("data-batch-remove", html)
        self.assertIn("method: 'DELETE'", html)
        self.assertIn("batchEditMetadata", html)
        self.assertIn("batchRemoveImage", html)
        self.assertIn("batchSharedSettings", html)
        self.assertIn("Серии", html)
        self.assertIn("Series", html)
        self.assertIn("Новая серия", html)
        self.assertIn("Run Series", html)
        self.assertIn("Назад к серии", html)
        self.assertNotIn("Пакеты", html)
        self.assertNotIn("Пакетная обработка", html)
        self.assertNotIn("New Batch", html)
        self.assertNotIn("Run Batch", html)
        self.assertNotIn("Назад к Batch", html)
        self.assertIn('id="settingsTab"', html)
        self.assertIn('id="settingsView"', html)
        self.assertIn('id="statusTab"', html)
        self.assertIn('id="statusView"', html)
        self.assertIn('id="apiTab"', html)
        self.assertIn('id="apiView"', html)
        utility_tabs = html.split('class="tabs utility-tabs"', 1)[1].split('</nav>', 1)[0]
        self.assertIn('id="statusTab"', utility_tabs)
        self.assertNotIn("hidden-status-tab", utility_tabs)
        self.assertNotIn('aria-hidden="true"', utility_tabs)
        self.assertLess(utility_tabs.index('id="statusTab"'), utility_tabs.index('id="apiTab"'))
        self.assertLess(utility_tabs.index('id="apiTab"'), utility_tabs.index('id="settingsTab"'))
        self.assertIn('id="apiDocsNav"', html)
        self.assertIn('id="apiDocsList"', html)
        self.assertIn('id="statusCards"', html)
        self.assertIn('id="statusHealthTable"', html)
        self.assertIn('id="statusStorageTable"', html)
        self.assertIn('id="statusSystemLog"', html)
        self.assertIn('id="statusAccessLog"', html)
        self.assertIn('id="refreshStatusBtn"', html)
        self.assertIn("const API_REFERENCE", html)
        self.assertIn("REST API documentation for the v2 UI", html)
        self.assertIn("Документация REST API для v2 UI", html)
        self.assertIn("function renderApiDocs()", html)
        self.assertIn("function runApiSandbox(endpointId)", html)
        self.assertIn("function attachApiSandboxHandlers()", html)
        self.assertIn("data-api-run", html)
        self.assertIn("api-request-url", html)
        self.assertIn("api-request-body", html)
        self.assertIn("api-file-input", html)
        self.assertIn("endpoint.download", html)
        self.assertIn("multipart/form-data", html)
        self.assertIn("/api/uploads/{upload_id}/preprocess", html)
        self.assertIn("/api/runs/start", html)
        self.assertIn("/api/runs/{run_id}/files", html)
        self.assertIn("/api/runs/{run_id}/artifacts.zip", html)
        self.assertIn("/api/runs/{run_id}/metrics.csv", html)
        self.assertIn("/api/runs/{run_id}/report.pdf", html)
        self.assertIn("/api/batches/{batch_id}", html)
        self.assertIn("/api/batches/{batch_id}/run", html)
        self.assertIn("/api/batches/{batch_id}/cancel", html)
        self.assertIn("/api/batches/{batch_id}/results.csv", html)
        self.assertIn("/api/history", html)
        self.assertIn('id="settingsLanguage"', html)
        self.assertIn('id="settingsTheme"', html)
        self.assertIn('id="settingsShowTiling"', html)
        self.assertIn('id="settingsPassword"', html)
        self.assertIn('id="settingsClearPassword"', html)
        self.assertIn('id="settingsPasswordState"', html)
        self.assertIn('id="settingsBackend"', html)
        self.assertIn('id="settingsCheckpoint"', html)
        self.assertIn('id="settingsCheckpoint" type="hidden"', html)
        self.assertIn('id="settingsCheckpointDisplay"', html)
        self.assertIn('id="settingsTalcBackend"', html)
        self.assertIn("Бэкенд сегментации талька", html)
        self.assertIn("Talc segmentation backend", html)
        self.assertIn('id="settingsGrainBackend"', html)
        self.assertIn("Бэкенд классификации зерен", html)
        self.assertIn("Grain classification backend", html)
        self.assertIn("grain_backend: 'heuristic'", html)
        self.assertIn("backend: 'ml'", html)
        self.assertIn("talc_backend: 'ml'", html)
        self.assertIn("settingsTalcBackendHeuristic", html)
        self.assertIn("heuristics", html)
        self.assertIn("ML-модель", html)
        self.assertIn("ML model", html)
        self.assertIn('id="settingsTalcCheckpoint"', html)
        self.assertIn('id="settingsTalcCheckpoint" type="hidden"', html)
        self.assertIn('id="settingsTalcCheckpointDisplay"', html)
        self.assertIn('id="settingsTalcThresholdField"', html)
        self.assertIn('id="settingsTalcThreshold"', html)
        self.assertIn("Порог вероятности ML-талька", html)
        self.assertIn("ML talc probability threshold", html)
        self.assertIn("function updateSettingsRuntimeControls()", html)
        self.assertIn("$('settingsTalcThreshold').disabled = !talcUsesMl", html)
        self.assertIn("settingsTalcThresholdHintHeuristic", html)
        self.assertIn("$('settingsTalcBackend').addEventListener('change', updateSettingsRuntimeControls)", html)
        self.assertIn('id="testRuntimeBtn"', html)
        self.assertIn("settings-runtime-actions", html)
        self.assertIn("Проверить все", html)
        self.assertIn("Test All", html)
        self.assertIn('id="runtimeTestStatus"', html)
        self.assertIn('id="removeAllHistoryBtn"', html)
        self.assertIn('id="settingsPreprocessingEnabled"', html)
        self.assertIn('id="settingsMetaProject"', html)
        self.assertIn('id="saveSettingsBtn"', html)
        self.assertIn('id="resetSettingsBtn"', html)
        self.assertIn("/api/settings", html)
        self.assertIn("/api/status", html)
        self.assertIn("/api/auth/status", html)
        self.assertIn("/api/auth/login", html)
        self.assertIn("/api/auth/logout", html)
        self.assertIn("/api/runtime/test", html)
        self.assertIn("function loadAppSettings()", html)
        self.assertIn("function testRuntimeFromPage()", html)
        self.assertIn("function removeAllHistoryFromSettings()", html)
        self.assertIn('id="configSulfideBackend"', html)
        self.assertIn('id="configTalcBackend"', html)
        self.assertIn('id="configGrainBackend"', html)
        self.assertIn("ML Sulfide (SegFormer-B2)", html)
        self.assertIn("ML Talc SegFormer-B0", html)
        self.assertIn("ML (Grade-CNN)", html)
        self.assertIn('value="grain_aggregate" disabled', html)
        self.assertIn("function runConfigurationFromRun(run)", html)
        self.assertIn("function loadSystemStatus(options = {})", html)
        self.assertIn("function renderSystemStatus(payload", html)
        self.assertIn("const backendSubvalue = app.backend === 'ml' ? (app.checkpoint || '') : ''", html)
        self.assertIn("function statusModelDetails(labelKey, model)", html)
        self.assertIn("white-space: pre-line", html)
        self.assertIn("].join('\\n\\n');", html)
        self.assertIn("settingsRuntimeTestOkModels", html)
        self.assertIn("function shortCheckpointName", html)
        self.assertIn("return `${parts[parts.length - 1]} (${parts[parts.length - 2]})`", html)
        self.assertIn("function setCheckpointDisplay", html)
        self.assertIn("settingsCheckpointNotConfigured", html)
        self.assertIn("function renderStatusLogs(logs)", html)
        self.assertIn("function saveSettingsObject(settings", html)
        self.assertIn("settingsPasswordEnabled", html)
        self.assertIn("settingsPasswordDisabled", html)
        self.assertIn("settingsRemoveAllHistory", html)
        self.assertIn("settingsHistoryRemoved", html)
        self.assertIn("ore-pipeline-app-settings-v0.1", html)
        self.assertIn('value="ru"', html)
        self.assertIn("Русский", html)
        self.assertIn("Russian", html)
        self.assertIn("English", html)
        self.assertIn("const DEFAULT_LANGUAGE = 'ru'", html)
        self.assertIn("orePipelineLanguage", html)
        self.assertIn("const PREPROCESS_STORAGE_KEY = 'orePipelinePreprocessPreset'", html)
        self.assertIn("const DEFAULT_PREPROCESS_PRESET", html)
        self.assertIn('id="preprocessingEnabled"', html)
        self.assertNotIn('id="preprocessingEnabled" checked', html)
        self.assertIn("preprocessing_enabled: false", html)
        self.assertIn('id="editPreprocessBtn"', html)
        self.assertIn('id="preprocessDialog"', html)
        self.assertIn('id="preprocessSummary"', html)
        self.assertIn("preprocessing_enabled", html)
        self.assertIn("panorama_scaling_mode", html)
        self.assertIn("panorama_max_side_px", html)
        self.assertIn("panorama_scale_factor", html)
        self.assertIn("function updatePreprocessSummary()", html)
        self.assertIn("function updatePanoramaScalingControls", html)
        self.assertIn("function panoramaScalingSummaryItem", html)
        self.assertIn("function preprocessingEnabledForView()", html)
        self.assertIn("preprocessingSummaryDisabled", html)
        self.assertIn("Preprocessing will be skipped on Start.", html)
        self.assertIn('id="illumination" checked', html)
        self.assertIn('id="denoise" checked', html)
        self.assertIn('id="contrast" checked', html)
        self.assertIn('id="panoramaScaling" checked', html)
        self.assertIn('id="panoramaScalingMode"', html)
        self.assertIn('id="panoramaMaxSidePx"', html)
        self.assertIn('id="panoramaScaleFactor"', html)
        self.assertIn(".preprocess-dialog { max-width: calc(100vw - 24px); box-sizing: border-box; overflow-x: hidden; }", html)
        self.assertIn(".preprocess-dialog .modal-foot { flex-wrap: wrap; }", html)
        self.assertIn(".preprocess-dialog .help-dot::after { left: auto; right: 0; transform: translate(0, 4px); }", html)
        self.assertIn(".panorama-scaling-controls { min-width: 0; padding-left: 24px; }", html)
        self.assertIn(".panorama-scaling-controls { grid-template-columns: 1fr; padding-left: 0; }", html)
        self.assertIn('id="settingsPanoramaScalingMode"', html)
        self.assertIn('id="settingsPanoramaMaxSidePx"', html)
        self.assertIn('id="settingsPanoramaScaleFactor"', html)
        augmentation_panel = html.split('id="augmentationEnabled"', 1)[1].split('<div class="panel">', 1)[0]
        self.assertIn('id="editAugmentationBtn"', augmentation_panel)
        self.assertIn('id="applyAugmentationBtn"', augmentation_panel)
        self.assertIn('id="augmentationSummary"', augmentation_panel)
        self.assertNotIn('id="preprocessingEnabled"', augmentation_panel)
        preprocessing_panel = html.split('id="preprocessingEnabled"', 1)[1].split('<div class="panel">', 1)[0]
        self.assertIn('id="editPreprocessBtn"', preprocessing_panel)
        self.assertIn('id="applyPreprocessBtn"', preprocessing_panel)
        self.assertNotIn('id="illumination"', preprocessing_panel)
        self.assertNotIn('id="denoise"', preprocessing_panel)
        self.assertNotIn('id="contrast"', preprocessing_panel)
        self.assertNotIn('id="panoramaScaling"', preprocessing_panel)
        self.assertIn('id="stopBtn"', html)
        self.assertIn('data-i18n="stop"', html)
        self.assertIn("function updateRunControls(run = state.run)", html)
        self.assertIn("ACTIVE_RUN_STATUSES", html)
        self.assertIn("/cancel", html)
        self.assertIn("statusCanceling", html)
        self.assertIn("statusCanceled", html)
        self.assertIn("function storedPreprocessPreset()", html)
        self.assertIn("function savePreprocessPreset()", html)
        self.assertIn("function applyLanguage(language)", html)
        self.assertIn("function localizedRunText(run)", html)
        self.assertIn("function localizedMetricLabel(row)", html)
        self.assertIn("function decisionRationale(run)", html)
        self.assertIn('id="decisionRationale"', html)
        self.assertIn('id="metricsDenominatorNote"', html)
        self.assertLess(html.index('id="textOutput"'), html.index('id="metricsTable"'))
        self.assertIn(".result-grid { display: grid; grid-template-columns: minmax(0, 1fr);", html)
        self.assertIn('class="panel metrics-panel"', html)
        self.assertIn('class="panel sulfide-grains-section"', html)
        self.assertIn('class="panel run-technical-section"', html)
        self.assertLess(html.index('class="panel metrics-panel"'), html.index('class="panel sulfide-grains-section"'))
        self.assertLess(html.index('class="panel sulfide-grains-section"'), html.index('class="panel run-technical-section"'))
        metrics_panel = html.split('class="panel metrics-panel"', 1)[1].split('class="panel sulfide-grains-section"', 1)[0]
        self.assertIn('id="metricsTable"', metrics_panel)
        self.assertIn('id="metricsDenominatorNote"', metrics_panel)
        self.assertIn('id="csvLink"', metrics_panel)
        self.assertIn('id="pdfLink"', metrics_panel)
        self.assertIn('id="runFilesBtn"', metrics_panel)
        self.assertNotIn('id="sulfideGrainsTable"', metrics_panel)
        self.assertNotIn('id="runTechDetails"', metrics_panel)
        self.assertIn('class="metrics-table"', html)
        self.assertIn("metricsDenominatorNote", html)
        self.assertIn("Сульфиды, тальк, кластеры талька и остальное считаются от проанализированной области", html)
        self.assertIn("Sulfides, talc, talc cluster areas, and other use analyzed area as denominator", html)
        self.assertIn("metricsHeaderAreaPx", html)
        self.assertIn("metricsHeaderPhysicalArea", html)
        self.assertIn("metricOtherFraction", html)
        self.assertIn("metricArtifactFraction", html)
        self.assertIn("metric-level-0", html)
        self.assertIn("metric-level-1", html)
        self.assertIn("metric-level-2", html)
        self.assertIn("data-metric-key", html)
        self.assertIn("function formatPhysicalArea(row)", html)
        self.assertIn("row.area_um2", html)
        self.assertIn("function renderHistoryTable(runs)", html)
        self.assertIn('id="historyModeButtons"', html)
        self.assertIn('data-history-mode="all"', html)
        self.assertIn('data-history-mode="single"', html)
        self.assertIn('data-history-mode="batches"', html)
        self.assertIn("historyModeAllRuns", html)
        self.assertIn("historyModeSingleRuns", html)
        self.assertIn("historyModeBatches", html)
        self.assertIn("function renderHistoryPage()", html)
        self.assertIn("function renderBatchHistoryTable(batches)", html)
        self.assertIn("function batchCountsText(batch)", html)
        self.assertIn("data-open-batch", html)
        self.assertIn("data-delete-batch", html)
        self.assertIn("historyOpenBatch", html)
        self.assertIn("historyNoBatches", html)
        self.assertIn("function removeBatch(batchId)", html)
        self.assertIn("confirmRemoveBatch", html)
        self.assertIn("statusBatchRemoved", html)
        self.assertIn("state.historyMode === 'single'", html)
        self.assertIn("fetch('/api/batches')", html)
        self.assertIn("function renderHistoryThumbnail(run)", html)
        self.assertIn("function renderHistoryStatus(run)", html)
        self.assertIn("function runErrorDetails(run)", html)
        self.assertIn("function runProgressPercent(run)", html)
        self.assertIn("function formatDurationSeconds(seconds)", html)
        self.assertIn("historyElapsed", html)
        self.assertIn("statusElapsed", html)
        self.assertIn("elapsed_seconds", html)
        self.assertNotIn("history-progress-bar", html)
        self.assertIn("function statusActiveJobsText(history)", html)
        self.assertIn("function setStatusPolling(enabled)", html)
        self.assertIn("function statusGpuDeviceDetails(device)", html)
        self.assertIn("statusActiveOperations", html)
        self.assertIn("active_operations", html)
        self.assertIn("statusGpuMetal", html)
        self.assertIn("statusGpuMpsAvailable", html)
        self.assertIn("function openHistoryPreview(url, title)", html)
        self.assertIn("history-row-media", html)
        self.assertIn("history-row-load", html)
        self.assertIn("history-row-text", html)
        self.assertIn('grid-template-columns: 68px minmax(0, 1fr)', html)
        self.assertIn('id="historyPreviewDialog"', html)
        self.assertIn('id="historyPreviewImage"', html)
        self.assertIn('data-preview-run', html)
        self.assertIn("historyThumbnail", html)
        self.assertIn("historyPreviewTitle", html)
        self.assertIn("function removeRun(runId)", html)
        self.assertIn("data-delete-run", html)
        self.assertIn("historyFilename", html)
        self.assertIn("historyStatus", html)
        self.assertIn("historyStatusDone", html)
        self.assertIn("historyStatusError", html)
        self.assertIn("history-status-info", html)
        self.assertIn("historyTileProgress", html)
        self.assertIn("historyOreClassification", html)
        self.assertIn("historyNonSulfides", html)
        self.assertIn("historyRemove", html)
        self.assertIn("const PAGE_SLUGS = {workspace: '/workspace', batch: '/batch', history: '/history', historySeries: '/history_series', settings: '/settings', status: '/status', api: '/api'}", html)
        self.assertIn("function historySlugForMode(mode)", html)
        self.assertIn("window.location.pathname === PAGE_SLUGS.historySeries", html)
        self.assertIn("window.location.pathname === PAGE_SLUGS.api", html)
        self.assertIn("state.historyMode = 'batches';", html)
        self.assertIn("window.history.pushState({page: 'history', historyMode: state.historyMode}, '', slug)", html)
        self.assertIn("function setPage(page, options = {})", html)
        self.assertIn("document.body.dataset.page = nextPage", html)
        self.assertIn('body[data-page="history"] main > aside', html)
        self.assertIn('body[data-page="api"] main > aside', html)
        self.assertIn("function resetWindowScroll()", html)
        self.assertIn("resetWindowScroll();", html)
        self.assertIn("window.history.pushState", html)
        self.assertIn("popstate", html)
        self.assertIn('data-i18n="appTitle"', html)
        self.assertIn("function renderUploadCard(upload)", html)
        self.assertIn("function resetPageForClearedImage()", html)
        self.assertIn("function applyPresetToControls(preset, options = {})", html)
        self.assertIn("нормализация освещения", html)
        self.assertIn("шумоподавление", html)
        self.assertIn("коррекция контраста", html)
        self.assertIn("масштабирование для панорамных снимков", html)
        settings_preprocess = html.split('data-i18n="settingsPreprocessDefaults"', 1)[1].split(
            'data-i18n="settingsMetadataDefaults"',
            1,
        )[0]
        self.assertIn('class="settings-grid settings-preprocess-main"', settings_preprocess)
        self.assertIn('class="settings-section-divider" aria-hidden="true"', settings_preprocess)
        self.assertIn('class="settings-scale-group"', settings_preprocess)
        self.assertLess(settings_preprocess.index('id="settingsContrast"'), settings_preprocess.index('class="settings-section-divider"'))
        self.assertLess(settings_preprocess.index('class="settings-section-divider"'), settings_preprocess.index('id="settingsPanoramaScaling"'))
        self.assertLess(settings_preprocess.index('id="settingsPanoramaScaling"'), settings_preprocess.index('id="settingsPanoramaScalingMode"'))
        self.assertIn("граница по длинной стороне", html)
        self.assertIn("Коэффициент, x", html)
        self.assertIn("панорама до {value} px", html)
        self.assertIn("illumination normalization", html)
        self.assertIn("noise reduction", html)
        self.assertIn("contrast correction", html)
        self.assertIn("panorama image scaling", html)
        self.assertIn("longest side bound", html)
        self.assertIn("Scale factor, x", html)
        self.assertIn("panorama {value}x", html)
        self.assertEqual(html.count('class="help-dot"'), 4)
        self.assertIn('data-i18n-tooltip="illuminationNormalizationHelp"', html)
        self.assertIn('data-i18n-tooltip="denoiseHelp"', html)
        self.assertIn('data-i18n-tooltip="contrastCorrectionHelp"', html)
        self.assertIn('data-i18n-tooltip="panoramaScalingHelp"', html)
        self.assertIn("Balances uneven lighting before segmentation.", html)
        self.assertIn("Suppresses small image noise while preserving larger ore structures.", html)
        self.assertIn("function applyLanguage(language)", html)
        self.assertIn("[data-i18n-tooltip]", html)
        self.assertIn('id="showTiling"', html)
        self.assertIn("показать тайлы", html)
        self.assertIn("show tiling", html)
        self.assertIn('id="overlayOpacity"', html)
        self.assertIn('id="boundaryOnly"', html)
        self.assertIn('id="contourWidth"', html)
        self.assertIn('id="contourWidthValue"', html)
        self.assertIn("function clampContourWidth(value)", html)
        self.assertIn("function boundaryCanvasForImage(image, key, width = 1)", html)
        self.assertIn("boundaryCanvasForImage(tinted, tintKey, options.contourWidth)", html)
        self.assertIn(":contour:${contourWidth}", html)
        self.assertIn("state.overlayOpacity", html)
        self.assertIn("state.boundaryOnly", html)
        self.assertIn("state.contourWidth", html)
        self.assertIn("contours only", html)
        self.assertIn("contour width", html)
        self.assertIn("function drawTilingGrid(display)", html)
        self.assertIn("function tilingManifest()", html)
        self.assertIn("Сравнение:", html)
        self.assertIn('class="primary-view-controls"', html)
        self.assertIn(".viewer-toolbar .segmented { overflow: visible; flex-wrap: nowrap; scrollbar-width: none; }", html)
        self.assertIn('data-mode="original"', html)
        self.assertIn('data-mode="augmented"', html)
        self.assertIn('data-mode="preprocessed"', html)
        self.assertIn('data-mode="sulfide"', html)
        self.assertIn('data-mode="final"', html)
        self.assertIn("artifact_overlay", html)
        self.assertNotIn('data-mode="artefacts"', html)
        self.assertNotIn("layer === 'artefacts'", html)
        self.assertLess(html.index('data-mode="original"'), html.index('data-mode="augmented"'))
        self.assertLess(html.index('data-mode="augmented"'), html.index('data-mode="preprocessed"'))
        self.assertLess(html.index('data-mode="preprocessed"'), html.index('data-mode="sulfide"'))
        self.assertLess(html.index('data-mode="sulfide"'), html.index('data-mode="final"'))
        primary_controls = html.split('class="primary-view-controls"', 1)[1].split('class="side-by-side-control"', 1)[0]
        self.assertIn('id="viewModeButtons"', primary_controls)
        self.assertNotIn('id="showBackground"', primary_controls)
        self.assertNotIn('id="showOrdinary"', primary_controls)
        self.assertNotIn('id="showFine"', primary_controls)
        self.assertNotIn('id="showTalc"', primary_controls)
        self.assertNotIn('id="showSulfide"', primary_controls)
        self.assertNotIn('id="showArtifacts"', html)
        self.assertIn('class="segmentation-legend-overlay hidden"', html)
        self.assertIn('class="segmentation-legend-panel left"', html)
        self.assertIn('class="segmentation-legend-panel right"', html)
        self.assertIn(".class-toggles { display: flex; flex-direction: column; gap: 6px; align-items: flex-start; padding: 0; }", html)
        self.assertIn(".class-toggles[hidden] { display: none; }", html)
        self.assertIn("legend-percent", html)
        self.assertIn('id="segmentationClassToggles"', html)
        self.assertIn('id="primaryClassLegend"', html)
        self.assertIn('id="sideClassLegend"', html)
        self.assertIn('id="primarySulfideClassToggles"', html)
        self.assertIn('id="primaryFinalClassToggles"', html)
        self.assertIn('id="sideSulfideClassToggles"', html)
        self.assertIn('id="sideFinalClassToggles"', html)
        self.assertIn('data-i18n="leftViewLegend"', html)
        self.assertIn('data-i18n="rightViewLegend"', html)
        sulfide_controls = html.split('id="primarySulfideClassToggles"', 1)[1].split('id="primaryFinalClassToggles"', 1)[0]
        self.assertIn('data-legend-toggle="showSulfide"', sulfide_controls)
        self.assertIn('data-legend-toggle="showNonSulfide"', sulfide_controls)
        self.assertIn('data-legend-toggle="showSulfideArtifacts"', sulfide_controls)
        self.assertIn("classSulfides", sulfide_controls)
        self.assertIn("classNonSulfides", sulfide_controls)
        self.assertIn("classArtefacts", sulfide_controls)
        final_controls = html.split('id="primaryFinalClassToggles"', 1)[1].split('id="sideClassLegend"', 1)[0]
        self.assertIn('data-legend-toggle="showOrdinary"', final_controls)
        self.assertIn('data-legend-toggle="showFine"', final_controls)
        self.assertIn('data-legend-toggle="showTalc"', final_controls)
        self.assertIn('data-legend-toggle="showTalcClusters"', final_controls)
        self.assertIn('data-legend-toggle="showFinalArtifacts"', final_controls)
        self.assertIn('data-legend-toggle="showBackground"', final_controls)
        self.assertIn("--talc-cluster: #40dcff;", html)
        self.assertIn('showTalcClusters: false', html)
        self.assertIn('state.classVisibility.showTalcClusters = false;', html)
        self.assertNotIn('data-legend-toggle="showTalcClusters" checked', final_controls)
        self.assertIn('class="segmentation-legend-divider" aria-hidden="true"', final_controls)
        self.assertEqual(final_controls.count('class="segmentation-legend-divider" aria-hidden="true"'), 2)
        self.assertLess(final_controls.index('data-legend-toggle="showTalc"'), final_controls.index('class="segmentation-legend-divider"'))
        self.assertLess(final_controls.index('class="segmentation-legend-divider"'), final_controls.index('data-legend-toggle="showTalcClusters"'))
        self.assertLess(final_controls.index('data-legend-toggle="showFinalArtifacts"'), final_controls.rindex('class="segmentation-legend-divider"'))
        self.assertLess(final_controls.rindex('class="segmentation-legend-divider"'), final_controls.index('data-legend-toggle="showBackground"'))
        side_final_controls = html.split('id="sideFinalClassToggles"', 1)[1].split('id="splitterOverlay"', 1)[0]
        self.assertEqual(side_final_controls.count('class="segmentation-legend-divider" aria-hidden="true"'), 2)
        self.assertNotIn('data-legend-toggle="showTalcClusters" checked', side_final_controls)
        self.assertIn("classOrdinaryShort", final_controls)
        self.assertIn("classFineShort", final_controls)
        self.assertIn("classTalc", final_controls)
        self.assertIn("classTalcClusters", final_controls)
        self.assertIn("classArtefacts", final_controls)
        self.assertIn("--artifact", html)
        self.assertIn("--sulfide", html)
        self.assertIn("--non-sulfide", html)
        self.assertIn("ARTIFACT_COLOR = (198, 60, 255, 180)", Path(__file__).resolve().parents[1].joinpath("apps/ore_pipeline_web.py").read_text(encoding="utf-8"))
        self.assertIn("TALC_CLUSTER_COLOR = (64, 220, 255, 165)", Path(__file__).resolve().parents[1].joinpath("apps/ore_pipeline_web.py").read_text(encoding="utf-8"))
        self.assertIn("function tintedOverlayCanvasForImage(image, key, color)", html)
        self.assertIn("tintColor: cssColor('--artifact')", html)
        self.assertIn("function visibleCompositeLayers()", html)
        self.assertIn("function updateSegmentationToggleVisibility()", html)
        self.assertIn("function setLegendPanel(panelId, sulfideId, finalId, layer)", html)
        self.assertIn("function classVisible(key)", html)
        self.assertIn("function legendClassFraction(toggleKey, layer)", html)
        self.assertIn("function syncLegendPercentages()", html)
        self.assertIn("function syncClassVisibilityControls()", html)
        self.assertIn("const text = legendPercentText(legendClassFraction(input.dataset.legendToggle, layer))", html)
        self.assertIn("ordinary_sulfide_area_px", html)
        self.assertIn("fine_sulfide_area_px", html)
        self.assertIn("artifact_fraction_image", html)
        self.assertIn("talc_cluster_fraction", html)
        self.assertIn("talc_cluster_overlay", html)
        self.assertIn("if (layer === 'sulfide')", html)
        self.assertIn("if (toggleKey === 'showNonSulfide') return Math.max(0, 1 - sulfide);", html)
        self.assertIn("const showSulfide = classVisible('showSulfide')", html)
        self.assertIn("const showNonSulfide = classVisible('showNonSulfide')", html)
        self.assertIn("showSulfide ? baseLayerKey(display) : 'non_sulfide_base'", html)
        self.assertIn("if (showSulfide)", html)
        self.assertIn("if (classVisible('showSulfideArtifacts'))", html)
        self.assertIn("if (classVisible('showFinalArtifacts'))", html)
        self.assertIn("if (classVisible('showTalcClusters'))", html)
        self.assertIn("showImage: classVisible('showBackground')", html)
        self.assertIn("await drawOverlay(display.sulfide_overlay", html)
        self.assertIn("await drawOverlay(display.artifact_overlay", html)
        self.assertIn("function artifactOverlayColor(alpha = 180)", html)
        self.assertIn("const artifactColor = artifactOverlayColor(175)", html)
        self.assertIn("state.editor.layer === 'artifact') color = artifactColor", html)
        self.assertIn("фиолетовая кисть", html)
        self.assertIn("violet brush", html)
        self.assertLess(html.index('id="viewModeButtons"'), html.index('<div class="viewer-shell"'))
        self.assertLess(html.index('<div class="viewer-shell"'), html.index('id="segmentationClassToggles"'))
        self.assertLess(html.index('id="segmentationClassToggles"'), html.index('id="splitterOverlay"'))
        self.assertLess(html.index('id="splitterOverlay"'), html.index('id="mainCanvas"'))
        self.assertIn('class="splitter-overlay hidden"', html)
        self.assertIn('class="splitter-line"', html)
        self.assertIn('class="splitter-handle"', html)
        self.assertIn("function updateSplitterOverlay()", html)
        self.assertIn("function clampSideBySideSplitter(value)", html)
        self.assertIn("return Math.max(0, Math.min(1, number));", html)
        self.assertIn("state.splitter = clampSideBySideSplitter(point.x / canvas.width);", html)
        self.assertNotIn("Math.max(0.12, Math.min(0.88", html)
        self.assertIn("overlay.style.left = `${Math.round(state.splitter * 10000) / 100}%`;", html)
        self.assertLess(html.index('<div class="viewer-shell"'), html.index('class="viewer-options-row"'))
        self.assertLess(html.index('class="viewer-options-row"'), html.index('id="resultPanel"'))
        viewer_options = html.split('class="viewer-options-row"', 1)[1].split('id="resultPanel"', 1)[0]
        self.assertIn('id="showTiling"', viewer_options)
        self.assertIn('id="boundaryOnly"', viewer_options)
        self.assertIn('id="contourWidth"', viewer_options)
        self.assertIn('id="contourWidthValue"', viewer_options)
        self.assertIn('data-i18n="contourWidth"', viewer_options)
        self.assertIn('id="overlayOpacity"', viewer_options)
        self.assertIn('data-i18n="mouseWheelZoomHint"', viewer_options)
        self.assertIn("Колесо мыши - масштаб", viewer_options)
        self.assertIn("Mouse wheel - zoom in / out", html)
        self.assertIn('data-i18n="mouseWheelPanHint"', viewer_options)
        self.assertIn("Нажатие колеса - панорама", viewer_options)
        self.assertIn("Mouse wheel press - pan", html)
        self.assertNotIn('id="showOrdinary"', viewer_options)
        self.assertIn('id="zoomWidget"', html)
        self.assertIn(".zoom-widget-row { display: grid; grid-template-columns: 32px; gap: 6px; justify-content: center; }", html)
        self.assertIn(".zoom-widget button[data-tooltip]::after", html)
        self.assertIn(".zoom-widget button[data-tooltip]:hover::after", html)
        self.assertLess(html.index('id="fitViewBtn"'), html.index('id="actualSizeBtn"'))
        self.assertNotIn("grid-template-columns: 32px 32px", html)
        self.assertIn('id="fitViewBtn" type="button" title="Вписать" aria-label="Вписать" data-i18n-title="fitView" data-i18n-aria-label="fitView" data-i18n-tooltip="fitView"', html)
        self.assertIn('id="actualSizeBtn" type="button" title="Фактический размер" aria-label="Фактический размер" data-i18n-title="actualSize" data-i18n-aria-label="actualSize" data-i18n-tooltip="actualSize"', html)
        self.assertIn('id="zoomInBtn" type="button" title="Увеличить" aria-label="Увеличить" data-i18n-title="zoomIn" data-i18n-aria-label="zoomIn" data-i18n-tooltip="zoomIn"', html)
        self.assertIn('id="zoomLevel"', html)
        self.assertIn('id="zoomOutBtn" type="button" title="Уменьшить" aria-label="Уменьшить" data-i18n-title="zoomOut" data-i18n-aria-label="zoomOut" data-i18n-tooltip="zoomOut"', html)
        self.assertIn("function fitMainView()", html)
        self.assertIn("function actualSizeMainView()", html)
        self.assertIn("function zoomMainView(factor)", html)
        self.assertIn("fitViewBtn').addEventListener('click', fitMainView)", html)
        self.assertIn("actualSizeBtn').addEventListener('click', actualSizeMainView)", html)
        self.assertIn('id="sideLayerButtons"', html)
        self.assertNotIn('&lt;---&gt;', html)
        self.assertNotIn('<--->', html)
        self.assertNotIn('side-divider', html)
        self.assertIn('data-side-layer="none"', html)
        self.assertIn('data-side-layer="augmented"', html)
        self.assertNotIn('data-side-layer="artefacts"', html)
        self.assertIn('data-side-layer="preprocessed"', html)
        self.assertIn("function updateViewControls()", html)
        self.assertIn("function setSideLayer(layer)", html)
        self.assertNotIn('id="leftLayer"', html)
        self.assertNotIn('id="rightLayer"', html)
        self.assertIn('id="shareComparisonBtn"', html)
        self.assertIn('data-i18n="shareComparison"', html)
        self.assertIn("Поделиться сравнением", html)
        self.assertIn("Share comparison", html)
        self.assertIn("shareComparisonNoLayer", html)
        self.assertIn("function renderComparisonLayer(display, layer)", html)
        self.assertIn("function exportSideBySideComparison()", html)
        self.assertIn("canvasToBlobAsync(output)", html)
        self.assertIn("comparison_${safeFilenamePart(runPart)}_${safeFilenamePart(state.viewMode)}_${safeFilenamePart(state.sideLayer)}.png", html)
        self.assertLess(html.index('id="shareComparisonBtn"'), html.index('id="fixBtn"'))
        self.assertIn("Исправить", html)
        self.assertIn("#fixBtn { background: var(--danger)", html)
        self.assertIn("button.primary:disabled, button.danger:disabled, #fixBtn:disabled", html)
        self.assertIn("Исправить и перезапустить", html)
        self.assertIn("История запусков", html)
        self.assertIn("Text output", html)
        self.assertIn("Metrics", html)
        self.assertIn("Total sulfide fraction", html)
        self.assertIn("Ore classified as", html)
        self.assertIn("historyLoad", html)
        self.assertIn("Загрузить", html)
        self.assertNotIn(">Open</button>", html)
        self.assertIn('id="editLayerTabs"', html)
        self.assertIn('data-layer="artifact"', html)
        self.assertIn('data-layer="sulfide"', html)
        self.assertIn('data-layer="final"', html)
        self.assertIn("artefactsLayer", html)
        self.assertIn("artefactsLayerShort", html)
        self.assertIn("sulfideLayerShort", html)
        self.assertIn("finalLayerShort", html)
        self.assertIn("editorArtifactHelp", html)
        self.assertIn("saveArtefacts", html)
        self.assertIn("statArtefacts", html)
        self.assertIn("statOfSulfides", html)
        self.assertIn("/artifact-mask", html)
        self.assertIn("function ensureUploadPreparedForArtifactEditor()", html)
        self.assertIn("function editorLayerAvailable(layer)", html)
        self.assertIn("function preferredEditorLayer()", html)
        self.assertIn("state.viewMode === 'final'", html)
        self.assertIn("edit_layer: state.editor.layer", html)
        self.assertNotIn('id="editLayer">', html)
        self.assertIn('id="editorTopToolbar"', html)
        self.assertIn('id="brushToolBtn"', html)
        self.assertIn("Кисть: левая кнопка рисует, правая стирает.", html)
        self.assertIn("Перемещение двигает вид.", html)
        self.assertNotIn("Панорама перемещает вид.", html)
        self.assertIn('id="undoEditBtn"', html)
        self.assertIn('id="redoEditBtn"', html)
        self.assertNotIn('id="zoomInEditBtn"', html)
        self.assertNotIn('id="zoomOutEditBtn"', html)
        self.assertNotIn('id="fitEditBtn"', html)
        self.assertIn('id="editorZoomWidget"', html)
        self.assertIn('id="fitEditViewBtn" type="button" title="Вписать" aria-label="Вписать" data-i18n-title="fitView" data-i18n-aria-label="fitView" data-i18n-tooltip="fitView"', html)
        self.assertIn('id="actualSizeEditBtn" type="button" title="Фактический размер" aria-label="Фактический размер" data-i18n-title="actualSize" data-i18n-aria-label="actualSize" data-i18n-tooltip="actualSize"', html)
        self.assertIn('id="zoomInEditViewBtn" type="button" title="Увеличить" aria-label="Увеличить" data-i18n-title="zoomIn" data-i18n-aria-label="zoomIn" data-i18n-tooltip="zoomIn"', html)
        self.assertIn('id="editorZoomLevel"', html)
        self.assertIn('id="zoomOutEditViewBtn" type="button" title="Уменьшить" aria-label="Уменьшить" data-i18n-title="zoomOut" data-i18n-aria-label="zoomOut" data-i18n-tooltip="zoomOut"', html)
        editor_left_panel = html.split('<dialog id="fixDialog"', 1)[1].split('<div class="panel editor-side"', 1)[0]
        self.assertIn('class="editor-main"', editor_left_panel)
        self.assertIn('id="editorZoomWidget"', editor_left_panel)
        self.assertIn('class="viewer-options-hints"', editor_left_panel)
        self.assertIn('data-i18n="mouseWheelZoomHint"', editor_left_panel)
        self.assertIn('data-i18n="mouseWheelPanHint"', editor_left_panel)
        self.assertIn("function actualSizeEditorView()", html)
        self.assertIn("function updateEditorZoomWidget()", html)
        self.assertIn("function editorZoomPercent()", html)
        self.assertIn("fitEditViewBtn').addEventListener('click', fitEditorView)", html)
        self.assertIn("actualSizeEditBtn').addEventListener('click', actualSizeEditorView)", html)
        self.assertIn("Колесо мыши - масштаб", html)
        self.assertIn("Нажатие колеса - панорама", html)
        self.assertIn('id="panToolBtn"', html)
        self.assertIn('id="brushSize" type="range" min="2" max="240" value="28"', html)
        self.assertIn('id="brushSizeValue">28 px</output>', html)
        self.assertIn("function updateBrushSizeLabel()", html)
        self.assertIn("function editorBrushRadius()", html)
        self.assertIn("Number($('brushSize').value || 28) / 2", html)
        self.assertIn('id="editorStats"', html)
        self.assertIn("function refreshRunForEditor()", html)
        self.assertIn("await refreshRunForEditor()", html)
        self.assertIn("function editorBasePreview()", html)
        self.assertIn("function updateEditorStats()", html)
        self.assertIn("editorLoading", html)
        self.assertIn("editorLoadFailed", html)
        self.assertIn('id="themeSelect"', html)
        self.assertIn('value="dark"', html)
        self.assertIn(':root[data-theme="dark"]', html)
        self.assertIn("orePipelineTheme", html)


class OrePipelineOpenApiTest(unittest.TestCase):
    # Concrete non-templated segments that must still appear in the handler
    # source, so a renamed route trips this test instead of silently drifting
    # from the OpenAPI document.
    ROUTE_TOKENS = {
        "/api/auth/status",
        "/api/auth/login",
        "/api/auth/logout",
        "/api/status",
        "/api/settings",
        "/api/runtime/test",
        "/api/uploads",
        "/preprocess",
        "/artifact-mask",
        "/api/batches",
        "/results.csv",
        "/items",
        "/metadata",
        "/run",
        "/cancel",
        "/api/runs",
        "/api/runs/start",
        "/prepare",
        "/start",
        "/fix",
        "/files",
        "/metrics.csv",
        "/report.pdf",
        "/artifacts.zip",
        "/api/history",
        "/artifacts/",
    }

    def _make_server(self) -> tuple[OrePipelineHTTPServer, threading.Thread, str, int]:
        root = Path(tempfile.mkdtemp(prefix="test_ore_openapi_"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        store = OrePipelineStore(
            workspace_dir=root / "workspace",
            backend="heuristic",
            checkpoint=None,
            processing_max_side=256,
            panorama_max_side=128,
            preview_max_sides=(128, 256),
        )
        server = OrePipelineHTTPServer(("127.0.0.1", 0), store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        host, port = server.server_address[:2]
        return server, thread, host, port

    def test_document_is_valid_openapi_31_with_resolvable_refs(self) -> None:
        doc = build_openapi_document()
        self.assertEqual(doc["openapi"], "3.1.0")
        self.assertIn("paths", doc)
        self.assertTrue(doc["info"]["title"])
        self.assertTrue(doc["info"]["version"])
        text = json.dumps(doc)  # serializes cleanly for the wire
        for ref in set(re.findall(r'"\$ref": "(#[^"]+)"', text)):
            node: object = doc
            for part in ref.lstrip("#/").split("/"):
                self.assertIn(part, node, msg=f"unresolved $ref: {ref}")
                node = node[part]

    def test_document_passes_formal_openapi_spec_validation(self) -> None:
        try:
            from openapi_spec_validator import validate as validate_openapi
        except ImportError:  # pragma: no cover - optional dev dependency.
            self.skipTest("openapi-spec-validator not installed (see requirements-dev.txt)")
        # Raises OpenAPIValidationError with a descriptive message on any schema
        # violation, failing this test.
        validate_openapi(build_openapi_document())

    def test_documented_route_tokens_exist_in_handler_source(self) -> None:
        source = Path(ore_pipeline_web.__file__).read_text(encoding="utf-8")
        for token in self.ROUTE_TOKENS:
            self.assertIn(token, source, msg=f"route token {token!r} not found in handler source")

    def test_openapi_route_served_without_authentication(self) -> None:
        server, _thread, host, port = self._make_server()
        # Even with a UI password configured, the spec route must stay open so
        # tooling can read it before logging in.
        server.store.save_app_settings({"auth": {"password": "secret-pass"}})
        self.assertTrue(server.store.auth_enabled())
        connection = http.client.HTTPConnection(host, port, timeout=5)
        connection.request("GET", "/api/openapi.json")
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        content_type = response.getheader("Content-Type", "")
        connection.close()
        self.assertEqual(response.status, HTTPStatus.OK)
        self.assertIn("application/json", content_type)
        payload = json.loads(body)
        self.assertEqual(payload["openapi"], "3.1.0")
        self.assertIn("/api/openapi.json", payload["paths"])
        self.assertEqual(payload, build_openapi_document())


if __name__ == "__main__":
    unittest.main()
