from __future__ import annotations

import base64
import http.client
import io
import json
import shutil
import sys
import threading
import time
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from apps.ore_pipeline_web import OrePipelineHTTPServer, OrePipelineStore, RunCancelled, render_html_page  # noqa: E402


def mask_data_url(mask: np.ndarray) -> str:
    handle = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8), mode="L").save(handle, format="PNG")
    return "data:image/png;base64," + base64.b64encode(handle.getvalue()).decode("ascii")


class OrePipelineWebTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / "outputs/test_ore_pipeline_web"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.image_path = self.root / "sample.png"
        rgb = np.full((120, 160, 3), (54, 62, 52), dtype=np.uint8)
        cv2.circle(rgb, (44, 46), 18, (226, 225, 212), -1)
        cv2.rectangle(rgb, (95, 35), (135, 82), (215, 214, 205), -1)
        cv2.rectangle(rgb, (12, 82), (52, 112), (66, 108, 70), -1)
        Image.fromarray(rgb, mode="RGB").save(self.image_path)
        self.store = OrePipelineStore(
            workspace_dir=self.root / "workspace",
            backend="heuristic",
            checkpoint=None,
            processing_max_side=256,
            panorama_max_side=128,
            preview_max_sides=(128, 256),
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_upload_preprocess_run_and_download_artifacts(self) -> None:
        upload = self.store.register_upload_from_path(self.image_path)
        self.assertEqual(upload["width"], 160)
        self.assertIn("original", upload["display"])

        preprocessed = self.store.prepare_upload(
            upload["upload_id"],
            {
                "illumination_normalization": True,
                "denoise": True,
                "contrast_correction": True,
                "panorama_scaling": True,
            },
        )
        self.assertTrue(preprocessed["preprocess"]["source_scaled_for_processing"])
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
        self.assertIn("sulfide", run["masks"])
        self.assertIn("ordinary_overlay", run["display"])

        history = self.store.list_runs()["runs"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["run_id"], run["run_id"])
        self.assertEqual(history[0]["summary"]["ore_class"], run["summary"]["ore_class"])
        self.assertEqual(history[0]["metrics"][0]["key"], "sulfide_fraction")
        self.assertTrue(history[0]["thumbnail"]["thumbnail_url"])
        self.assertTrue(history[0]["thumbnail"]["preview_url"])
        thumbnail_id = history[0]["thumbnail"]["thumbnail_url"].split("/")[2]
        self.assertTrue(self.store.artifact_path(thumbnail_id).exists())

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

            for page in ("/workspace", "/history"):
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
        self.assertEqual(self.store._read_run(run_id)["status"], "canceling")
        with self.assertRaises(RunCancelled):
            self.store._check_cancelled(run_id)

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
        self.assertIn("SUPPORTED_UPLOAD_EXTENSIONS", html)
        self.assertIn("function uploadFileWithProgress(file)", html)
        self.assertIn("XMLHttpRequest", html)
        self.assertIn("startPreviewPreparationProgress", html)
        self.assertIn("uploadProgressPreparing", html)
        self.assertIn("function handleSelectedFile(file)", html)
        self.assertIn("function isSupportedUploadFile(file)", html)
        self.assertIn("invalidImageFormat", html)
        self.assertIn("Unsupported file format", html)
        self.assertIn('id="languageSelect"', html)
        self.assertIn('value="ru"', html)
        self.assertIn("Русский", html)
        self.assertIn("Russian", html)
        self.assertIn("English", html)
        self.assertIn("const DEFAULT_LANGUAGE = 'ru'", html)
        self.assertIn("orePipelineLanguage", html)
        self.assertIn("const PREPROCESS_STORAGE_KEY = 'orePipelinePreprocessPreset'", html)
        self.assertIn("const DEFAULT_PREPROCESS_PRESET", html)
        self.assertIn('id="illumination" checked', html)
        self.assertIn('id="denoise" checked', html)
        self.assertIn('id="contrast" checked', html)
        self.assertIn('id="panoramaScaling" checked', html)
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
        self.assertIn("function renderHistoryTable(runs)", html)
        self.assertIn("function renderHistoryThumbnail(run)", html)
        self.assertIn("function openHistoryPreview(url, title)", html)
        self.assertIn('id="historyPreviewDialog"', html)
        self.assertIn('id="historyPreviewImage"', html)
        self.assertIn('data-preview-run', html)
        self.assertIn("historyThumbnail", html)
        self.assertIn("historyPreviewTitle", html)
        self.assertIn("function removeRun(runId)", html)
        self.assertIn("data-delete-run", html)
        self.assertIn("historyFilename", html)
        self.assertIn("historyOreClassification", html)
        self.assertIn("historyNonSulfides", html)
        self.assertIn("historyRemove", html)
        self.assertIn("const PAGE_SLUGS = {workspace: '/workspace', history: '/history'}", html)
        self.assertIn("function setPage(page, options = {})", html)
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
        self.assertIn("illumination normalization", html)
        self.assertIn("noise reduction", html)
        self.assertIn("contrast correction", html)
        self.assertIn("panorama image scaling", html)
        self.assertIn('id="showTiling"', html)
        self.assertIn("показать тайлы", html)
        self.assertIn("show tiling", html)
        self.assertIn("function drawTilingGrid(display)", html)
        self.assertIn("function tilingManifest()", html)
        self.assertIn("Сравнение:", html)
        self.assertIn('id="sideLayerButtons"', html)
        self.assertIn('data-side-layer="none"', html)
        self.assertIn('data-side-layer="preprocessed"', html)
        self.assertIn("function updateViewControls()", html)
        self.assertIn("function setSideLayer(layer)", html)
        self.assertNotIn('id="leftLayer"', html)
        self.assertNotIn('id="rightLayer"', html)
        self.assertIn("Исправить", html)
        self.assertIn("#fixBtn { background: var(--danger)", html)
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
        self.assertIn('data-layer="sulfide"', html)
        self.assertIn('data-layer="final"', html)
        self.assertNotIn('id="editLayer">', html)
        self.assertIn('id="editorTopToolbar"', html)
        self.assertIn('id="brushToolBtn"', html)
        self.assertIn("Кисть: левая кнопка рисует, правая стирает.", html)
        self.assertIn('id="undoEditBtn"', html)
        self.assertIn('id="redoEditBtn"', html)
        self.assertIn('id="zoomInEditBtn"', html)
        self.assertIn('id="zoomOutEditBtn"', html)
        self.assertIn('id="fitEditBtn"', html)
        self.assertIn('id="panToolBtn"', html)
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


if __name__ == "__main__":
    unittest.main()
