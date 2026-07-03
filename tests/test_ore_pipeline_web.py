from __future__ import annotations

import base64
import csv
import http.client
import io
import json
import shutil
import sys
import threading
import time
import unittest
import zipfile
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
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

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
        self.assertIn("ordinary_overlay", run["display"])

        history = self.store.list_runs()["runs"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["run_id"], run["run_id"])
        self.assertEqual(history[0]["summary"]["ore_class"], run["summary"]["ore_class"])
        metric_keys = [row["key"] for row in history[0]["metrics"]]
        self.assertEqual(metric_keys[:4], ["analyzed_fraction", "sulfide_fraction", "ordinary_sulfide_fraction", "fine_sulfide_fraction"])
        self.assertIn("other_fraction", metric_keys)
        self.assertIn("artifact_fraction_image", metric_keys)
        self.assertEqual(history[0]["metrics"][0]["level"], 0)
        self.assertEqual(next(row for row in history[0]["metrics"] if row["key"] == "sulfide_fraction")["level"], 1)
        self.assertEqual(next(row for row in history[0]["metrics"] if row["key"] == "ordinary_sulfide_fraction")["parent_key"], "sulfide_fraction")
        self.assertTrue(history[0]["thumbnail"]["thumbnail_url"])
        self.assertTrue(history[0]["thumbnail"]["preview_url"])
        thumbnail_id = history[0]["thumbnail"]["thumbnail_url"].split("/")[2]
        self.assertTrue(self.store.artifact_path(thumbnail_id).exists())

    def test_run_files_payload_and_zip_endpoint_include_image_dimensions(self) -> None:
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
        files_payload = self.store.run_files_payload(run["run_id"])
        files_by_path = {file["path"]: file for file in files_payload["files"]}
        self.assertIn("run.json", files_by_path)
        self.assertIn("input/preprocessed.png", files_by_path)
        self.assertTrue(files_by_path["input/preprocessed.png"]["is_image"])
        self.assertEqual(files_by_path["input/preprocessed.png"]["width"], 128)
        self.assertEqual(files_by_path["input/preprocessed.png"]["height"], 96)
        self.assertIn("reports/ore_summary.json", files_by_path)
        self.assertFalse(files_by_path["reports/ore_summary.json"]["is_image"])
        self.assertNotIn("reports/run_artifacts.zip", files_by_path)

        zip_path = self.store.run_zip_path(run["run_id"])
        self.assertTrue(zip_path.exists())
        with zipfile.ZipFile(zip_path) as archive:
            names = set(archive.namelist())
        self.assertIn("run.json", names)
        self.assertIn("input/preprocessed.png", names)
        self.assertIn("reports/ore_summary.json", names)
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

            for page in ("/workspace", "/batch", "/batch/batch_demo", "/history", "/settings"):
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

    def test_app_settings_are_persisted_and_exposed_by_api(self) -> None:
        settings = self.store.save_app_settings(
            {
                "language": "en",
                "theme": "dark",
                "show_tiling": True,
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
        self.assertFalse(settings["preprocess"]["preprocessing_enabled"])
        self.assertEqual(settings["preprocess"]["panorama_scaling_mode"], "scale_factor")
        self.assertEqual(settings["preprocess"]["panorama_max_side_px"], 4096)
        self.assertEqual(settings["preprocess"]["panorama_scale_factor"], 0.25)
        self.assertEqual(settings["metadata_defaults"]["project"], "system-default-project")
        self.assertNotIn("sample_id", settings["metadata_defaults"])

        restarted = OrePipelineStore(
            workspace_dir=self.root / "workspace",
            backend="heuristic",
            checkpoint=None,
            processing_max_side=256,
            panorama_max_side=128,
            preview_max_sides=(128, 256),
        )
        self.assertEqual(restarted.app_settings()["metadata_defaults"]["om_instrument"], "scope-1")
        self.assertEqual(restarted.app_settings()["preprocess"]["panorama_scaling_mode"], "scale_factor")
        self.assertEqual(restarted.app_settings()["preprocess"]["panorama_scale_factor"], 0.25)

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

            connection = http.client.HTTPConnection(host, port, timeout=5)
            body = json.dumps({"theme": "neon"}).encode("utf-8")
            connection.request("PUT", "/api/settings", body=body, headers={"Content-Type": "application/json", "Content-Length": str(len(body))})
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()
            self.assertEqual(response.status, 400)
            self.assertIn("settings.theme", payload["error"])
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
        self.assertIn('id="metadataDialog"', html)
        self.assertIn('id="runFilesBtn"', html)
        self.assertIn('id="runFilesDialog"', html)
        self.assertIn('id="runFilesTable"', html)
        self.assertIn('id="runFilesZipLink"', html)
        self.assertIn("Просмотреть файлы", html)
        self.assertIn("View files", html)
        self.assertIn("Download ZIP", html)
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
        self.assertIn("buttonId: 'applyAugmentationBtn'", html)
        self.assertIn("augmentation: augmentationPayload()", html)
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
        self.assertIn('id="settingsLanguage"', html)
        self.assertIn('id="settingsTheme"', html)
        self.assertIn('id="settingsShowTiling"', html)
        self.assertIn('id="settingsPreprocessingEnabled"', html)
        self.assertIn('id="settingsMetaProject"', html)
        self.assertIn('id="saveSettingsBtn"', html)
        self.assertIn('id="resetSettingsBtn"', html)
        self.assertIn("/api/settings", html)
        self.assertIn("function loadAppSettings()", html)
        self.assertIn("function saveSettingsObject(settings", html)
        self.assertIn("ore-pipeline-app-settings-v0.1", html)
        self.assertIn('value="ru"', html)
        self.assertIn("Русский", html)
        self.assertIn("Russian", html)
        self.assertIn("English", html)
        self.assertIn("const DEFAULT_LANGUAGE = 'ru'", html)
        self.assertIn("orePipelineLanguage", html)
        self.assertIn("const PREPROCESS_STORAGE_KEY = 'orePipelinePreprocessPreset'", html)
        self.assertIn("const DEFAULT_PREPROCESS_PRESET", html)
        self.assertIn('id="preprocessingEnabled" checked', html)
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
        self.assertIn('class="metrics-table"', html)
        self.assertIn("metricsDenominatorNote", html)
        self.assertIn("Сульфиды, тальк и остальное считаются от проанализированной области", html)
        self.assertIn("Sulfides, talc, and other use analyzed area as denominator", html)
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
        self.assertIn("historyOpenBatch", html)
        self.assertIn("historyNoBatches", html)
        self.assertIn("state.historyMode === 'single'", html)
        self.assertIn("fetch('/api/batches')", html)
        self.assertIn("function renderHistoryThumbnail(run)", html)
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
        self.assertIn("historyOreClassification", html)
        self.assertIn("historyNonSulfides", html)
        self.assertIn("historyRemove", html)
        self.assertIn("const PAGE_SLUGS = {workspace: '/workspace', batch: '/batch', history: '/history', settings: '/settings'}", html)
        self.assertIn("function setPage(page, options = {})", html)
        self.assertIn("document.body.dataset.page = nextPage", html)
        self.assertIn('body[data-page="history"] aside', html)
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
        self.assertIn("function boundaryCanvasForImage(image, key)", html)
        self.assertIn("state.overlayOpacity", html)
        self.assertIn("state.boundaryOnly", html)
        self.assertIn("contours only", html)
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
        self.assertIn('data-legend-toggle="showFinalArtifacts"', final_controls)
        self.assertIn('data-legend-toggle="showBackground"', final_controls)
        self.assertIn("classOrdinaryShort", final_controls)
        self.assertIn("classFineShort", final_controls)
        self.assertIn("classTalc", final_controls)
        self.assertIn("classArtefacts", final_controls)
        self.assertIn("--artifact", html)
        self.assertIn("--sulfide", html)
        self.assertIn("--non-sulfide", html)
        self.assertIn("ARTIFACT_COLOR = (198, 60, 255, 180)", Path(__file__).resolve().parents[1].joinpath("apps/ore_pipeline_web.py").read_text(encoding="utf-8"))
        self.assertIn("function tintedOverlayCanvasForImage(image, key, color)", html)
        self.assertIn("tintColor: cssColor('--artifact')", html)
        self.assertIn("function visibleCompositeLayers()", html)
        self.assertIn("function updateSegmentationToggleVisibility()", html)
        self.assertIn("function setLegendPanel(panelId, sulfideId, finalId, layer)", html)
        self.assertIn("function classVisible(key)", html)
        self.assertIn("function syncClassVisibilityControls()", html)
        self.assertIn("if (layer === 'sulfide')", html)
        self.assertIn("showImage: classVisible('showNonSulfide')", html)
        self.assertIn("if (classVisible('showSulfide'))", html)
        self.assertIn("if (classVisible('showSulfideArtifacts'))", html)
        self.assertIn("if (classVisible('showFinalArtifacts'))", html)
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
        self.assertLess(html.index('id="segmentationClassToggles"'), html.index('id="mainCanvas"'))
        self.assertLess(html.index('<div class="viewer-shell"'), html.index('class="viewer-options-row"'))
        self.assertLess(html.index('class="viewer-options-row"'), html.index('id="resultPanel"'))
        viewer_options = html.split('class="viewer-options-row"', 1)[1].split('id="resultPanel"', 1)[0]
        self.assertIn('id="showTiling"', viewer_options)
        self.assertIn('id="boundaryOnly"', viewer_options)
        self.assertIn('id="overlayOpacity"', viewer_options)
        self.assertNotIn('id="showOrdinary"', viewer_options)
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
