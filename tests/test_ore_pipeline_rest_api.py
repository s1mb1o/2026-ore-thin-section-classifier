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
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from apps.ore_pipeline_web import OrePipelineHTTPServer, OrePipelineStore  # noqa: E402


def image_mask_data_url(path: Path) -> str:
    data = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def blank_mask_data_url(width: int, height: int) -> str:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((0, 0, min(12, width - 1), min(12, height - 1)), fill=255)
    handle = io.BytesIO()
    mask.save(handle, format="PNG")
    return "data:image/png;base64," + base64.b64encode(handle.getvalue()).decode("ascii")


class OrePipelineRestApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / "outputs/test_ore_pipeline_rest_api"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.image_path = self.root / "sample.png"
        self.image_path_2 = self.root / "sample_2.png"
        self._write_sample_image(self.image_path, size=(160, 120), variant=1)
        self._write_sample_image(self.image_path_2, size=(140, 100), variant=2)
        self.store = OrePipelineStore(
            workspace_dir=self.root / "workspace",
            backend="heuristic",
            checkpoint=None,
            processing_max_side=256,
            panorama_max_side=128,
            preview_max_sides=(128, 256),
            # Pin heuristic backends so the API-contract tests do not depend on the
            # ML talc default (DEFAULT_TALC_BACKEND is "ml" whenever the SegFormer-B0
            # checkpoint is present). Loading/running SegFormer takes ~20s+ on CPU/MPS,
            # far longer than the 10s client timeouts here, causing false failures.
            # This mirrors OrePipelineWebTest, which pins talc_backend="heuristic".
            talc_backend="heuristic",
            grain_backend="heuristic",
        )
        self.server = OrePipelineHTTPServer(("127.0.0.1", 0), self.store)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.server_address[:2]

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_sample_image(self, path: Path, *, size: tuple[int, int], variant: int) -> None:
        width, height = size
        rgb = np.full((height, width, 3), (48 + variant * 4, 58, 51), dtype=np.uint8)
        image = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(image)
        draw.ellipse((24, 24, 66, 66), fill=(228, 226, 216))
        draw.rectangle((width - 58, 28, width - 18, min(height - 18, 78)), fill=(214, 212, 202))
        draw.rectangle((12, height - 38, 52, height - 8), fill=(70, 112, 72))
        image.save(path)

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        request_headers = dict(headers or {})
        request_body = body
        if json_body is not None:
            request_body = json.dumps(json_body).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        if request_body is not None:
            request_headers["Content-Length"] = str(len(request_body))
        connection = http.client.HTTPConnection(self.host, self.port, timeout=10)
        connection.request(method, path, body=request_body, headers=request_headers)
        response = connection.getresponse()
        data = response.read()
        response_headers = dict(response.getheaders())
        connection.close()
        return response.status, response_headers, data

    def json_request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        expected: int = 200,
    ) -> dict[str, Any]:
        status, response_headers, data = self.request(method, path, json_body=json_body, body=body, headers=headers)
        self.assertEqual(status, expected, data.decode("utf-8", errors="replace"))
        self.assertIn("application/json", response_headers.get("Content-Type", ""))
        return json.loads(data.decode("utf-8"))

    def multipart_upload(self, path: Path) -> dict[str, Any]:
        boundary = f"ore-rest-api-test-{time.time_ns()}"
        payload = b"".join(
            [
                f"--{boundary}\r\n".encode("ascii"),
                f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode("utf-8"),
                b"Content-Type: image/png\r\n\r\n",
                path.read_bytes(),
                b"\r\n",
                f"--{boundary}--\r\n".encode("ascii"),
            ]
        )
        return self.json_request(
            "POST",
            "/api/uploads",
            body=payload,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

    def wait_for_run(self, run_id: str, *, timeout: float = 10.0) -> dict[str, Any]:
        deadline = time.time() + timeout
        last_payload: dict[str, Any] | None = None
        while time.time() < deadline:
            last_payload = self.json_request("GET", f"/api/runs/{run_id}")
            if last_payload.get("status") in {"complete", "failed", "canceled"}:
                self.assertEqual(last_payload.get("status"), "complete")
                return last_payload
            time.sleep(0.05)
        self.fail(f"run {run_id} did not complete; last payload={last_payload}")

    def wait_for_batch(self, batch_id: str, *, timeout: float = 10.0) -> dict[str, Any]:
        deadline = time.time() + timeout
        last_payload: dict[str, Any] | None = None
        while time.time() < deadline:
            last_payload = self.json_request("GET", f"/api/batches/{batch_id}")
            if last_payload.get("status") in {"complete", "failed", "partial", "canceled"}:
                self.assertEqual(last_payload.get("status"), "complete")
                return last_payload
            time.sleep(0.05)
        self.fail(f"batch {batch_id} did not complete; last payload={last_payload}")

    def start_completed_run(self) -> dict[str, Any]:
        upload = self.multipart_upload(self.image_path)
        run = self.json_request(
            "POST",
            "/api/runs/start",
            json_body={
                "upload_id": upload["upload_id"],
                "preprocessing_enabled": True,
                "illumination_normalization": True,
                "denoise": True,
                "contrast_correction": True,
                "panorama_scaling": False,
            },
        )
        return self.wait_for_run(run["run_id"])

    def test_health_settings_and_error_contracts(self) -> None:
        status, headers, _ = self.request("GET", "/workspace")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers.get("Content-Type", ""))

        status_payload = self.json_request("GET", "/api/status")
        self.assertEqual(status_payload["schema_version"], "ore-pipeline-status-v0.1")
        self.assertIn(status_payload["health"]["overall"], {"ok", "warning", "error"})
        self.assertEqual(status_payload["app"]["backend"], "heuristic")
        self.assertTrue(any(entry.get("path") == "/workspace" for entry in status_payload["logs"]["access"]))

        settings = self.json_request("GET", "/api/settings")
        self.assertEqual(settings["schema_version"], "ore-pipeline-app-settings-v0.1")
        saved = self.json_request(
            "PUT",
            "/api/settings",
            json_body={"language": "en", "theme": "dark", "runtime": {"backend": "heuristic"}},
        )
        self.assertEqual(saved["language"], "en")
        self.assertEqual(saved["theme"], "dark")
        self.assertEqual(saved["runtime"]["backend"], "heuristic")
        runtime_probe = self.json_request("POST", "/api/runtime/test", json_body={"runtime": {"backend": "heuristic"}})
        self.assertTrue(runtime_probe["ok"])
        self.assertEqual(runtime_probe["backend"], "heuristic")

        invalid_settings = self.json_request("PUT", "/api/settings", json_body={"theme": "neon"}, expected=400)
        self.assertIn("settings.theme", invalid_settings["error"])
        invalid_json = self.json_request(
            "POST",
            "/api/runs/start",
            body=b"{",
            headers={"Content-Type": "application/json"},
            expected=400,
        )
        self.assertIn("invalid JSON", invalid_json["error"])
        bad_upload = self.json_request(
            "POST",
            "/api/uploads",
            body=b"not multipart",
            headers={"Content-Type": "application/octet-stream"},
            expected=400,
        )
        self.assertIn("multipart/form-data", bad_upload["error"])

    def test_upload_run_and_artifact_download_endpoints(self) -> None:
        upload = self.multipart_upload(self.image_path)
        self.assertEqual(upload["schema_version"], "ore-pipeline-upload-v0.1")
        self.assertEqual(upload["width"], 160)
        self.assertTrue(upload["upload_id"])

        upload_readback = self.json_request("GET", f"/api/uploads/{upload['upload_id']}")
        self.assertEqual(upload_readback["upload_id"], upload["upload_id"])
        preprocessed = self.json_request(
            "POST",
            f"/api/uploads/{upload['upload_id']}/preprocess",
            json_body={
                "preprocessing_enabled": True,
                "illumination_normalization": True,
                "denoise": True,
                "contrast_correction": True,
                "panorama_scaling": False,
            },
        )
        self.assertEqual(preprocessed["upload_id"], upload["upload_id"])
        self.assertIn("preprocessed", preprocessed["display"])

        artifact_payload = self.json_request(
            "POST",
            f"/api/uploads/{upload['upload_id']}/artifact-mask",
            json_body={
                "mask_png": blank_mask_data_url(
                    int(preprocessed["preprocess"]["width"]),
                    int(preprocessed["preprocess"]["height"]),
                ),
                "comment": "rest api suite",
            },
        )
        self.assertIn("artifact_mask", artifact_payload)

        run = self.json_request(
            "POST",
            "/api/runs/start",
            json_body={
                "upload_id": upload["upload_id"],
                "preprocessing_enabled": True,
                "illumination_normalization": True,
                "denoise": True,
                "contrast_correction": True,
                "panorama_scaling": False,
                "curated_metadata": {"domain": {"project": "rest-api-suite"}},
            },
        )
        completed = self.wait_for_run(run["run_id"])
        self.assertEqual(completed["input"]["curated_metadata"]["domain"]["project"], "rest-api-suite")

        run_history = self.json_request("GET", "/api/runs")
        self.assertTrue(any(item["run_id"] == completed["run_id"] for item in run_history["runs"]))
        files = self.json_request("GET", f"/api/runs/{completed['run_id']}/files")
        self.assertGreater(files["file_count"], 0)
        self.assertTrue(any(item.get("is_image") for item in files["files"]))

        status, headers, csv_data = self.request("GET", f"/api/runs/{completed['run_id']}/metrics.csv")
        self.assertEqual(status, 200)
        self.assertIn("text/csv", headers.get("Content-Type", ""))
        metrics_csv = csv_data.decode("utf-8")
        self.assertIn("metric,key", metrics_csv)
        self.assertIn("sulfide_fraction", metrics_csv)

        status, headers, pdf_data = self.request("GET", f"/api/runs/{completed['run_id']}/report.pdf")
        self.assertEqual(status, 200)
        self.assertIn("application/pdf", headers.get("Content-Type", ""))
        self.assertTrue(pdf_data.startswith(b"%PDF"))

        status, headers, zip_data = self.request("GET", f"/api/runs/{completed['run_id']}/artifacts.zip")
        self.assertEqual(status, 200)
        self.assertIn("application/zip", headers.get("Content-Type", ""))
        self.assertIn("attachment", headers.get("Content-Disposition", ""))
        with zipfile.ZipFile(io.BytesIO(zip_data)) as archive:
            names = set(archive.namelist())
        self.assertIn("run.json", names)
        self.assertIn("reports/ore_summary.json", names)

        cancel_complete = self.json_request("POST", f"/api/runs/{completed['run_id']}/cancel")
        self.assertEqual(cancel_complete["status"], "complete")
        removed = self.json_request("DELETE", f"/api/runs/{completed['run_id']}")
        self.assertEqual(removed["removed_run_id"], completed["run_id"])

    def test_prepare_start_and_fix_run_endpoints(self) -> None:
        completed = self.start_completed_run()
        prepared = self.json_request(
            "POST",
            f"/api/runs/{completed['run_id']}/prepare",
            json_body={
                "changed_step": "preprocess",
                "preprocessing_enabled": True,
                "illumination_normalization": True,
                "denoise": False,
                "contrast_correction": True,
                "panorama_scaling": False,
            },
        )
        self.assertEqual(prepared["status"], "prepared")
        self.assertEqual(prepared["derivation"]["parent_run_id"], completed["run_id"])

        restarted = self.json_request("POST", f"/api/runs/{prepared['run_id']}/start", json_body={})
        restarted = self.wait_for_run(restarted["run_id"])
        self.assertEqual(restarted["status"], "complete")

        final_mask = self.store.runs_dir / restarted["run_id"] / "masks/final_mask.png"
        edited = self.json_request(
            "POST",
            f"/api/runs/{restarted['run_id']}/fix",
            json_body={
                "edit_layer": "final",
                "mask_png": image_mask_data_url(final_mask),
                "comment": "rest api suite final edit",
            },
        )
        self.assertEqual(edited["status"], "complete")
        self.assertEqual(edited["derivation"]["edit_layer"], "final")
        self.assertEqual(edited["derivation"]["parent_run_id"], restarted["run_id"])

    def test_series_lifecycle_endpoints(self) -> None:
        upload_a = self.multipart_upload(self.image_path)
        upload_b = self.multipart_upload(self.image_path_2)
        batch = self.json_request(
            "POST",
            "/api/batches",
            json_body={
                "settings": {
                    "preprocess": {"preprocessing_enabled": True, "panorama_scaling": False},
                    "augmentation": {"enabled": False},
                }
            },
        )
        self.assertEqual(batch["status"], "draft")
        cancel_draft = self.json_request("POST", f"/api/batches/{batch['batch_id']}/cancel")
        self.assertEqual(cancel_draft["status"], "draft")

        batch = self.json_request(
            "POST",
            f"/api/batches/{batch['batch_id']}/items",
            json_body={"upload_ids": [upload_a["upload_id"], upload_b["upload_id"]]},
        )
        self.assertEqual(len(batch["items"]), 2)
        first_item = batch["items"][0]
        second_item = batch["items"][1]

        batch = self.json_request(
            "PUT",
            f"/api/batches/{batch['batch_id']}/items/{first_item['item_id']}/metadata",
            json_body={"curated_metadata": {"domain": {"sample_id": "REST-A"}}},
        )
        self.assertEqual(batch["items"][0]["curated_metadata"]["domain"]["sample_id"], "REST-A")

        batch = self.json_request("DELETE", f"/api/batches/{batch['batch_id']}/items/{second_item['item_id']}")
        self.assertEqual(len(batch["items"]), 1)
        batch = self.json_request(
            "PUT",
            f"/api/batches/{batch['batch_id']}/settings",
            json_body={
                "settings": {
                    "preprocess": {"preprocessing_enabled": True, "panorama_scaling": False},
                    "augmentation": {"enabled": False},
                }
            },
        )
        self.assertEqual(batch["settings"]["preprocess"]["panorama_scaling"], False)

        queued = self.json_request(
            "POST",
            f"/api/batches/{batch['batch_id']}/run",
            json_body={
                "preprocess": {"preprocessing_enabled": True, "panorama_scaling": False},
                "augmentation": {"enabled": False},
            },
        )
        completed = self.wait_for_batch(queued["batch_id"])
        self.assertEqual(completed["item_counts"]["complete"], 1)
        self.assertTrue(completed["items"][0]["run_id"])

        batch_history = self.json_request("GET", "/api/batches")
        self.assertTrue(any(item["batch_id"] == completed["batch_id"] for item in batch_history["batches"]))
        status, headers, csv_data = self.request("GET", f"/api/batches/{completed['batch_id']}/results.csv")
        self.assertEqual(status, 200)
        self.assertIn("text/csv", headers.get("Content-Type", ""))
        self.assertIn("ore_class", csv_data.decode("utf-8"))

        removed = self.json_request("DELETE", f"/api/batches/{completed['batch_id']}")
        self.assertEqual(removed["removed_batch_id"], completed["batch_id"])
        self.assertEqual(len(removed["removed_run_ids"]), 1)


if __name__ == "__main__":
    unittest.main()
