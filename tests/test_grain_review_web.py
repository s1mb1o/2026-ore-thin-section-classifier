"""Tests for the grain review / labeling app (apps/grain_review_web.py).

Covers the store (manifest paging, filters, annotate persistence) and the HTTP
surface (page HTML, /api/page, /crops/ file serving, /api/annotate, error
paths). The crop-path sandbox escape itself is covered in
tests/test_grain_pipeline.py; here we only add the sandbox-adjacent 404 cases.
"""
from __future__ import annotations

import csv
import http.client
import json
import sys
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps"))

import grain_review_web  # noqa: E402


def build_dataset(dataset_dir: Path, grains: list[dict[str, str]]) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "grain_uid",
        "grade_label",
        "heuristic_label",
        "crop_path",
        "area_px",
        "dark_inside_ratio",
        "solidity",
    ]
    with (dataset_dir / "grains_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(grains)
    for grain in grains:
        crop = dataset_dir / grain["crop_path"]
        crop.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 10), (90, 90, 90)).save(crop)


def default_grains() -> list[dict[str, str]]:
    rows = []
    for i, grade in enumerate(
        ["ordinary_intergrowth", "ordinary_intergrowth", "fine_intergrowth", "talcose"]
    ):
        uid = f"run_{i}__c{i}"
        rows.append(
            {
                "grain_uid": uid,
                "grade_label": grade,
                "heuristic_label": "fine_intergrowth" if i % 2 else "ordinary_intergrowth",
                "crop_path": f"crops/{grade}/{uid}.png",
                "area_px": str(400 + i),
                "dark_inside_ratio": "0.25",
                "solidity": "0.8",
            }
        )
    return rows


class GrainReviewStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self._tmp.name) / "grain_dataset"
        build_dataset(self.dataset_dir, default_grains())
        self.store = grain_review_web.GrainReviewStore(self.dataset_dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_missing_manifest_exits(self) -> None:
        with self.assertRaises(SystemExit):
            grain_review_web.GrainReviewStore(Path(self._tmp.name) / "empty")

    def test_stats_start_unlabeled(self) -> None:
        stats = self.store.stats()
        self.assertEqual(stats["total"], 4)
        self.assertEqual(stats["labeled"], 0)
        self.assertEqual(stats["counts"]["ordinary_intergrowth"], 0)

    def test_page_pagination_and_item_payload(self) -> None:
        page = self.store.page(offset=1, limit=2, grade="all", view="all")
        self.assertEqual(page["filtered_total"], 4)
        self.assertEqual(len(page["items"]), 2)
        item = page["items"][0]
        self.assertEqual(item["grain_uid"], "run_1__c1")
        self.assertTrue(item["crop_url"].startswith("/crops/"))
        self.assertIsNone(item["label"])

    def test_page_grade_filter(self) -> None:
        page = self.store.page(offset=0, limit=60, grade="fine_intergrowth", view="all")
        self.assertEqual(page["filtered_total"], 1)
        self.assertEqual(page["items"][0]["grade_label"], "fine_intergrowth")

    def test_page_view_filter_tracks_labels(self) -> None:
        self.store.annotate("run_0__c0", "ordinary_intergrowth")
        unlabeled = self.store.page(offset=0, limit=60, grade="all", view="unlabeled")
        labeled = self.store.page(offset=0, limit=60, grade="all", view="labeled")
        self.assertEqual(unlabeled["filtered_total"], 3)
        self.assertEqual(labeled["filtered_total"], 1)
        self.assertEqual(labeled["items"][0]["label"], "ordinary_intergrowth")

    def test_annotate_persists_and_reloads(self) -> None:
        stats = self.store.annotate("run_2__c2", "fine_intergrowth")
        self.assertEqual(stats["labeled"], 1)
        self.assertEqual(stats["counts"]["fine_intergrowth"], 1)
        payload = json.loads((self.dataset_dir / "annotations.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], "grain-annotations-v0.1")
        self.assertEqual(payload["labels"]["run_2__c2"]["label"], "fine_intergrowth")
        # a fresh store picks the annotation up from disk
        reloaded = grain_review_web.GrainReviewStore(self.dataset_dir)
        self.assertEqual(reloaded.labels["run_2__c2"]["label"], "fine_intergrowth")

    def test_annotate_none_removes_label(self) -> None:
        self.store.annotate("run_0__c0", "uncertain")
        stats = self.store.annotate("run_0__c0", None)
        self.assertEqual(stats["labeled"], 0)
        payload = json.loads((self.dataset_dir / "annotations.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["labels"], {})

    def test_annotate_rejects_unknown_uid_and_bad_label(self) -> None:
        with self.assertRaises(grain_review_web.ApiError) as ctx:
            self.store.annotate("nope", "uncertain")
        self.assertEqual(ctx.exception.status, HTTPStatus.NOT_FOUND)
        with self.assertRaises(grain_review_web.ApiError) as ctx:
            self.store.annotate("run_0__c0", "talcose")
        self.assertEqual(ctx.exception.status, HTTPStatus.BAD_REQUEST)

    def test_crop_file_missing_is_not_found(self) -> None:
        with self.assertRaises(grain_review_web.ApiError) as ctx:
            self.store.crop_file("ordinary_intergrowth/missing.png")
        self.assertEqual(ctx.exception.status, HTTPStatus.NOT_FOUND)


class GrainReviewHTTPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dataset_dir = Path(cls._tmp.name) / "grain_dataset"
        build_dataset(cls.dataset_dir, default_grains())
        cls.store = grain_review_web.GrainReviewStore(cls.dataset_dir)
        cls.server = grain_review_web.GrainReviewHTTPServer(("127.0.0.1", 0), cls.store)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls._tmp.cleanup()

    def request(self, method: str, path: str, body: bytes | None = None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {"Content-Type": "application/json"} if body is not None else {}
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response, data

    def test_root_serves_review_page(self) -> None:
        response, data = self.request("GET", "/")
        self.assertEqual(response.status, HTTPStatus.OK)
        self.assertIn("text/html", response.getheader("Content-Type"))
        html = data.decode("utf-8")
        self.assertIn("Разметка зёрен", html)
        self.assertIn("/api/annotate", html)

    def test_api_page_returns_items_and_stats(self) -> None:
        response, data = self.request("GET", "/api/page?offset=0&limit=2&grade=all&view=all")
        self.assertEqual(response.status, HTTPStatus.OK)
        payload = json.loads(data)
        self.assertEqual(len(payload["items"]), 2)
        self.assertEqual(payload["filtered_total"], 4)
        self.assertEqual(payload["stats"]["total"], 4)

    def test_api_page_limit_is_clamped(self) -> None:
        response, data = self.request("GET", "/api/page?offset=0&limit=99999")
        self.assertEqual(response.status, HTTPStatus.OK)
        self.assertEqual(json.loads(data)["limit"], 200)

    def test_crops_endpoint_serves_png(self) -> None:
        response, data = self.request("GET", "/crops/ordinary_intergrowth/run_0__c0.png")
        self.assertEqual(response.status, HTTPStatus.OK)
        self.assertEqual(response.getheader("Content-Type"), "image/png")
        self.assertTrue(data.startswith(b"\x89PNG"))

    def test_crops_endpoint_rejects_traversal(self) -> None:
        response, _ = self.request("GET", "/crops/../grains_manifest.csv")
        self.assertEqual(response.status, HTTPStatus.NOT_FOUND)

    def test_unknown_get_and_post_paths_are_not_found(self) -> None:
        response, _ = self.request("GET", "/api/unknown")
        self.assertEqual(response.status, HTTPStatus.NOT_FOUND)
        response, _ = self.request("POST", "/api/unknown", body=b"{}")
        self.assertEqual(response.status, HTTPStatus.NOT_FOUND)

    def test_annotate_roundtrip_over_http(self) -> None:
        body = json.dumps({"grain_uid": "run_3__c3", "label": "uncertain"}).encode()
        response, data = self.request("POST", "/api/annotate", body=body)
        self.assertEqual(response.status, HTTPStatus.OK)
        payload = json.loads(data)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["stats"]["counts"]["uncertain"], 1)
        # clearing again keeps the suite order-independent
        clear = json.dumps({"grain_uid": "run_3__c3", "label": None}).encode()
        response, data = self.request("POST", "/api/annotate", body=clear)
        self.assertEqual(json.loads(data)["stats"]["labeled"], 0)

    def test_annotate_bad_label_is_bad_request(self) -> None:
        body = json.dumps({"grain_uid": "run_0__c0", "label": "wrong"}).encode()
        response, _ = self.request("POST", "/api/annotate", body=body)
        self.assertEqual(response.status, HTTPStatus.BAD_REQUEST)

    def test_invalid_json_and_empty_body_are_bad_requests(self) -> None:
        response, _ = self.request("POST", "/api/annotate", body=b"{not json")
        self.assertEqual(response.status, HTTPStatus.BAD_REQUEST)
        response, _ = self.request("POST", "/api/annotate", body=b"")
        self.assertEqual(response.status, HTTPStatus.BAD_REQUEST)

    def test_non_object_json_body_is_bad_request(self) -> None:
        response, _ = self.request("POST", "/api/annotate", body=b"[1, 2]")
        self.assertEqual(response.status, HTTPStatus.BAD_REQUEST)


if __name__ == "__main__":
    unittest.main()
