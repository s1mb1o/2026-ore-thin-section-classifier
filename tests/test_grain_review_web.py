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
    for grain in grains:
        if not grain.get("source_dataset_path"):
            grain["source_dataset_path"] = str(dataset_dir / grain["image_rel_path"])
    fieldnames = [
        "grain_uid",
        "run_id",
        "grade_label",
        "heuristic_label",
        "image_rel_path",
        "source_dataset_path",
        "component_id",
        "crop_path",
        "area_px",
        "dark_inside_ratio",
        "solidity",
        "compactness",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
    ]
    with (dataset_dir / "grains_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(grains)
    batch_dir = dataset_dir / "batch"
    (dataset_dir / "dataset_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "grain-dataset-v0.1",
                "batch_dir": str(batch_dir),
                "params": {"crop_pad_px": 10, "crop_max_side": 256},
            }
        ),
        encoding="utf-8",
    )
    for grain in grains:
        crop = dataset_dir / grain["crop_path"]
        crop.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 10), (90, 90, 90)).save(crop)
        source = Path(grain["source_dataset_path"])
        source.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), (120, 115, 100)).save(source)
        mask_dir = batch_dir / "runs" / grain["grade_label"] / grain["run_id"] / "binary_sulfide"
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask = Image.new("L", (64, 48), 0)
        x, y = int(grain["bbox_x"]), int(grain["bbox_y"])
        w, h = int(grain["bbox_w"]), int(grain["bbox_h"])
        for yy in range(y, min(y + h, mask.height)):
            for xx in range(x, min(x + w, mask.width)):
                mask.putpixel((xx, yy), 255)
        mask.save(mask_dir / "sulfide_mask.png")


def default_grains() -> list[dict[str, str]]:
    rows = []
    for i, grade in enumerate(
        ["ordinary_intergrowth", "ordinary_intergrowth", "fine_intergrowth", "talcose"]
    ):
        uid = f"run_{i}__c{i}"
        rows.append(
            {
                "grain_uid": uid,
                "run_id": f"run_{i}",
                "grade_label": grade,
                "heuristic_label": "fine_intergrowth" if i % 2 else "ordinary_intergrowth",
                "image_rel_path": f"source_images/source_{i}.jpg",
                "source_dataset_path": "",
                "component_id": str(i),
                "crop_path": f"crops/{grade}/{uid}.png",
                "area_px": str(400 + i),
                "dark_inside_ratio": "0.25",
                "solidity": "0.8",
                "compactness": "0.25",
                "bbox_x": str(5 + i),
                "bbox_y": str(6 + i),
                "bbox_w": "18",
                "bbox_h": "12",
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
        self.assertTrue(item["source_url"].startswith("/source/"))
        self.assertEqual(item["bbox"], {"x": 6.0, "y": 7.0, "w": 18.0, "h": 12.0})
        self.assertEqual(
            item["heuristic_scores"],
            {"ordinary": 67, "fine": 33, "fine_votes": 1, "total_votes": 3},
        )
        self.assertEqual(
            [(row["label"], row["rule"], row["matched"]) for row in item["heuristic_rows"]],
            [
                ("Тёмное внутри", "≥ 0.18", True),
                ("Выпуклость", "≤ 0.62", False),
                ("Компактность", "≤ 0.12", False),
            ],
        )
        self.assertIsInstance(item["review_value"]["score"], int)
        self.assertIsNone(item["label"])

    def test_page_review_value_sort_prioritizes_valuable_manual_review_cases(self) -> None:
        self.store.grains[0].update(
            {
                "area_px": "120",
                "dark_inside_ratio": "0.02",
                "solidity": "0.92",
                "compactness": "0.30",
            }
        )
        self.store.grains[1].update(
            {
                "area_px": "900",
                "dark_inside_ratio": "0.17",
                "solidity": "0.61",
                "compactness": "0.13",
            }
        )

        manifest = self.store.page(offset=0, limit=1, grade="all", view="all")
        valuable = self.store.page(offset=0, limit=1, grade="all", view="all", sort="review_value")

        self.assertEqual(manifest["items"][0]["grain_uid"], "run_0__c0")
        self.assertEqual(valuable["sort"], "review_value")
        self.assertEqual(valuable["items"][0]["grain_uid"], "run_1__c1")
        self.assertIn("близко к порогам", valuable["items"][0]["review_value"]["reasons"])

    def test_small_fine_ordinary_shape_context_is_flagged(self) -> None:
        self.store.grains[2].update(
            {
                "area_px": "100",
                "dark_inside_ratio": "0.02",
                "solidity": "0.88",
                "compactness": "0.30",
            }
        )

        page = self.store.page(offset=0, limit=4, grade="all", view="all", focus="run_2__c2")
        item = next(item for item in page["items"] if item["grain_uid"] == "run_2__c2")

        self.assertTrue(item["small_fine_context"]["matched"])
        self.assertEqual(item["small_fine_context"]["threshold_px"], self.store.small_area_threshold_px)
        self.assertIn("мелкое в тонких", item["review_value"]["reasons"])
        self.assertGreaterEqual(item["review_value"]["small_context"], 20)

    def test_page_rejects_unknown_sort(self) -> None:
        with self.assertRaises(grain_review_web.ApiError) as ctx:
            self.store.page(offset=0, limit=1, grade="all", view="all", sort="nope")
        self.assertEqual(ctx.exception.status, HTTPStatus.BAD_REQUEST)

    def test_page_focus_adjusts_offset_to_include_requested_grain(self) -> None:
        page = self.store.page(offset=0, limit=2, grade="all", view="all", focus="run_2__c2")
        self.assertEqual(page["offset"], 2)
        self.assertTrue(page["focus_found"])
        self.assertEqual(page["items"][0]["grain_uid"], "run_2__c2")

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
        self.assertIn("Tinder mode", html)
        self.assertIn("← тонкое", html)
        self.assertIn("/api/annotate", html)

    def test_tinder_slug_serves_review_page(self) -> None:
        response, data = self.request("GET", "/tinder/run_2__c2")
        self.assertEqual(response.status, HTTPStatus.OK)
        self.assertIn("text/html", response.getheader("Content-Type"))
        self.assertIn("Tinder mode", data.decode("utf-8"))

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

    def test_source_endpoint_serves_manifest_source_image(self) -> None:
        response, data = self.request("GET", "/source/run_0__c0")
        self.assertEqual(response.status, HTTPStatus.OK)
        self.assertEqual(response.getheader("Content-Type"), "image/jpeg")
        self.assertTrue(data.startswith(b"\xff\xd8"))

    def test_source_endpoint_rejects_unknown_uid(self) -> None:
        response, _ = self.request("GET", "/source/does-not-exist")
        self.assertEqual(response.status, HTTPStatus.NOT_FOUND)

    def test_contour_endpoint_serves_png_overlay(self) -> None:
        response, data = self.request("GET", "/contours/run_0__c0")
        self.assertEqual(response.status, HTTPStatus.OK)
        self.assertEqual(response.getheader("Content-Type"), "image/png")
        self.assertTrue(data.startswith(b"\x89PNG"))
        self.assertGreater(len(data), 50)

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
