from __future__ import annotations

import base64
import io
import json
import shutil
import sys
import threading
import urllib.parse
import unittest
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from apps.talc_review_web import TalcReviewHTTPServer, TalcReviewStore, render_html_page  # noqa: E402
from ore_classifier.talc_blue_line_converter import read_mask  # noqa: E402


def mask_data_url(mask: np.ndarray) -> str:
    handle = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8), mode="L").save(handle, format="PNG")
    return "data:image/png;base64," + base64.b64encode(handle.getvalue()).decode("ascii")


class TalcReviewWebTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / "outputs/test_talc_review_web"
        shutil.rmtree(self.root, ignore_errors=True)
        self.annotated_dir = self.root / "annotated"
        self.original_dir = self.root / "original"
        self.workspace_dir = self.root / "workspace"
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        self.original_dir.mkdir(parents=True, exist_ok=True)
        self.image_name = "sample_blue.JPG"
        annotated = np.full((90, 120, 3), (54, 66, 48), dtype=np.uint8)
        cv2.rectangle(annotated, (22, 18), (90, 72), (0, 0, 255), thickness=4)
        original = np.full((90, 120, 3), (54, 66, 48), dtype=np.uint8)
        Image.fromarray(annotated, mode="RGB").save(self.annotated_dir / self.image_name)
        Image.fromarray(original, mode="RGB").save(self.original_dir / self.image_name)
        self.store = TalcReviewStore(
            annotated_dir=self.annotated_dir,
            original_dir=self.original_dir,
            workspace_dir=self.workspace_dir,
            conversion_dir=None,
            sulfide_mask_dir=None,
            silicate_mask_dir=None,
            reconvert=True,
            limit=None,
            sam2_model_id="facebook/sam2.1-hiera-tiny",
            sam2_device="cpu",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_pairs_annotated_file_with_same_filename_original(self) -> None:
        manifest = self.store.manifest_payload()
        self.assertEqual(manifest["sample_count"], 1)
        sample = manifest["samples"][0]
        self.assertEqual(sample["image_name"], self.image_name)
        self.assertTrue(sample["has_original"])
        self.assertNotEqual(sample["status"], "missing_original")

    def test_first_sample_open_creates_current_mask_from_autodetected_mask(self) -> None:
        sample_id = self.store.manifest_payload()["samples"][0]["sample_id"]
        payload = self.store.sample_payload(sample_id)
        current_path = Path(payload["image"]["sample_dir"]) / "current_talc_mask.png"
        final_path = Path(payload["summary"]["paths"]["final_talc_mask"])
        self.assertTrue(current_path.exists())
        np.testing.assert_array_equal(read_mask(current_path), read_mask(final_path))
        self.assertEqual(payload["metrics"]["current_talc_pixels"], int(np.count_nonzero(read_mask(final_path))))

    def test_save_review_writes_current_reviewed_overlay_patch_and_summary(self) -> None:
        sample_id = self.store.manifest_payload()["samples"][0]["sample_id"]
        payload = self.store.sample_payload(sample_id)
        mask = np.zeros((payload["image"]["height"], payload["image"]["width"]), dtype=np.uint8)
        mask[10:30, 12:42] = 255

        result = self.store.save_current_mask(
            sample_id,
            {
                "mask_png": mask_data_url(mask),
                "edits": [{"type": "test_rectangle", "x1": 12, "y1": 10, "x2": 42, "y2": 30}],
                "reviewer": "unit-test",
                "notes": "synthetic",
            },
            reviewed=True,
        )

        self.assertTrue(result["reviewed"])
        reviewed = result["review_summary"]["paths"]
        for key in ["reviewed_talc_mask", "reviewed_ignore_mask", "reviewed_overlay", "review_patch"]:
            self.assertTrue(Path(reviewed[key]).exists(), key)
        reviewed_mask = read_mask(Path(reviewed["reviewed_talc_mask"]))
        self.assertEqual(int(np.count_nonzero(reviewed_mask)), 600)
        patch = json.loads(Path(reviewed["review_patch"]).read_text(encoding="utf-8"))
        self.assertEqual(patch["reviewer"], "unit-test")
        self.assertEqual(patch["original_image_path"], str((self.original_dir / self.image_name).resolve()))
        refreshed = self.store.manifest_payload()["samples"][0]
        self.assertEqual(refreshed["review_state"], "reviewed")

    def test_reset_restores_autodetected_mask(self) -> None:
        sample_id = self.store.manifest_payload()["samples"][0]["sample_id"]
        payload = self.store.sample_payload(sample_id)
        mask = np.zeros((payload["image"]["height"], payload["image"]["width"]), dtype=np.uint8)
        self.store.save_current_mask(sample_id, {"mask_png": mask_data_url(mask), "edits": []}, reviewed=False)

        reset_payload = self.store.reset_current_mask(sample_id)
        current_path = Path(reset_payload["image"]["sample_dir"]) / "current_talc_mask.png"
        final_path = Path(reset_payload["summary"]["paths"]["final_talc_mask"])
        np.testing.assert_array_equal(read_mask(current_path), read_mask(final_path))

    def test_http_manifest_and_sample_endpoints(self) -> None:
        server = TalcReviewHTTPServer(("127.0.0.1", 0), self.store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address[:2]
            with urllib.request.urlopen(f"http://{host}:{port}/api/manifest", timeout=5) as response:
                manifest = json.loads(response.read().decode("utf-8"))
            sample_id = manifest["samples"][0]["sample_id"]
            with urllib.request.urlopen(f"http://{host}:{port}/api/samples/{urllib.parse.quote(sample_id)}", timeout=5) as response:
                sample = json.loads(response.read().decode("utf-8"))
            self.assertEqual(sample["sample"]["sample_id"], sample_id)
            self.assertIn("current_mask", sample["urls"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_page_includes_persistent_theme_selector_and_dark_mode_css(self) -> None:
        markup = render_html_page()
        self.assertIn('id="themeSelect"', markup)
        self.assertIn('value="system"', markup)
        self.assertIn(':root[data-theme="dark"]', markup)
        self.assertIn("talcReviewTheme", markup)
        self.assertIn("right-click a point to delete it", markup)
        self.assertIn("viewer.addEventListener('contextmenu'", markup)
        self.assertIn("event.button === 2", markup)
        self.assertIn("const insertAt = nearestPolygonSegment(point)", markup)
        self.assertIn("Polygon point inserted.", markup)
        self.assertIn('id="protectSulfides"', markup)
        self.assertIn('id="subtractSulfidesBtn"', markup)
        self.assertIn("removeSulfidePixelsFromMask", markup)
        self.assertIn("protect_sulfides", markup)
        self.assertIn("subtract_sulfides", markup)
        self.assertIn("Current on sulfide px", markup)
        self.assertNotIn("enforceSulfideProtection('undo')", markup)


if __name__ == "__main__":
    unittest.main()
