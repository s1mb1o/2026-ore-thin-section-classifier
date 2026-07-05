from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import shutil
import subprocess
import sys
import threading
import urllib.parse
import unittest
import urllib.request
from unittest import mock
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

    def test_sample_open_recovers_unreadable_current_mask(self) -> None:
        sample_id = self.store.manifest_payload()["samples"][0]["sample_id"]
        payload = self.store.sample_payload(sample_id)
        sample_dir = Path(payload["image"]["sample_dir"])
        current_path = sample_dir / "current_talc_mask.png"
        final_path = Path(payload["summary"]["paths"]["final_talc_mask"])
        current_path.write_bytes(b"not a png")

        recovered_payload = self.store.sample_payload(sample_id)

        np.testing.assert_array_equal(read_mask(current_path), read_mask(final_path))
        state = json.loads((sample_dir / "working_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["source"], "browser_review_union_recovered")
        self.assertTrue(state["recovery_reason"].startswith("unreadable_"))
        self.assertTrue(list(sample_dir.glob("current_talc_mask.recovered.*.png")))
        self.assertEqual(
            recovered_payload["metrics"]["current_talc_pixels"],
            int(np.count_nonzero(read_mask(final_path))),
        )

    def test_concurrent_first_open_creates_one_valid_current_mask(self) -> None:
        sample_id = self.store.manifest_payload()["samples"][0]["sample_id"]
        payload = self.store.sample_payload(sample_id)
        sample_dir = Path(payload["image"]["sample_dir"])
        current_path = sample_dir / "current_talc_mask.png"
        final_path = Path(payload["summary"]["paths"]["final_talc_mask"])
        current_path.unlink()
        state_path = sample_dir / "working_state.json"
        state_path.unlink(missing_ok=True)
        self.store.refresh_samples()

        def open_sample() -> int:
            opened = self.store.sample_payload(sample_id)
            return int(opened["metrics"]["current_talc_pixels"])

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            counts = list(pool.map(lambda _: open_sample(), range(8)))

        expected = int(np.count_nonzero(read_mask(final_path)))
        self.assertEqual(counts, [expected] * 8)
        np.testing.assert_array_equal(read_mask(current_path), read_mask(final_path))

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
                "view_settings": {
                    "brightness_threshold_luma": 90,
                    "brightness_visible_pixels": 123,
                    "brightness_visible_total_pixels": 1000,
                    "brightness_visible_fraction": 0.123,
                    "brightness_threshold_formula": "luma = 0.299*R + 0.587*G + 0.114*B",
                    "background_mode": "original",
                    "background_visible": False,
                    "blank_white_visible": True,
                },
            },
            reviewed=True,
        )

        self.assertTrue(result["reviewed"])
        reviewed = result["review_summary"]["paths"]
        for key in [
            "reviewed_talc_mask",
            "reviewed_positive_bag_mask",
            "reviewed_talc_node_mask",
            "reviewed_not_talc_mask",
            "reviewed_ignore_mask",
            "reviewed_overlay",
            "review_patch",
        ]:
            self.assertTrue(Path(reviewed[key]).exists(), key)
        reviewed_mask = read_mask(Path(reviewed["reviewed_talc_mask"]))
        self.assertEqual(int(np.count_nonzero(reviewed_mask)), 600)
        self.assertEqual(int(np.count_nonzero(read_mask(Path(reviewed["reviewed_positive_bag_mask"])))), 600)
        self.assertEqual(int(np.count_nonzero(read_mask(Path(reviewed["reviewed_talc_node_mask"])))), 0)
        self.assertEqual(int(np.count_nonzero(read_mask(Path(reviewed["reviewed_not_talc_mask"])))), 0)
        self.assertEqual(result["not_talc_pixels"], 0)
        patch = json.loads(Path(reviewed["review_patch"]).read_text(encoding="utf-8"))
        self.assertEqual(patch["schema_version"], "talc-review-web-patch-v0.3")
        self.assertEqual(patch["reviewer"], "unit-test")
        self.assertEqual(patch["original_image_path"], str((self.original_dir / self.image_name).resolve()))
        self.assertIn("not_talc", patch["class_definitions"])
        self.assertIsNone(patch["model_talc_mask_path"])
        self.assertEqual(patch["human_review_masks"], [])
        self.assertEqual(patch["view_settings"]["brightness_threshold_luma"], 90)
        self.assertEqual(patch["view_settings"]["brightness_visible_pixels"], 123)
        self.assertEqual(patch["view_settings"]["brightness_visible_total_pixels"], 1000)
        self.assertAlmostEqual(patch["view_settings"]["brightness_visible_fraction"], 0.123)
        self.assertEqual(patch["view_settings"]["background_mode"], "original")
        self.assertFalse(patch["view_settings"]["background_visible"])
        self.assertTrue(patch["view_settings"]["blank_white_visible"])
        refreshed = self.store.manifest_payload()["samples"][0]
        self.assertEqual(refreshed["review_state"], "reviewed")

    def test_save_review_writes_positive_bag_and_talc_node_classes(self) -> None:
        sample_id = self.store.manifest_payload()["samples"][0]["sample_id"]
        payload = self.store.sample_payload(sample_id)
        shape = (payload["image"]["height"], payload["image"]["width"])
        positive_bag = np.zeros(shape, dtype=np.uint8)
        talc_node = np.zeros(shape, dtype=np.uint8)
        positive_bag[10:30, 12:42] = 255
        talc_node[40:50, 60:80] = 255
        union = positive_bag.copy()
        union[talc_node > 0] = 255

        result = self.store.save_current_mask(
            sample_id,
            {
                "mask_png": mask_data_url(union),
                "positive_bag_mask_png": mask_data_url(positive_bag),
                "talc_node_mask_png": mask_data_url(talc_node),
                "edits": [{"type": "similar_talc_add", "target_class": "talc_node"}],
            },
            reviewed=True,
        )

        reviewed = result["review_summary"]["paths"]
        np.testing.assert_array_equal(read_mask(Path(reviewed["reviewed_positive_bag_mask"])), positive_bag)
        np.testing.assert_array_equal(read_mask(Path(reviewed["reviewed_talc_node_mask"])), talc_node)
        np.testing.assert_array_equal(read_mask(Path(reviewed["reviewed_talc_mask"])), union)
        self.assertEqual(result["positive_bag_pixels"], 600)
        self.assertEqual(result["talc_node_pixels"], 200)
        self.assertEqual(result["current_talc_pixels"], 800)

    def test_positive_bag_and_talc_node_masks_can_overlap(self) -> None:
        sample_id = self.store.manifest_payload()["samples"][0]["sample_id"]
        payload = self.store.sample_payload(sample_id)
        shape = (payload["image"]["height"], payload["image"]["width"])
        positive_bag = np.zeros(shape, dtype=np.uint8)
        talc_node = np.zeros(shape, dtype=np.uint8)
        positive_bag[10:30, 12:42] = 255
        talc_node[14:24, 18:30] = 255
        union = positive_bag.copy()

        result = self.store.save_current_mask(
            sample_id,
            {
                "mask_png": mask_data_url(union),
                "positive_bag_mask_png": mask_data_url(positive_bag),
                "talc_node_mask_png": mask_data_url(talc_node),
                "edits": [{"type": "similar_talc_add", "target_class": "talc_node"}],
            },
            reviewed=True,
        )

        reviewed = result["review_summary"]["paths"]
        np.testing.assert_array_equal(read_mask(Path(reviewed["reviewed_positive_bag_mask"])), positive_bag)
        np.testing.assert_array_equal(read_mask(Path(reviewed["reviewed_talc_node_mask"])), talc_node)
        np.testing.assert_array_equal(read_mask(Path(reviewed["reviewed_talc_mask"])), union)
        self.assertEqual(result["positive_bag_pixels"], 600)
        self.assertEqual(result["talc_node_pixels"], 120)
        self.assertEqual(result["current_talc_pixels"], 600)

    def test_save_review_writes_not_talc_hard_negative_class(self) -> None:
        sample_id = self.store.manifest_payload()["samples"][0]["sample_id"]
        payload = self.store.sample_payload(sample_id)
        shape = (payload["image"]["height"], payload["image"]["width"])
        positive_bag = np.zeros(shape, dtype=np.uint8)
        talc_node = np.zeros(shape, dtype=np.uint8)
        not_talc = np.zeros(shape, dtype=np.uint8)
        positive_bag[10:30, 12:42] = 255
        talc_node[20:40, 20:50] = 255
        not_talc[25:35, 30:45] = 255
        expected_talc_node = talc_node.copy()
        expected_talc_node[not_talc > 0] = 0
        expected_union = positive_bag.copy()
        expected_union[expected_talc_node > 0] = 255

        result = self.store.save_current_mask(
            sample_id,
            {
                "mask_png": mask_data_url(expected_union),
                "positive_bag_mask_png": mask_data_url(positive_bag),
                "talc_node_mask_png": mask_data_url(talc_node),
                "not_talc_mask_png": mask_data_url(not_talc),
                "edits": [{"type": "brush", "target_class": "not_talc"}],
            },
            reviewed=True,
        )

        reviewed = result["review_summary"]["paths"]
        np.testing.assert_array_equal(read_mask(Path(reviewed["reviewed_not_talc_mask"])), not_talc)
        np.testing.assert_array_equal(read_mask(Path(reviewed["reviewed_talc_node_mask"])), expected_talc_node)
        np.testing.assert_array_equal(read_mask(Path(reviewed["reviewed_talc_mask"])), expected_union)
        self.assertEqual(result["not_talc_pixels"], int(np.count_nonzero(not_talc)))
        self.assertEqual(result["talc_node_pixels"], int(np.count_nonzero(expected_talc_node)))
        summary = json.loads((Path(reviewed["reviewed_talc_mask"]).parent / "review_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["schema_version"], "talc-review-web-summary-v0.3")
        self.assertEqual(summary["reviewed_not_talc_pixels"], int(np.count_nonzero(not_talc)))
        self.assertIn("reviewed_not_talc_mask", summary["paths"])

    def test_sample_payload_exposes_model_and_teammate_human_masks(self) -> None:
        sample_id = self.store.manifest_payload()["samples"][0]["sample_id"]
        payload = self.store.sample_payload(sample_id)
        sample_dir = Path(payload["image"]["sample_dir"])
        shape = (payload["image"]["height"], payload["image"]["width"])
        model_mask = np.zeros(shape, dtype=np.uint8)
        teammate_mask = np.zeros(shape, dtype=np.uint8)
        model_mask[8:18, 10:20] = 255
        teammate_mask[30:42, 50:65] = 255
        Image.fromarray(model_mask, mode="L").save(sample_dir / "model_talc_mask.png")
        teammate_dir = sample_dir / "human_reviews" / "alex"
        teammate_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(teammate_mask, mode="L").save(teammate_dir / "reviewed_talc_node_mask.png")

        refreshed = self.store.sample_payload(sample_id)

        self.assertTrue(refreshed["metrics"]["has_model_talc_mask"])
        self.assertEqual(refreshed["metrics"]["human_review_mask_count"], 1)
        self.assertIsNotNone(refreshed["urls"]["model_talc_mask"])
        self.assertEqual(len(refreshed["urls"]["human_review_masks"]), 1)
        self.assertEqual(refreshed["urls"]["human_review_masks"][0]["label"], "alex")
        self.assertIn("human_reviews/alex/reviewed_talc_node_mask.png", refreshed["urls"]["human_review_masks"][0]["path"])
        self.assertIn("neural_model_runner", refreshed)
        self.assertIn("checkpoint_exists", refreshed["neural_model_runner"])

    def test_talcose_heuristic_qa_writes_sample_artifacts(self) -> None:
        sample_id = self.store.manifest_payload()["samples"][0]["sample_id"]

        result = self.store.run_talcose_heuristic(
            sample_id,
            {
                "k_threshold": 0.85,
                "classify_threshold": 0.01,
                "proc_width": 240,
            },
        )

        self.assertEqual(result["schema_version"], "talc-review-web-talcose-heuristic-v0.1")
        self.assertEqual(result["sample_id"], sample_id)
        self.assertIn(result["result"]["ore_class"], ("talcose_ore", "not_talcose"))
        for key in ["zone_mask", "flake_mask", "overlay", "result_json"]:
            self.assertTrue(Path(result["paths"][key]).exists(), key)
            self.assertIsNotNone(result["urls"][key], key)
        refreshed = self.store.sample_payload(sample_id)
        self.assertIsNotNone(refreshed["non_neural_talcose_qa"])
        self.assertEqual(refreshed["non_neural_talcose_qa"]["schema_version"], "talc-zone-heuristic-v1")
        self.assertIsNotNone(refreshed["urls"]["talcose_heuristic_overlay"])

    def test_neural_model_run_writes_sample_model_mask(self) -> None:
        sample_id = self.store.manifest_payload()["samples"][0]["sample_id"]
        payload = self.store.sample_payload(sample_id)
        sample_dir = Path(payload["image"]["sample_dir"])
        shape = (payload["image"]["height"], payload["image"]["width"])
        fake_checkpoint = self.root / "fake_talc_checkpoint.pt"
        fake_checkpoint.write_bytes(b"fake checkpoint")
        self.store.talc_checkpoint = fake_checkpoint
        self.store.talc_tile_size = 64
        self.store.talc_stride = 48
        self.store.talc_batch_size = 1
        self.store.talc_device = "cpu"

        def fake_run(cmd, cwd, stdout, stderr, check):  # noqa: ANN001
            self.assertEqual(cwd, ROOT)
            self.assertIn(str(ROOT / "scripts/infer_talc_segmentation.py"), cmd)
            self.assertIn("--checkpoint", cmd)
            self.assertEqual(cmd[cmd.index("--checkpoint") + 1], str(fake_checkpoint.resolve()))
            self.assertIn("--threshold", cmd)
            self.assertEqual(cmd[cmd.index("--threshold") + 1], "0.33")
            if "--sulfide-mask" in cmd:
                self.assertTrue(Path(cmd[cmd.index("--sulfide-mask") + 1]).exists())
            out_dir = Path(cmd[cmd.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            mask = np.zeros(shape, dtype=np.uint8)
            mask[12:22, 30:45] = 255
            Image.fromarray(mask, mode="L").save(out_dir / "talc_mask.png")
            Image.fromarray(mask, mode="L").save(out_dir / "confidence.png")
            Image.fromarray(mask, mode="L").save(out_dir / "confidence_non_sulfide.png")
            Image.fromarray(np.zeros((*shape, 3), dtype=np.uint8), mode="RGB").save(out_dir / "overlay_preview.jpg")
            summary = {
                "schema_version": "binary-talc-inference-v0.1",
                "paths": {
                    "talc_mask": str(out_dir / "talc_mask.png"),
                    "confidence": str(out_dir / "confidence.png"),
                    "confidence_non_sulfide": str(out_dir / "confidence_non_sulfide.png"),
                    "overlay_preview": str(out_dir / "overlay_preview.jpg"),
                },
            }
            (out_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            stdout.write("fake run\n")
            return subprocess.CompletedProcess(cmd, 0)

        with mock.patch("apps.talc_review_web.subprocess.run", side_effect=fake_run):
            result = self.store.run_neural_talc_model(sample_id, {"threshold": 0.33})

        self.assertEqual(result["schema_version"], "talc-review-web-neural-model-v0.1")
        self.assertEqual(result["sample_id"], sample_id)
        self.assertEqual(result["threshold"], 0.33)
        self.assertEqual(result["model_talc_pixels"], 150)
        self.assertEqual(Path(result["paths"]["model_talc_mask"]), sample_dir / "model_talc_mask.png")
        self.assertTrue((sample_dir / "model_talc_mask.png").exists())
        self.assertTrue((sample_dir / "qa/neural_talc_model/summary.json").exists())
        self.assertIsNotNone(result["urls"]["model_talc_mask"])
        self.assertIsNotNone(result["urls"]["overlay_preview"])
        refreshed = self.store.sample_payload(sample_id)
        self.assertTrue(refreshed["metrics"]["has_model_talc_mask"])
        self.assertIsNotNone(refreshed["urls"]["model_talc_mask"])

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
            with urllib.request.urlopen(f"http://{host}:{port}/sample/{urllib.parse.quote(sample_id)}", timeout=5) as response:
                html = response.read().decode("utf-8")
            self.assertIn("Talc samples", html)
            self.assertIn("requestedSampleFromLocation", html)
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
        self.assertIn("Select the Edit radio in Segmentation classes", markup)
        self.assertIn("Brush: left mouse adds the selected class, right mouse erases it.", markup)
        self.assertIn("Brush (B): left mouse draws the selected class, right mouse erases it", markup)
        self.assertIn('<input id="brushSize" type="range" min="2" max="240" value="28"', markup)
        self.assertIn("Fill (F): click an area bounded by blue lines, sulfides, existing selected-class regions, or image edges", markup)
        self.assertIn("Similar: add positive talc seeds and negative non-talc seeds to preview luma/color/texture-similar talc pixels", markup)
        self.assertIn("Fill: click an empty area bounded by blue lines, sulfides, existing selected-class regions, or the image edge.", markup)
        self.assertIn("Similar: add + seeds for confirmed talc and - seeds for dark non-talc", markup)
        self.assertIn("function sampleUrlSlug(sample)", markup)
        self.assertIn(".replace(/[хХ×]/g, 'x')", markup)
        self.assertIn("function requestedSampleSlugFromLocation()", markup)
        self.assertIn("function requestedSampleFromLocation()", markup)
        self.assertIn("const path = `/sample/${encodeURIComponent(sampleUrlSlug(sample))}`;", markup)
        self.assertIn("window.history.replaceState(statePayload, '', path)", markup)
        self.assertIn("window.addEventListener('popstate'", markup)
        self.assertIn("const initialSample = requestedSampleFromLocation() || state.samples[0];", markup)
        self.assertIn("if (options.updateUrl !== false) updateSampleUrl(sampleId);", markup)
        self.assertIn('aria-keyshortcuts="B"', markup)
        self.assertIn('aria-keyshortcuts="F"', markup)
        self.assertLess(markup.index('data-tool="brush"'), markup.index('data-tool="fill"'))
        self.assertLess(markup.index('data-tool="fill"'), markup.index('data-tool="similar"'))
        self.assertLess(markup.index('data-tool="similar"'), markup.index('data-tool="rectangle"'))
        self.assertLess(markup.index('data-tool="rectangle"'), markup.index('data-tool="polygon"'))
        self.assertLess(markup.index('data-tool="polygon"'), markup.index('data-tool="sam2"'))
        for tool, label in (
            ("brush", "Brush"),
            ("fill", "Fill"),
            ("similar", "Similar"),
            ("rectangle", "Rectangle"),
            ("polygon", "Polygon"),
        ):
            tool_button = markup.split(f'data-tool="{tool}"', 1)[1].split("</button>", 1)[0]
            self.assertIn('class="tool-button icon-tool', tool_button)
            self.assertIn(f'aria-label="{label}"', tool_button)
            self.assertIn("<svg ", tool_button)
            self.assertIn(f'<span class="visually-hidden">{label}</span>', tool_button)
            self.assertIn("data-tooltip=", tool_button)
        self.assertIn(".icon-tool { width: 34px; height: 34px; min-width: 34px; padding: 0; display: inline-flex;", markup)
        self.assertIn(".icon-tool svg { width: 18px; height: 18px;", markup)
        similar_button = markup.split('data-tool="similar"', 1)[1].split("</button>", 1)[0]
        self.assertIn('class="magic-wand-icon"', similar_button)
        self.assertIn('d="M4 20 14.5 9.5"', similar_button)
        self.assertIn('id="toolTooltip"', markup)
        self.assertIn('class="tool-tooltip hidden"', markup)
        self.assertIn('role="tooltip"', markup)
        self.assertIn(".tool-tooltip {\n  position: fixed;", markup)
        self.assertIn("function showToolTooltip(target)", markup)
        self.assertIn("function hideToolTooltip()", markup)
        self.assertIn("function findToolTooltipTarget(node)", markup)
        self.assertIn("function handleToolTooltipOver(event)", markup)
        self.assertIn("document.addEventListener('pointerover', handleToolTooltipOver);", markup)
        self.assertIn("document.addEventListener('pointermove', handleToolTooltipOver);", markup)
        self.assertIn("document.addEventListener('mouseover', handleToolTooltipOver);", markup)
        self.assertIn("document.addEventListener('mousemove', handleToolTooltipOver);", markup)
        self.assertIn("document.addEventListener('focusin', handleToolTooltipOver);", markup)
        self.assertIn("document.addEventListener('pointerout', handleToolTooltipOut);", markup)
        self.assertIn("document.addEventListener('mouseout', handleToolTooltipOut);", markup)
        self.assertIn("document.addEventListener('focusout', handleToolTooltipOut);", markup)
        self.assertIn("document.addEventListener('scroll', hideToolTooltip, true);", markup)
        self.assertIn("const label = button.getAttribute('aria-label') || button.textContent.trim();", markup)
        self.assertIn('class="toolbar-separator"', markup)
        self.assertNotIn('id="zoomInBtn"', markup)
        self.assertNotIn('id="zoomOutBtn"', markup)
        self.assertNotIn('id="fitBtn"', markup)
        self.assertNotIn('id="zoomValue"', markup)
        self.assertNotIn("document.getElementById('zoomInBtn')", markup)
        self.assertNotIn("document.getElementById('zoomOutBtn')", markup)
        self.assertNotIn("document.getElementById('fitBtn')", markup)
        self.assertNotIn("document.getElementById('zoomValue')", markup)
        self.assertIn('id="zoomWidget"', markup)
        self.assertIn('class="zoom-widget"', markup)
        self.assertIn('id="zoomFitWidgetBtn"', markup)
        self.assertIn('id="zoomActualWidgetBtn"', markup)
        self.assertIn('id="zoomInWidgetBtn"', markup)
        self.assertIn('id="zoomWidgetValue"', markup)
        self.assertIn('id="zoomOutWidgetBtn"', markup)
        self.assertIn("position: fixed", markup)
        self.assertIn("left: var(--zoom-widget-left, 12px)", markup)
        self.assertIn("bottom: var(--zoom-widget-bottom, 12px)", markup)
        self.assertIn("min-width: 56px", markup)
        self.assertIn("function updateZoomWidgetPosition()", markup)
        self.assertIn("els.zoomWidget.style.setProperty('--zoom-widget-left'", markup)
        self.assertIn("Math.round(rect.left + 12)", markup)
        self.assertIn(".zoom-widget-row, .zoom-widget-main { display: grid; grid-template-columns: 32px", markup)
        self.assertIn(".zoom-widget button {\n  width: 32px;\n  height: 32px;", markup)
        self.assertIn(".zoom-widget .zoom-level {\n  min-width: 32px;", markup)
        self.assertIn("font-size: 11px", markup)
        self.assertIn('function actualSizeView()', markup)
        self.assertIn("els.zoomWidgetValue.textContent = zoomText", markup)
        self.assertIn("els.zoomActualWidgetBtn.addEventListener('click', actualSizeView)", markup)
        self.assertIn('class="viewer-options-row"', markup)
        self.assertIn('class="viewer-options-hints"', markup)
        self.assertLess(markup.index('id="viewerWrap"'), markup.index('class="viewer-options-row"'))
        self.assertLess(markup.index('class="viewer-options-row"'), markup.index('id="statusLine"'))
        viewer_options = markup.split('class="viewer-options-row"', 1)[1].split('id="statusLine"', 1)[0]
        self.assertIn("Mouse wheel - zoom in / out", viewer_options)
        self.assertIn("Mouse wheel press - pan", viewer_options)
        self.assertIn('<rect x="7" y="3" width="10" height="18" rx="5"></rect>', viewer_options)
        self.assertIn('class="review-actions"', markup)
        self.assertIn(".topbar { flex: 0 0 auto;", markup)
        self.assertIn(".topbar-controls { grid-column: 2; min-width: 0; display: flex;", markup)
        self.assertIn("flex-wrap: wrap", markup)
        self.assertIn(".toolbar { flex: 1 1 520px;", markup)
        self.assertIn(".review-actions { flex: 0 0 auto;", markup)
        self.assertIn(".topbar-title, .topbar-controls { grid-column: 1; }", markup)
        self.assertNotIn(".topbar-controls { display: contents; }", markup)
        self.assertIn('id="saveBtn"', markup)
        self.assertIn('Save &amp; Next', markup)
        self.assertIn('id="nextBtn"', markup)
        self.assertIn('id="downloadViewBtn"', markup)
        self.assertIn('class="plain-button icon-tool download-icon-button"', markup)
        self.assertIn('aria-label="Download"', markup)
        self.assertIn('<span class="visually-hidden">Download</span>', markup)
        self.assertIn('<path d="M12 3v12"></path>', markup)
        self.assertIn(".review-actions .download-icon-button { width: 34px;", markup)
        self.assertIn("Download current image with enabled classes and layers", markup)
        self.assertIn('class="plain-button"', markup)
        self.assertIn("async function goToNextSample()", markup)
        self.assertIn("els.nextBtn.addEventListener('click'", markup)
        self.assertIn("els.downloadViewBtn.addEventListener('click', downloadCurrentImageWithLayers)", markup)
        self.assertIn("function drawEnabledClassesAndLayers(targetCtx)", markup)
        self.assertIn("drawComparisonOverlay(targetCtx)", markup)
        self.assertIn("drawClusterOverlay(targetCtx)", markup)
        self.assertIn("function downloadCurrentImageWithLayers()", markup)
        self.assertIn("exportCanvas.width = state.imageW", markup)
        self.assertIn("exportCanvas.height = state.imageH", markup)
        self.assertIn("drawEnabledClassesAndLayers(exportCtx)", markup)
        self.assertIn("_enabled_layers.png", markup)
        self.assertIn("link.download = downloadName", markup)
        self.assertIn(".plain-button { background: transparent;", markup)
        self.assertLess(markup.index('class="review-actions"'), markup.index('id="viewerWrap"'))
        self.assertLess(markup.index('id="saveBtn"'), markup.index('id="viewerWrap"'))
        self.assertLess(markup.index('id="saveNextBtn"'), markup.index('id="viewerWrap"'))
        self.assertLess(markup.index('id="saveNextBtn"'), markup.index('id="nextBtn"'))
        self.assertLess(markup.index('id="nextBtn"'), markup.index('id="viewerWrap"'))
        self.assertLess(markup.index('id="nextBtn"'), markup.index('id="downloadViewBtn"'))
        self.assertLess(markup.index('id="downloadViewBtn"'), markup.index('id="viewerWrap"'))
        export_function = markup.split("function downloadCurrentImageWithLayers()", 1)[1].split("function describeUnavailableBackground()", 1)[0]
        self.assertNotIn("drawBrushCursor", export_function)
        self.assertNotIn("drawShapeGuides", export_function)
        self.assertLess(markup.index('id="notesInput"'), markup.index('id="resetBtn"'))
        self.assertNotIn("Save and next", markup)
        self.assertIn('value="sulfide"', markup)
        self.assertIn("Sulfide mask (sulfide/non-sulfide mask segmentation)", markup)
        self.assertIn("Mask-only background", markup)
        self.assertIn('id="viewerTopWidgets"', markup)
        self.assertIn('class="viewer-top-widgets"', markup)
        self.assertIn("position: fixed", markup)
        self.assertIn("top: var(--viewer-top-widgets-top, 10px)", markup)
        self.assertIn("left: var(--viewer-top-widgets-left, 10px)", markup)
        self.assertIn("width: var(--viewer-top-widgets-width, 0px)", markup)
        self.assertIn("visibility: var(--viewer-top-widgets-visibility, hidden)", markup)
        self.assertIn("align-content: flex-start", markup)
        self.assertIn("flex-wrap: wrap", markup)
        self.assertIn("max-width: min(286px, 100%)", markup)
        self.assertIn("min-width: 0", markup)
        self.assertIn("function updateViewerTopWidgetsPosition()", markup)
        self.assertIn("function updateViewerOverlayPositions()", markup)
        self.assertIn("els.viewerTopWidgets.style.setProperty('--viewer-top-widgets-left'", markup)
        self.assertIn("els.viewerTopWidgets.style.setProperty('--viewer-top-widgets-visibility', 'visible')", markup)
        self.assertIn("const workPane = wrap.closest('.work-pane');", markup)
        self.assertIn("const workRect = workPane ? workPane.getBoundingClientRect() : rect;", markup)
        self.assertIn("const viewportPadding = 8;", markup)
        self.assertIn("const viewerInset = 10;", markup)
        self.assertIn("Math.max(rect.left, workRect.left) + viewerInset", markup)
        self.assertIn("Math.min(rect.right, workRect.right) - viewerInset", markup)
        self.assertIn("window.innerWidth - viewportPadding", markup)
        self.assertIn("Math.max(0, visibleRight - visibleLeft)", markup)
        self.assertIn("requestAnimationFrame(updateViewerOverlayPositions)", markup)
        self.assertIn('class="segmentation-class-widget"', markup)
        self.assertIn('aria-label="Visible segmentation classes"', markup)
        self.assertIn("Segmentation classes", markup)
        self.assertIn("Show", markup)
        self.assertIn("Edit", markup)
        self.assertIn('class="viewer-layer-widget"', markup)
        self.assertIn('aria-label="Display layers"', markup)
        self.assertIn("Display layers", markup)
        self.assertIn('class="viewer-layer-row"', markup)
        self.assertIn('id="layerBackground"', markup)
        self.assertIn('aria-label="Show background image"', markup)
        self.assertIn("Background", markup)
        self.assertIn('id="layerBlankWhite"', markup)
        self.assertIn('aria-label="Show blank white background"', markup)
        self.assertIn("Blank White", markup)
        self.assertIn('id="layerLines"', markup)
        self.assertIn('aria-label="Show Original blue lines"', markup)
        self.assertIn('class="class-swatch blue-lines"', markup)
        self.assertIn("Original blue lines", markup)
        self.assertIn(".class-swatch.blue-lines { background: #2563eb; }", markup)
        self.assertIn('id="layerSulfides"', markup)
        self.assertIn('aria-label="Show Sulfides"', markup)
        self.assertIn('class="class-swatch sulfides"', markup)
        self.assertIn("Sulfides", markup)
        self.assertIn(".class-swatch.sulfides { background: #f97316; }", markup)
        self.assertLess(markup.index('class="segmentation-class-widget"'), markup.index('class="viewer-layer-widget"'))
        self.assertLess(markup.index('class="viewer-layer-widget"'), markup.index('id="viewerCanvas"'))
        self.assertLess(markup.index('class="viewer-layer-widget"'), markup.index('id="layerBackground"'))
        self.assertLess(markup.index('id="layerBackground"'), markup.index('id="layerBlankWhite"'))
        self.assertLess(markup.index('id="layerBlankWhite"'), markup.index('id="layerLines"'))
        self.assertLess(markup.index('class="viewer-layer-widget"'), markup.index('id="layerLines"'))
        self.assertLess(markup.index('class="viewer-layer-widget"'), markup.index('id="layerClusterAreas"'))
        self.assertLess(markup.index('class="viewer-layer-widget"'), markup.index('id="layerSulfides"'))
        segmentation_widget = markup.split('class="segmentation-class-widget"', 1)[1].split('class="viewer-layer-widget"', 1)[0]
        self.assertNotIn('id="layerBackground"', segmentation_widget)
        self.assertNotIn('id="layerBlankWhite"', segmentation_widget)
        self.assertNotIn('id="layerLines"', segmentation_widget)
        self.assertNotIn('id="layerClusterAreas"', segmentation_widget)
        self.assertNotIn('id="layerSulfides"', segmentation_widget)
        self.assertNotIn("Original blue lines", segmentation_widget)
        self.assertNotIn("Talc cluster areas", segmentation_widget)
        self.assertNotIn("Sulfides", segmentation_widget)
        side_layers = markup.split('<div class="layers">', 1)[1].split('<div class="guard-controls">', 1)[0]
        self.assertNotIn('id="layerBlankWhite"', side_layers)
        self.assertNotIn('id="layerLines"', side_layers)
        self.assertNotIn("Original blue lines", side_layers)
        self.assertIn('id="positiveBagPct"', markup)
        self.assertIn('id="talcNodePct"', markup)
        self.assertIn('id="notTalcPct"', markup)
        self.assertIn('id="layerClusterAreas"', markup)
        self.assertIn('id="clusterAreaPct"', markup)
        self.assertIn("lines: document.getElementById('layerLines')", markup)
        self.assertIn("Original blue lines layer is not available.", markup)
        self.assertIn("sulfides: document.getElementById('layerSulfides')", markup)
        self.assertIn("state.staticTints.sulfide", markup)
        self.assertIn("Sulfides layer is not available.", markup)
        self.assertIn("sulfide: buildTintFromImage(sulfideMask, [249, 115, 22, 125])", markup)
        self.assertIn('id="runNeuralModelBtn"', markup)
        self.assertIn('id="neuralTalcThreshold"', markup)
        self.assertIn("ML talc probability threshold", markup)
        self.assertIn(">Run model</button>", markup)
        self.assertIn("async function runNeuralModelQa()", markup)
        self.assertIn("/neural-model", markup)
        self.assertIn("currentNeuralTalcThreshold()", markup)
        self.assertIn("syncNeuralTalcThresholdFromSample()", markup)
        self.assertIn("runNeuralModelQa().catch", markup)
        self.assertIn("Neural model mask is not available for this sample. Run model to generate it.", markup)
        self.assertIn('id="talcThresholdStatus"', markup)
        self.assertIn("Target talc >= 10% visible px", markup)
        self.assertIn("TALC_VISIBLE_THRESHOLD_FRACTION = 0.10", markup)
        self.assertIn("function updateSegmentationClassWidgetMetrics", markup)
        self.assertIn("under 10% by", markup)
        self.assertIn("target >=10% met", markup)
        self.assertIn('name="editTargetClass"', markup)
        self.assertIn('id="editTargetPositiveBag"', markup)
        self.assertIn('id="editTargetTalcNode"', markup)
        self.assertIn('id="editTargetNotTalc"', markup)
        self.assertIn('value="positive_bag"', markup)
        self.assertIn('value="talc_node"', markup)
        self.assertIn('value="not_talc"', markup)
        self.assertIn('class="class-swatch positive-bag"', markup)
        self.assertIn('class="class-swatch talc"', markup)
        self.assertIn('class="class-swatch not-talc"', markup)
        self.assertIn('class="class-swatch cluster"', markup)
        self.assertIn("Positive bag", markup)
        self.assertIn("Talc", markup)
        self.assertIn("Not Talc", markup)
        self.assertIn("Talc cluster areas", markup)
        self.assertIn('id="brightnessThreshold"', markup)
        self.assertIn('id="brightnessThresholdValue"', markup)
        self.assertIn('id="brightnessVisibleValue"', markup)
        self.assertIn('id="brightnessThreshold90Btn"', markup)
        self.assertIn('id="brightnessThresholdOffBtn"', markup)
        self.assertIn("Dark pixel preview threshold", markup)
        self.assertIn("Visible pixels:", markup)
        self.assertIn("Luma = 0.299 R + 0.587 G + 0.114 B", markup)
        self.assertIn("BRIGHTNESS_THRESHOLD_STORAGE_KEY", markup)
        self.assertIn("BRIGHTNESS_THRESHOLD_FORMULA", markup)
        self.assertIn("function brightnessFilteredBackground(base)", markup)
        self.assertIn("function setBrightnessVisibleStats", markup)
        self.assertIn("function brightnessVisibleStatsPayload()", markup)
        self.assertIn("brightness_visible_pixels", markup)
        self.assertIn("brightness_visible_fraction", markup)
        self.assertIn("const luma = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]", markup)
        self.assertIn("if (threshold <= 0)", markup)
        self.assertIn("if (luma <= threshold)", markup)
        self.assertIn('id="clusterOverlayToggle"', markup)
        self.assertIn("Show talc cluster areas", markup)
        self.assertIn('id="clusterSource"', markup)
        self.assertIn('value="talc_node"', markup)
        self.assertIn('value="union"', markup)
        self.assertIn('id="clusterRadius"', markup)
        self.assertIn('id="clusterDensity"', markup)
        self.assertIn('id="clusterOpacity"', markup)
        self.assertIn('id="clusterStats"', markup)
        self.assertIn("CLUSTER_OVERLAY_STORAGE_KEY", markup)
        self.assertIn("function updateClusterLayerWidget", markup)
        self.assertIn("els.clusterLayerToggle.addEventListener('change'", markup)
        self.assertIn("function clusterOverlayCanvasForCurrentSettings()", markup)
        self.assertIn("function drawClusterOverlay(targetCtx = ctx)", markup)
        self.assertIn("drawClusterOverlay(targetCtx);", markup)
        self.assertIn("new Uint32Array((width + 1) * (height + 1))", markup)
        self.assertIn("min_density_percent", markup)
        self.assertIn("sulfide_excluded_pixels", markup)
        self.assertIn("non_sulfide_pixels", markup)
        self.assertIn("highlighted_pixels", markup)
        self.assertIn("talc_cluster_overlay", markup)
        self.assertIn("clusterOverlayRebuildDeferred", markup)
        self.assertIn("Highlighted ${formatInt(stats.highlightedPixels)} non-sulfide px", markup)
        self.assertIn("sulfideData && isMaskDataActive(sulfideData, pixel, 0)", markup)
        self.assertIn("sulfideExcludedPixels += 1", markup)
        self.assertIn("view_settings: viewSettingsPayload()", markup)
        self.assertIn("background_visible", markup)
        self.assertIn("blank_white_visible", markup)
        self.assertIn("const showBackground = !els.layers.background || els.layers.background.checked", markup)
        self.assertIn("const showBlankWhite = !showBackground && Boolean(els.layers.blankWhite && els.layers.blankWhite.checked);", markup)
        self.assertIn("if (!showBackground)", markup)
        self.assertIn("targetCtx.fillStyle = '#ffffff';", markup)
        self.assertIn("targetCtx.fillRect(0, 0, state.imageW, state.imageH);", markup)
        self.assertIn("if (els.layers.background && !els.layers.background.checked) return null;", markup)
        self.assertIn('id="modelQaStats"', markup)
        self.assertIn('id="comparisonModeSelect"', markup)
        self.assertIn('id="currentComparisonControls"', markup)
        self.assertIn('id="heuristicComparisonControls"', markup)
        self.assertIn('id="neuralComparisonControls"', markup)
        self.assertIn('id="heuristicLayerLegend"', markup)
        self.assertIn('id="heuristicComparisonLegend"', markup)
        self.assertIn('id="heuristicNeuralComparisonLegend"', markup)
        self.assertIn('id="neuralLayerLegend"', markup)
        self.assertIn('id="neuralComparisonLegend"', markup)
        self.assertIn('id="runTalcoseHeuristicBtn"', markup)
        self.assertIn('id="heuristicQaStats"', markup)
        self.assertIn("Comparison mode", markup)
        self.assertIn('<option value="current">Current</option>', markup)
        self.assertIn('<option value="heuristic">Heuristic</option>', markup)
        self.assertIn('<option value="neural_model">Neural Model</option>', markup)
        self.assertIn('<option value="current_vs_heuristic">Current vs Heuristic</option>', markup)
        self.assertIn('<option value="current_vs_neural">Current vs Neural Model</option>', markup)
        self.assertIn('<option value="heuristic_vs_neural">Heuristic vs Neural Model</option>', markup)
        self.assertNotIn("Default annotation", markup)
        self.assertNotIn("Current Talc annotation vs Heuristic", markup)
        self.assertNotIn("Current Talc annotation vs Neural Model", markup)
        self.assertIn("Run non-neural classifier", markup)
        self.assertIn("function selectedComparisonMode()", markup)
        self.assertIn("function modelStandaloneEnabled()", markup)
        self.assertIn("function heuristicStandaloneEnabled()", markup)
        self.assertIn("function heuristicNeuralComparisonEnabled()", markup)
        self.assertIn("function heuristicComparisonCanvasForCurrentState()", markup)
        self.assertIn("function heuristicNeuralComparisonCanvasForCurrentState()", markup)
        self.assertIn("function modelStandaloneCanvasForCurrentState()", markup)
        self.assertIn("function heuristicStandaloneCanvasForCurrentState()", markup)
        self.assertIn("function drawComparisonOverlay(targetCtx = ctx)", markup)
        self.assertIn("function drawModelStandaloneOverlay(targetCtx = ctx)", markup)
        self.assertIn("function drawHeuristicStandaloneOverlay(targetCtx = ctx)", markup)
        self.assertIn("function drawHeuristicNeuralComparisonOverlay(targetCtx = ctx)", markup)
        self.assertIn("function runTalcoseHeuristicQa()", markup)
        self.assertIn("talcose-heuristic", markup)
        self.assertIn("heuristic-vs-neural", markup)
        self.assertIn("heuristic_vs_neural_enabled", markup)
        self.assertIn("heuristic_vs_neural_stats", markup)
        self.assertIn(".qa-swatch.qa-heuristic-only { background: #ec4899; }", markup)
        self.assertIn("const HEURISTIC_QA_RGB = [236, 72, 153];", markup)
        self.assertIn("const HEURISTIC_QA_ALPHA = 150;", markup)
        self.assertNotIn(".qa-swatch.qa-heuristic-only { background: #f97316; }", markup)
        self.assertIn("neural only", markup)
        self.assertIn("heuristic only", markup)
        self.assertIn("current only", markup)
        self.assertIn("agreement", markup)
        self.assertIn("sulfide conflict", markup)
        self.assertIn("function modelHumanQaCanvasForCurrentState()", markup)
        self.assertIn("function drawHeuristicComparisonOverlay(targetCtx = ctx)", markup)
        self.assertIn("const humanData = captureTalcNodeData().data;", markup)
        self.assertIn("const currentData = captureTalcNodeData().data;", markup)
        self.assertIn("model_talc_mask", markup)
        self.assertIn("talcose_heuristic_zone_mask", markup)
        self.assertIn("human_review_masks", markup)
        self.assertIn('id="assetWarnings"', markup)
        self.assertIn("baseMode === 'sulfide'", markup)
        self.assertIn("state.images.sulfideMask", markup)
        self.assertIn("state.images = { original, annotated, qa, sulfideMask, modelMask, heuristicZoneMask", markup)
        self.assertIn("function prepareFillBoundaries(rawLines, closedLines)", markup)
        self.assertIn("function fillAtPoint(point)", markup)
        self.assertIn("state.fillBoundaryLoaded", markup)
        self.assertIn("const sulfideBoundaryData = hasSulfideGuard()", markup)
        self.assertIn("boundaryLabels.push('sulfide_pixels')", markup)
        self.assertIn("isMaskDataActive(sulfideBoundaryData, pixel, 0)", markup)
        self.assertIn("fillAtPoint(point).catch", markup)
        self.assertIn("closed_blue_stroke", markup)
        self.assertIn("boundaries: boundaryLabels", markup)
        self.assertIn('id="brushParams"', markup)
        self.assertIn('id="similarParams"', markup)
        self.assertIn('id="similarStrictness"', markup)
        self.assertIn('id="similarStrictnessValue"', markup)
        self.assertIn('id="similarApplyBtn"', markup)
        self.assertIn('id="similarClearBtn"', markup)
        self.assertIn('id="sam2Params"', markup)
        self.assertIn('id="sam2ApplyBtn"', markup)
        self.assertIn("Load SAM2", markup)
        self.assertIn('aria-label="Tool parameters"', markup)
        self.assertIn('aria-pressed="true"', markup)
        self.assertIn("function selectTool(tool, options = {})", markup)
        self.assertIn("other.setAttribute('aria-pressed', 'false')", markup)
        self.assertIn("function updateToolParams()", markup)
        self.assertIn("els.similarParams.classList.toggle('hidden', state.tool !== 'similar')", markup)
        self.assertIn("viewer.addEventListener('wheel'", markup)
        self.assertIn("zoomBy(event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP, event)", markup)
        self.assertIn("viewPan:", markup)
        self.assertIn("function isMiddleButtonEvent(event)", markup)
        self.assertIn("event.button === 1 || (typeof event.buttons === 'number' && (event.buttons & 4) === 4)", markup)
        self.assertIn("function isMiddleButtonHeld(event)", markup)
        self.assertIn("--pan-gutter-x", markup)
        self.assertIn("const PAN_GUTTER_MIN_PX = 240", markup)
        self.assertIn("function updatePanGutter(options = {})", markup)
        self.assertIn("function resetViewPanOrigin()", markup)
        self.assertIn("function startViewPan(event)", markup)
        self.assertIn("function updateViewPan(event)", markup)
        self.assertIn("function finishViewPan(event = null)", markup)
        self.assertIn("if (isMiddleButtonEvent(event))", markup)
        self.assertIn("viewer.addEventListener('mousedown'", markup)
        self.assertIn("document.addEventListener('mousemove'", markup)
        self.assertIn("document.addEventListener('mouseup'", markup)
        self.assertIn("wrap.scrollLeft = state.panGutter.x", markup)
        self.assertIn("state.panGutter.x + anchorImagePoint.x * state.zoom - anchorOffset.x", markup)
        self.assertIn("window.addEventListener('resize'", markup)
        self.assertIn("wrap.scrollLeft = state.viewPan.scrollLeft - dx", markup)
        self.assertIn("viewer.addEventListener('auxclick'", markup)
        self.assertIn("viewer.addEventListener('pointercancel'", markup)
        self.assertNotIn('id="zoomSlider"', markup)
        self.assertNotIn('class="canvas-controls"', markup)
        self.assertIn("hoverPoint", markup)
        self.assertIn("talcNodeCanvas", markup)
        self.assertIn("baseTalcNodeCanvas", markup)
        self.assertIn("talcNodeTintCanvas", markup)
        self.assertIn("function activeEditClass()", markup)
        self.assertIn("function editClassContexts(targetClass = activeEditClass())", markup)
        self.assertIn("function setEditClass(targetClass, options = {})", markup)
        self.assertIn("els.editTargets.forEach", markup)
        self.assertIn("drawMaskLine(point, point, strokeMode, target.baseCtx)", markup)
        self.assertIn("target_class: targetClass", markup)
        self.assertIn("boundaryLabels.push(`current_${targetClass}_regions`)", markup)
        self.assertIn("function classEditType(targetClass, baseType)", markup)
        self.assertIn("return `${baseType}_not_talc`", markup)
        self.assertIn("rasterizeShape(editClassContexts(shape.targetClass).ctx, shape)", markup)
        self.assertIn("function combinedMaskCanvas()", markup)
        self.assertIn("positive_bag_mask_png", markup)
        self.assertIn("talc_node_mask_png", markup)
        self.assertIn("not_talc_mask_png", markup)
        self.assertIn("current_not_talc_mask", markup)
        self.assertIn("target_class: 'positive_bag'", markup)
        self.assertIn("target_class: 'talc_node'", markup)
        self.assertIn("target_class: 'not_talc'", markup)
        self.assertIn('id="layerTalcNode"', markup)
        self.assertIn('id="layerNotTalc"', markup)
        self.assertIn("current_positive_bag_mask", markup)
        self.assertIn("current_talc_node_mask", markup)
        self.assertIn("similarTalcPreview", markup)
        self.assertIn("MAX_SIMILAR_TALC_REGION_FRACTION", markup)
        self.assertIn("function computeSimilarTalcPreview(point = null)", markup)
        self.assertIn("function applySimilarTalcPreview(options = {})", markup)
        self.assertIn("function cleanupSimilarTalcCandidates(candidate, width, height)", markup)
        self.assertIn("function collectSeedPatchSamples(seedX, seedY, sourceData, sulfideData)", markup)
        self.assertIn("function collectSimilarSamplesFromSeeds(points, sourceData, currentData, sulfideData)", markup)
        self.assertIn("function collectNegativeSeedSamples(points, sourceData, sulfideData)", markup)
        self.assertIn("function collectNotTalcMaskSamples(sourceData, notTalcData, sulfideData)", markup)
        self.assertIn("function localTextureAtPixel(sourceData, pixelIndex)", markup)
        self.assertIn("function similarFeatureDistanceToStats(item, stats)", markup)
        self.assertIn("function drawSimilarTalcPreview()", markup)
        self.assertIn("drawSimilarTalcPreview();", markup)
        self.assertIn('id="similarPositiveSeedBtn"', markup)
        self.assertIn('id="similarNegativeSeedBtn"', markup)
        self.assertIn("setSimilarSeedMode('positive')", markup)
        self.assertIn("negativeSeeds", markup)
        self.assertIn("source_tool: 'similar_talc'", markup)
        self.assertIn("type: 'similar_talc_add'", markup)
        self.assertIn("overlapping_positive_bag_pixels", markup)
        self.assertIn("excluded_existing_talc_pixels", markup)
        self.assertIn("source_kind: sourceKind", markup)
        self.assertIn("source_kind: preview.stats ? preview.stats.source_kind : null", markup)
        self.assertIn("positive_bag_kept: positiveBagKept", markup)
        self.assertIn("seed patch + filtered positive bag", markup)
        self.assertIn("negative_seed_count", markup)
        self.assertIn("not_talc_negative_samples", markup)
        self.assertIn("texture_tolerance", markup)
        self.assertIn("excluded_not_talc_pixels", markup)
        self.assertIn("excluded_negative_seed_pixels", markup)
        self.assertIn("const strictnessLooseness = (100 - strictness) / 99", markup)
        self.assertIn("const sulfideData = hasSulfideGuard()", markup)
        self.assertIn("if (sulfideData && isMaskDataActive(sulfideData, pixel, 0))", markup)
        self.assertIn("state.tool === 'similar'", markup)
        self.assertIn("clearSimilarTalcPreview", markup)
        self.assertIn("similar_talc_strictness", markup)
        self.assertIn("Apply the visible Similar preview to the talc-node class.", markup)
        self.assertIn("Press Apply Similar or Save to add talc nodes.", markup)
        self.assertIn("await applySimilarTalcPreview({ autosave: false })", markup)
        self.assertIn("Applying Similar preview before save", markup)
        save_review_index = markup.index("async function saveReview")
        self.assertLess(
            markup.index("await applySimilarTalcPreview({ autosave: false })", save_review_index),
            markup.index("mask_png: combined.toDataURL('image/png')", save_review_index),
        )
        self.assertIn("function drawBrushCursor()", markup)
        self.assertIn("drawBrushCursor();", markup)
        self.assertIn("function updateViewerCursor(point = null)", markup)
        self.assertIn("function canvasCursorForPoint(point = null)", markup)
        self.assertIn("viewer.addEventListener('pointerleave'", markup)
        self.assertIn("function strokeModeForPointer(event)", markup)
        self.assertIn("if (event.button === 2) return 'eraser';", markup)
        self.assertIn("source_tool: state.tool", markup)
        self.assertNotIn('data-tool="eraser"', markup)
        self.assertNotIn("Eraser removes talc.", markup)
        self.assertNotIn('id="applyPolygonBtn"', markup)
        self.assertNotIn('id="cancelPolygonBtn"', markup)
        self.assertNotIn('id="applyRectBtn"', markup)
        self.assertNotIn('id="cancelRectBtn"', markup)
        self.assertIn("click the first point to close", markup)
        self.assertIn("right-click a polygon point to remove it", markup)
        self.assertIn("right-click elsewhere to cancel the current polygon", markup)
        self.assertIn("right-click cancels the current rectangle", markup)
        self.assertIn("Press Delete to remove the selected completed polygon or rectangle.", markup)
        self.assertIn("SAM2 point: hover without moving to preview, then press Apply SAM2.", markup)
        self.assertIn("SAM2 box applies after drawing the box.", markup)
        self.assertIn("Shapes stay editable until another image is opened or the sample is saved.", markup)
        self.assertIn("baseMaskCanvas", markup)
        self.assertIn("function addPolygonShape(points)", markup)
        self.assertIn("function addRectangleShape(rect)", markup)
        self.assertIn("function finishShapeDrag()", markup)
        self.assertIn("function removePolygonShapePoint(hit)", markup)
        self.assertIn("function deleteSelectedShape()", markup)
        self.assertIn("function isTextEditingTarget(target)", markup)
        self.assertIn("const shortcutAllowed = !isTextEditingTarget(event.target) && !event.metaKey && !event.ctrlKey && !event.altKey", markup)
        self.assertIn("key === 'b' || key === 'f'", markup)
        self.assertIn("const tool = key === 'b' ? 'brush' : 'fill'", markup)
        self.assertIn("selectTool(tool, { shortcut: key.toUpperCase() })", markup)
        self.assertIn("polygon_point_remove", markup)
        self.assertIn("polygon_shape_delete", markup)
        self.assertIn("rectangle_shape_delete", markup)
        self.assertIn("shapeById(state.activeShapeId)", markup)
        self.assertIn("event.key === 'Delete' || event.key === 'Backspace'", markup)
        self.assertIn("!isTextEditingTarget(event.target)", markup)
        self.assertIn("MAX_SAM2_REGION_FRACTION", markup)
        self.assertIn("SAM2_POINT_HOVER_PREVIEW_DELAY_MS = 2000", markup)
        self.assertIn("sam2Preview", markup)
        self.assertIn("function clearSam2Preview(options = {})", markup)
        self.assertIn("function updateSam2ApplyButton()", markup)
        self.assertIn("function scheduleSam2PointHoverPreview(point)", markup)
        self.assertIn("function requestSam2PointHoverPreview(promptGeometry, key)", markup)
        self.assertIn("function applySam2PointPreviewOrRun()", markup)
        self.assertIn("function fetchSam2Mask(promptGeometry, runningMessage)", markup)
        self.assertIn("function applySam2MaskResult(maskResult)", markup)
        self.assertIn("function drawSam2ResultPreview()", markup)
        self.assertIn("drawSam2ResultPreview();", markup)
        self.assertIn("setTimeout(() =>", markup)
        self.assertIn("Running SAM2 point preview", markup)
        self.assertIn("SAM2 point preview ready; press Apply SAM2 to add it.", markup)
        self.assertIn("Hold still for SAM2 point preview, then press Apply SAM2.", markup)
        self.assertIn("Run & Apply", markup)
        self.assertIn("Apply the visible SAM2 point preview to the talc mask.", markup)
        self.assertIn("function drawSam2PromptPreview()", markup)
        self.assertIn("drawSam2PromptPreview();", markup)
        self.assertIn("ctx.setLineDash([Math.max(8, 8 / state.zoom)", markup)
        self.assertIn("state.tool === 'brush' || state.tool === 'sam2' || state.tool === 'similar'", markup)
        self.assertIn("els.sam2PromptMode.addEventListener('change'", markup)
        self.assertIn("function positiveMaskStats(img)", markup)
        self.assertIn("function isPositiveMaskPixel(src, i)", markup)
        self.assertIn("SAM2 mask covers", markup)
        self.assertIn("mask_fraction: maskResult.stats.fraction", markup)
        self.assertLess(markup.index('value="rectangle_xyxy"'), markup.index('value="point_xy"'))
        self.assertNotIn("runSam2({ type: 'point_xy', x: Math.round(point.x), y: Math.round(point.y) })", markup)
        self.assertIn("flattenShapesToBase(true)", markup)
        self.assertNotIn("fillPolygon(state.polygon.points)", markup)
        self.assertIn("viewer.addEventListener('contextmenu'", markup)
        self.assertIn("event.button === 2", markup)
        self.assertIn("const insertAt = nearestPolygonSegment(point)", markup)
        self.assertIn("Polygon point inserted.", markup)
        self.assertIn('id="protectSulfides"', markup)
        self.assertIn('id="subtractSulfidesBtn"', markup)
        self.assertIn("removeSulfidePixelsFromMask", markup)
        self.assertIn("protect_sulfides", markup)
        self.assertIn("subtract_sulfides", markup)
        self.assertIn("Current talc on sulfide", markup)
        self.assertIn("Working mask saved", markup)
        self.assertIn("Autosave failed", markup)
        self.assertIn("function nextVisibleSampleId(currentId)", markup)
        self.assertIn("function canLeaveCurrentSample(targetSampleId)", markup)
        self.assertIn("nextInVisibleQueue", markup)
        self.assertIn("statusLabel(sample.status)", markup)
        self.assertIn("reviewStateLabel(sample.review_state)", markup)
        self.assertNotIn("sample.status.includes('review')", markup)
        self.assertNotIn("enforceSulfideProtection('undo')", markup)


if __name__ == "__main__":
    unittest.main()
