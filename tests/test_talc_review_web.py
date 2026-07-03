from __future__ import annotations

import base64
import concurrent.futures
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
                    "brightness_threshold_formula": "luma = 0.299*R + 0.587*G + 0.114*B",
                    "background_mode": "original",
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
            "reviewed_ignore_mask",
            "reviewed_overlay",
            "review_patch",
        ]:
            self.assertTrue(Path(reviewed[key]).exists(), key)
        reviewed_mask = read_mask(Path(reviewed["reviewed_talc_mask"]))
        self.assertEqual(int(np.count_nonzero(reviewed_mask)), 600)
        self.assertEqual(int(np.count_nonzero(read_mask(Path(reviewed["reviewed_positive_bag_mask"])))), 600)
        self.assertEqual(int(np.count_nonzero(read_mask(Path(reviewed["reviewed_talc_node_mask"])))), 0)
        patch = json.loads(Path(reviewed["review_patch"]).read_text(encoding="utf-8"))
        self.assertEqual(patch["reviewer"], "unit-test")
        self.assertEqual(patch["original_image_path"], str((self.original_dir / self.image_name).resolve()))
        self.assertEqual(patch["view_settings"]["brightness_threshold_luma"], 90)
        self.assertEqual(patch["view_settings"]["background_mode"], "original")
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
        self.assertIn("Select the Edit radio in Segmentation classes", markup)
        self.assertIn("Brush: left mouse adds the selected class, right mouse erases it.", markup)
        self.assertIn("Brush (B): left mouse draws the selected class, right mouse erases it", markup)
        self.assertIn('<input id="brushSize" type="range" min="2" max="240" value="28"', markup)
        self.assertIn("Fill (F): click an area bounded by blue lines, sulfides, existing selected-class regions, or image edges", markup)
        self.assertIn("Similar: click a known talc grain to preview intensity-similar talc-node pixels", markup)
        self.assertIn("Fill: click an empty area bounded by blue lines, sulfides, existing selected-class regions, or the image edge.", markup)
        self.assertIn("Similar: click a confirmed talc grain to preview luma/color-similar non-sulfide pixels", markup)
        self.assertIn('aria-keyshortcuts="B"', markup)
        self.assertIn('aria-keyshortcuts="F"', markup)
        self.assertLess(markup.index('data-tool="brush"'), markup.index('data-tool="fill"'))
        self.assertLess(markup.index('data-tool="fill"'), markup.index('data-tool="similar"'))
        self.assertLess(markup.index('data-tool="similar"'), markup.index('data-tool="rectangle"'))
        self.assertLess(markup.index('data-tool="rectangle"'), markup.index('data-tool="polygon"'))
        self.assertLess(markup.index('data-tool="polygon"'), markup.index('data-tool="sam2"'))
        self.assertIn('class="toolbar-separator"', markup)
        self.assertIn('id="zoomInBtn"', markup)
        self.assertIn('id="zoomOutBtn"', markup)
        self.assertIn('id="fitBtn"', markup)
        self.assertIn('id="zoomValue"', markup)
        self.assertIn('class="review-actions"', markup)
        self.assertIn('id="saveBtn"', markup)
        self.assertIn('Save &amp; Next', markup)
        self.assertIn('id="nextBtn"', markup)
        self.assertIn('class="plain-button"', markup)
        self.assertIn("async function goToNextSample()", markup)
        self.assertIn("els.nextBtn.addEventListener('click'", markup)
        self.assertIn(".plain-button { background: transparent;", markup)
        self.assertLess(markup.index('class="review-actions"'), markup.index('id="viewerWrap"'))
        self.assertLess(markup.index('id="saveBtn"'), markup.index('id="viewerWrap"'))
        self.assertLess(markup.index('id="saveNextBtn"'), markup.index('id="viewerWrap"'))
        self.assertLess(markup.index('id="saveNextBtn"'), markup.index('id="nextBtn"'))
        self.assertLess(markup.index('id="nextBtn"'), markup.index('id="viewerWrap"'))
        self.assertLess(markup.index('id="notesInput"'), markup.index('id="resetBtn"'))
        self.assertNotIn("Save and next", markup)
        self.assertIn('value="sulfide"', markup)
        self.assertIn("Sulfide mask (sulfide/non-sulfide mask segmentation)", markup)
        self.assertIn("Mask-only background", markup)
        self.assertIn('class="segmentation-class-widget"', markup)
        self.assertIn('aria-label="Visible segmentation classes"', markup)
        self.assertIn("Segmentation classes", markup)
        self.assertIn("Show", markup)
        self.assertIn("Edit", markup)
        self.assertIn('name="editTargetClass"', markup)
        self.assertIn('id="editTargetPositiveBag"', markup)
        self.assertIn('id="editTargetTalcNode"', markup)
        self.assertIn('value="positive_bag"', markup)
        self.assertIn('value="talc_node"', markup)
        self.assertIn('class="class-swatch positive-bag"', markup)
        self.assertIn('class="class-swatch talc"', markup)
        self.assertIn("Positive bag", markup)
        self.assertIn("Talc", markup)
        self.assertIn('id="brightnessThreshold"', markup)
        self.assertIn('id="brightnessThresholdValue"', markup)
        self.assertIn('id="brightnessThreshold90Btn"', markup)
        self.assertIn('id="brightnessThresholdOffBtn"', markup)
        self.assertIn("Dark pixel preview threshold", markup)
        self.assertIn("Luma = 0.299 R + 0.587 G + 0.114 B", markup)
        self.assertIn("BRIGHTNESS_THRESHOLD_STORAGE_KEY", markup)
        self.assertIn("BRIGHTNESS_THRESHOLD_FORMULA", markup)
        self.assertIn("function brightnessFilteredBackground(base)", markup)
        self.assertIn("const luma = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]", markup)
        self.assertIn("if (threshold <= 0)", markup)
        self.assertIn("if (luma > threshold)", markup)
        self.assertIn("view_settings: viewSettingsPayload()", markup)
        self.assertIn('id="assetWarnings"', markup)
        self.assertIn("baseMode === 'sulfide'", markup)
        self.assertIn("state.images.sulfideMask", markup)
        self.assertIn("state.images = { original, annotated, qa, sulfideMask }", markup)
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
        self.assertIn("function startViewPan(event)", markup)
        self.assertIn("function updateViewPan(event)", markup)
        self.assertIn("function finishViewPan(event = null)", markup)
        self.assertIn("if (event.button === 1)", markup)
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
        self.assertIn("boundaryLabels.push(targetClass === 'talc_node' ? 'current_talc_node_regions' : 'current_positive_bag_regions')", markup)
        self.assertIn("type: targetClass === 'talc_node' ? 'polygon_add_talc_node' : 'polygon_add_positive_bag'", markup)
        self.assertIn("type: targetClass === 'talc_node' ? 'rectangle_add_talc_node' : 'rectangle_add_positive_bag'", markup)
        self.assertIn("const targetCtx = normalizeEditClass(shape.targetClass) === 'talc_node' ? talcNodeCtx : maskCtx", markup)
        self.assertIn("function combinedMaskCanvas()", markup)
        self.assertIn("positive_bag_mask_png", markup)
        self.assertIn("talc_node_mask_png", markup)
        self.assertIn("target_class: 'positive_bag'", markup)
        self.assertIn("target_class: 'talc_node'", markup)
        self.assertIn('id="layerTalcNode"', markup)
        self.assertIn("current_positive_bag_mask", markup)
        self.assertIn("current_talc_node_mask", markup)
        self.assertIn("similarTalcPreview", markup)
        self.assertIn("MAX_SIMILAR_TALC_REGION_FRACTION", markup)
        self.assertIn("function computeSimilarTalcPreview(point)", markup)
        self.assertIn("function applySimilarTalcPreview(options = {})", markup)
        self.assertIn("function cleanupSimilarTalcCandidates(candidate, width, height)", markup)
        self.assertIn("function collectSeedPatchSamples(seedX, seedY, sourceData, sulfideData)", markup)
        self.assertIn("function similarFeatureDistanceToStats(item, stats)", markup)
        self.assertIn("function drawSimilarTalcPreview()", markup)
        self.assertIn("drawSimilarTalcPreview();", markup)
        self.assertIn("source_tool: 'similar_talc'", markup)
        self.assertIn("type: 'similar_talc_add'", markup)
        self.assertIn("overlapping_positive_bag_pixels", markup)
        self.assertIn("excluded_existing_talc_pixels", markup)
        self.assertIn("source_kind: sourceKind", markup)
        self.assertIn("source_kind: preview.stats ? preview.stats.source_kind : null", markup)
        self.assertIn("positive_bag_kept: positiveBagKept", markup)
        self.assertIn("seed patch + filtered positive bag", markup)
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
        self.assertIn("state.tool === 'brush' || state.tool === 'sam2'", markup)
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
