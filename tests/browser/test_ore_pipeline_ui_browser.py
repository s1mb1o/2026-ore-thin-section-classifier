"""In-browser smoke tests for the main ore-pipeline single-page UI.

Boots ``OrePipelineHTTPServer`` (heuristic backend) and drives the rendered
page: home load, navigation between the five pages, and a full single-image
upload -> heuristic run -> results-render flow.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from conftest import serve_in_thread, shutdown_server


@pytest.fixture
def ore_server(tmp_path):
    import apps.ore_pipeline_web as ore_pipeline_web

    store = ore_pipeline_web.OrePipelineStore(
        workspace_dir=tmp_path / "workspace",
        backend="heuristic",
        checkpoint=None,
        processing_max_side=256,
        panorama_max_side=128,
        preview_max_sides=(128, 256),
    )
    server = ore_pipeline_web.OrePipelineHTTPServer(("127.0.0.1", 0), store)
    base_url, thread = serve_in_thread(server)
    try:
        yield base_url
    finally:
        shutdown_server(server, thread)


@pytest.fixture
def sample_image(tmp_path):
    """A small optical-microscopy-like tile with bright metallic + green regions."""
    rgb = np.full((120, 160, 3), (54, 62, 52), dtype=np.uint8)
    cv2.circle(rgb, (44, 46), 18, (226, 225, 212), -1)  # bright sulfide-like blob
    cv2.rectangle(rgb, (95, 35), (135, 82), (215, 214, 205), -1)
    cv2.rectangle(rgb, (12, 82), (52, 112), (66, 108, 70), -1)  # green talc-like patch
    path = tmp_path / "sample.png"
    Image.fromarray(rgb, mode="RGB").save(path)
    return path


def test_home_page_loads_without_console_errors(page, ore_server):
    page.goto(ore_server, wait_until="networkidle")
    assert "шлиф" in page.title().lower() or "шлиф" in page.title()
    # default landing page is the single-image workspace
    assert page.locator("body").get_attribute("data-page") == "workspace"
    assert page.locator("#workspaceTab").get_attribute("class") and "active" in page.locator("#workspaceTab").get_attribute("class")
    assert page.locator("#dropZone").is_visible()
    assert page.console_errors == []


@pytest.mark.parametrize(
    ("tab_id", "view_id", "page_name"),
    [
        ("batchTab", "#batchView", "batch"),
        ("historyTab", "#historyView", "history"),
        ("settingsTab", "#settingsView", "settings"),
        ("statusTab", "#statusView", "status"),
        ("apiTab", "#apiView", "api"),
        ("workspaceTab", "#workspaceView", "workspace"),
    ],
)
def test_navigation_switches_pages(page, ore_server, tab_id, view_id, page_name):
    page.goto(ore_server, wait_until="networkidle")
    page.locator(f"#{tab_id}").click()
    page.wait_for_selector(f"{view_id}:not(.hidden)")
    assert page.locator("body").get_attribute("data-page") == page_name
    assert page.locator(view_id).is_visible()
    assert page.console_errors == []


def test_api_page_renders_endpoint_docs(page, ore_server):
    page.goto(ore_server, wait_until="networkidle")
    page.locator("#apiTab").click()
    page.wait_for_selector("#apiView:not(.hidden)")
    # the API docs nav is populated by renderApiDocs() from the JS descriptor list
    page.wait_for_function("document.querySelectorAll('#apiDocsNav a, #apiDocsNav button').length > 0")
    assert page.console_errors == []


def _set_range(page, control_id, value):
    """Set a range input's value and fire the input/change handlers, as a drag would."""
    page.evaluate(
        """([id, v]) => {
            const el = document.getElementById(id);
            el.value = String(v);
            el.dispatchEvent(new Event('input', {bubbles: true}));
            el.dispatchEvent(new Event('change', {bubbles: true}));
        }""",
        [control_id, value],
    )


def _run_augmentation_brightness(page, run_id):
    """Read the brightness actually stored in the given run's immutable run.json."""
    return page.evaluate(
        """async (id) => {
            const r = await fetch(`/api/runs/${id}`);
            const j = await r.json();
            return j.augmentation.settings.color.brightness_pct;
        }""",
        run_id,
    )


def test_param_change_after_apply_without_reapply_is_applied_on_start(page, ore_server, sample_image):
    """Regression: a result-affecting change made after Apply but before Start must still apply.

    Flow (all via real clicks): upload -> Start -> complete; open Augmentation, set
    brightness=10, Apply (creates a `prepared` run snapshotting brightness=10); then change
    brightness to 45 WITHOUT pressing Apply again and press Start. The fixed Start handler
    detects the drifted settings and re-prepares the run in place, so the completed run must
    reflect brightness=45 (not the stale 10), and no duplicate history entry is created.
    """
    page.goto(ore_server, wait_until="networkidle")
    page.set_input_files("#fileInput", str(sample_image))
    page.wait_for_function("!document.getElementById('startBtn').disabled")

    # 1) fresh run -> complete
    page.locator("#startBtn").click()
    page.wait_for_function("typeof state !== 'undefined' && state.run && state.run.status === 'complete'", timeout=60_000)
    first_run_id = page.evaluate("state.run.run_id")

    # 2) enable augmentation, set brightness=10 in the dialog, close it, Apply -> prepared (snapshot v1).
    #    The brightness slider lives inside the modal <dialog>; the Apply button lives in the side
    #    panel (inert while the modal is open), so the realistic flow opens the dialog to adjust the
    #    slider, closes it, then presses Apply.
    page.locator("#editAugmentationBtn").click()
    page.wait_for_selector("#augmentationDialog[open]")
    page.evaluate("() => { const c = document.getElementById('augmentationEnabled'); c.checked = true; c.dispatchEvent(new Event('change', {bubbles: true})); }")
    _set_range(page, "augBrightness", 10)
    page.locator("#closeAugmentationBtn").click()
    page.wait_for_selector("#augmentationDialog[open]", state="hidden")
    page.locator("#applyAugmentationBtn").click()
    page.wait_for_function("typeof state !== 'undefined' && state.run && state.run.status === 'prepared'", timeout=60_000)
    prepared_run_id = page.evaluate("state.run.run_id")
    assert prepared_run_id != first_run_id
    runs_after_apply = page.evaluate("async () => (await (await fetch('/api/runs')).json()).runs.length")

    # 3) change brightness to 45 in the dialog WITHOUT pressing Apply again, close it, then Start
    page.locator("#editAugmentationBtn").click()
    page.wait_for_selector("#augmentationDialog[open]")
    _set_range(page, "augBrightness", 45)
    page.locator("#closeAugmentationBtn").click()
    page.wait_for_selector("#augmentationDialog[open]", state="hidden")
    page.locator("#startBtn").click()

    # 4) wait for the re-prepared run to complete
    page.wait_for_function(
        "(firstId) => (typeof state !== 'undefined' && state.run && state.run.status === 'complete' && state.run.run_id !== firstId)",
        arg=first_run_id,
        timeout=60_000,
    )
    final_run_id = page.evaluate("state.run.run_id")

    # 5) the latest change (45) must have been applied, not the Apply-time snapshot (10)
    assert _run_augmentation_brightness(page, final_run_id) == 45.0
    # re-prepared in place: same id as the prepared run, and Start's re-prepare added no new run
    assert final_run_id == prepared_run_id
    runs_after = page.evaluate("async () => (await (await fetch('/api/runs')).json()).runs.length")
    assert runs_after == runs_after_apply  # Start reused the prepared run in place, no duplicate
    assert page.console_errors == []


def test_upload_and_heuristic_run_flow(page, ore_server, sample_image):
    page.goto(ore_server, wait_until="networkidle")

    # Uploading through the file input triggers the change handler -> /api/uploads.
    page.set_input_files("#fileInput", str(sample_image))

    # The Start button is disabled until an upload is registered in state.
    page.wait_for_function("!document.getElementById('startBtn').disabled")
    page.locator("#startBtn").click()

    # pollRun() un-hides the result panel and fills the metrics table on completion.
    page.wait_for_selector("#resultPanel:not(.hidden)", timeout=60_000)
    page.wait_for_function("document.querySelectorAll('#metricsTable tbody tr').length > 0", timeout=60_000)

    assert page.locator("#resultPanel").is_visible()
    assert page.locator("#textOutput").inner_text().strip() != ""
    assert page.locator("#metricsTable tbody tr").count() > 0
    # a completed run is appended to history
    page.locator("#historyTab").click()
    page.wait_for_selector("#historyView:not(.hidden)")
    page.wait_for_function("document.querySelectorAll('#historyPageList tr, #historyPageList [data-load-run]').length > 0")
    assert page.console_errors == []
