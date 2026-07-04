"""In-browser smoke tests for the talc mask review single-page UI.

Boots ``TalcReviewHTTPServer`` against a one-sample workspace and verifies the
rendered review app: page load, sample-queue population, auto-load of the first
sample into the viewer, and basic tool switching.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from conftest import serve_in_thread, shutdown_server

IMAGE_NAME = "sample_blue.JPG"


@pytest.fixture
def talc_server(tmp_path):
    import apps.talc_review_web as talc_review_web

    annotated_dir = tmp_path / "annotated"
    original_dir = tmp_path / "original"
    workspace_dir = tmp_path / "workspace"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    original_dir.mkdir(parents=True, exist_ok=True)

    annotated = np.full((90, 120, 3), (54, 66, 48), dtype=np.uint8)
    cv2.rectangle(annotated, (22, 18), (90, 72), (0, 0, 255), thickness=4)  # blue markup
    original = np.full((90, 120, 3), (54, 66, 48), dtype=np.uint8)
    Image.fromarray(annotated, mode="RGB").save(annotated_dir / IMAGE_NAME)
    Image.fromarray(original, mode="RGB").save(original_dir / IMAGE_NAME)

    store = talc_review_web.TalcReviewStore(
        annotated_dir=annotated_dir,
        original_dir=original_dir,
        workspace_dir=workspace_dir,
        conversion_dir=None,
        sulfide_mask_dir=None,
        silicate_mask_dir=None,
        reconvert=True,
        limit=None,
        sam2_model_id="facebook/sam2.1-hiera-tiny",
        sam2_device="cpu",
    )
    server = talc_review_web.TalcReviewHTTPServer(("127.0.0.1", 0), store)
    base_url, thread = serve_in_thread(server)
    try:
        yield base_url
    finally:
        shutdown_server(server, thread)


def test_review_page_loads_and_populates_queue(page, talc_server):
    page.goto(talc_server, wait_until="networkidle")
    assert page.title() == "Talc mask review"
    # loadManifest(true) fills the sample queue and auto-loads the first sample.
    page.wait_for_selector("#sampleList .sample-card")
    assert page.locator("#sampleList .sample-card").count() >= 1
    assert "total" in page.locator("#queueStats").inner_text()
    assert page.console_errors == []


def test_first_sample_auto_loads_into_viewer(page, talc_server):
    page.goto(talc_server, wait_until="networkidle")
    # loadSample() sets the title to the image name and hides the empty state.
    page.wait_for_function(
        "document.getElementById('sampleTitle').textContent.trim() !== 'Loading...'"
        " && document.getElementById('sampleTitle').textContent.trim() !== ''"
    )
    assert page.locator("#sampleTitle").inner_text().strip() == IMAGE_NAME
    # emptyState gets display:none via the .hidden class once the sample renders.
    page.wait_for_function("document.getElementById('emptyState').classList.contains('hidden')")
    assert page.locator("#viewerCanvas").count() == 1
    assert page.console_errors == []


def test_tool_switch_reveals_similar_params(page, talc_server):
    page.goto(talc_server, wait_until="networkidle")
    page.wait_for_selector("#sampleList .sample-card")
    # The similar-region tool group starts hidden; selecting the tool reveals it.
    similar_tool = page.locator("[data-tool='similar']")
    if similar_tool.count() == 0:
        pytest.skip("similar tool control not present in this build")
    similar_tool.first.click()
    page.wait_for_selector("#similarParams:not(.hidden)")
    assert page.locator("#similarParams").is_visible()
    assert page.console_errors == []


def test_cluster_overlay_controls_update_live_stats(page, talc_server):
    page.goto(talc_server, wait_until="networkidle")
    page.wait_for_selector("#sampleList .sample-card")
    page.wait_for_function("document.getElementById('emptyState').classList.contains('hidden')")

    page.locator("#clusterSource").select_option("union")
    page.locator("#clusterDensity").evaluate("(el) => { el.value = '1'; el.dispatchEvent(new Event('input', { bubbles: true })); }")
    page.locator("#clusterOverlayToggle").check()

    page.wait_for_function(
        "() => document.getElementById('clusterStats').textContent.trim() !== 'Cluster overlay is off.'"
    )
    assert page.locator("#clusterStats").inner_text().strip() != "Cluster overlay is off."
    assert page.console_errors == []
