"""In-browser smoke tests for the grain review / labeling single-page UI.

Boots ``GrainReviewHTTPServer`` against a small synthetic grain dataset and
drives the rendered page with headless Chromium: grid population, click-to-label
persistence into ``annotations.json``, label toggle-off, and keyboard labeling.
"""

from __future__ import annotations

import csv
import json

import pytest
from PIL import Image

from conftest import serve_in_thread, shutdown_server


@pytest.fixture
def grain_server(tmp_path):
    import apps.grain_review_web as grain_review_web

    dataset_dir = tmp_path / "grain_dataset"
    grains = []
    for i, grade in enumerate(["ordinary_intergrowth", "fine_intergrowth", "talcose"]):
        uid = f"run_{i}__c{i}"
        grains.append(
            {
                "grain_uid": uid,
                "run_id": f"run_{i}",
                "grade_label": grade,
                "heuristic_label": "fine_intergrowth" if i % 2 else "ordinary_intergrowth",
                "image_rel_path": f"source_images/source_{i}.jpg",
                "source_dataset_path": str(dataset_dir / f"source_images/source_{i}.jpg"),
                "component_id": str(i),
                "crop_path": f"crops/{grade}/{uid}.png",
                "area_px": str(500 + i),
                "dark_inside_ratio": "0.3",
                "solidity": "0.75",
                "compactness": "0.18",
                "bbox_x": str(3 + i),
                "bbox_y": str(4 + i),
                "bbox_w": "10",
                "bbox_h": "8",
            }
        )
    dataset_dir.mkdir(parents=True)
    fieldnames = list(grains[0].keys())
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
        Image.new("RGB", (16, 12), (110, 105, 95)).save(crop)
        source = dataset_dir / grain["image_rel_path"]
        source.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80, 60), (130, 120, 100)).save(source)
        mask_dir = batch_dir / "runs" / grain["grade_label"] / grain["run_id"] / "binary_sulfide"
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask = Image.new("L", (80, 60), 0)
        x, y = int(grain["bbox_x"]), int(grain["bbox_y"])
        w, h = int(grain["bbox_w"]), int(grain["bbox_h"])
        for yy in range(y, min(y + h, mask.height)):
            for xx in range(x, min(x + w, mask.width)):
                mask.putpixel((xx, yy), 255)
        mask.save(mask_dir / "sulfide_mask.png")

    store = grain_review_web.GrainReviewStore(dataset_dir)
    server = grain_review_web.GrainReviewHTTPServer(("127.0.0.1", 0), store)
    base_url, thread = serve_in_thread(server)
    try:
        yield base_url, dataset_dir
    finally:
        shutdown_server(server, thread)


def read_labels(dataset_dir):
    path = dataset_dir / "annotations.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("labels", {})


def test_page_loads_grid_and_progress(page, grain_server):
    base_url, _ = grain_server
    page.goto(base_url, wait_until="networkidle")
    assert "Разметка зёрен" in page.title()
    page.wait_for_selector(".card")
    assert page.locator(".card").count() == 3
    assert page.locator("#sort option[value='review_value']").inner_text() == "ценные для проверки"
    page.locator("#sort").select_option("review_value")
    page.wait_for_function("document.querySelector('#sort').value === 'review_value'")
    # crop images resolve through /crops/ (a broken URL would leave naturalWidth 0)
    page.wait_for_function(
        "[...document.querySelectorAll('.card img')].every(img => img.complete && img.naturalWidth > 0)"
    )
    assert "Ценность проверки:" in page.locator("#detail .verdict").inner_text()
    assert "размечено 0/3" in page.locator("#prog").inner_text()
    assert page.console_errors == []


def test_three_choice_buttons_use_fine_uncertain_ordinary_order(page, grain_server):
    base_url, _ = grain_server
    page.goto(base_url, wait_until="networkidle")
    page.wait_for_selector(".card")

    card_labels = page.locator(".card").first.locator(".row button").evaluate_all(
        "(buttons) => buttons.map((button) => button.textContent.trim())"
    )
    assert card_labels == ["тонкое", "?", "рядовое"]

    detail_labels = page.locator("#detail .dbtns button").evaluate_all(
        "(buttons) => buttons.map((button) => button.textContent.trim())"
    )
    assert detail_labels == ["тонкое (F)", "? (U)", "рядовое (O)"]
    assert page.console_errors == []


def test_heuristic_score_is_shown_in_detail_and_tinder_mode(page, grain_server):
    base_url, _ = grain_server
    page.goto(base_url, wait_until="networkidle")
    page.wait_for_selector(".card")

    detail_text = page.locator("#detail .verdict").inner_text()
    assert "Признак" in detail_text
    assert "Порог тонкого" in detail_text
    assert "Тёмное внутри" in detail_text
    assert "≥ 0.18" in detail_text
    assert "Счёт эвристики: рядовое 67% · тонкое 33%" in detail_text

    page.locator("#mode").select_option("tinder")
    page.wait_for_selector(".tinder-stage")
    tinder_text = page.locator(".tinder-verdict").inner_text()
    assert "Компактность" in tinder_text
    assert "рядовое" in tinder_text
    assert "тонкое" in tinder_text
    assert "Счёт эвристики: рядовое 67% · тонкое 33%" in tinder_text
    assert page.console_errors == []


def test_grain_crop_contour_checkbox_toggles_overlay(page, grain_server):
    base_url, _ = grain_server
    page.goto(base_url, wait_until="networkidle")
    page.wait_for_selector(".card")

    toggle = page.locator("#detail .contour-toggle")
    background = page.locator("#detail .background-toggle")
    assert toggle.is_enabled()
    assert background.is_checked()
    toggle.check()
    page.wait_for_function(
        "document.querySelector('#detail .crop-viewer').classList.contains('show-contour')"
    )
    page.wait_for_function(
        "document.querySelector('#detail .contour-overlay').complete && document.querySelector('#detail .contour-overlay').naturalWidth > 0"
    )
    background.uncheck()
    page.wait_for_function(
        "document.querySelector('#detail .crop-viewer').classList.contains('hide-background')"
    )
    assert page.locator("#detail .crop-viewer").evaluate(
        "node => node.classList.contains('show-contour')"
    )
    assert page.console_errors == []


def test_click_labeling_persists_and_toggles_off(page, grain_server):
    base_url, dataset_dir = grain_server
    page.goto(base_url, wait_until="networkidle")
    page.wait_for_selector(".card")

    # click 'рядовое' on the first card -> POST /api/annotate -> annotations.json
    page.locator(".card").first.locator("button.a-ord").click()
    page.wait_for_function("document.querySelector('.card button.a-ord').classList.contains('on')")
    labels = read_labels(dataset_dir)
    assert labels["run_0__c0"]["label"] == "ordinary_intergrowth"
    page.wait_for_function("document.getElementById('prog').textContent.includes('размечено 1/3')")

    # clicking the same active label again clears it
    page.locator(".card").first.locator("button.a-ord").click()
    page.wait_for_function("!document.querySelector('.card button.a-ord').classList.contains('on')")
    page.wait_for_function("document.getElementById('prog').textContent.includes('размечено 0/3')")
    assert read_labels(dataset_dir) == {}
    assert page.console_errors == []


def test_keyboard_labeling_assigns_selected_card(page, grain_server):
    base_url, dataset_dir = grain_server
    page.goto(base_url, wait_until="networkidle")
    page.wait_for_selector(".card")

    # first card is selected by default; F labels it fine_intergrowth and advances
    page.keyboard.press("f")
    page.wait_for_function("document.querySelector('.card button.a-fine').classList.contains('on')")
    labels = read_labels(dataset_dir)
    assert labels["run_0__c0"]["label"] == "fine_intergrowth"
    # selection auto-advanced to the second card; U marks it uncertain
    page.keyboard.press("u")
    page.wait_for_function(
        "document.querySelectorAll('.card')[1].querySelector('button.a-unc').classList.contains('on')"
    )
    labels = read_labels(dataset_dir)
    assert labels["run_1__c1"]["label"] == "uncertain"
    page.wait_for_function("document.getElementById('prog').textContent.includes('размечено 2/3')")
    assert page.console_errors == []


def test_tinder_mode_arrow_actions_and_source_context(page, grain_server):
    base_url, dataset_dir = grain_server
    page.goto(base_url, wait_until="networkidle")
    page.wait_for_selector(".card")

    page.locator("#mode").select_option("tinder")
    page.wait_for_selector(".tinder-stage")
    page.wait_for_function("document.querySelector('#sourceImg').complete && document.querySelector('#sourceImg').naturalWidth > 0")
    page.wait_for_function("document.querySelector('#grainBox').getAttribute('width') === '10'")

    # In Tinder mode, the arrow keys are decisions: left=fine, up=postpone,
    # down=uncertain, right=ordinary. Up must advance without writing a label.
    page.keyboard.press("ArrowLeft")
    page.wait_for_function("document.getElementById('prog').textContent.includes('размечено 1/3')")
    labels = read_labels(dataset_dir)
    assert labels["run_0__c0"]["label"] == "fine_intergrowth"

    page.keyboard.press("ArrowUp")
    labels = read_labels(dataset_dir)
    assert "run_1__c1" not in labels

    page.keyboard.press("ArrowDown")
    page.wait_for_function("document.getElementById('prog').textContent.includes('размечено 2/3')")
    labels = read_labels(dataset_dir)
    assert labels["run_2__c2"]["label"] == "uncertain"
    assert page.console_errors == []


def test_tinder_slug_direct_loads_and_updates_current_grain(page, grain_server):
    base_url, dataset_dir = grain_server
    page.goto(f"{base_url}/tinder/run_1__c1", wait_until="networkidle")
    page.wait_for_selector(".tinder-stage")

    assert page.locator("#mode").input_value() == "tinder"
    assert page.locator(".zoom-panel h2").inner_text() == "Зерно · run_1__c1"
    assert page.evaluate("window.location.pathname") == "/tinder/run_1__c1"

    page.keyboard.press("ArrowUp")
    page.wait_for_function("window.location.pathname === '/tinder/run_2__c2'")
    assert "run_1__c1" not in read_labels(dataset_dir)
    assert page.locator(".zoom-panel h2").inner_text() == "Зерно · run_2__c2"
    assert page.console_errors == []


def test_view_filter_hides_labeled_cards(page, grain_server):
    base_url, _ = grain_server
    page.goto(base_url, wait_until="networkidle")
    page.wait_for_selector(".card")
    page.locator(".card").first.locator("button.a-unc").click()
    page.wait_for_function("document.getElementById('prog').textContent.includes('размечено 1/3')")

    page.locator("#view").select_option("unlabeled")
    page.wait_for_function("document.querySelectorAll('.card').length === 2")
    page.locator("#view").select_option("labeled")
    page.wait_for_function("document.querySelectorAll('.card').length === 1")
    assert page.console_errors == []
