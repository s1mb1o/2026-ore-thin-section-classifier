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
                "grade_label": grade,
                "heuristic_label": "fine_intergrowth" if i % 2 else "ordinary_intergrowth",
                "crop_path": f"crops/{grade}/{uid}.png",
                "area_px": str(500 + i),
                "dark_inside_ratio": "0.3",
                "solidity": "0.75",
            }
        )
    dataset_dir.mkdir(parents=True)
    fieldnames = list(grains[0].keys())
    with (dataset_dir / "grains_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(grains)
    for grain in grains:
        crop = dataset_dir / grain["crop_path"]
        crop.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 12), (110, 105, 95)).save(crop)

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
    # crop images resolve through /crops/ (a broken URL would leave naturalWidth 0)
    page.wait_for_function(
        "[...document.querySelectorAll('.card img')].every(img => img.complete && img.naturalWidth > 0)"
    )
    assert "размечено 0/3" in page.locator("#prog").inner_text()
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
