#!/usr/bin/env python3
"""Capture presentation screenshots from the two running local apps via Playwright.

Main ore-pipeline UI must be running on 127.0.0.1:8230, talc review on :8231.
Screenshots are written to presentation/assets/screens/.
Run with the system python that has playwright + chromium installed.
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent / "assets" / "screens"
OUT.mkdir(parents=True, exist_ok=True)

MAIN = "http://127.0.0.1:8230"
TALC = "http://127.0.0.1:8231"

ORDINARY_RUN = "run_20260703_203236_872638000_6c5d2f09"  # row_ore -> рядовая
FINE_RUN = "run_20260703_195241_691810000_75c57d1b"      # hard_to_process -> труднообогатимая

VW, VH = 1680, 1050


def shot(page, name, full=False):
    path = OUT / f"{name}.png"
    page.screenshot(path=str(path), full_page=full)
    print(f"  saved {path.name} ({'full' if full else 'viewport'})")


def load_run_and_shot(page, run_id, name):
    page.goto(f"{MAIN}/history", wait_until="networkidle")
    page.wait_for_timeout(1200)
    # The "Load" button is hidden until row hover; force the click, with a JS fallback.
    clicked = False
    try:
        page.locator(f'[data-load-run="{run_id}"]').first.click(timeout=5000, force=True)
        clicked = True
    except Exception as e:
        print(f"  ~ force click failed ({e}); trying JS loadRun()")
    if not clicked:
        try:
            page.evaluate("(id) => loadRun(id)", run_id)
            clicked = True
        except Exception as e:
            print(f"  ! JS loadRun failed: {e}")
            return False
    # loadRun populates the workspace result panel; wait for it
    try:
        page.wait_for_selector("#resultPanel", state="visible", timeout=15000)
    except Exception:
        # maybe it didn't switch views; go to workspace explicitly (state persists in JS)
        try:
            page.goto(f"{MAIN}/workspace", wait_until="networkidle")
            page.wait_for_selector("#resultPanel", state="visible", timeout=8000)
        except Exception:
            print("  ~ resultPanel never became visible")
    page.wait_for_timeout(2800)  # let viewer image + overlay render
    shot(page, name)
    return True


def try_select_layer(page, label_substr, name):
    """Best-effort: click a layer control whose text contains label_substr."""
    for sel in [
        f'button:has-text("{label_substr}")',
        f'label:has-text("{label_substr}")',
        f'[data-layer*="{label_substr}"]',
    ]:
        try:
            el = page.locator(sel).first
            if el.count() and el.is_visible():
                el.click(timeout=3000)
                page.wait_for_timeout(1500)
                shot(page, name)
                return True
        except Exception:
            continue
    print(f"  ~ layer control '{label_substr}' not found (skipped {name})")
    return False


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": VW, "height": VH}, device_scale_factor=2)
        page = ctx.new_page()

        # --- Main app: static-ish pages (full page) ---
        for slug, name, full in [
            ("/status", "main_status", True),
            ("/api", "main_api", True),
            ("/settings", "main_settings", True),
            ("/history", "main_history", True),
            ("/batch", "main_series", True),
        ]:
            print(f"[main] {slug}")
            try:
                page.goto(f"{MAIN}{slug}", wait_until="networkidle")
                page.wait_for_timeout(1600)
                shot(page, name, full=full)
            except Exception as e:
                print(f"  ! {slug} failed: {e}")

        # --- Main app: workspace with a run loaded (the money shots) ---
        print("[main] workspace + fine run")
        if load_run_and_shot(page, FINE_RUN, "main_workspace_fine"):
            try_select_layer(page, "сульфид", "main_sulfide_layer")
        print("[main] workspace + ordinary run")
        load_run_and_shot(page, ORDINARY_RUN, "main_workspace_ordinary")

        # --- Talc review app ---
        print("[talc] canvas")
        try:
            page.goto(f"{TALC}/", wait_until="networkidle")
            page.wait_for_timeout(3000)  # first sample + mask render
            shot(page, "talc_canvas")
            # best-effort dark theme
            for sel in ['select:has(option:has-text("Dark"))', 'select:has(option:has-text("Тёмная"))']:
                try:
                    el = page.locator(sel).first
                    if el.count():
                        el.select_option(label="Dark")
                        page.wait_for_timeout(1200)
                        shot(page, "talc_dark")
                        break
                except Exception:
                    continue
        except Exception as e:
            print(f"  ! talc failed: {e}")

        ctx.close()
        browser.close()
    print("done")


if __name__ == "__main__":
    sys.exit(main())
