"""Shared fixtures for in-browser UI smoke tests (Playwright + headless Chromium).

These tests boot the real stdlib HTTP servers on an ephemeral port in a daemon
thread and drive the rendered single-page apps with a real browser, so they
exercise the JavaScript, fetch calls, and DOM the HTTP-level tests never touch.

The whole module is skipped (never errored) when Playwright or its Chromium
build is unavailable, so `pytest tests/` stays green on machines without them.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Skip the entire browser suite (collection-safe) when Playwright is absent.
sync_api = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="session")
def browser():
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        try:
            instance = playwright.chromium.launch()
        except PlaywrightError as exc:  # chromium build not installed
            pytest.skip(f"chromium not available: {exc}")
        try:
            yield instance
        finally:
            instance.close()


@pytest.fixture
def page(browser):
    """A fresh page that records JS console errors and uncaught exceptions.

    The recorded messages are exposed on ``page.console_errors`` so tests can
    assert the UI rendered without runtime errors.
    """
    context = browser.new_context(viewport={"width": 1280, "height": 900})
    pg = context.new_page()
    pg.console_errors = []
    pg.on("pageerror", lambda exc: pg.console_errors.append(f"pageerror: {exc}"))
    pg.on(
        "console",
        lambda msg: pg.console_errors.append(f"console.error: {msg.text}")
        if msg.type == "error"
        else None,
    )
    try:
        yield pg
    finally:
        context.close()


def serve_in_thread(server):
    """Start ``server.serve_forever`` in a daemon thread and return its base URL."""
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    return f"http://{host}:{port}", thread


def shutdown_server(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
