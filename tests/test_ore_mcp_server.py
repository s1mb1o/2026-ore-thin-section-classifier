"""Fast smoke tests for the ore MCP server (apps/ore_mcp_server.py).

These deliberately do NOT load any model checkpoint (which is slow and requires the
weights to be present). They cover the MCP protocol surface — module import, tool
registration, config resolution, and input validation — all of which run before the
lazy model load. A real end-to-end classification is exercised manually (SMOKE_TESTS.md).
"""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "apps" / "ore_mcp_server.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ore_mcp_server", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tools_are_registered():
    module = _load_module()
    tools = asyncio.run(module.mcp.list_tools())
    names = {t.name for t in tools}
    assert {"classify_thin_section", "get_config"} <= names
    classify = next(t for t in tools if t.name == "classify_thin_section")
    assert classify.inputSchema.get("required") == ["image_path"]


def test_get_config_does_not_load_model():
    module = _load_module()
    cfg = module.get_config()
    assert cfg["model_loaded"] is False
    assert cfg["talc_backend"] in {"ml_model", "heuristic"}
    assert set(cfg) >= {
        "device",
        "out_root",
        "sulfide_checkpoint",
        "talc_backend",
        "grade_checkpoint",
        "model_loaded",
    }


def test_classify_rejects_missing_image_before_model_load():
    module = _load_module()
    with pytest.raises(ValueError, match="does not exist"):
        module.classify_thin_section("this/path/does/not/exist_xyz.png")
    # The bad-path guard must run before any checkpoint is touched.
    assert module.get_config()["model_loaded"] is False
