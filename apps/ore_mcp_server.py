#!/usr/bin/env python3
"""MCP server for the ore thin-section classifier (Option A — resident model).

Exposes the in-process ore pipeline (``ResidentSulfidePipeline.run_image``) as MCP
tools so an assistant (Claude Code, Claude Desktop, any MCP client) can classify an
optical-microscopy ore thin section and get the sulfide phases, talc source, ore-class
metrics, an optional learned ordinary/fine grade opinion, and artifact paths back inline.

The checkpoint(s) load **once**, lazily on the first ``classify_thin_section`` call, and
stay warm for the life of the server process — no per-image reload. This is the whole
point of Option A over shelling out to the CLI per image.

Run (stdio transport):

    .venv/bin/python apps/ore_mcp_server.py

Register with Claude Code (absolute paths required):

    claude mcp add ore-classifier -- \
        /abs/path/2026_Nornikel_Hackaton_v2/.venv/bin/python \
        /abs/path/2026_Nornikel_Hackaton_v2/apps/ore_mcp_server.py

Configuration via environment variables (all optional; defaults mirror the web app):

    ORE_MCP_CHECKPOINT        binary-sulfide segmentation checkpoint (.pt)
    ORE_MCP_TALC_CHECKPOINT   talc segmentation checkpoint (.pt); heuristic talc if unset/missing
    ORE_MCP_GRADE_CHECKPOINT  grade-classifier checkpoint (.pt); grade branch skipped if unset/missing
    ORE_MCP_DEVICE            torch device string (default: "auto")
    ORE_MCP_OUT_ROOT          base dir for run artifacts (default: <repo>/outputs/mcp_runs)
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp.server.fastmcp import FastMCP  # noqa: E402

# Defaults mirror apps/ore_pipeline_web.py so the MCP tool and the web UI agree on models.
DEFAULT_CHECKPOINT = ROOT / "models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt"
DEFAULT_TALC_CHECKPOINT = ROOT / "outputs/talc_segformer_folds/segformer_b0_full_20260703/fold_00/segformer_b0/best.pt"
DEFAULT_GRADE_CHECKPOINT = ROOT / "models/grade_classifier/effb3_ordfine_ppaug_20260704/best.pt"


def _env_checkpoint(var: str, default: Path) -> Path | None:
    """Resolve a checkpoint path from ``var`` (falling back to ``default``); None if missing."""
    raw = os.environ.get(var, "").strip()
    path = Path(raw).expanduser() if raw else default
    if not path.is_absolute():
        path = ROOT / path
    return path if path.exists() else None


class _Engine:
    """Lazily loads and holds the warm pipeline + optional grade model for the process."""

    def __init__(self) -> None:
        self._pipeline = None
        self._grade_model = None
        self._grade_loaded = False
        self.device = os.environ.get("ORE_MCP_DEVICE", "auto").strip() or "auto"
        self.out_root = Path(os.environ.get("ORE_MCP_OUT_ROOT", "").strip() or (ROOT / "outputs/mcp_runs"))

    @property
    def pipeline(self):
        if self._pipeline is None:
            from ore_classifier.resident_pipeline import ResidentSulfidePipeline

            checkpoint = _env_checkpoint("ORE_MCP_CHECKPOINT", DEFAULT_CHECKPOINT)
            if checkpoint is None:
                raise RuntimeError(
                    "No sulfide checkpoint found. Set ORE_MCP_CHECKPOINT to a valid .pt file "
                    f"(default {DEFAULT_CHECKPOINT} does not exist)."
                )
            talc_checkpoint = _env_checkpoint("ORE_MCP_TALC_CHECKPOINT", DEFAULT_TALC_CHECKPOINT)
            self._pipeline = ResidentSulfidePipeline(
                checkpoint=checkpoint,
                device=self.device,
                talc_checkpoint=talc_checkpoint,
            )
        return self._pipeline

    @property
    def grade_model(self):
        """The grade classifier if a checkpoint is available, else None (loaded once)."""
        if not self._grade_loaded:
            self._grade_loaded = True
            grade_checkpoint = _env_checkpoint("ORE_MCP_GRADE_CHECKPOINT", DEFAULT_GRADE_CHECKPOINT)
            if grade_checkpoint is not None:
                from ore_classifier.grade_classifier import load_grade_model

                self._grade_model = load_grade_model(grade_checkpoint, device=self.device)
        return self._grade_model


_engine = _Engine()
mcp = FastMCP("ore-classifier")


def _read_json(path: str | None) -> Any:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@mcp.tool()
def classify_thin_section(image_path: str, out_dir: str | None = None) -> dict[str, Any]:
    """Classify an optical-microscopy ore thin section (шлиф).

    Runs the full in-process pipeline on a single image: tiled binary-sulfide
    segmentation -> analyzed area -> talc (ML model if configured, else color heuristic)
    -> deterministic ore-class component analysis, plus an optional learned ordinary/fine
    grade opinion. The model stays warm across calls.

    Args:
        image_path: Absolute (or repo-relative) path to the thin-section image (RGB).
        out_dir: Optional directory for run artifacts (masks, overlays, CSVs). Defaults to
            a fresh subdir under ORE_MCP_OUT_ROOT.

    Returns:
        A dict with: result_quality, degradations, checkpoint, talc_source, the inlined
        ore_summary (phase/class breakdown and metrics), the grade_branch opinion (or null),
        an artifacts map of output file paths, and the out_dir used.
    """
    src = Path(image_path).expanduser()
    if not src.is_absolute():
        src = ROOT / src
    if not src.exists():
        raise ValueError(f"image_path does not exist: {src}")

    if out_dir:
        run_dir = Path(out_dir).expanduser()
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
    else:
        run_dir = _engine.out_root / f"{src.stem}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = _engine.pipeline.run_image(src, run_dir)

    grade_branch = None
    grade_model = _engine.grade_model
    if grade_model is not None:
        from PIL import Image

        from ore_classifier.grade_classifier import predict_grade

        grade_branch = predict_grade(grade_model, Image.open(src))

    return {
        "image": str(src),
        "out_dir": str(run_dir),
        "result_quality": summary.get("result_quality"),
        "degradations": summary.get("degradations", []),
        "checkpoint": summary.get("checkpoint"),
        "talc_source": summary.get("talc_source"),
        "ore_summary": _read_json(summary.get("paths", {}).get("ore_summary")),
        "grade_branch": grade_branch,
        "artifacts": summary.get("paths", {}),
    }


@mcp.tool()
def get_config() -> dict[str, Any]:
    """Report the resolved model checkpoints, device, and output root for this server.

    Does not load the models — safe to call before the first classification to confirm
    which checkpoints will be used.
    """
    sulfide = _env_checkpoint("ORE_MCP_CHECKPOINT", DEFAULT_CHECKPOINT)
    talc = _env_checkpoint("ORE_MCP_TALC_CHECKPOINT", DEFAULT_TALC_CHECKPOINT)
    grade = _env_checkpoint("ORE_MCP_GRADE_CHECKPOINT", DEFAULT_GRADE_CHECKPOINT)
    return {
        "device": _engine.device,
        "out_root": str(_engine.out_root),
        "sulfide_checkpoint": str(sulfide) if sulfide else None,
        "talc_checkpoint": str(talc) if talc else None,
        "talc_backend": "ml_model" if talc else "heuristic",
        "grade_checkpoint": str(grade) if grade else None,
        "model_loaded": _engine._pipeline is not None,
    }


if __name__ == "__main__":
    mcp.run()
