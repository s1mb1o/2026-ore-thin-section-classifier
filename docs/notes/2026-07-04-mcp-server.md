# MCP server for the ore classifier (Option A — resident model)

**Date:** 2026-07-04
**File:** `apps/ore_mcp_server.py`
**Status:** working, smoke-tested locally (SegFormer-B2 sulfide + SegFormer-B0 talc + EfficientNet-B3 grade).

## What it is

A [Model Context Protocol](https://modelcontextprotocol.io) server (stdio transport, built on
the `mcp` Python SDK's `FastMCP`) that exposes the in-process ore pipeline as MCP tools. An
assistant (Claude Code, Claude Desktop, any MCP client) can classify an optical-microscopy ore
thin section (шлиф) and get the sulfide phases, talc source, ore-class metrics, an optional
learned ordinary/fine grade opinion, and artifact paths back inline.

This is **Option A** from the design discussion: the MCP tool wraps
`ResidentSulfidePipeline.run_image` directly and holds the model **warm** in-process, rather than
shelling out to `scripts/run_ore_pipeline.py` per image (which would reload the checkpoint every
call — the cold-start that would bite a live demo).

## Tools

- `classify_thin_section(image_path, out_dir=None) -> dict` — full pipeline on one image. Returns
  `result_quality`, `degradations`, `checkpoint`, `talc_source`, the inlined `ore_summary`
  (phase/class breakdown + metrics), the `grade_branch` opinion (or null), an `artifacts` map of
  output file paths (masks, overlays, CSVs), and the `out_dir` used.
- `get_config() -> dict` — reports the resolved checkpoints, device, and output root **without**
  loading models. Safe to call first to confirm which weights will be used.

Model loading is **lazy**: the first `classify_thin_section` call loads the checkpoint(s); every
later call reuses the warm model. `get_config` never triggers a load.

## Configuration (env vars, all optional)

Defaults mirror `apps/ore_pipeline_web.py` so the MCP tool and the web UI agree on models.

| Var | Meaning | Default |
| --- | --- | --- |
| `ORE_MCP_CHECKPOINT` | binary-sulfide segmentation `.pt` | `models/binary_sulfide/segformer_b2_.../best.pt` |
| `ORE_MCP_TALC_CHECKPOINT` | talc segmentation `.pt` (heuristic if unset/missing) | `outputs/talc_segformer_folds/.../best.pt` |
| `ORE_MCP_GRADE_CHECKPOINT` | grade-classifier `.pt` (branch skipped if unset/missing) | `models/grade_classifier/effb3_ordfine_ppaug_.../best.pt` |
| `ORE_MCP_DEVICE` | torch device | `auto` |
| `ORE_MCP_OUT_ROOT` | base dir for run artifacts | `outputs/mcp_runs` (gitignored) |

## Run / register

```bash
# stdio server (for manual test, exits on EOF):
.venv/bin/python apps/ore_mcp_server.py

# register with Claude Code (absolute paths required):
claude mcp add ore-classifier -- \
    /abs/path/2026_Nornikel_Hackaton_v2/.venv/bin/python \
    /abs/path/2026_Nornikel_Hackaton_v2/apps/ore_mcp_server.py
```

Dependency: `mcp>=1.2.0` (added to `requirements.txt`; installed into `.venv`).

## Verified

- `tests/test_ore_mcp_server.py` — tool registration, `get_config` without model load, bad-path
  guard runs before any checkpoint touch. 3 passed.
- Manual end-to-end: `classify_thin_section` on
  `data/external/lumenstone/full/S2_v2/.../test_01.jpg` returned `ore_class=talcose_ore`,
  `talc_source=ml_model`, a grade branch (`ordinary_intergrowth`, conf ≈1.0), and wrote the full
  artifact set under `outputs/mcp_runs/`. Second `get_config` confirmed `model_loaded=true` (warm).

## Not covered (deliberate scope)

- No image-**bytes**/base64 input yet — takes a file **path** (simplest; right for a local demo).
  A remote MCP client would need a bytes variant.
- No parity with the web app's upload preprocessing, artifact-mask exclusion, batch runs, or run
  history — those live in `OrePipelineStore` and are out of scope for a thin classifier tool.
