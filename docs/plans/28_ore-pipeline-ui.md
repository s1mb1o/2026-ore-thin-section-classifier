# Ore Pipeline UI Plan

Date: 2026-07-03

## Goal

Create a local browser UI for the official OM-only ore-classifier pipeline:

```text
image upload
-> display-scaled original preview
-> preprocessing preview
-> immutable run artifact
-> sulfide/non-sulfide mask
-> ordinary/fine intergrowth plus talc mask
-> metrics, text conclusion, CSV/PDF exports
-> edit-and-recalculate derived runs
-> run history
```

## Approaches Considered

1. Extend the current Streamlit QA page.
   - Pro: fastest form/table work.
   - Con: weak fit for draggable splitters, mask painting, side-by-side canvas, and large-image preview pyramids.

2. Build a new local `http.server` + vanilla JS/canvas app.
   - Pro: matches the implemented talc browser-review pattern, supports drag/drop, pan/zoom, splitters, client-side mask editing, and immutable artifact endpoints.
   - Con: more frontend code to maintain than Streamlit.

3. Keep CLI-only inference plus separate review tools.
   - Pro: no new UI complexity.
   - Con: does not satisfy the requested drop-zone, preview, progress, edit/recalculate, export, and history workflow.

Decision: use approach 2.

## Implementation Plan

1. Add `apps/ore_pipeline_web.py`.
   - Local HTTP app with generated HTML/CSS/vanilla JS.
   - Upload endpoint accepts PNG, JPEG, TIFF, and RAW extensions.
   - RAW decode uses optional `rawpy`; PNG/JPEG/TIFF use Pillow.
   - Store uploads under `outputs/ore_pipeline_ui/uploads/`.

2. Add large-image display handling.
   - Keep original image as the immutable source artifact.
   - Build display preview pyramids for original, preprocessed, and mask overlay layers.
   - Use lower-scale previews by default and switch to less-downscaled previews during zoom.

3. Add preprocessing controls.
   - `нормализация освещения`
   - `шумоподавление`
   - `коррекция контраста`
   - `масштабирование для панорамных снимков`
   - Store preset metadata and the preprocessed analysis image.

4. Add run creation and progress.
   - `Start` creates a new immutable run directory under `outputs/ore_pipeline_ui/runs/`.
   - Run metadata records original input, preprocess preset, preprocessed artifact, backend, stage, progress, and ETA.
   - Default local backend is heuristic for smoke/demo reliability; optional `--backend ml --checkpoint ...` calls the existing ML pipeline.

5. Add result visualization.
   - Switch views: original, preprocessed, sulfide/non-sulfide, final, side-by-side.
   - Side-by-side uses a draggable splitter.
   - Final layer has class visibility toggles: background, ordinary intergrowth, fine intergrowth, talc.
   - Viewer supports pan and zoom.

6. Add metrics and exports.
   - Metrics table includes total sulfide fraction, ordinary/fine intergrowth fractions, talc fraction, component count, and analyzed area.
   - `Save to CSV` exports run metrics.
   - `Save PDF Report` generates a compact local PDF report.
   - Text output follows the requested Russian conclusion style.

7. Add edit-and-recalculate.
   - `Fix me` opens a mask editor with brush painting.
   - Editable layers: sulfide/non-sulfide and final segmentation.
   - Changes require a comment and create a new run.
   - Sulfide edits copy prerequisites, replace the sulfide mask from the edit, and rerun final segmentation/metrics.
   - Final edits copy prerequisites and sulfide mask, replace final masks, and recalculate metrics/report only.
   - Parent run remains immutable.

8. Add History page.
   - List all runs from the workspace.
   - Show parent/edit lineage and open past runs.

## Done Criteria

- New UI app starts with `python3 apps/ore_pipeline_web.py --host 127.0.0.1 --port 0`.
- Focused tests cover upload, preprocessing, complete run, CSV/PDF exports, sulfide edit rerun, final edit recalculation, history, and required UI controls.
- `COMMANDS.md`, `SMOKE_TESTS.md`, `apps/README.md`, `ChangeLog.md`, and `docs/session-sync.md` mention the new UI.
