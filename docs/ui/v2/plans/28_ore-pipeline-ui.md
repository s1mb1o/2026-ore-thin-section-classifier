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
   - Runtime augmentation is implemented for this v2 UI as `[ ] Augmentation [Edit]`
     in the input/left panel; grouped settings persist between browser runs and
     enabled augmentation is applied before preprocessing.

4. Add metadata editing.
   - `Edit Metadata...` opens a modal for the selected image.
   - Domain metadata is editable; raw image/header/EXIF metadata is read-only.
   - Session defaults live in browser localStorage for repeated operator fields.
   - Saved metadata is sent as `curated_metadata` to `POST /api/runs/start`.
   - Runs write `metadata/curated_metadata.json` and edit-derived runs inherit
     parent metadata.
   - Detailed contract: `docs/ui/v2/specs/ore-pipeline-metadata-editor-v0.1.md`.
   - Hardening plan: `docs/ui/v2/plans/30_ore-pipeline-metadata-editor.md`.

5. Add run creation and progress.
   - `Start` creates a new immutable run directory under `outputs/ore_pipeline_ui/runs/`.
   - Run metadata records original input, preprocess preset, preprocessed artifact, backend, stage, progress, ETA, and elapsed runtime.
   - Default local backend is heuristic for smoke/demo reliability; optional `--backend ml --checkpoint ...` calls the existing ML pipeline.

6. Add result visualization.
   - Switch views: original, preprocessed, sulfide/non-sulfide, final, side-by-side.
   - Side-by-side uses a draggable splitter.
   - Final layer has class visibility toggles: background, ordinary intergrowth, fine intergrowth, talc.
   - Viewer supports pan and zoom.

7. Add metrics and exports.
   - Metrics table includes total sulfide fraction, ordinary/fine intergrowth fractions, talc fraction, component count, and analyzed area.
   - The result panel also lists each classified sulfide grain from `component_features.csv`
     with ordinary/fine type, pixel area, sulfide-area share, and a checkbox
     that strokes the selected grain union on the viewer through the generated
     RGB component label-map artifact.
   - After the grain table, the result panel shows a technical details widget
     from `run.json` and `reports/runtime.json`: backend/model provenance, ML
     checkpoints only for ML stages, tile dimensions/progress, elapsed time,
     stage results, and present run artifacts.
   - Scale-aware extension: `docs/ui/v2/specs/ore-pipeline-scale-metrics-v0.1.md`
     and `docs/ui/v2/plans/33_ore-pipeline-scale-metrics.md` define pixel areas,
     calibrated physical areas, and `microns_per_pixel` / `pixel_size_um`
     handling for the result table and `metrics.csv`.
   - `Save to CSV` exports run metrics, pixel areas, and calibrated physical
     areas when scale is available.
   - `Save PDF Report` generates a compact local PDF report.
   - Text output follows the requested Russian conclusion style.

8. Add edit-and-recalculate.
   - `Fix me` opens a mask editor with brush painting.
   - Editable layers: sulfide/non-sulfide and final segmentation.
   - Changes require a comment and create a new run.
   - Sulfide edits copy prerequisites, replace the sulfide mask from the edit, and rerun final segmentation/metrics.
   - Final edits copy prerequisites and sulfide mask, replace final masks, and recalculate metrics/report only.
   - Parent run remains immutable.

9. Add History page.
   - List all runs from the workspace.
   - Show parent/edit lineage, text-only progress percent/elapsed time, and open past runs.

## Implemented V2 Runtime Augmentation

This augmentation work targets this v2 ore pipeline UI directly:

```text
apps/ore_pipeline_web.py
src/ore_classifier/augmentation.py
```

Required UI placement:

```text
Input image
#########
#########

[ ] Augmentation [Edit]
```

Required pipeline order:

```text
original upload -> augmentation -> preprocessing -> run artifact -> sulfide/final inference
```

When the checkbox is enabled for a run, preprocessing consumes the augmented
image. The run metadata records the exact augmentation settings and the stored
augmented artifact.

Required viewer order:

```text
original -> augmented -> preprocessed -> sulfide -> final
```

The `augmented` image type appears only for runs that actually created an
augmented artifact, and it sits between `original` and `preprocessed` for
pipeline debugging. The grouped settings popup uses the same
`ore-pipeline-augmentation-v0.1` schema exposed by `src/ore_classifier/augmentation.py`
so runtime debugging and model-training augmentation can stay reproducible.
The implemented groups cover color/tone, acquisition noise, and domain-specific
grinding/polishing artifacts: scratches, polishing haze, and pits/dust specks.

## Done Criteria

- New UI app starts with `python3 apps/ore_pipeline_web.py --host 127.0.0.1 --port 0`.
- Focused tests cover upload, runtime augmentation before preprocessing, preprocessing, complete run, CSV/PDF exports, sulfide edit rerun, final edit recalculation, history, and required UI controls.
- `COMMANDS.md`, `SMOKE_TESTS.md`, `apps/README.md`, `ChangeLog.md`, and `docs/session-sync.md` mention the new UI.
