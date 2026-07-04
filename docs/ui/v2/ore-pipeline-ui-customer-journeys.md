# Ore Pipeline UI Customer Journeys

Date: 2026-07-04

This document describes practical use cases for the v2 ore pipeline UI. It is written as customer journeys rather than implementation notes.

## Actors

- Lab operator: prepares images, enters metadata, starts analysis, exports results.
- Geologist or expert reviewer: checks masks, marks artefacts, corrects segmentation, and interprets grain-level outputs.
- Process engineer or supervisor: reviews history, compares runs, and collects evidence for a decision.
- System administrator: configures runtime/backend defaults, monitors health, and manages stored history.
- Demo presenter: runs a stable walkthrough for stakeholders or jury review.

## Journey 1: Single Image From Upload To Report

Goal: classify one optical thin-section image and export evidence.

1. Open `/workspace`.
2. Drop or select a PNG/JPEG/TIFF/RAW-extension image.
3. Confirm the thumbnail, filename, and dimensions are shown.
4. Open `Edit Metadata...` and enter sample, microscope, and scale data if available.
5. Keep or adjust Augmentation and Preprocessing settings.
6. Press `Start`.
7. Watch stage/progress/elapsed time until completion.
8. Review the text conclusion and rationale.
9. Review the hierarchical metrics table.
10. Open the sulfide-grain card and check grains of interest to outline them on the viewer.
11. Review the technical details card below the grain table: backend/model source, tiles, elapsed time, stage outputs, and artifact paths.
12. Switch between original, preprocessed, sulfide, final, and side-by-side views.
13. Save CSV, PDF report, or open `View files` and download the ZIP.

Success criteria:

- the run reaches `complete`;
- result layers are available;
- metrics and grain rows are populated;
- technical details show runtime provenance without heuristic checkpoint noise;
- exported files are downloadable;
- the run appears in History.

## Journey 2: Large Panorama Analysis

Goal: process a panorama or very large image without blocking the browser on full-resolution display.

1. Open `/workspace` and upload the panorama.
2. Verify upload progress appears in the input panel, not in the Run progress bar.
3. Keep panorama scaling enabled or configure the longest-side bound/factor in Preprocessing.
4. Press `Apply` for Preprocessing if a preview refresh is needed.
5. Press `Start`.
6. Use `show tiling` below the viewer to inspect analysis tiles.
7. Use pan/zoom and side-by-side comparison to inspect original/preprocessed/sulfide/final layers.
8. Confirm the technical details widget reports analysis dimensions and tile count/progress.
9. Export the run files or report after completion.

Success criteria:

- browser remains responsive;
- preview layers use display-scale images;
- tiling overlay is available when a tile manifest exists;
- the output records the analysis dimensions, tile counts, operation time, and preprocessing settings.

## Journey 3: Mark Artefacts Before Running

Goal: exclude grinding/polishing defects before segmentation starts.

1. Upload an image.
2. Press `Fix me` before `Start`.
3. Use the Artefacts layer in the editor.
4. Brush over polishing scratches, dust, or grinding defects.
5. Add a comment if needed.
6. Save artefacts or use the editor action that prepares the next run.
7. Press `Start` from the Workspace.
8. Confirm artefact regions are excluded from sulfide/final overlays and metric denominators.

Success criteria:

- artefact regions use the violet/magenta artefact color;
- marked pixels are excluded in the next run;
- the original upload remains unchanged.

## Journey 4: Expert Fix And Recalculate After A Run

Goal: correct a segmentation error without mutating the completed parent run.

1. Load or complete a run.
2. Press `Fix me`.
3. Choose Artefacts, sulfide/non-sulfide, or final segmentation.
4. Use Brush for draw/erase, Pan for navigation, Undo/Redo for correction control, and Fit/Zoom for inspection.
5. Add a change comment.
6. Press `Fix and Restart`.
7. Review the newly created derived run.

Expected behavior:

- artefact or sulfide edits rerun downstream final segmentation and metrics;
- final segmentation edits recalculate metrics/report only;
- every correction creates a new immutable run;
- parent run artifacts remain available in History.

## Journey 5: Grain-Level Review Of Sulfide Components

Goal: inspect which sulfide grains drive ordinary/fine intergrowth metrics.

1. Complete or load a run.
2. Scroll to the sulfide-grain card below metrics.
3. Sort visually by type, area, or share using the table values.
4. Check one or more grain rows.
5. Inspect the combined outline on the viewer.
6. Switch between sulfide, final, original, and side-by-side views while keeping the outline visible.
7. Use the final segmentation class checkboxes to focus on ordinary/fine/talc/background overlays.

Success criteria:

- each row has component ID, type, area, and share;
- multiple checked grains render as one union outline;
- unchecking rows removes them from the outline;
- no mask or metric data is changed by checkbox selection.

## Journey 6: Compare Original Against Result Layers

Goal: visually validate model output against image evidence.

1. Complete or load a run.
2. Set primary view to `original`, `preprocessed`, `sulfide`, or `final`.
3. Set Side-by-side to a comparison layer.
4. Drag the splitter left/right.
5. Toggle class visibility in the left/right overlay legends.
6. Adjust opacity and optionally enable `contours only`.
7. Pan/zoom to inspect uncertain regions.

Success criteria:

- unavailable layers stay disabled;
- splitter remains visible above overlays;
- class controls show only for segmentation layers;
- left and right class legends control the selected visible side.

## Journey 7: Process A Series Of Images

Goal: process multiple images with shared settings.

1. Open `/batch`.
2. Use `Add images` in the Gallery panel.
3. For each draft card, edit metadata if needed.
4. Remove mistaken draft images before running.
5. Confirm shared augmentation/preprocessing settings.
6. Press `Run Series`.
7. Watch cards progress sequentially.
8. Load any child run to inspect results.
9. Return to the Series page with `Back to Series`.
10. Download `batch_results.csv` when complete.

Success criteria:

- only one item runs at a time;
- completed cards keep `100%`;
- child runs appear as immutable ordinary runs;
- Series history opens through `/history_series`.

## Journey 8: History Review And Reuse

Goal: reopen old work, compare outputs, or clean history.

1. Open `/history`.
2. Choose all runs, standalone runs, or Series.
3. Use thumbnails to preview runs quickly.
4. Press `Load` to reopen a run in Workspace.
5. Tune parameters and press `Start` to create a new run if needed.
6. Use `Remove` for obsolete completed/failed runs.
7. Open Series entries from `/history_series` for grouped details.

Success criteria:

- progress is shown as percent text, not a bar;
- standalone mode excludes Series child runs;
- `Load` restores enough input state for tuning;
- removing a run does not delete uploaded source images.

## Journey 9: Configure Runtime And Verify ML Readiness

Goal: choose heuristic or ML backend safely.

1. Open `/settings`.
2. Set binary sulfide runtime backend/checkpoint.
3. Set talc source/checkpoint/threshold if ML talc is used.
4. Press Runtime `Test`.
5. Save settings only after the test result is acceptable.
6. Return to Workspace and start new runs.

Success criteria:

- heuristic test returns immediately;
- ML test checks selected checkpoint loading without creating a run;
- missing checkpoints are rejected;
- changes are blocked while active jobs are running;
- completed runs record backend and checkpoint provenance.

## Journey 10: Operator Checks System Health

Goal: verify the service is ready before a demo or processing session.

1. Open `/status`.
2. Press `Refresh`.
3. Check health, CPU, RAM, Flash, GPU, backend/model state, active jobs, history size, and storage rows.
4. Review System log and Access log for recent failures.
5. If storage is too large, use Settings history removal after exporting needed runs.

Success criteria:

- status loads without the workflow sidebar;
- GPU is either reported or clearly not detected;
- active jobs are visible;
- logs are bounded and sanitized.

## Journey 11: API-Oriented Integration Smoke

Goal: confirm the browser service can be driven by scripts.

1. Open `/api`.
2. Run `GET /api/status` in the sandbox.
3. Use upload sandbox with multipart field `file`.
4. Run preprocessing for the upload.
5. Start a run and poll `GET /api/runs/{run_id}`.
6. Download metrics, PDF, files list, or ZIP.

Success criteria:

- JSON endpoints show readable request/response bodies;
- binary downloads show status/content type/size/disposition;
- `GET /api/runs/{run_id}` includes masks, display previews, metrics, sulfide-grain rows, and downloads.

## Journey 12: Customer Demo Flow

Goal: present the system in a controlled stakeholder demo.

1. Start on `/workspace` in Russian dark mode.
2. Load a prepared representative image or use a pre-generated History run.
3. Show upload metadata and preprocessing/augmentation settings.
4. Show layer switching and side-by-side comparison.
5. Show final text conclusion and hierarchical metrics.
6. Check several sulfide-grain rows to demonstrate component-level explainability.
7. Show the technical details card to explain what backend/model paths ran, how many tiles were processed, operation time, and returned stage results.
8. Open `View files` and show immutable artifacts.
9. Download the PDF report.
10. Open `/status` to show deployment/runtime health.
11. Open `/history` to show auditability.

Success criteria:

- no live long-running panorama processing is required unless already smoke-tested;
- all shown claims match the actual run output;
- exported evidence is available during the demo.

## Common Failure Paths

Unsupported file format:

- User drops an unsupported file.
- UI shows a warning and does not start upload.

Large-image preparation:

- User uploads a large panorama.
- Input-panel progress shows upload/preview preparation.
- Run progress remains idle until `Start`.

Runtime unavailable:

- ML checkpoint is missing or unloadable.
- Runtime `Test` reports the failure before a production run is started.

Correction after completion:

- User edits a completed run.
- The app creates a derived run, preserving the parent run.

History cleanup:

- User tries to remove all history during an active job.
- The app rejects the action until jobs are terminal.
