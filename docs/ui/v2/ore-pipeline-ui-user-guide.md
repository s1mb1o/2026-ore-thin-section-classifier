# Ore Pipeline UI User Guide

Date: 2026-07-04

This document describes how the v2 ore pipeline browser UI works from the user's point of view. It covers `apps/ore_pipeline_web.py` and the direct-loadable UI routes served by that app.

## Purpose

The UI supports optical-microscopy thin-section analysis for the Nornickel task `Скажи мне, кто твой шлиф`:

```text
image upload
-> optional augmentation
-> optional preprocessing
-> binary sulfide/non-sulfide segmentation
-> ordinary/fine sulfide intergrowth classification
-> talc detection
-> metrics, report, files, and immutable history
```

The app is designed for high-resolution optical images, including panorama-scale sources. Original files are kept as input artifacts; browser display uses generated preview pyramids so the UI does not load huge images directly at full resolution.

## Navigation

The top navigation uses slug routes:

- `/workspace`: single-image work area.
- `/batch`: Series work area for processing several images sequentially.
- `/history`: run history.
- `/history_series`: Series history mode.
- `/settings`: persistent system-wide settings.
- `/status`: system status and logs.
- `/api`: REST API reference and live sandbox.

`/` redirects to `/workspace`. The UI has persistent language and theme selectors. Russian is the default language, and English translations should cover all user-facing text.

## Workspace Layout

The Workspace page has a left input/sidebar and a main image/result area.

The left sidebar contains:

- input image drop zone;
- metadata editor button after image upload;
- augmentation controls;
- preprocessing controls;
- run controls;
- compact recent history cards.

The main area contains:

- layer selector and side-by-side selector;
- image canvas with pan, zoom, segmentation legends, and draggable split-view divider;
- viewer options row below the image;
- text conclusion, metrics, sulfide-grain table, technical run details, and export buttons after a run completes.

## Image Upload

The drop zone accepts click-to-open and drag/drop. Supported file classes are PNG, JPEG, TIFF, and common RAW-extension files. Unsupported files show an inline warning and do not start an upload.

After an image is selected, the drop zone shows a thumbnail, filename, dimensions, and an `x` clear button. Clearing the image resets upload, run, result, progress, view, and edit state while keeping persisted history on disk.

Large images show progress only in the input panel during upload and preview preparation. The Run progress bar is reserved for `Start` and active run processing.

## Metadata

`Edit Metadata...` opens a modal with three tabs:

- Domain: curated project, microscope/camera, objective, scale, sample, operator, and review metadata.
- Raw: read-only upload/header metadata.
- Session Defaults: reusable defaults for repeated work.

Scale metadata is conservative. Physical area appears in the result table and CSV only when a positive pixel size is supplied with calibrated confidence.

Saved metadata is included in the next run request and written into the immutable run under `metadata/curated_metadata.json`. Derived edit runs inherit parent metadata.

## Augmentation

Augmentation is controlled by:

```text
[ ] Augmentation [Edit] [Apply]
```

`Edit` opens color/tone, acquisition-noise, and grinding/polishing artifact parameters. `Apply` checks the box, saves the settings, refreshes preview artifacts, and switches the viewer to the augmented layer when available.

If no completed run is loaded, `Apply` updates the current upload/pre-run state. If a completed run is loaded, `Apply` creates or updates a new immutable `prepared` run with prerequisites rebuilt up to the augmentation step and downstream masks/metrics cleared.

## Preprocessing

Preprocessing is controlled by:

```text
[x] Preprocessing [Edit] [Apply]
```

`Edit` opens illumination normalization, denoising, contrast correction, and panorama scaling settings. Panorama scaling is explicit: either a longest-side bound or scale factor.

If preprocessing is unchecked, `Start` skips preprocessing and the preprocessed view/side-by-side option stays disabled. If preprocessing is checked, the run records the exact preset used. For large images, the UI keeps original source artifacts and uses analysis/display-scale images for browser work.

## Run Controls

`Start` creates or continues an immutable run. While a run is queued/running, `Start` is replaced by a red `Stop` button. `Stop` requests cooperative cancellation and terminates ML subprocesses when applicable.

If `Start` is pressed while a previous completed run is loaded, stale text, metrics, exports, sulfide/final layers, side-by-side comparison, and selected grain outlines are cleared immediately before the new run result arrives.

Run progress includes percent, current stage, ETA when available, and elapsed time. ML tiled inference can report tile progress.

## Viewer

The primary layer selector is:

```text
original | augmented | preprocessed | sulfide | final
```

The side-by-side selector is:

```text
Side-by-side: none | augmented | preprocessed | sulfide | final
```

Unavailable layers are greyed out. Side-by-side shows the selected comparison layer on the right with a draggable vertical splitter. The viewer supports pan and zoom.

Segmentation class legends appear over the image, not in the toolbar:

- image-only layers hide class controls;
- sulfide layer shows sulfides, non-sulfides, and artefacts;
- final layer shows ordinary, fine, talc, artefacts, and background;
- when side-by-side is active, left and right legends are shown independently at the top-left and top-right.

Viewer-level options below the image are:

- show tiling;
- contours only;
- opacity.

## Results

After completion, the result area appears below the image.

The text output states the ore classification, talc fraction, and dominant intergrowth type. The rationale line shows denominator-aware shares and margins.

The metrics table is hierarchical:

```text
Analyzed area fraction
- Total sulfide fraction
-- Ordinary intergrowth fraction
-- Fine intergrowth fraction
- Talc fraction
- Other analyzed area
Image artefact fraction
```

Rows with area semantics include pixel area and, when calibrated scale is available, physical area.

The sulfide-grain table appears after metrics. Each row represents a classified sulfide connected component from `reports/component_features.csv` and shows:

- checkbox for viewer outline;
- component ID;
- ordinary or fine intergrowth type;
- pixel area;
- percent share of total sulfide area.

Checking one or more grains draws one combined cyan outline over the current image view. This is visual-only and does not modify masks, metrics, or run artifacts.

The technical details widget appears after the sulfide-grain table. It summarizes run provenance from `run.json` and `reports/runtime.json`:

- run id, status, stage, timestamps, and elapsed processing time;
- effective backend and model/rule sources for sulfide, talc, and final segmentation;
- model checkpoints only for ML-backed stages, not for heuristic or rule-only stages;
- analysis dimensions, tile count, tile size, stride, and ML tile progress when available;
- stage outputs returned by the sulfide, talc, final segmentation, and artefact paths;
- masks, reports, runtime provenance file, and grain CSV artifacts present in the immutable run.

Older runs that do not contain a field show `n/a` instead of inventing a value.

## Exports

The result export row contains:

- Save CSV: downloads hierarchical metrics with pixel and optional physical areas.
- Save PDF Report: downloads the current five-page lab-style demonstration report.
- View files: opens a file browser for the immutable run.

The file browser lists all run files with sortable filename, type, size, and image-dimension columns. Image files show `WxH`. Each row has a `View` action backed by `/artifacts/...`, and `Download ZIP` downloads the entire run directory.

## Edit And Recalculate

`Fix me` is available after image upload, even before `Start`.

Before a completed run exists, the editor supports artefact masking only. Artefact regions mark grinding/polishing defects to exclude from later segmentation and metric denominators.

After a run completes, the editor supports:

- Artefacts;
- sulfide/non-sulfide;
- final segmentation.

The editor has Brush/Pan, Undo/Redo, Zoom in/out, Fit view, brush size, comment, and live statistics. Brush left-draws and right-erases. Artefacts use the same violet/magenta color as the main viewer.

`Fix and Restart` always creates a new immutable run:

- artefact or sulfide edits rerun downstream final segmentation and metrics;
- final segmentation edits recalculate metrics/report without replacing the parent sulfide mask.

## History

The left sidebar shows compact recent runs with thumbnail, `Load`, filename, date, run id, and conclusion text.

The History page has three modes:

- all runs;
- standalone runs only;
- Series.

Run history is table-based. Progress is text-only percent with optional muted details for status, stage, tiles, ETA, and elapsed time. `Load` restores a run into the Workspace for review/tuning. `Remove` deletes a selected completed/failed run artifact from history while keeping uploads.

Clicking a thumbnail opens a preview popup.

## Series

The Series page supports grouped multi-image work. `Add images` lives in the Gallery section and accepts multiple files. Draft cards have `Edit Metadata...` and `Remove`. `Run Series` processes items sequentially with shared augmentation/preprocessing/runtime settings.

Each completed card has `Load`, which opens the ordinary run result with `Back to Series` navigation. Series history is available via `/history_series` and opens persisted Series detail pages through `/batch/{batch_id}`.

## Settings

Settings are server-backed and persisted under `outputs/ore_pipeline_ui/settings/app_settings.json`.

The Settings page controls:

- default language;
- theme;
- binary sulfide backend/checkpoint;
- talc source/checkpoint/threshold;
- preprocessing defaults;
- default tiling overlay;
- repeated session metadata defaults;
- history removal.

Runtime changes apply to new runs, validate checkpoint paths, and are blocked while a run or Series job is active. The Runtime `Test` button probes unsaved settings without saving them or creating a run.

`Remove all history` deletes persisted run and Series artifact folders while preserving uploads and app settings; it rejects active jobs.

## Status And API

The Status page shows health, CPU, GPU when available, RAM, Flash, history size, run/Series counts, backend/model state, active jobs, bounded system events, and access logs.

The API page documents service endpoints and provides live sandboxes for status, upload, preprocessing, runs, artifacts, Series, and settings. Binary downloads show status/content-type/size/disposition instead of raw bytes.

## Artifact Model

Original upload artifacts live under:

```text
outputs/ore_pipeline_ui/uploads/
```

Immutable run artifacts live under:

```text
outputs/ore_pipeline_ui/runs/
```

A completed run typically contains:

- `run.json`;
- `input/` original, augmented, and preprocessed artifacts;
- `masks/` sulfide, final, talc, analyzed, artefact, and component-label masks;
- `display/` preview pyramids;
- `reports/` summary, metrics CSV, runtime provenance, PDF, and ZIP artifacts.

Run metadata should be treated as immutable after completion. Corrections produce derived runs.
