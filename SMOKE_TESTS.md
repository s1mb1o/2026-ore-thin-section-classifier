# Smoke Tests

## Current Structural Checks

Run from the v2 root:

```bash
test -L dataset
python3 - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("dataset/_download_manifest.json").read_text())
assert manifest["download_status"] == "complete"
assert manifest["file_count"] == 1236
assert manifest["local_verified_count"] == 1236
assert manifest["local_verified_size_bytes"] == 3018194503
print("dataset manifest ok")
PY
```

## Ore Pipeline UI Smoke

Run from the v2 root:

```bash
python3 -m py_compile apps/ore_pipeline_web.py
python3 -m unittest discover -s tests -p 'test_ore_pipeline_web.py' -v
```

```bash
python3 apps/ore_pipeline_web.py \
  --host 127.0.0.1 \
  --port 0
```

Expected:

- Focused tests pass for upload, repeated same-image upload ID allocation, runtime augmentation before preprocessing, preprocessing, complete run artifacts, metrics CSV, PDF report, run file listing/ZIP download, batch sequential runs with per-image metadata, sulfide edit rerun, final segmentation edit recalculation, history, Status diagnostics, API documentation/sandboxes, and required UI controls.
- The app prints a local URL such as `http://127.0.0.1:<port>/`.
- The top navigation visibly includes `Workspace` / `Рабочее место`, `Series` / `Серии`, `History` / `История`, `Status` / `Статус`, `API`, and `Settings` / `Настройки`.
- `/` redirects to `/workspace`; `/workspace`, `/batch`, `/batch/{batch_id}`, `/history`, `/history_series`, `/settings`, `/status`, and `/api` can be opened directly, tab clicks update the URL, selecting Series mode on History pushes `/history_series`, and browser back/forward restores the matching page/history mode.
- The Settings page saves system-wide defaults through `/api/settings` into `outputs/ore_pipeline_ui/settings/app_settings.json`; changing language/theme/runtime backend/checkpoint/default preprocessing/show-tiling/session metadata defaults, saving, and reloading the app keeps those defaults. Runtime backend/checkpoint changes apply immediately to new runs, reject missing ML checkpoints, and are blocked while a run or Series job is active. The History panel has a red `Remove all history` action that confirms, calls `DELETE /api/history`, removes saved runs and Series while keeping uploads/settings, and rejects active jobs. In `Предобработка по умолчанию`, panorama scaling is a separate group below a divider, with normalization, denoise, and contrast remaining above it.
- The Settings page Runtime `Test` button checks the currently selected backend/checkpoint before saving. Heuristic should report success immediately. ML should validate the checkpoint path and report whether the model loader can load it, without creating a run or changing saved settings.
- The Status page opens through `Status` / `Статус` and `/status`, hides the workflow sidebar, and `Refresh` calls `/api/status`. The page shows overall health, CPU load, GPU utilization when `nvidia-smi` sees an NVIDIA GPU or a clear not-detected state otherwise, RAM, Flash/disk usage, backend/checkpoint state, uptime, history size, run/series counts, active jobs, health checks, storage rows for runs/series/uploads/workspace, plus bounded System log and Access log panels.
- The API page opens through `API` and `/api`, hides the workflow sidebar, and shows grouped endpoint navigation, method/path cards, localized request/response examples, and live sandboxes for health, upload, preprocessing, runs/jobs, artifacts, Series, and settings. `GET /api/status` should return JSON in the sandbox; `POST /api/uploads` should require a selected file and send it as multipart field `file`; PDF/ZIP sandboxes should show status, content type, size, and disposition instead of raw binary.
- The `Theme` selector supports `System`, `Light`, and `Dark`; explicit dark/light choices update panels, controls, toolbar, modal, history rows, and viewer backgrounds and persist across reloads.
- The `Language` selector defaults to `Русский`; the Russian UI labels are visible on first load, and switching to English updates controls, preprocessing labels, result/history summaries, metrics, and editor labels without clearing the current run/history state.
- The `Series` / `Серии` tab opens the v2 multi-image page. The `Gallery` panel contains `Add images`, which accepts multiple supported files and creates a persisted series gallery; each draft card has `Edit Metadata...` and `Remove`, and the page shows the current shared preprocessing/augmentation settings from the left panel. The technical route remains `/batch`.
- `Run Series` processes the gallery one image at a time. Only the active card advances its progress at a time, completed cards keep progress at 100%, and the finished series exposes `outputs/ore_pipeline_ui/batches/{batch_id}/reports/batch_results.csv`.
- Each processed Series card keeps `Load`; loading opens the ordinary run results view, shows `Back to Series`, and browser Back or that button returns to `/batch/{batch_id}`.
- The drop zone accepts click-to-upload and drag/drop for PNG, JPEG, TIFF, and RAW-extension files.
- Dropping or selecting an unsupported file extension shows an inline warning in the upload panel and does not start an upload; supported extensions remain PNG, JPEG, TIFF, and RAW-camera variants.
- Large uploads show progress only in the input panel: byte-transfer progress during upload, then a preview-preparation phase while the server decodes and scales display pyramids. The Run progress bar stays at idle/zero until `Start` creates an active run. Manual preprocessing preview updates also use only the input-panel preparation progress.
- After image selection, the drop zone shows a thumbnail, filename, dimensions, and an `×` clear button; clearing resets upload/run/result/progress/view state and returns to `Waiting for image`.
- After image selection, `Редактировать метаданные...` / `Edit Metadata...` is enabled. The popup has Domain, Raw, and Session Defaults tabs; Domain starts with `Session specific` fields such as Project and Microscope/camera, then `Sample specific` fields; Raw metadata is read-only; session defaults persist in browser local storage; `Scale value, um/px` is the manual scale-value input; setting a scale value without calibrated confidence shows a warning; `Exclude this image from training/validation sets` is clearly scoped to dataset exports; pressing `Save metadata` includes `curated_metadata` in the next `Start` request and writes `metadata/curated_metadata.json` under the immutable run.
- Augmentation is shown in the input/left panel as compact `[ ] Augmentation [Edit] [Apply]`. `Edit` opens grouped color/tone, acquisition-noise, and grinding/polishing artifact settings; the artifact group exposes scratches, scratch intensity, polishing haze, pits/dust specks, and pit/dust intensity. The saved settings persist in browser local storage between runs. `Apply` works like preprocessing apply: it checks augmentation, saves the settings, refreshes the upload debug previews, and switches to the `augmented` viewer layer. If no completed run is loaded, `Apply` keeps working on the current upload. If a completed run is loaded, `Apply` creates a new immutable `prepared` run with original + updated augmentation/preprocessing artifacts and no downstream masks/metrics yet; pressing `Start` continues that prepared run in place. Re-applying before `Start` updates the same prepared run rather than creating another one. When checked, `Apply` and `Start` create a deterministic geometry-preserving augmented image before preprocessing, and the immutable run records the exact settings plus `input/augmented.png`.
- For large images, augmentation and preprocessing are applied at the original source dimensions first. The UI then creates an analysis/display-scale `preprocessed.png` according to panorama scaling. With preprocessing enabled, upload metadata and immutable runs keep a full-size `preprocessed_full.png` artifact plus the smaller analysis image.
- Preview supports primary `original`, `augmented`, `preprocessed`, `sulfide`, and `final` layers with pan/zoom. `augmented` appears between `original` and `preprocessed` only when an augmented artifact exists; `augmented`, `preprocessed`, `sulfide`, and `final` stay greyed/disabled until the corresponding artifacts are ready. Artefact regions are not a separate primary layer; they render inside `sulfide` and `final` when the relevant artefact visibility checkbox is enabled.
- Segmentation class controls are contextual and sit as overlays on top of the image view. The primary/left layer legend appears in the top-left corner; the side-by-side/right layer legend appears in the top-right corner. `original`, `augmented`, and `preprocessed` hide class controls unless side-by-side is showing `sulfide` or `final`; `sulfide` shows only sulfides, non-sulfides, and artefacts; `final` shows only ordinary intergrowths, fine intergrowths, talc, artefacts, and background. In `sulfide`, turning off `sulfides` while keeping `non-sulfides` on must hide segmented sulfide pixels instead of showing the full raw base image.
- The `show tiling` checkbox is available when the current upload/run has a tile manifest and overlays the tile grid used for scaled/large-image analysis on top of original, augmented, preprocessed, sulfide, final, and side-by-side views.
- `show tiling`, `contours only`, and `opacity` are grouped in one viewer-options row below the image canvas.
- Side-by-side is a separate `none/augmented/preprocessed/sulfide/final` selector; choosing a ready comparison layer shows it on the right side with a dedicated DOM-overlay draggable left/right splitter that remains visible above segmentation overlays and legends.
- Preprocessing is shown as compact `[x] Предобработка [Настроить...] [Применить]` / `[x] Preprocessing [Edit...] [Apply]`. `Настроить...` opens the detailed settings popup with illumination normalization `(?)`, denoising `(?)`, contrast correction `(?)`, and panorama scaling `(?)`; hovering or focusing each `(?)` shows a short localized hint. Panorama scaling exposes an explicit mode/value pair: longest-side bound in pixels or scale factor, and the summary shows the active rule such as `панорама до 1800 px` / `panorama to 1800 px`. `Применить` checks preprocessing and refreshes the preprocessed preview. If a completed run is loaded, `Применить` creates a new immutable `prepared` run that keeps original/prerequisite artifacts up to the changed preprocessing step and clears downstream masks/metrics until `Start`; if the prepared run has not been started yet, another `Применить` updates that same prepared run. Unchecking preprocessing and pressing `Start` skips preprocessing, records `preprocessing_enabled=false`, and leaves the preprocessed viewer button and side-by-side option disabled. Turning panorama scaling off falls back to normal processing size; it does not disable the independent tiling manifest/overlay.
- `Start` creates an immutable run under `outputs/ore_pipeline_ui/runs/` with original input, preprocess preset, preprocessed image, masks, metrics, and report artifacts.
- Pressing `Start` while a previous completed run is loaded immediately hides the old text output, metrics table, CSV/PDF/file links, `sulfide`/`final` viewer layers, and side-by-side comparison; the viewer shows the best available input layer until the new run finishes.
- If `Edit Metadata...` or the API supplies calibrated `microns_per_pixel` / `pixel_size_um`, `scale_source`, and `scale_confidence=calibrated`, the result metrics table shows pixel area plus physical area, and `/api/runs/{run_id}/metrics.csv` contains matching hierarchy fields, `area_px`, `area_um2`, `area_mm2`, and scale provenance columns. Without calibrated scale, physical-area cells stay empty.
- The result section shows text output first and the metrics table below it at full width. The table is hierarchical: analyzed area, total sulfides with ordinary/fine children, talc, other analyzed area, and image artefacts.
- After the metrics table, the result section shows a sulfide-grain table. Each row lists a component ID, ordinary/fine intergrowth type, pixel area, equivalent diameter, perimeter, percent share of total sulfide area, liberation proxy, contact-count proxy, and locked/composite proxy flag. The note above the table states that liberation/contact/locked-composite values are OM-mask proxies, not chemistry-based MLA analysis. Checking one or more rows strokes the union of those selected grains on the current image view; unchecking all rows removes the outline without changing immutable run masks or metrics.
- After the sulfide-grain table, the result section shows technical run details from `run.json` and `reports/runtime.json`: run id/status/stage, elapsed time, effective backend/model sources, ML checkpoints only for ML stages, analysis dimensions, tile count/size/stride/progress, sulfide/talc/final/artefact stage outputs, and present mask/report artifacts.
- The result export row has `Save CSV`, `Save PDF Report`, and `View files`. `Save PDF Report` downloads a five-page A4 lab-style demonstration report with `Паспорт исследования`, conclusion, quantitative metrics table, original and preprocessed photo-documentation with preprocessing details, a two-color sulfide/non-sulfide map, full final class segmentation, ordinary/fine/talc final-overlay-plus-mask side-by-side rows, and method/QC/expert-review fields. `View files` opens a popup with all immutable run files, sortable filename/type/size/image-dimension headers, byte sizes, and `WxH` dimensions for image files; image, CSV, and JSON rows open a second preview popup, while other file types show a row-level `Download` action. `Download ZIP` downloads `/api/runs/{run_id}/artifacts.zip` with the run contents.
- While a run is queued/running, the `Start` control is replaced by a red `Stop` button. Pressing `Stop` disables that button, shows stopping status, cancels the active run through `/api/runs/{run_id}/cancel`, and leaves the run in terminal `canceled` state without enabling result editing.
- Result view shows layer-specific class toggles; artefact overlays use a distinct violet/magenta color inside the `sulfide` and `final` views instead of the red/green/blue class colors.
- `Fix me` opens the editor; sulfide/non-sulfide and final segmentation are selected with tabs, artefact edits use the same violet/magenta color as the main viewer, the top toolbar contains Brush/Pan, Undo/Redo, Zoom in/out, Fit view, and Brush size, Brush left-draws and right-erases, and the right panel shows live pixel/% statistics for sulfide, non-sulfide, ordinary intergrowth, fine intergrowth, and talc.
- The edit dialog refreshes the selected run before loading masks so image and segmentation layers appear even after a server restart or stale browser state; missing prerequisites show a localized editor error rather than a blank canvas.
- `Fix and Restart` creates a new derived run instead of mutating the parent.
- The left-sidebar History cards show a small thumbnail on the left with `Load` directly underneath it, while filename/run/date/classification text uses the remaining card width.
- The History page has three modes: all runs, standalone single runs, and series. All-runs and single-run modes show a table with thumbnail, filename, date, text-only progress percent, ore classification, sulfides, non-sulfides, ordinary intergrowth, fine intergrowth, talc, and Actions; no progress bar is shown in the history table, while muted details can show status/stage, tile counts, ETA, and elapsed time. Single-run mode excludes runs created inside Series. Series mode uses the `/history_series` slug, lists persisted series from the existing batch store, and exposes `Open` plus `Remove`; `Open` navigates to `/batch/{batch_id}`, while `Remove` confirms and deletes the persisted series plus its child immutable runs. Clicking a run thumbnail opens a preview popup, `Load` restores the original upload/preprocessing preset for tuning, and run-level `Remove` deletes the selected completed/failed run artifact from history.

## Ore Pipeline UI Docker Smoke

Run from the v2 root on a Docker-capable host:

```bash
python3 -m unittest discover -s tests -p 'test_ore_pipeline_docker.py' -v
docker compose -f docker-compose.ore-pipeline-ui.yml build
docker compose -f docker-compose.ore-pipeline-ui.yml up
```

On the organizer VM, prefix the Docker commands with `sudo` if Docker socket
access is still restricted for the default user.

Expected:

- The focused Docker artifact test passes.
- The image builds without copying `dataset`, `outputs`, or model checkpoints into the image context.
- The container logs `Ore pipeline UI: http://0.0.0.0:8080/`.
- `http://127.0.0.1:8080/workspace` opens the same v2 ore pipeline UI as the local Python launch.
- Uploads, settings, runs, CSV files, and PDF reports persist under `outputs/ore_pipeline_ui` after container restart.
- The default backend is heuristic and works without GPU packages. ML mode is only expected after using an ML-capable image or installing the full Torch/Transformers stack and mounting `models`.

## Augmentation Review Gallery Smoke

Run from the v2 root:

```bash
python3 -m py_compile scripts/generate_augmentation_review_gallery.py
python3 scripts/generate_augmentation_review_gallery.py \
  --per-label 1 \
  --max-side 720 \
  --overwrite
```

Expected:

- `outputs/augmentation_review/index.html` opens as a static local HTML file.
- The gallery contains one source image for each deconflicted official class and nine cards per source: original, UI default, color/tone, acquisition noise, scratches, polishing haze, pits/dust, combined moderate, and combined stress.
- Each non-original card includes the exact v2 augmentation settings JSON used to create that preview.

## Talc Blue-Line Converter Smoke

Run from the v2 root:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

```bash
python3 scripts/convert_talc_blue_lines.py \
  --input "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
  --output-dir outputs/talc_blue_line_conversion \
  --summary-json outputs/talc_blue_line_conversion_summary.json
```

Preferred browser/canvas review:

```bash
python3 apps/talc_review_web.py \
  --conversion-dir outputs/talc_blue_line_conversion \
  --host 127.0.0.1 \
  --port 0
```

Legacy Streamlit review after installing `requirements-ui.txt`:

```bash
streamlit run apps/talc_review_streamlit.py -- \
  --conversion-dir outputs/talc_blue_line_conversion
```

Expected:

- Unit tests pass.
- `outputs/talc_blue_line_conversion/manifest.json` exists with `42` samples.
- Current status counts are `31` `candidate_ok`, `9` `needs_manual_review`, and `2` `sulfide_overlap_review_required`.
- Each sample directory contains source image copy, blue stroke masks, talc candidates, sulfide/overlap/ignore masks, optional-silicate derived masks, final talc mask, QA overlay, and `conversion_summary.json`.
- The talc converter accepts `--silicate-mask-dir` and writes `silicate_support_mask.png`, `silicate_supported_talc_mask.png`, `silicate_unsupported_talc_mask.png`, `talc_positive_core_mask.png`, and `silicate_hard_negative_mask.png`; synthetic unit tests verify supported talc, unsupported uncertain talc, and silicate hard negatives.
- Preferred browser review shows original blue annotation lines explicitly, edits talc class masks directly, keeps Brush/Fill/Rectangle/Polygon/SAM2 edits in the `positive_bag` class, keeps Similar output in the `talc_node` class, supports undo, autosave, default-on sulfide protection, manual sulfide subtraction, top-right `Save`, `Save & Next`, and transparent `Next`, and saves reviewed class masks plus the compatibility union under each sample's `reviewed/` directory. Polygon/rectangle regions stay editable while the current image is open and are flattened on save. The legacy Streamlit review remains available only as a fallback.
- The `SAM2 assist` canvas tool shows model/device controls, `Load/check SAM2`, draggable point/box prompts, and `Run SAM2`; without local `torch` and `sam2`, it should report missing optional dependencies rather than blocking the rest of the app.

## Talc Browser Review App Smoke

Run from the v2 root:

```bash
python3 -m py_compile apps/talc_review_web.py
python3 -m unittest discover -s tests -p 'test_talc_review_web.py' -v
```

```bash
python3 apps/talc_review_web.py \
  --conversion-dir outputs/talc_blue_line_conversion \
  --host 127.0.0.1 \
  --port 0
```

Expected:

- Focused tests pass and cover same-filename annotated/original pairing, first-open `current_talc_mask.png` creation, reviewed save artifacts, reset behavior, and basic HTTP endpoints.
- The app prints a local URL such as `http://127.0.0.1:<port>/`.
- `/api/manifest` reports `42` samples for the full conversion workspace.
- The `Theme` selector supports `System`, `Light`, and `Dark`; explicit dark/light choices update the UI immediately and persist across reloads.
- Opening a sample auto-creates or reuses `current_positive_bag_mask.png`, `current_talc_node_mask.png`, and compatibility union `current_talc_mask.png`.
- The viewer shows a compact over-image segmentation class widget with checked `Positive bag` and `Talc` visibility toggles plus an `Edit` radio for each class; turning visibility off hides only that class overlay, while the selected `Edit` class controls Brush, Fill, Rectangle, and Polygon. The same widget shows live visible-pixel percentages for `Positive bag` and confirmed `Talc`, a separated display-only `Talc cluster areas` row with its own visibility toggle and highlighted-area percentage, and flags confirmed `Talc` below the known talcose threshold of `10%` visible pixels.
- The top toolbar is ordered `Brush`, `Fill`, `Similar`, `Rectangle`, `Polygon`, `SAM2`, `Undo`, `Zoom In`, `Zoom Out`, `Fit`, followed by active-tool parameters and a visible zoom percentage, with `Save`, `Save & Next`, and transparent `Next` pinned at the top right; Brush shows brush size with a `2-240 px` range, Similar shows `Strictness` plus `Apply Similar`/`Clear Preview`, and SAM2 shows prompt mode and `Load SAM2`.
- Mouse wheel over the canvas zooms in/out anchored around the cursor without changing mask geometry.
- Holding the mouse wheel / middle button and dragging over the canvas pans the zoomed view without drawing, erasing, moving shapes, or changing the selected tool.
- Sample cards and the header show human-readable statuses such as `Candidate OK`, `Needs manual review`, `Working draft`, and `Reviewed`; raw enum strings such as `candidate_ok` or `needs_manual_review` are not shown as user-facing labels.
- Autosaved edits show `Working mask saved`, `Saving working mask...`, or `Autosave failed` in the metrics panel instead of a stale yes/no unsaved flag.
- `Save & Next` advances through the currently visible filtered/search queue, not the unfiltered manifest order.
- `Next` advances through the same currently visible filtered/search queue without saving and has no filled button background.
- Switching samples warns before discarding an unfinished polygon/rectangle draft or failed local autosave state.
- The canvas edits talc class masks: Brush left-draw/right-erase, Fill, polygon, and rectangle write the selected `Edit` class (`Positive bag` or `Talc`), optional SAM2 prompt tools edit `positive_bag`, and Similar edits `talc_node`.
- `Background` includes `Sulfide mask (sulfide/non-sulfide mask segmentation)` and `Mask-only background`; selecting a missing background or layer shows a visible warning instead of silently dropping it.
- `Dark pixel preview threshold` is available beside the background controls. `255` leaves the photo unchanged, `90` is a quick talc-candidate starting preset, `0` paints the photo background white, moving the slider reports the visible-pixel percentage/count (`luma <= threshold`) for the active photo background, and moving it never changes the current talc mask pixels.
- `Show talc cluster areas` is available beside the background controls and as the separated `Talc cluster areas` row in the over-image widget. Turning either toggle on highlights locally dense talc regions without changing mask pixels; source can be `Talc class` or `Positive bag + Talc`, and radius, minimum local talc percentage, and opacity sliders update the overlay, widget percentage, and stats.
- `Fill` adds the selected `Edit` class to the clicked connected area bounded by raw/closed blue annotation strokes, sulfide pixels, existing selected-class regions, and the image edge; it autosaves, is undoable, and still clips newly added pixels against sulfides when protection is enabled.
- `Similar` left-clicks a confirmed talc grain to preview luma/color-similar non-sulfide pixels in yellow, `Strictness` recomputes that preview, right-click or `Clear Preview` discards it, and `Apply Similar` merges it into the `talc_node` class with autosave/undo. `Save` and `Save & Next` also apply an active Similar preview before writing reviewed outputs. Clicking inside an existing positive bag still uses the clicked seed patch as the anchor; nearby bag pixels may refine calibration only after passing seed-similarity filtering, so `Strictness=100` should not match broad matrix-heavy regions. Similar may mark pixels inside `positive_bag`; the bag remains a rough containing region, not confirmed talc.
- Keyboard shortcuts select tools without changing text fields: `B` selects Brush and `F` selects Fill.
- In Brush mode, left mouse draws the selected `Edit` class and right mouse erases it without opening the browser context menu.
- In Brush mode, hovering over the image shows a circle matching the current brush draw/erase area; changing brush size updates the circle.
- Polygon drawing supports click-to-add points and closes by clicking the first point; right-click on a polygon point removes it, while right-click elsewhere cancels the current draft without opening the browser context menu.
- Completed polygon regions fill immediately, stay editable while the current image is open, and support point drag, edge-click-to-insert, whole-shape move, and live mask updates.
- Rectangle drawing fills on drag or on the second corner click; right-click cancels the current draft, and completed rectangle regions stay editable while the current image is open with corner/edge drag, whole-shape move, and live mask updates.
- Pressing Delete/Backspace removes the selected completed polygon or rectangle, while text fields keep normal delete behavior.
- No `Apply Polygon`, `Cancel Polygon`, `Apply Rectangle`, or `Cancel Rectangle` controls are shown; `Save` flattens live polygon/rectangle regions into the reviewed mask.
- `Protect sulfides while drawing` is enabled by default; add tools cannot add new talc pixels on the sulfide mask, `Current talc on sulfide` is shown for existing overlap with pixels and percent, and `Subtract sulfides from mask` removes overlap and autosaves.
- `Save` writes `reviewed/reviewed_positive_bag_mask.png`, `reviewed/reviewed_talc_node_mask.png`, compatibility union `reviewed/reviewed_talc_mask.png`, `reviewed/reviewed_ignore_mask.png`, `reviewed/reviewed_overlay.png`, `reviewed/review_patch.json`, and `reviewed/review_summary.json`.
- `SAM2 box` is the default SAM2 canvas prompt; point mode remains available. SAM2 box/polygon prompt results are clipped to the prompt bounds, and SAM2 masks covering more than half the image are rejected with a clear status message instead of filling the whole canvas.
- In SAM2 mode, hovering over the image shows a dashed orange preview of the proposed box/point area; dragging a SAM2 box keeps the dashed preview visible.
- In `SAM2 point` mode, keeping the cursor still over the image for about two seconds requests a SAM2 mask as an orange preview overlay without changing the talc mask; `Apply SAM2` applies that preview, or runs and applies the current hover point if no preview is ready yet.
- If SAM2 dependencies are missing, `Load SAM2` and the SAM2 canvas tool report the missing optional dependency without blocking manual editing.

## Heuristic Segmentation Smoke

Run from the v2 root:

```bash
python3 -m unittest discover -s heuristic_segmentation/tests -p 'test_*.py' -v
```

```bash
python3 heuristic_segmentation/run_heuristic_segmentation.py \
  --image "dataset/Фото руд по сортам. ч1/Рядовые руды/DSCN2176.JPG" \
  --output-dir outputs/heuristic_segmentation_smoke \
  --max-side 900 \
  --overwrite
```

Expected:

- Unit tests pass.
- The CLI writes `class_mask.png`, `sulfide_mask.png`, `talc_candidate_mask.png`, `analyzed_mask.png`, `overlay.png`, `components.csv`, `metrics.json`, `run_summary.json`, and `batch_summary.json`.
- Smoke metrics currently include `sulfide_fraction 0.164864`, `talc_candidate_fraction 0.000708`, and `70` sulfide components for `DSCN2176.JPG` at `--max-side 900`.
- The overlay is nonblank and displays ordinary intergrowth in green, fine intergrowth in red/orange, and talc candidates in blue.

## Reusable Demo Libraries Smoke

Run from the v2 root:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Expected:

- Unit tests pass; current local result is `55` tests.
- Coverage includes `source_fusion`, `review_queue`, `curation`, `component_reports`, `report_cards`, and `scribble_classifier`.
- These tests use synthetic inputs and do not require GPU, Streamlit, SAM2, or external datasets.

## Binary Sulfide Training Smoke

Run from the v2 root:

```bash
python3 scripts/build_official_manifest.py \
  --dataset-root dataset \
  --out outputs/commit_smoke_official_manifest.json
```

```bash
python3 scripts/build_binary_sulfide_dataset.py \
  --out-dir outputs/commit_smoke_binary_sulfide_dataset \
  --tile-size 128 \
  --stride 128 \
  --max-lumenstone-images 1 \
  --max-official-images-per-label 1 \
  --max-tiles-per-source 2 \
  --max-total-tiles 12 \
  --downscale-max-side 512 \
  --overwrite
```

```bash
python3 scripts/train_binary_sulfide.py \
  --dataset-manifest outputs/commit_smoke_binary_sulfide_dataset/manifest.json \
  --model resunet \
  --out-dir outputs/commit_smoke_train_resunet \
  --epochs 1 \
  --batch-size 2 \
  --num-workers 0 \
  --base-channels 8 \
  --device cpu \
  --max-steps-per-epoch 1
```

```bash
python3 scripts/train_binary_sulfide.py \
  --dataset-manifest outputs/commit_smoke_binary_sulfide_dataset/manifest.json \
  --model segformer_b0 \
  --pretrained-model random \
  --allow-random-init \
  --out-dir outputs/commit_smoke_train_segformer_b0 \
  --epochs 1 \
  --batch-size 1 \
  --num-workers 0 \
  --device cpu \
  --max-steps-per-epoch 1
```

Expected:

- Official manifest command reports `1236` images.
- Binary sulfide smoke dataset writes a manifest and at least one train and one val tile.
- ResUNet and SegFormer smoke commands each write `train_log.csv`, `last.pt`, and `best.pt`.

## Binary Sulfide Evaluation And Pipeline Smoke

Run from the v2 root after the binary smoke commands above:

```bash
python3 scripts/evaluate_binary_sulfide.py \
  --dataset-manifest outputs/commit_smoke_binary_sulfide_dataset/manifest.json \
  --checkpoint outputs/commit_smoke_train_resunet/best.pt \
  --split val \
  --batch-size 2 \
  --num-workers 0 \
  --hausdorff-max-items 4 \
  --out-json outputs/commit_smoke_train_resunet/eval_metrics.json
```

```bash
python3 scripts/run_ore_pipeline.py \
  --image outputs/commit_smoke_binary_sulfide_dataset/tiles/train/images/official_heuristic_014709cbad4a_384_0.jpg \
  --checkpoint outputs/commit_smoke_train_resunet/best.pt \
  --out-dir outputs/commit_smoke_ore_pipeline_auto_talc \
  --tile-size 64 \
  --stride 32 \
  --batch-size 2 \
  --device cpu \
  --preview-max-side 256 \
  --auto-talc-candidate
```

```bash
python3 scripts/run_official_batch.py \
  --split-json outputs/official_balanced_eval_split.json \
  --dataset-root dataset \
  --checkpoint outputs/commit_smoke_train_resunet/best.pt \
  --out-dir outputs/commit_smoke_official_batch \
  --max-total 1 \
  --tile-size 512 \
  --stride 512 \
  --batch-size 2 \
  --device cpu \
  --preview-max-side 256 \
  --overwrite
```

```bash
python3 scripts/evaluate_ore_classification.py \
  --summary-csv outputs/commit_smoke_official_batch/summary.csv \
  --out-json outputs/commit_smoke_official_batch/ore_classification_metrics.json \
  --out-md outputs/commit_smoke_official_batch/ore_classification_metrics.md
```

```bash
python3 scripts/build_official_balanced_eval_split.py \
  --official-manifest outputs/official_manifest.json \
  --out-json outputs/official_balanced_eval_split.json \
  --out-csv outputs/official_balanced_eval_split.csv
```

Preferred deconflicted split:

```bash
python3 scripts/audit_official_labels.py \
  --official-manifest outputs/official_manifest.json \
  --dataset-root dataset \
  --out-dir outputs/official_label_audit

python3 scripts/build_official_balanced_eval_split.py \
  --official-manifest outputs/official_manifest.json \
  --label-audit-json outputs/official_label_audit/summary.json \
  --exclude-conflicts \
  --dedupe-sha256 \
  --out-json outputs/official_balanced_eval_split_deconflicted.json \
  --out-csv outputs/official_balanced_eval_split_deconflicted.csv
```

Expected:

- Evaluation JSON includes `iou_sulfide`, `f1_sulfide`, `auc_sulfide`, `hausdorff_px_mean`, and `hd95_px_mean`.
- The pipeline writes `binary_sulfide/sulfide_mask.png`, `binary_sulfide/confidence.png`, `binary_sulfide/analyzed_mask.png`, `binary_sulfide/overlay_preview.jpg`, `talc_candidate/talc_candidate_mask.png`, `talc_candidate/talc_candidate_summary.json`, `ore_analysis/ore_summary.json`, `ore_analysis/component_features.csv`, `ore_analysis/analyzed_mask.png`, and `ore_analysis/intergrowth_overlay_preview.jpg`.
- UI runs write runtime provenance in `run.json.runtime` and `reports/runtime.json`; ML runs record the effective binary sulfide checkpoint plus checkpoint metadata/device/tile settings when available, while heuristic runs explicitly record no model checkpoint for heuristic/rule-only steps.
- `ore_summary.json` reports analyzed-denominator `sulfide_fraction` / `talc_fraction`, full-image `*_fraction_image`, `analyzed_fraction`, `talc_margin`, `intergrowth_margin`, `needs_expert_review`, and `warnings`.
- Raw balanced official split contains `387` labelled images: `129` ordinary, `129` fine, `129` talcose. Preferred deconflicted split contains `345` labelled images: `115` per class after excluding label-conflict hashes and duplicate hashes. The `14` panoramas remain listed separately as unlabelled stress-test images.
- The one-image official batch smoke writes `summary.csv`, `summary.json`, `failures.json`, and image-level classification metrics JSON/Markdown. On a one-class smoke sample, AUC may be `null`; full balanced evaluation is needed for meaningful F1/AUC.
- `scripts/merge_official_batch_shards.py` combines class-sharded `run_official_batch.py` outputs, rejects duplicate `run_id` values, and writes combined `summary.csv`, `summary.json`, and `failures.json`.
- `scripts/evaluate_ore_feature_classifier.py` cross-validates image-level classifiers from `summary.csv` plus per-run `component_features.csv`; use the full balanced split, not the one-image smoke, for meaningful F1/AUC.

## Ore Rule Calibration Smoke

Run from the v2 root after an official batch has written `summary.csv` and
per-run `ore_analysis/component_features.csv` files:

```bash
python3 scripts/calibrate_ore_rules.py \
  --summary-csv outputs/commit_smoke_official_batch/summary.csv \
  --out-json outputs/commit_smoke_official_batch/ore_rule_calibration.json \
  --out-md outputs/commit_smoke_official_batch/ore_rule_calibration.md
```

Expected:

- The command writes `ore_rule_calibration.json` and optional Markdown.
- Output includes `best_config`, `best_metrics`, `top_results`, and an explicit note that calibration uses image-level labels, not pixel-level geological ground truth.
- `scripts/analyze_ore_from_masks.py`, `scripts/run_ore_pipeline.py`, and `scripts/run_official_batch.py` accept `--rule-config-json` with either this calibration artifact or a direct config object; explicit CLI flags still override the file.
- On one-class smoke data, AUC may be `null`; use the full balanced batch for meaningful F1/AUC calibration.

## Manual Review Pack Smoke

Run from the v2 root after a compatible B2 checkpoint is available:

```bash
python3 scripts/prepare_manual_review_pack.py \
  --per-label 1 \
  --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt \
  --out-dir outputs/manual_review/smoke_b2_review_pack \
  --device auto \
  --batch-size 1 \
  --overwrite
```

Expected:

- `review_manifest.csv`, `review_manifest.json`, `feedback_template.csv`, and `review_candidates.csv` exist.
- Each `runs/*` directory contains `pipeline_summary.json`, `review_panel.jpg`, `source_preview.jpg`, `binary_sulfide/confidence_heatmap.jpg`, sulfide mask/overlay, and ordinary/fine overlay.
- The generated pack can be opened with:

```bash
streamlit run apps/sulfide_qa_streamlit.py -- \
  --runs-dir outputs/manual_review/smoke_b2_review_pack/runs \
  --review-dir outputs/manual_review/smoke_b2_review_pack/reviews
```

Note: zelda-trained SegFormer checkpoints may use a different Transformers key namespace than the local runtime. The shared loader now applies a strict all-keys/all-shapes namespace remap; if this smoke fails after a dependency upgrade, re-run Runtime `Test` in Settings and inspect the checkpoint-loader error before falling back to zelda.

## MLflow Debug Tracking (optional)

The training scripts have optional MLflow tracking, off by default. Verify it stays
non-invasive and that the enabled path logs a run.

No-op path (mlflow not required): `--help` lists the flags and training runs unchanged
without `--mlflow`.

```bash
python scripts/train_grade_classifier.py --help | grep -- --mlflow   # flags present
python - <<'PY'
import argparse, sys; sys.path.insert(0, "src")
from ore_classifier.tracking import mlflow_run
with mlflow_run(argparse.Namespace(mlflow=False), params={"lr": 1e-3}) as r:
    assert r.enabled is False
    r.log_metrics({"loss": 0.5}, step=1); r.log_artifact("nope.json")  # must not raise
print("mlflow no-op path ok")
PY
```

Enabled path (needs `pip install -r requirements-dev.txt`): a short capped training run
writes an experiment under `./mlruns` (gitignored) that `mlflow ui` can browse.

```bash
python scripts/train_grade_classifier.py --out-dir outputs/smoke_grade_mlflow \
  --epochs 1 --limit 32 --max-steps-per-epoch 2 --mlflow --mlflow-experiment smoke
test -d mlruns && echo "mlruns store created"
```

## Planned Pipeline Checks

- Streamlit QA app opens the pseudo-label manifest and saves a JSON patch.
