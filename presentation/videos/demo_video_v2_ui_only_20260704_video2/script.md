# VIDEO #2 - Main v2 Ore Pipeline UI Director Script

Purpose: record a concrete walkthrough of `apps/ore_pipeline_web.py` on the
official sample `2550382-1-10x`. This replaces the previous broad UI-only tour
with an operator-grade script: each scene defines the route, the exact UI state,
the screenshot to show, and how that screenshot is reached.

The verified local reference run used:

- sample: `dataset/Фото руд по сортам. ч1/Оталькованные руды/2550382-1 10x.JPG`
- workspace: `outputs/ore_pipeline_ui_video2_20260705`
- parent run: `run_20260705_013329_516251000_1643dd89`
- artifact-edit child run: `edit_20260705_013353_461997000_0088678f`
- series: `batch_20260705_013902_291751000`
- augmentation: disabled
- preprocessing: disabled

## What Changed From The Previous Script

Previous script: `presentation/videos/demo_video_v2_ui_only_20260704/script_ru.md`.

Keep from the previous version:

- Workspace, History, Series, Settings, Status, and API pages.
- Viewer layers: original, sulfide, final, and side-by-side.
- Text output, metrics, sulfide-grain table, files, and technical details.

Add or correct for VIDEO #2:

- Use one concrete sample: `2550382-1 10x.JPG`.
- Show `Edit Metadata...` and `Configuration...`.
- Explicitly show `Augmentation` and `Preprocessing` off before Start.
- Show a real end-to-end run and its immutable run id.
- Show `Fix me`, edit `Artefacts`, and explain the new child run created after
  `Fix and Restart`.
- Show how an artifact exclusion changes downstream fractions and component
  counts.
- Select one `Sulfide grains` row and show the stroked grain in the viewer.
- Open the API page only. Do not use the API playground in the recording.

## How To Run The Demo App

Start from the v2 repo root:

```bash
cd /Volumes/T7_2TB/Projects-T7_2TB/2026_Nornikel_Hackaton_v2

.venv/bin/python apps/ore_pipeline_web.py \
  --workspace-dir outputs/ore_pipeline_ui_video2_20260705 \
  --host 127.0.0.1 \
  --port 0 \
  --backend heuristic \
  --talc-backend heuristic \
  --grain-backend heuristic
```

Open the printed URL. The verified reference URL was
`http://127.0.0.1:57314/`, but the port changes when `--port 0` is used.

For a production ML recording, keep the same UI sequence but start the app with
the default ML checkpoints or the explicit ML command from `COMMANDS.md`. The
screenplay below intentionally uses the fast heuristic backend so the full
sequence can be rehearsed without waiting on model load.

## Most Important Code Paths

Use these functions when explaining how `apps/ore_pipeline_web.py` works:

- `main()` (`apps/ore_pipeline_web.py:7293`) parses CLI flags, builds
  `OrePipelineStore`, and starts the threaded HTTP server.
- `OrePipelineStore.__init__()` (`apps/ore_pipeline_web.py:1675`) owns
  workspace directories, runtime defaults, job registries, system logs, and
  allowed artifact roots.
- `register_upload_from_bytes()` / `register_upload_from_path()`
  (`apps/ore_pipeline_web.py:2154`, `apps/ore_pipeline_web.py:2165`) create an
  immutable upload and call `_register_upload_file()`.
- `_register_upload_file()` (`apps/ore_pipeline_web.py:2188`) reads dimensions,
  protects against decode bombs, builds previews, extracts raw metadata, and
  writes `upload.json`.
- `prepare_upload()` (`apps/ore_pipeline_web.py:2280`) applies runtime
  augmentation and preprocessing, builds analysis-scale images, writes
  `augmentation.json` / `preprocess.json`, and records tiling metadata. In this
  script both gates are off, so the output stays close to the original image.
- `start_run()` (`apps/ore_pipeline_web.py:2429`) creates the immutable run,
  attaches curated metadata and runtime provenance, queues the worker, and
  returns the first run payload.
- `_run_job()` (`apps/ore_pipeline_web.py:4220`) executes the run stages and
  delegates to `_run_heuristic_backend()` or `_run_ml_backend()`.
- `_write_run_outputs()` (`apps/ore_pipeline_web.py:4605`) persists masks,
  display layers, reports, summaries, and metadata.
- `_finalize_run_metadata()` (`apps/ore_pipeline_web.py:4958`) writes final
  status, metrics, runtime provenance, report links, and GIS/export metadata.
- `run_payload()` (`apps/ore_pipeline_web.py:3627`) is the main read model for
  the UI: display URLs, masks, downloads, metrics, history, and
  `sulfide_grains`.
- `_sulfide_grains_payload()` (`apps/ore_pipeline_web.py:3661`) enriches the
  component CSV into the `Sulfide grains` table and creates the component label
  map used for row-to-view highlighting.
- `create_edit_run()` (`apps/ore_pipeline_web.py:2644`) is the server side of
  `Fix and Restart`: it copies parent inputs, applies the edited mask layer,
  recalculates outputs, and writes a new immutable child run.
- `create_batch()` / `run_batch()` (`apps/ore_pipeline_web.py:3250`,
  `apps/ore_pipeline_web.py:3463`) implement the Series page: each item becomes
  a normal immutable run.
- `status_payload()` and `test_runtime()` (`apps/ore_pipeline_web.py:2744`,
  `apps/ore_pipeline_web.py:2975`) power the Status and runtime readiness
  checks.
- `OrePipelineHandler._handle_get()` / `_handle_post()` (`apps/ore_pipeline_web.py:6800`,
  `apps/ore_pipeline_web.py:6895`) map `/workspace`, `/history`, `/batch`,
  `/status`, `/api`, `/settings`, and all `/api/*` endpoints to the store.

## Screen Markup Contract

Each scene uses this markup:

```screen
route: browser path to open
capture_file_placeholder: screenshot shown in the video
navigation:
  - exact actions to reach the state
ui_state:
  - controls that must be set before capture
must_contain:
  - visible evidence required in the frame
do_not:
  - recording constraints
```

All reference screenshots are stored in:

`presentation/videos/demo_video_v2_ui_only_20260704_video2/screenshots/`

## 00. Preflight [00:00-00:20]

```screen
route: /workspace
capture_filename: no screenshot
navigation:
  - start the app with the command above
  - switch Language to English for the recording
  - use the isolated workspace `outputs/ore_pipeline_ui_video2_20260705`
  - keep browser chrome out of the final crop
ui_state:
  sample: `2550382-1 10x.JPG`
  Augmentation: off
  Preprocessing: off
  runtime: heuristic / heuristic / heuristic for rehearsal
must_contain:
  - none; this is an operator setup step
```

Narration: "This video shows the main v2 ore-pipeline UI on one real official
sample. We keep augmentation and preprocessing disabled so the path from image
to masks, metrics, and immutable run artifacts is easy to audit."

## 01. History As The Entry Point [00:20-00:42]

```screen
route: /history
capture_filename: screenshots/01_history_page.png
navigation:
  - open `/history`
  - verify the table contains `2550382-1_10x.jpg`
  - show parent run `run_20260705_013329_516251000_1643dd89`
ui_state:
  history_mode: all runs
must_contain:
  - rows with status `Done`
  - parent run and artifact-edit child run
  - visible `Load` actions
```

Narration: "History is a practical starting point for recording. Every result is
an immutable run, so we can reload the exact state without recomputing the
pipeline."

## 02. Open Image In Workspace [00:42-01:10]

```screen
route: /workspace
capture_filename: screenshots/02_workspace_loaded_parent.png
navigation:
  - from `/history`, click `Load` for `run_20260705_013329_516251000_1643dd89`
  - alternatively upload `2550382-1 10x.JPG` on `/workspace`
ui_state:
  view_mode: final
  side_by_side: none
  Augmentation: off
  Preprocessing: off
must_contain:
  - Input image card with `2550382-1 10x.JPG`
  - disabled `augmented` and `preprocessed` layer buttons
  - `Augmentation is off.`
  - `Preprocessing will be skipped on Start.`
  - final segmentation legend and viewer
```

Narration: "The workspace has the input image on the left and the viewer on the
right. Notice that augmentation and preprocessing are off. The augmented and
preprocessed layers are disabled because this run intentionally skips them."

## 03. Edit Metadata [01:10-01:35]

```screen
route: /workspace
capture_filename: screenshots/03_edit_metadata_dialog.png
navigation:
  - click `Edit Metadata...`
  - keep the `Domain` tab visible first
  - briefly switch to `Raw` if recording live, then return to `Domain`
ui_state:
  metadata.sample_id: `2550382-1-10x`
  metadata.magnification: `10x`
must_contain:
  - metadata dialog
  - domain fields
  - raw image metadata tab
  - session-defaults tab or equivalent control
```

Narration: "Metadata is curated before the run and stored with the result. The
same dialog also exposes raw image metadata, so the reviewer can separate
operator input from file-derived facts."

## 04. Configuration [01:35-02:00]

```screen
route: /workspace
capture_filename: screenshots/04_configuration_dialog.png
navigation:
  - close Metadata
  - click `Configuration...`
ui_state:
  sulfide_backend: heuristic for rehearsal
  talc_backend: heuristic for rehearsal
  grain_backend: heuristic for rehearsal
must_contain:
  - runtime backend selectors
  - talc source selector
  - grain-classification selector
  - reset/save controls
```

Narration: "Configuration is per-run. It lets us override sulfide segmentation,
talc detection, and grain classification for the next run without rewriting old
results. Completed runs keep their original runtime provenance."

## 05. End-To-End Pipeline Run [02:00-02:35]

```screen
route: /workspace
capture_filename: screenshots/02_workspace_loaded_parent.png
navigation:
  - on a fresh upload, make sure `Augmentation` is unchecked
  - make sure `Preprocessing` is unchecked
  - click `Start`
  - wait until status reaches `complete`
ui_state:
  parent_run_id: `run_20260705_013329_516251000_1643dd89`
  result_class: `hard_to_process_ore`
  sulfide_fraction: `10.8668%`
  talc_fraction: `0.6108%`
  component_count: `91`
must_contain:
  - final segmentation layer
  - completed run id in the status/history area
  - Start button available for creating a new run from changed settings
```

Narration: "Start creates an immutable run. This sample is classified as
hard-to-process ore: talc is well below the ten percent gate, and the fine
intergrowth share is slightly above the ordinary share."

## 06. Text Output And Metrics [02:35-03:05]

```screen
route: /workspace
capture_filename: screenshots/05_text_output_metrics.png
navigation:
  - scroll to `Text output`
  - keep `Metrics` table directly below it
ui_state:
  run_id: `run_20260705_013329_516251000_1643dd89`
must_contain:
  - `Text output`
  - rationale sentence with sulfide, talc, ordinary/fine, and warning margin
  - `Metrics` table
  - `Save to CSV` and `Save PDF Report`
```

Narration: "The UI does not just paint a mask. It writes a compact decision
rationale and exposes the numeric basis: analyzed area, sulfide fraction,
ordinary and fine intergrowths, talc fraction, cluster areas, and image
artefacts."

## 07. Sulfide Grains Table [03:05-03:35]

```screen
route: /workspace
capture_filename: screenshots/06_sulfide_grains_selected_row.png
navigation:
  - scroll to `Sulfide grains`
  - select the first row checkbox
ui_state:
  selected_grain: first visible `data-grain-id`
  total_grains: `91`
must_contain:
  - `Sulfide grains` section
  - table with component id, type, area, diameter, perimeter, share, liberation,
    contacts, and locked/composite proxy
  - selected row checkbox
```

Narration: "The grain table expands the sulfide mask into connected components.
Each row is a grain-level proxy report, not a chemistry claim: area, shape,
contacts, liberation proxy, and whether the grain looks locked or composite."

## 08. Selected Grain Stroked In Viewer [03:35-03:55]

```screen
route: /workspace
capture_filename: screenshots/07_selected_grain_stroked_in_view.png
navigation:
  - after selecting one grain row, scroll back to the viewer
  - keep the same row selected
ui_state:
  view_mode: final
  selected_grain_overlay: on
must_contain:
  - viewer with selected grain outline/stroke
  - final segmentation layer still visible
  - left legend still active
```

Narration: "A checked grain row is stroked directly in the viewer. This is the
review bridge between table metrics and the actual image region that produced
them."

## 09. Fix Me UI [03:55-04:25]

```screen
route: /workspace
capture_filename: screenshots/08_fix_me_artifact_editor.png
navigation:
  - click `Fix me`
  - choose or keep the `Artefacts` edit layer
  - show brush and pan controls
ui_state:
  edit_layer: Artefacts
  brush_size: default
must_contain:
  - Fix dialog
  - artifact mask editor
  - `Fix and Restart` button
  - pixel statistics for artefacts vs clean area
```

Narration: "Fix me is the correction surface. In this scene we use the
Artefacts layer: a scratch or polishing defect can be excluded from the analyzed
area before the downstream fractions are recomputed."

## 10. Fix And Restart Creates A New Run [04:25-04:55]

```screen
route: /workspace
capture_filename: screenshots/09_fix_restart_new_run_loaded.png
navigation:
  - in Fix dialog, draw a small artefact area
  - click `Fix and Restart`
  - wait for the new child run to load
ui_state:
  parent_run_id: `run_20260705_013329_516251000_1643dd89`
  child_run_id: `edit_20260705_013353_461997000_0088678f`
  derivation.operation: `recalculate_from_artifact_edit`
must_contain:
  - child run loaded in Workspace
  - final mask after recalculation
  - history/sidebar showing lineage or new run row
```

Narration: "The correction never overwrites the original run. Fix and Restart
creates a new run with a parent pointer and the edit operation recorded. The
old run remains available for audit and comparison."

## 11. Artefact Edit Changes Downstream Metrics [04:55-05:20]

```screen
route: /workspace
capture_filename: screenshots/09_fix_restart_new_run_loaded.png
navigation:
  - compare parent and child summaries
  - use the child run after artifact edit
ui_state:
  parent:
    sulfide_fraction: `0.1086683511`
    talc_fraction: `0.0061082489`
    component_count: `91`
  child:
    sulfide_fraction: `0.1062389252`
    talc_fraction: `0.0060190161`
    artifact_fraction_image: `0.0070179735`
    component_count: `93`
must_contain:
  - child run metrics
  - artefact fraction no longer zero
```

Narration: "Because the artefact mask changes the analyzed area and masks, the
fractions move. In the reference run, the image artefact fraction becomes about
zero point seven percent; sulfide and talc fractions are recalculated in the
child run."

## 12. Side-By-Side Comparison [05:20-05:45]

```screen
route: /workspace
capture_filename: screenshots/10_side_by_side_final_vs_sulfide.png
navigation:
  - load the artifact-edit child run
  - set left layer to `final`
  - set `Side-by-side` to `sulfide`
ui_state:
  left_layer: final
  right_layer: sulfide
must_contain:
  - side-by-side splitter
  - final segmentation on the left
  - sulfide mask on the right
  - `Share comparison` button enabled
```

Narration: "Side-by-side mode is the fastest way to explain how the final class
layer relates to the underlying sulfide mask. The splitter keeps both views
inside the same image coordinate frame."

## 13. Run Technical Details [05:45-06:10]

```screen
route: /workspace
capture_filename: screenshots/11_run_technical_details.png
navigation:
  - load the parent run
  - scroll to `Run technical details`
ui_state:
  run_id: `run_20260705_013329_516251000_1643dd89`
must_contain:
  - run id, status, stage, timestamps, elapsed time
  - runtime/model card
  - tiling/card analysis size
  - masks and report paths
```

Narration: "Technical details are the provenance panel. The viewer result is
not a screenshot-only story: run id, runtime, model source, tiling metadata,
masks, reports, and runtime JSON are all tied to the saved run."

## 14. Series Page [06:10-06:35]

```screen
route: /batch/batch_20260705_013902_291751000
capture_filename: screenshots/12_series_page_completed_batch.png
navigation:
  - open `/batch/batch_20260705_013902_291751000`
  - show the completed one-image series
ui_state:
  batch_id: `batch_20260705_013902_291751000`
  status: complete
  child_run_id: `run_20260705_013902_657117000_1643dd89`
must_contain:
  - Series page
  - completed item card/table
  - child run load action
  - results CSV link if visible
```

Narration: "Series is the batch workflow. Each item becomes the same kind of
immutable run as a single-image workflow, so a series result can still be opened
and audited image by image."

## 15. Status Page [06:35-06:55]

```screen
route: /status
capture_filename: screenshots/13_status_page.png
navigation:
  - open `/status`
ui_state:
  no_active_job_expected: true
must_contain:
  - service health
  - backend/checkpoint readiness
  - CPU, GPU or Metal, RAM, flash
  - active jobs and system events
```

Narration: "The Status page is the operations screen. It tells the presenter and
operator what backend is active, whether checkpoints are available, what the
machine resources look like, and whether any run is still executing."

## 16. API Page - Open Only [06:55-07:15]

```screen
route: /api
capture_filename: screenshots/14_api_page_no_playground_use.png
navigation:
  - open `/api`
  - do not click sandbox/playground controls
ui_state:
  playground_used: false
must_contain:
  - endpoint list
  - `/api/uploads`
  - `/api/runs/start`
  - `/api/runs/{run_id}`
  - `/api/batches`
  - `/api/settings`
do_not:
  - do not run examples
  - do not send playground requests
```

Narration: "The API page documents the same operations available in the browser:
upload, start, poll run, read artifacts, use Series, settings, and runtime
checks. For this video we only open the documentation page and do not run the
playground."

## 17. Settings Page [07:15-07:40]

```screen
route: /settings
capture_filename: screenshots/15_settings_page.png
navigation:
  - open `/settings`
ui_state:
  show_runtime_defaults: true
  show_preprocessing_defaults: true
must_contain:
  - language/theme settings
  - runtime backend defaults
  - preprocessing defaults
  - history/security/system controls if visible
```

Narration: "Settings are server-backed defaults for new work. They do not
rewrite completed runs. That separation is important: a reviewer can change the
next runtime configuration while old run provenance stays intact."

## 18. Closing Frame [07:40-08:00]

```screen
route: /history
capture_filename: screenshots/01_history_page.png
navigation:
  - return to `/history`
  - show parent, edit child, and series child run rows
ui_state:
  history_contains:
    - `run_20260705_013329_516251000_1643dd89`
    - `edit_20260705_013353_461997000_0088678f`
    - `run_20260705_013902_657117000_1643dd89`
must_contain:
  - immutable run rows
  - `Done` statuses
  - load actions
```

Narration: "The full demonstration stays inside one app: image, metadata,
configuration, run, masks, metrics, grain-level evidence, fix-and-restart,
series, status, API docs, and settings. The output is reproducible because each
result is stored as a run with artifacts and provenance."

## Verification Commands Used

The reference run was created through the same API endpoints that the browser UI
calls:

```bash
BASE=http://127.0.0.1:57314
SAMPLE='dataset/Фото руд по сортам. ч1/Оталькованные руды/2550382-1 10x.JPG'

curl -F "file=@${SAMPLE}" "$BASE/api/uploads"

curl -H 'Content-Type: application/json' \
  -d '{
    "upload_id": "...",
    "preprocessing_enabled": false,
    "illumination_normalization": false,
    "denoise": false,
    "contrast_correction": false,
    "panorama_scaling": false,
    "augmentation": {"enabled": false},
    "runtime": {
      "backend": "heuristic",
      "talc_backend": "heuristic",
      "grain_backend": "heuristic"
    }
  }' \
  "$BASE/api/runs/start"
```

The UI route screenshots were then reached by normal browser navigation and
visible UI actions (`Load`, `Edit Metadata...`, `Configuration...`, `Fix me`,
grain-row checkbox, side-by-side buttons, and page tabs).
