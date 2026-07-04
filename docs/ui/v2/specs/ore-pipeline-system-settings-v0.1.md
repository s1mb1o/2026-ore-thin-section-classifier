# Ore Pipeline System Settings v0.1

Date: 2026-07-03

## Scope

Add a v2 ore-pipeline Settings page for system-wide, long-lasting defaults in
`apps/ore_pipeline_web.py`.

System-wide means the settings are stored by the local server under the ore
pipeline workspace and are shared by browsers using the same app workspace.
Long-lasting means the settings survive browser reloads, browser local-storage
clears, and app restarts.

## Storage

Persist settings as JSON:

```text
outputs/ore_pipeline_ui/settings/app_settings.json
```

The file must be non-run artifact state. Immutable run folders still record the
exact effective values used by each run.

## Settings

The first version covers settings already present in the UI:

- language: `ru` or `en`, default `ru`;
- theme: `system`, `light`, or `dark`, default `system`;
- runtime backend:
  - `heuristic` or `ml`;
  - checkpoint path for the ML backend;
  - Settings keeps the checkpoint path as a saved runtime value but shows only a
    shortened checkpoint name in the Runtime panel, never the full filesystem
    path;
  - talc source: `ml` by default for the trained talc segmentation model, or
    `heuristic` for the existing optical candidate fallback;
  - talc checkpoint path and probability threshold for the talc ML source;
    the threshold input is disabled when talc source is `heuristic`, because the
    heuristic talc path does not consume the ML probability cutoff;
  - saving applies the new runtime immediately for new runs;
  - already-created immutable runs keep their recorded backend and checkpoint;
  - changing runtime is rejected while a run or Series job is active;
  - `Test All` checks the selected, possibly unsaved runtime values; for
    `heuristic` it verifies the built-in heuristic backend is available, and
    for each `ml` source it verifies that the configured checkpoint exists and
    can be loaded through the same model loader used by runs without creating a
    run;
- default preprocessing preset:
  - preprocessing disabled by default;
  - illumination normalization;
  - denoise;
  - contrast correction;
  - panorama scaling;
- default `show tiling` checkbox;
- metadata session defaults for repeated fields:
  - project;
  - microscope/camera;
  - objective;
  - scale source;
  - scale value, micrometers per pixel;
  - scale confidence;
  - review status.
- history maintenance:
  - remove all saved run and Series history;
  - keep uploaded source images and app settings intact;
  - reject removal while a run, Series job, or foreground operation is active.

## UI

Add a direct-loadable `/settings` slug page and a `Settings` navigation tab.

The page shows:

- language and theme selectors;
- runtime backend selector and shortened checkpoint display;
- talc source selector, shortened talc checkpoint display, and ML talc
  probability threshold input that is enabled only for `ML model`;
- runtime `Test All` action with a localized success/failure status line;
- preprocessing default checkboxes;
- show-tiling default checkbox;
- metadata session-default inputs;
- a destructive `Remove all history` action in a separate History panel;
- `Save settings` and `Reset to defaults` actions;
- a clear status line after save/reset/load failure.

## API

Add:

- `GET /api/settings` -> current merged settings, including effective runtime;
- `PUT /api/settings` -> validate, persist, and apply settings.
- `POST /api/runtime/test` -> validate a supplied runtime selection without
  saving it; selected ML sources perform bounded subprocess checkpoint-load
  probes and return `ok/status/details/models`.
- `DELETE /api/history` -> remove all persisted run and Series artifact folders
  while leaving uploads and settings intact.

Invalid settings should return `400` with a clear error.
Runtime changes during active jobs should return `409` to avoid changing backend
selection for work that is already queued/running.
Runtime ML tests should return a non-OK test payload when imports or checkpoint
loading fail, and should return `409` when an active job is already running.
History removal during active jobs should return `409`.

## Acceptance

- `/settings` loads the same app shell and selects the Settings page.
- Settings are loaded before workspace defaults are applied.
- Saving settings updates the server-side JSON file.
- Saving backend/checkpoint and talc source/checkpoint/threshold updates the
  live server runtime for the next run.
- ML backend save validates that the binary checkpoint path exists; talc ML
  save validates that the talc checkpoint path exists.
- The Settings page `Test All` button can test unsaved runtime form values.
- Heuristic runtime tests return success without creating a run.
- ML runtime tests do not mutate saved settings, do not create a run, and
  report per-model checkpoint-loader success or failure on the same `auto`
  device selection used by real pipeline inference.
- New immutable runs snapshot `backend`, `checkpoint`, `talc_backend`,
  `talc_checkpoint`, and `talc_threshold` in `run.json` runtime provenance.
- `Remove all history` asks for confirmation, removes run/Series history,
  clears the loaded run/result state in the browser, and keeps uploads/settings.
- After app restart, `GET /api/settings` returns saved values.
- Browser-local values may remain as a fallback, but server settings are the
  preferred source when available.
