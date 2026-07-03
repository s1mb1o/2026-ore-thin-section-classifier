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
- default preprocessing preset:
  - preprocessing enabled;
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

## UI

Add a direct-loadable `/settings` slug page and a `Settings` navigation tab.

The page shows:

- language and theme selectors;
- preprocessing default checkboxes;
- show-tiling default checkbox;
- metadata session-default inputs;
- `Save settings` and `Reset to defaults` actions;
- a clear status line after save/reset/load failure.

## API

Add:

- `GET /api/settings` -> current merged settings;
- `PUT /api/settings` -> validate and persist settings.

Invalid settings should return `400` with a clear error.

## Acceptance

- `/settings` loads the same app shell and selects the Settings page.
- Settings are loaded before workspace defaults are applied.
- Saving settings updates the server-side JSON file.
- After app restart, `GET /api/settings` returns saved values.
- Browser-local values may remain as a fallback, but server settings are the
  preferred source when available.
