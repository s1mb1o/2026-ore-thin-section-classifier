# Ore Pipeline Preprocessing Control Implementation Plan

Date: 2026-07-03

Spec:

```text
docs/ui/v2/specs/ore-pipeline-preprocessing-control-v0.1.md
```

## Current State

`apps/ore_pipeline_web.py` currently renders four preprocessing checkboxes directly in the sidebar:

- illumination normalization;
- noise reduction;
- contrast correction;
- panorama image scaling.

`Start` always calls `prepare_upload(...)`, so a preprocessed artifact and preprocessed display layer are always created.

## Implementation Plan

1. Add a versioned preset gate.
   - Extend browser preset JSON with `preprocessing_enabled`.
   - Accept compatibility alias `enabled`.
   - Keep default enabled.

2. Update server preprocessing.
   - Make `prepare_upload(...)` support `preprocessing_enabled=false`.
   - When disabled, create an internal analysis-scale original copy for pipeline compatibility.
   - Do not add a user-facing `display.preprocessed` layer when disabled.
   - Record `preprocess.enabled` in upload and run metadata.

3. Update UI.
   - Replace the visible checklist with one row: checkbox `Preprocessing`, `Edit...`, `Apply`.
   - Add a preprocessing settings dialog with the four existing settings.
   - Add a compact `(?)` help control beside each setting label with a localized hover/focus description.
   - Replace ambiguous panorama scaling on/off behavior with explicit controls:
     - enabled/off;
     - longest-side pixel bound;
     - scale factor.
   - Make `Apply` check `Preprocessing` and run the existing preview endpoint.
   - Persist the main checkbox plus settings in local storage.

4. Update viewer behavior.
   - Disable `preprocessed` layer and side-by-side option when no preprocessed preview exists or the current tuning choice has preprocessing unchecked.
   - Draw sulfide/final overlays over original display when preprocessing is disabled.
   - Keep tiling independent from panorama scaling; turning panorama scaling off falls back to normal processing size and does not disable tile-manifest creation.

5. Update tests and docs.
   - Add focused tests for skipped preprocessing and UI controls.
   - Cover the four preprocessing help controls in the UI test.
   - Cover explicit panorama scaling bounds/factor metadata.
   - Update `SMOKE_TESTS.md`, `ChangeLog.md`, and `docs/session-sync.md`.

## Verification

Run:

```bash
python3 -m py_compile apps/ore_pipeline_web.py tests/test_ore_pipeline_web.py
python3 -m unittest discover -s tests -p 'test_ore_pipeline_web.py' -v
python3 -m unittest discover -s tests -p 'test_*.py' -v
git diff --check -- apps/ore_pipeline_web.py tests/test_ore_pipeline_web.py ChangeLog.md SMOKE_TESTS.md docs/session-sync.md docs/ui/v2/specs/ore-pipeline-preprocessing-control-v0.1.md docs/ui/v2/plans/29_ore-pipeline-preprocessing-control.md
```
