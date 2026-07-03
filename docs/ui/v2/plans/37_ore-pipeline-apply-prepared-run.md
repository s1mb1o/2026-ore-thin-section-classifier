# Ore Pipeline Apply Prepared Run Implementation Plan

Date: 2026-07-03

Spec:

```text
docs/ui/v2/specs/ore-pipeline-apply-prepared-run-v0.1.md
```

## Current State

`Apply` in `Augmentation` and `Preprocessing` refreshes upload previews before a
run starts. Loading a completed run also restores its upload/settings so the
user can tune parameters, but completed run artifacts must remain immutable.

## Plan

1. Add a backend prepared-run state.
   - Accept completed and prepared runs as sources.
   - For completed sources, create a new run directory.
   - For prepared sources, reuse and rebuild the same run directory before
     `Start`.
   - Record derivation metadata with parent run, changed step, and operation.

2. Rebuild prerequisites only.
   - Use the existing upload preparation path for original, augmentation, and
     preprocessing artifacts.
   - Preserve compatible artefact masks.
   - Clear sulfide/final masks, metrics, reports, and result text.
   - Generate display pyramids only for available prepared layers.

3. Add API endpoints.
   - `POST /api/runs/{run_id}/prepare` for Apply after a completed/prepared run.
   - `POST /api/runs/{run_id}/start` to continue a prepared run in place.

4. Update browser behavior.
   - Detect completed/prepared current runs inside the shared Apply handler.
   - Keep existing upload-preview Apply behavior when no completed run is active.
   - Clear result panels for prepared runs and show only ready layers.
   - Start prepared runs through the new endpoint instead of creating another
     fresh run.

5. Add regression tests.
   - Applying after a completed run creates a new prepared run.
   - Starting the prepared run completes the same run id.
   - Re-applying before Start updates the same prepared run.
   - Static UI contract includes prepared-run statuses and API calls.

## Verification

Run:

```bash
python3 -m py_compile apps/ore_pipeline_web.py tests/test_ore_pipeline_web.py
python3 -m unittest discover -s tests -p 'test_ore_pipeline_web.py' -v
node --check /tmp/ore_pipeline_web_inline.js
git diff --check -- \
  apps/ore_pipeline_web.py \
  tests/test_ore_pipeline_web.py \
  SMOKE_TESTS.md \
  ChangeLog.md \
  docs/session-sync.md \
  docs/ui/v2/specs/ore-pipeline-apply-prepared-run-v0.1.md \
  docs/ui/v2/plans/37_ore-pipeline-apply-prepared-run.md
```
