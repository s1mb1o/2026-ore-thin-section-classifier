# Plan 31: Ore Pipeline System Settings

Date: 2026-07-03

Spec: `docs/ui/v2/specs/ore-pipeline-system-settings-v0.1.md`

## Goal

Add a Settings page to `apps/ore_pipeline_web.py` for system-wide,
long-lasting app defaults.

## Implementation

1. Add default settings constants plus server-side normalization/validation.
2. Store settings in `workspace_dir/settings/app_settings.json`.
3. Add `GET /api/settings` and `PUT /api/settings`.
4. Add runtime settings for `backend` and `checkpoint`.
5. Apply runtime changes immediately after a successful save, with active-job
   protection.
6. Add a runtime `Test All` path:
   - `POST /api/runtime/test`;
   - accept unsaved runtime form values;
   - return immediate success for `heuristic`;
   - for `ml`, validate checkpoint existence and run a bounded subprocess probe
     that loads the checkpoint through `ore_classifier.model_io` with the same
     `auto` device selection as real inference;
   - do not save settings or create a run.
7. Snapshot backend/checkpoint into every immutable run.
8. Add `/settings` to slug routing.
9. Add a Settings navigation tab and page section.
10. Load settings on page startup and apply them before workspace defaults.
11. Save/reset/test settings from the page.
12. Add a destructive History panel action:
   - `DELETE /api/history`;
   - remove all persisted run and Series artifact folders;
   - keep uploads and settings;
   - reject the request while any run, Series job, or foreground operation is
     active;
   - clear loaded run/result/history state in the browser after success.
13. Keep existing localStorage use as fallback only.

## Verification

- Add unit coverage for settings persistence and API validation.
- Add unit coverage for live runtime switching, restart persistence, and invalid
  ML checkpoint validation.
- Add unit coverage for `/api/runtime/test` heuristic success, ML probe success,
  ML probe failure, and missing checkpoint validation.
- Add unit coverage for bulk history removal and active-job rejection.
- Extend rendered-page assertions for `/settings` UI controls and routing.
- Run focused ore pipeline web tests.
- Run `git diff --check` on touched files.
