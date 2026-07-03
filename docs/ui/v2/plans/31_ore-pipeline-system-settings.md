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
4. Add `/settings` to slug routing.
5. Add a Settings navigation tab and page section.
6. Load settings on page startup and apply them before workspace defaults.
7. Save/reset settings from the page.
8. Keep existing localStorage use as fallback only.

## Verification

- Add unit coverage for settings persistence and API validation.
- Extend rendered-page assertions for `/settings` UI controls and routing.
- Run focused ore pipeline web tests.
- Run `git diff --check` on touched files.
