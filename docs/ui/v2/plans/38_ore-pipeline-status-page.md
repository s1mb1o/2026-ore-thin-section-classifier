# Ore Pipeline Status Page Implementation Plan

Date: 2026-07-03

Spec:

```text
docs/ui/v2/specs/ore-pipeline-status-page-v0.1.md
```

## Plan

1. Backend diagnostics.
   - Add stdlib-only helpers for CPU load, memory, disk, directory sizes, and
     optional NVIDIA GPU status through `nvidia-smi`.
   - Add `OrePipelineStore.status_payload()`.
   - Include app uptime, backend, checkpoint presence, history sizes, active
     jobs, and health checks.

2. API and routing.
   - Add `GET /api/status`.
   - Add direct-loadable `/status`.

3. UI.
   - Add `Status` / `Статус` tab.
   - Hide the workflow sidebar on `/status`.
   - Render operational cards, health checks, and storage/history tables.
   - Localize all labels in Russian and English.
   - Use explicit `Refresh` instead of background polling.

4. Tests and docs.
   - Cover the payload shape and `/api/status`.
   - Cover the `/status` slug route.
   - Cover required UI controls/static strings.
   - Update smoke tests, changelog, and session sync.

## Verification

Run:

```bash
python3 -m py_compile apps/ore_pipeline_web.py tests/test_ore_pipeline_web.py
python3 -m unittest discover -s tests -p 'test_ore_pipeline_web.py' -v
node --check outputs/test_ore_pipeline_web_inline.js
git diff --check -- \
  apps/ore_pipeline_web.py \
  tests/test_ore_pipeline_web.py \
  SMOKE_TESTS.md \
  ChangeLog.md \
  docs/session-sync.md \
  docs/ui/v2/specs/ore-pipeline-status-page-v0.1.md \
  docs/ui/v2/plans/38_ore-pipeline-status-page.md
```
