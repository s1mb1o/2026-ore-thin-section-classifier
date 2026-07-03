# Ore Pipeline REST API and Docs Page Implementation Plan

Date: 2026-07-03

Spec:

```text
docs/ui/v2/specs/ore-pipeline-rest-api-v0.1.md
```

## Implemented

1. Core REST service.
   - Upload input image: `POST /api/uploads`.
   - Read upload and preprocessing state: `GET /api/uploads/{upload_id}`.
   - Refresh preprocessing/augmentation artifacts:
     `POST /api/uploads/{upload_id}/preprocess`.
   - Define/start jobs as immutable runs: `POST /api/runs/start`.
   - Read jobs/results: `GET /api/runs`, `GET /api/runs/{run_id}`.
   - Cancel active jobs: `POST /api/runs/{run_id}/cancel`.
   - Export results: `files`, `metrics.csv`, `report.pdf`, and
     `artifacts.zip` endpoints.
   - Batch/Series lifecycle: create, read, add items, update metadata/settings,
     run sequentially, cancel, remove, and download CSV.
   - Health: `GET /api/status`.
   - Settings: `GET/PUT /api/settings`.

2. API documentation UI.
   - Added `API` tab and direct `/api` route.
   - Added an Ozon-docs-style endpoint navigation column plus endpoint cards.
   - Added localized Russian/English titles and summaries.
   - Added live sandboxes for core service-control endpoints.
   - Added file input handling for multipart upload.
   - Added JSON request editors and response panels.
   - Added binary-download handling for PDF/ZIP sandboxes.

3. Reliability.
   - Repeated same-file uploads now include nanosecond entropy under the store
     lock so upload IDs do not collide during fast batching or tests.

4. Tests and docs.
   - Extended `tests/test_ore_pipeline_web.py` for `/api`, API DOM controls,
     documented endpoint strings, binary sandbox handling, and upload ID
     uniqueness.
   - Updated smoke, changelog, and session handoff docs.

## Remaining Hardening

1. Add public cards for the lower-level edit-mask endpoints:
   `artifact-mask`, `prepare`, prepared-run `start`, and `fix`.
2. Add a non-destructive sandbox mode for destructive endpoints such as delete
   and cancel when the request references a real run or Series.
3. Split the API reference data into a small structured JSON/module if the
   endpoint list grows beyond the current single-file UI.
4. Add browser-level visual regression for `/api` once a browser test harness is
   already running for the v2 UI.

## Verification

Run:

```bash
python3 -m py_compile apps/ore_pipeline_web.py tests/test_ore_pipeline_web.py
python3 -m unittest discover -s tests -p 'test_ore_pipeline_web.py' -v
git diff --check -- \
  apps/ore_pipeline_web.py \
  tests/test_ore_pipeline_web.py \
  SMOKE_TESTS.md \
  ChangeLog.md \
  docs/session-sync.md \
  docs/ui/v2/specs/ore-pipeline-rest-api-v0.1.md \
  docs/ui/v2/plans/39_ore-pipeline-rest-api-docs.md
```
