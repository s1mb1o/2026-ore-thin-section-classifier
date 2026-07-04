# Ore Pipeline REST API v0.1

Date: 2026-07-03

## Scope

This spec applies to the v2 ore pipeline browser service:

```text
apps/ore_pipeline_web.py
```

It documents the local REST API used by the UI and exposed on the `API` page.
The API controls image upload, preprocessing, immutable run jobs, result
downloads, Series processing, app settings, and service health.

## Concepts

- Upload: an input image persisted under the current workspace with raw
  metadata and preview pyramids.
- Run: the service job object. Runs are immutable once completed; edit/apply
  actions create derived or prepared runs instead of mutating completed output.
- Runtime provenance: completed run payloads expose `runtime`, `tiling`,
  `tile_progress`, `elapsed_seconds`, `masks`, and `reports` fields so the UI
  can show backend/model sources, operation time, tile counts/progress, stage
  results, and artifact paths after the sulfide-grain table. Heuristic/rule-only
  stages keep checkpoint fields empty/null.
- Artifacts: masks, metrics, CSV/PDF reports, display layers, and full ZIP
  exports created by a run.
- Sulfide grain rows: per-connected-component result rows derived from
  `reports/component_features.csv`. `GET /api/runs/{run_id}` returns them as
  `sulfide_grains.items` with `component_id`, ordinary/fine type, `area_px`,
  and `share_percent`, plus a `sulfide_grains.label_map` URL for the generated
  RGB component-label PNG used by the browser to draw selected-grain outlines.
- Series: user-facing batch workflow for processing several uploads
  sequentially with shared settings. Route and storage names remain `batch`
  for compatibility.
- Settings: server-backed defaults shared by all browsers opening the same
  workspace, including binary sulfide backend/checkpoint and talc
  source/checkpoint/threshold.
- Status: read-only service health and resource diagnostics.

## Endpoint Contract

All JSON endpoints return UTF-8 JSON. Upload uses `multipart/form-data`.
Downloads return CSV, PDF, or ZIP payloads.

| Endpoint | Purpose | Implemented |
| --- | --- | --- |
| `GET /api/status` | Service health, backend/model status, CPU/GPU/RAM/flash, storage, active jobs | yes |
| `POST /api/uploads` | Upload one PNG/JPEG/TIFF/RAW-extension file as form field `file` | yes |
| `GET /api/uploads/{upload_id}` | Read upload metadata and previews | yes |
| `POST /api/uploads/{upload_id}/preprocess` | Refresh augmentation/preprocessing artifacts before a run | yes |
| `POST /api/uploads/{upload_id}/artifact-mask` | Save a pre-run artifact exclusion mask | yes |
| `POST /api/runs/start` | Create and optionally run a new immutable job from an upload | yes |
| `GET /api/runs` | List persisted run history with status/progress/elapsed timing | yes |
| `GET /api/runs/{run_id}` | Read run status, progress, elapsed time, metadata, masks, previews, metrics, sulfide-grain rows, downloads | yes |
| `POST /api/runs/{run_id}/cancel` | Request cooperative cancellation | yes |
| `POST /api/runs/{run_id}/prepare` | Create or update a prepared run from changed settings | yes |
| `POST /api/runs/{run_id}/start` | Continue a prepared run in place | yes |
| `POST /api/runs/{run_id}/fix` | Create a derived run from sulfide/final/artifact mask edits | yes |
| `GET /api/runs/{run_id}/files` | List immutable run files with sizes and image dimensions | yes |
| `GET /api/runs/{run_id}/metrics.csv` | Download hierarchical metrics CSV | yes |
| `GET /api/runs/{run_id}/report.pdf` | Generate/download the current PDF report | yes |
| `GET /api/runs/{run_id}/artifacts.zip` | Download the full run artifact archive | yes |
| `GET /api/batches` | List Series summaries | yes |
| `POST /api/batches` | Create a Series draft | yes |
| `GET /api/batches/{batch_id}` | Read a Series with items, settings, progress, and child runs | yes |
| `POST /api/batches/{batch_id}/items` | Add uploaded images to a draft Series | yes |
| `DELETE /api/batches/{batch_id}/items/{item_id}` | Remove a draft item before it has a run | yes |
| `PUT /api/batches/{batch_id}/items/{item_id}/metadata` | Save per-item curated metadata | yes |
| `PUT /api/batches/{batch_id}/settings` | Save shared Series settings before start | yes |
| `POST /api/batches/{batch_id}/run` | Start sequential Series processing | yes |
| `POST /api/batches/{batch_id}/cancel` | Request cooperative Series cancellation | yes |
| `GET /api/batches/{batch_id}/results.csv` | Download Series result CSV | yes |
| `DELETE /api/batches/{batch_id}` | Remove a completed/failed/canceled Series and child runs | yes |
| `GET /api/settings` | Read server-backed app defaults | yes |
| `PUT /api/settings` | Validate and persist app defaults | yes |
| `POST /api/runtime/test` | Validate unsaved runtime values and probe selected ML checkpoints | yes |

## API Documentation Page

The UI exposes a direct-loadable `API` / `/api` page:

- left-side endpoint navigation grouped by service area;
- method badges, route paths, localized summaries, request examples, and
  response examples;
- live sandboxes that send requests to the same app server;
- file upload sandbox for `POST /api/uploads`;
- JSON body editors for mutating endpoints;
- binary download sandboxes report HTTP status, content type, size, and
  disposition instead of dumping raw PDF/ZIP bytes.

The public reference intentionally starts with the stable service-control
endpoints: health, upload, preprocessing, runs, artifacts, Series, and settings.
Edit-mask endpoints exist and remain covered by focused feature specs.

## Validation Rules

- Unsupported upload extensions return `400`.
- Uploads above the configured size limit return `413`.
- Repeated same-file uploads must allocate distinct `upload_id` values.
- JSON bodies that should be objects return `400` when given another type.
- Prepared-run continuation is only valid for `prepared` runs.
- Cancellation is cooperative and only meaningful for queued/running jobs.
- Active Series cannot be removed through delete endpoints.
- ML binary sulfide runtime requires an existing `runtime.checkpoint`.
- Talc ML runtime requires an existing `runtime.talc_checkpoint`; talc
  threshold is clamped to the supported `0.01..0.99` range.

## Test Coverage

Focused coverage lives in:

```text
tests/test_ore_pipeline_web.py
```

The tests cover slug routing, API/status/settings payloads, run artifact
downloads, Series processing, API-page DOM/static contract, and repeated
same-file upload ID allocation.
