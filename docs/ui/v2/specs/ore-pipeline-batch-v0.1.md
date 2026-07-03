# Ore Pipeline Batch UI v0.1

Scope: this specification is explicitly for the v2 ore pipeline UI implemented in `apps/ore_pipeline_web.py`.

## Goal

Add a `Batch` page that groups multiple input images and runs the existing ore pipeline once per image. A batch is not a merged analysis: each image creates a separate immutable run, while the batch page provides shared setup, progress monitoring, and a gallery-oriented return point.

## User Contract

- The top navigation includes `Batch` beside `Workspace`, `History`, and `Settings`.
- `/batch` is direct-loadable, and `/batch/{batch_id}` reopens a persisted batch.
- `Add images` sits in the gallery section and accepts multiple PNG, JPEG, TIFF, and RAW-extension files.
- The batch gallery shows one card per image with thumbnail, filename, dimensions, item status, item progress, and `Edit Metadata...`.
- Draft image cards have `Remove`; removal deletes only the draft batch item reference, not immutable child runs.
- Metadata editing uses the same curated metadata model as single-image runs and is stored per batch item before execution.
- Preprocessing and runtime augmentation settings are shared by all images in the batch and come from the existing v2 left-panel controls at batch run time.
- `Run Batch` processes images sequentially. Only one item is active at a time.
- Progress is shown on the currently processed image card and retained on completed/failed/canceled items.
- When an item has created a run, its card keeps a `Load` button. `Load` opens that run in the normal run results view.
- Browser Back, or the visible `Back to Batch` result control, returns to the same batch page.

## Data Contract

Batch state is persisted under `outputs/ore_pipeline_ui/batches/{batch_id}/batch_summary.json`.

The persisted batch contains:

- `batch_id`, timestamps, status, aggregate progress, and item counts.
- Shared `settings.preprocess` and `settings.augmentation` used for the run.
- Ordered `items`, each with `item_id`, `index`, `upload_id`, source name/size/hash, status, progress, stage, optional `curated_metadata`, optional `run_id`, and optional error.

Each batch item run remains a normal immutable run under `outputs/ore_pipeline_ui/runs/{run_id}/run.json`. The run includes a `batch` link with `batch_id`, `item_id`, and `index`.

When the batch finishes, `reports/batch_results.csv` is written with item status, run id, ore class, key fractions, and error text.

## API Contract

- `GET /api/batches` lists persisted batches.
- `POST /api/batches` creates a draft batch.
- `GET /api/batches/{batch_id}` reads a batch with item upload previews.
- `POST /api/batches/{batch_id}/items` adds uploaded images to a draft batch.
- `DELETE /api/batches/{batch_id}/items/{item_id}` removes a draft item before it creates a run.
- `PUT /api/batches/{batch_id}/items/{item_id}/metadata` stores curated metadata for a draft item.
- `POST /api/batches/{batch_id}/run` starts sequential processing.
- `POST /api/batches/{batch_id}/cancel` requests cooperative cancellation.
- `GET /api/batches/{batch_id}/results.csv` downloads the batch summary CSV.

## Non-Goals

- No parallel execution in v0.1.
- No merged multi-image metrics beyond the CSV item summary.
- No editing item metadata after that item has already created a run.
- No old/v1 UI implementation.
