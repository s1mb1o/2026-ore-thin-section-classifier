# Ore Pipeline Batch UI Implementation Plan

Scope: this plan is explicitly for the v2 ore pipeline UI in `apps/ore_pipeline_web.py`.

Spec: `docs/ui/v2/specs/ore-pipeline-batch-v0.1.md`.

## Implementation Steps

1. Add server-side batch persistence under `outputs/ore_pipeline_ui/batches/`.
2. Expose JSON endpoints for draft creation, item addition, item metadata update, sequential run, cancellation, direct read, and CSV download.
3. Link each child immutable run back to its batch with `batch_id`, `item_id`, and item index.
4. Add `/batch` and `/batch/{batch_id}` as direct-loadable v2 UI slugs.
5. Add the `Batch` navigation tab and page with multi-image add, shared settings summary, gallery cards, per-item `Edit Metadata...`, `Run Batch`, progress, `Stop`, and per-item `Load`.
6. Reuse the existing v2 metadata modal for both single-image and batch-item targets.
7. Reuse the existing v2 preprocessing and augmentation controls as shared batch settings at run time.
8. Add result-view return context so `Load` opens normal run results and `Back to Batch` returns to the same persisted batch.
9. Add focused tests for sequential batch execution, per-item metadata persistence, route exposure, and required v2 UI controls.
10. Update v2 docs, smoke checklist, `ChangeLog.md`, and `docs/session-sync.md`.

## Verification

- `python3 -m py_compile apps/ore_pipeline_web.py`
- `python3 -m unittest discover -s tests -p 'test_ore_pipeline_web.py' -v`
- Browser smoke: open `/batch`, add multiple images, edit metadata on one card, run the batch, watch only one active card progress, load a child run, and return to `/batch/{batch_id}`.

## Current Status

Implemented in v2 as of 2026-07-03.
