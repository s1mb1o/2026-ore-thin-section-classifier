# Ore Pipeline Apply Prepared Run v0.1

Date: 2026-07-03

## Scope

This spec applies to the v2 ore pipeline UI:

```text
apps/ore_pipeline_web.py
```

It defines what happens when the user presses `Apply` in the `Augmentation` or
`Preprocessing` blocks after a run has already been evaluated.

## Goals

- Preserve the immutable-run model for completed runs.
- Allow users to tune augmentation or preprocessing from a loaded run without
  accidentally mutating that completed run.
- Avoid history noise while a newly prepared run has not been started yet.
- Reuse all safe prerequisite artifacts up to the changed step.
- Clear downstream segmentation, metrics, and report artifacts until `Start`
  evaluates them again.

## User Flow

### No Completed Run Loaded

If an image is uploaded but no run has been started yet:

1. User edits augmentation or preprocessing settings.
2. User presses `Apply`.
3. The current upload preview is refreshed.
4. The same not-yet-started working state remains active.

No immutable run is created only because `Apply` was pressed.

### Completed Run Loaded

If the current workspace state is loaded from a completed run:

1. User edits augmentation or preprocessing settings.
2. User presses `Apply`.
3. The backend creates a new immutable run with status `prepared`.
4. The new run keeps the original image and prerequisite artifacts up to the
   changed step.
5. Downstream sulfide/final masks, metrics, CSV, PDF, and result text are not
   copied.
6. The viewer shows only the available prepared layers.
7. Pressing `Start` continues this prepared run in place and completes the
   missing downstream pipeline.

### Prepared Run Not Started Yet

If the current workspace state is already a `prepared` run:

1. Pressing `Apply` again updates that same prepared run.
2. No additional run is added to history.
3. The prepared run remains mutable only until `Start` is pressed.
4. Once `Start` is pressed, it becomes a normal immutable evaluated run.

## Step Semantics

### Augmentation Apply

- Rebuild original display artifacts.
- Rebuild augmented artifacts when augmentation is enabled.
- Rebuild preprocessing artifacts from the updated augmentation state when
  preprocessing is enabled.
- Preserve upload-level or parent artefact masks when compatible with the new
  analysis size.
- Clear sulfide/final masks, metrics, and reports.

### Preprocessing Apply

- Keep the original image and current augmentation prerequisite.
- Rebuild preprocessing artifacts with the current preprocessing settings.
- If preprocessing is disabled, do not expose a user-facing preprocessed layer.
- Preserve upload-level or parent artefact masks when compatible with the new
  analysis size.
- Clear sulfide/final masks, metrics, and reports.

## API Contract

Create or update a prepared run:

```http
POST /api/runs/{run_id}/prepare
```

Request:

```json
{
  "changed_step": "augmentation",
  "preset": {},
  "augmentation_settings": {}
}
```

`changed_step` is either:

- `augmentation`
- `preprocess`

Response is the current run payload. A completed source run returns a new
`run_id`; a prepared source run returns the same `run_id`.

Start a prepared run:

```http
POST /api/runs/{run_id}/start
```

The response is the same run payload with normal queued/running/completed
progress semantics.

## Run Metadata

Prepared runs record derivation metadata:

```json
{
  "derivation": {
    "type": "apply_pipeline_settings",
    "parent_run_id": "run_...",
    "changed_step": "preprocess",
    "operation": "prepare_from_preprocessing_apply",
    "mutable_until_start": true
  },
  "status": "prepared",
  "stage": "prepared",
  "progress": 0
}
```

When a prepared run is started, the run keeps the same `run_id`; only the
downstream artifacts are evaluated and the status advances to terminal
`completed`, `failed`, or `canceled`.

## UI Contract

- `Apply` on a completed run switches the workspace to the new prepared run.
- The result panel is cleared for prepared runs.
- Unavailable downstream layers are greyed out.
- The status line tells the user that a new run was prepared and that `Start`
  continues evaluation.
- History may show the prepared run, but it must not pretend to have masks,
  metrics, or reports until evaluation finishes.
