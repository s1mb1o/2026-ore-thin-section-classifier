# Ore Pipeline Metadata Editor Implementation Plan

Date: 2026-07-03

Spec:

```text
docs/ui/v2/specs/ore-pipeline-metadata-editor-v0.1.md
```

Implementation target:

```text
apps/ore_pipeline_web.py
```

## Current State

The v2 ore pipeline UI already has a working metadata editor baseline:

- `CURATED_METADATA_SCHEMA_VERSION = "ore-pipeline-curated-metadata-v0.1"`;
- `extract_image_raw_metadata(...)` stores bounded raw image metadata in upload
  records;
- `normalize_curated_metadata_payload(...)` accepts object payloads, preserves
  unknown keys under `extra`, and rejects non-object payloads;
- the workspace renders `metadataBtn` and `metadataDialog`;
- the modal has `Domain`, `Raw`, and `Session Defaults` tabs;
- JavaScript sends `curated_metadata` to `POST /api/runs/start`;
- run creation writes `metadata/curated_metadata.json`;
- edit-derived runs inherit parent curated metadata;
- `tests/test_ore_pipeline_web.py` covers persistence, API validation,
  inherited metadata, and required page controls.

## Decision

Keep metadata editor logic inside the v2 `http.server` + vanilla JS app for
v0.1. Extract shared helpers only if another v2 page starts duplicating
substantial metadata UI logic.

Canonical storage:

```text
outputs/ore_pipeline_ui/runs/{run_id}/metadata/curated_metadata.json
outputs/ore_pipeline_ui/runs/{run_id}/run.json
```

Canonical API entry point:

```text
POST /api/runs/start
```

## Phase 1: Contract Hardening

Files:

```text
apps/ore_pipeline_web.py
tests/test_ore_pipeline_web.py
docs/ui/v2/specs/ore-pipeline-metadata-editor-v0.1.md
```

Tasks:

- Keep the Domain tab split into `Session specific` fields first and
  `Sample specific` fields second.
- Compare visible fields against the spec's Domain field groups.
- Keep the current compact v0.1 UI subset, but ensure the server accepts and
  preserves the wider schema.
- Ensure unknown client fields are preserved under `extra`.
- Ensure content-free payloads do not create empty metadata artifacts.
- Keep `raw_summary`, `session_defaults_applied`, and `warnings` normalized.

Acceptance:

- Existing metadata tests still pass.
- A payload with future schema fields round-trips without field loss.
- A payload with only empty `domain`, `raw_summary`, and `warnings` writes no
  metadata file.

## Phase 2: Raw Metadata Extraction

Files:

```text
apps/ore_pipeline_web.py
tests/test_ore_pipeline_web.py
```

Tasks:

- Keep extraction bounded to header-level reads for JPEG/PNG/TIFF where
  possible.
- Preserve:
  - original filename;
  - stored path;
  - extension;
  - file size;
  - digest;
  - width and height;
  - image format;
  - image mode;
  - DPI/JFIF density;
  - EXIF selected tags;
  - ICC/XMP presence;
  - warnings.
- Keep RAW-extension uploads conservative: record basic file facts and a warning
  when camera decoder metadata is unavailable.
- Add fixtures for EXIF-bearing JPEG, no-EXIF JPEG, BMP or TIFF, and a large
  header-only image if not already covered.

Acceptance:

- Missing EXIF is represented as a warning, not an error.
- Very large images do not require full pixel decode for metadata extraction.
- Raw extraction failures never block upload registration.

## Phase 3: Modal UI Hardening

Files:

```text
apps/ore_pipeline_web.py
tests/test_ore_pipeline_web.py
```

Tasks:

- Keep `Edit Metadata...` disabled until an upload exists.
- Keep `project` and `microscope/camera` at the top of the Domain tab as
  session-specific fields.
- Preserve unsaved edits while switching tabs.
- Ensure `Cancel` closes without changing the pending `curated_metadata` state.
- Ensure `Save` updates only the pending run payload.
- Keep raw metadata read-only.
- Keep session defaults browser-local under a stable localStorage key.
- Apply defaults only to allowlisted repeated fields.
- Keep text and controls responsive at narrow viewport widths.

Acceptance:

- HTML smoke coverage finds the button, dialog, tab controls, warning element,
  localStorage key, and submission function.
- Browser smoke can open the modal, save a field, cancel a second edit, and see
  that only the saved payload is submitted.

## Phase 4: Scale Guardrails

Files:

```text
apps/ore_pipeline_web.py
tests/test_ore_pipeline_web.py
```

Tasks:

- Keep the warning condition:
  - `pixel_size_um` is present; and
  - `scale_confidence` is not `calibrated` or `scale_source` is unavailable.
- Never auto-fill `pixel_size_um` from:
  - DPI;
  - JFIF density;
  - EXIF focal length;
  - digital zoom;
  - filename `5x` / `10x` hints.
- Store filename magnification only as `filename_magnification_hint`.
- Label `pixel_size_um` as the manual/calibrated scale value in um/px.
- Ensure result/report text uses pixel areas and fractions unless scale is
  calibrated.

Acceptance:

- Tests assert the UI contains the scale warning branch.
- A raw metadata object containing DPI does not populate `pixel_size_um`.
- Reports do not show physical area units when scale is unavailable.

## Phase 5: Run, History, And Edit-Derived Runs

Files:

```text
apps/ore_pipeline_web.py
tests/test_ore_pipeline_web.py
```

Tasks:

- Keep `POST /api/runs/start` accepting `curated_metadata`.
- Keep completed `run.json` exposing:

```json
{
  "input": {
    "curated_metadata": {},
    "curated_metadata_json": "..."
  }
}
```

- Keep edit-derived runs inheriting parent metadata.
- Make history/result rendering show a compact metadata summary when useful.
- Add an artifact link only if the artifact-serving layer supports it
  consistently; otherwise rely on `input.curated_metadata_json`.

Acceptance:

- API start with valid `curated_metadata` returns a completed run with metadata
  in `run.json`.
- API start with non-object `curated_metadata` returns `400`.
- Edit-derived runs preserve parent `domain.sample_id`.

## Phase 6: Documentation

Files:

```text
docs/ui/v2/specs/ore-pipeline-metadata-editor-v0.1.md
docs/ui/v2/plans/30_ore-pipeline-metadata-editor.md
docs/ui/v2/plans/28_ore-pipeline-ui.md
apps/README.md
docs/session-sync.md
ChangeLog.md
```

Tasks:

- Keep metadata editor documentation in the v2 repository.
- Link the spec/plan from the v2 ore pipeline UI plan and handoff docs.
- Keep app README focused on operator-visible behavior.

Acceptance:

- The v2 session handoff points future agents to this spec and plan.
- The verification commands below pass.

## Verification

Run:

```bash
python3 -m py_compile apps/ore_pipeline_web.py tests/test_ore_pipeline_web.py
python3 -m unittest discover -s tests -p 'test_ore_pipeline_web.py' -v
git diff --check -- apps/ore_pipeline_web.py tests/test_ore_pipeline_web.py docs/ui/v2/specs/ore-pipeline-metadata-editor-v0.1.md docs/ui/v2/plans/30_ore-pipeline-metadata-editor.md docs/ui/v2/plans/28_ore-pipeline-ui.md apps/README.md docs/session-sync.md ChangeLog.md
```

Optional browser smoke:

1. Start the app:

   ```bash
   python3 apps/ore_pipeline_web.py --host 127.0.0.1 --port 0
   ```

2. Upload a JPEG.
3. Open `Edit Metadata...`.
4. Confirm `Domain`, `Raw`, and `Session Defaults` tabs.
5. Save `sample_id`, `project`, and scale fields.
6. Start a run.
7. Confirm `run.json` and `metadata/curated_metadata.json`.
8. Create a `Fix and Restart` derived run and confirm metadata inheritance.
