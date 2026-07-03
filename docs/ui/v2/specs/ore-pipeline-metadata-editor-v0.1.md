# Ore Pipeline Metadata Editor v0.1

Date: 2026-07-03

## Scope

This spec applies only to the v2 local ore pipeline UI:

```text
apps/ore_pipeline_web.py
```

The metadata editor is the popup opened from the selected-image workspace by
`Edit Metadata...`. It lets an operator edit curated microscopy/sample metadata
while keeping raw file metadata read-only.

## Goals

- Add a clear `Edit Metadata...` action for the selected image.
- Show editable domain metadata separately from raw image metadata.
- Preserve EXIF, image headers, ICC/XMP markers, and uploaded source files
  without mutating them.
- Store curated metadata with each immutable run.
- Preserve metadata when `Fix and Restart` creates a derived run.
- Support browser-local session defaults for repeated operator/session fields.
- Keep physical scale conservative: no calibrated microns-per-pixel value is
  inferred from DPI, focal length, digital zoom, or filename `5x` / `10x` hints.

## Non-Goals

- Do not edit EXIF, XMP, ICC profiles, or source image files in place.
- Do not infer calibrated specimen scale from raw camera metadata alone.
- Do not add SEM/XRD metadata fields to the v2 OM-only workflow.
- Do not define a full LIMS/ELN schema.
- Do not add multi-user locking or server-backed operator profiles in v0.1.

## Current Baseline

The current v2 UI already has the core path:

- upload registration extracts bounded raw image metadata into `upload.json`;
- the workspace renders `metadataBtn` and `metadataDialog`;
- the modal has `Domain`, `Raw`, and `Session Defaults` tabs;
- browser JavaScript sends `curated_metadata` to `POST /api/runs/start`;
- server validation rejects non-object `curated_metadata` payloads;
- every non-empty curated payload is written to
  `metadata/curated_metadata.json` under the immutable run directory;
- `run.json` records `input.curated_metadata` and
  `input.curated_metadata_json`;
- edit-derived runs inherit parent curated metadata;
- focused tests live in `tests/test_ore_pipeline_web.py`.

## User Flow

1. User opens the v2 ore pipeline workspace.
2. User uploads or selects one OM image.
3. The UI extracts bounded raw metadata during upload registration.
4. User clicks `Edit Metadata...`.
5. A modal opens with three tabs:
   - `Domain`
   - `Raw`
   - `Session Defaults`
6. `Domain` shows editable curated fields.
7. `Raw` shows read-only file/header/EXIF/profile metadata for the selected
   upload.
8. `Session Defaults` shows browser-local values reusable for later images in
   the same operator session.
9. User saves.
10. The next `Start` sends `curated_metadata` in `POST /api/runs/start`.
11. The server writes `metadata/curated_metadata.json` under the run folder.
12. Result, history, and edit-derived runs preserve the metadata payload.

## UI Placement

The workspace sidebar should keep metadata near the selected image:

```text
Input image
###########
###########

[Edit Metadata...]
[x] Preprocessing [Edit...] [Apply]
```

The modal should be a compact tool surface, not a nested card. It must keep
`Save`, `Cancel`, and `Apply Session Defaults` actions visible at the bottom.

## Modal Contract

The modal must:

- keep tab selection stable while editing;
- preserve unsaved domain edits while switching tabs;
- close on `Cancel` without changing the pending run payload;
- save only on explicit `Save`;
- store session defaults in browser `localStorage`;
- apply defaults only to allowlisted editable fields;
- show scale warnings immediately when a pixel size is present without a
  calibrated scale source;
- display raw metadata as read-only table/key-value content;
- avoid hidden auto-conversions from raw metadata into calibrated values;
- fit at desktop and mobile widths without text overlap.

## Domain Fields

The v0.1 UI may expose a compact subset, but the server-side schema should
accept and preserve the broader domain payload below.

The Domain tab groups visible fields by how operators think about reuse:

- `Session specific`: values likely to repeat for a microscopy session, shown at
  the top of the tab. In v0.1 this includes `project`, `om_instrument`,
  `om_objective_magnification`, `scale_source`, `pixel_size_um` as
  `Scale value, um/px`, and `scale_confidence`.
- `Sample specific`: values tied to the current image/sample, shown below the
  session-specific block. In v0.1 this includes `sample_id`, `run_label`,
  `source_role`, `task_label`, `filename_magnification_hint`, `review_status`,
  `exclude_from_training`, and notes.

### Identity

- `sample_id`
- `run_label`
- `project`
- `campaign_id`
- `batch_id`
- `session_id`
- `source_dataset`
- `source_path`
- `source_role`
  - `original_image`
  - `annotation_image`
  - `panorama`
  - `class_folder_photo`
  - `derived_mask`
  - `unknown`

### Domain And Task

- `material_domain`
- `sample_type`
  - `polished_section`
  - `thin_section`
  - `annshliff`
  - `polished_grain_mount`
  - `panorama`
  - `powder`
  - `unknown`
- `modality`
  - `om`
  - `metadata_only`
- `official_folder_label`
- `task_label`
  - `ordinary_intergrowth`
  - `fine_intergrowth`
  - `talcose`
  - `talc_region`
  - `background_or_matrix`
  - `unknown`
- `exclude_from_training`: exclude this image from training/validation dataset
  exports; it does not exclude the image from the current run, report, or
  history.
- `review_status`
  - `unreviewed`
  - `reviewed`
  - `needs_manual_review`
  - `needs_manual_mask`
  - `bad_image`

### Talc Annotation Pairing

These fields are optional and mainly support the official talc annotation
workflow:

- `paired_original_path`
- `paired_annotation_path`
- `pairing_status`
  - `not_applicable`
  - `basename_match`
  - `same_dimensions`
  - `missing_pair`
  - `dimension_mismatch`
- `annotation_type`
  - `none`
  - `blue_line_talc_region`
  - `manual_mask`
  - `generated_mask`
  - `reviewed_mask`
- `annotation_source`
- `mask_conversion_method`
- `mask_review_status`
- `mask_reviewer`

### OM Acquisition

- `om_instrument`
- `om_camera_model`
- `om_objective_magnification`
- `om_camera_adapter`
- `om_illumination`
  - `reflected_light`
  - `transmitted_light`
  - `brightfield`
  - `unknown`
- `om_exposure_mode`
- `om_white_balance_mode`
- `filename_magnification_hint`
- `magnification_hint_source`
  - `filename`
  - `operator`
  - `raw_metadata`
  - `unknown`

### Scale

- `pixel_size_um`: scale value in micrometers per pixel, entered manually or
  from a calibrated source.
- `microns_per_pixel`: API synonym accepted by the v2 metrics path; when both
  fields are supplied, `microns_per_pixel` wins.
- `scale_source`
  - `unavailable`
  - `manual`
  - `visible_scale_bar`
  - `instrument_sidecar`
  - `calibration_slide`
  - `stage_transform`
- `scale_confidence`
  - `none`
  - `weak`
  - `calibrated`
- `calibration_id`
- `calibration_date`
- `scale_notes`

Scale rules:

- `pixel_size_um` is the explicit scale value input for manual scale entry.
- `pixel_size_um` must stay empty unless the source is calibrated or the
  operator explicitly enters a manual/calibrated value.
- DPI, JFIF density, EXIF focal length, digital zoom, and filename `5x` / `10x`
  tokens must not auto-fill `pixel_size_um`.
- Filename magnification may populate only `filename_magnification_hint`.
- If scale is unavailable, reports may show pixel areas and fractions but not
  absolute physical areas.
- If scale is calibrated, the result metrics table and `metrics.csv` may show
  physical areas derived from the current analysis mask grid; see
  `docs/ui/v2/specs/ore-pipeline-scale-metrics-v0.1.md`.

### Preparation And Quality

- `preparation`
  - `polished`
  - `etched`
  - `coated`
  - `mounted`
  - `unknown`
- `surface_quality`
- `known_artifacts`
- `quality_notes`
- `operator_notes`

## Session Defaults

These values commonly repeat across one operator session and are safe as
browser-local defaults:

- `project`
- `campaign_id`
- `session_id`
- `operator`
- `source_dataset`
- `material_domain`
- `sample_type`
- `modality`
- `official_folder_label`
- `om_instrument`
- `om_camera_model`
- `om_objective_magnification`
- `om_camera_adapter`
- `om_illumination`
- `om_exposure_mode`
- `om_white_balance_mode`
- `preparation`
- `surface_quality`
- `scale_source`
- `pixel_size_um`
- `microns_per_pixel`
- `scale_confidence`
- `calibration_id`
- `calibration_date`
- `review_status`

These should not be applied by default because they are sample-specific or
source-specific:

- `sample_id`
- `run_label`
- `source_path`
- `source_role`
- `task_label`
- `paired_original_path`
- `paired_annotation_path`
- `pairing_status`
- `annotation_type`
- `filename_magnification_hint`
- `exclude_from_training`
- free-text notes unless explicitly copied by the operator.

## Raw Tab

The `Raw` tab is read-only and should show bounded metadata from the selected
upload:

- original filename;
- stored path;
- extension;
- file size;
- digest;
- width and height;
- image format;
- mode/channel representation;
- DPI/JFIF density when present;
- EXIF presence and selected decoded tags;
- ICC profile presence;
- XMP marker presence;
- extraction warnings.

Raw metadata is provenance and debugging context. It is not a calibrated domain
schema and must not overwrite curated fields automatically.

## Stored Payload

The run payload uses this schema:

```json
{
  "schema_version": "ore-pipeline-curated-metadata-v0.1",
  "source": "metadata_editor",
  "generated_at": "2026-07-03T00:00:00Z",
  "domain": {},
  "raw_summary": {},
  "session_defaults_applied": {},
  "warnings": [],
  "extra": {}
}
```

Submission:

```http
POST /api/runs/start
Content-Type: application/json

{
  "upload_id": "...",
  "preprocessing": {},
  "curated_metadata": {}
}
```

Persistence:

```text
outputs/ore_pipeline_ui/runs/{run_id}/metadata/curated_metadata.json
outputs/ore_pipeline_ui/runs/{run_id}/run.json
```

`run.json` records:

```json
{
  "input": {
    "curated_metadata": {},
    "curated_metadata_json": "metadata/curated_metadata.json"
  }
}
```

API behavior:

- missing, empty, or content-free `curated_metadata` creates no metadata file;
- object payloads are normalized and unknown keys are preserved under `extra`;
- non-object payloads return a clear `400` error;
- edit-derived runs inherit parent metadata.

## Acceptance Criteria

- The workspace renders `Edit Metadata...` only when an upload is available.
- The modal has `Domain`, `Raw`, and `Session Defaults` tabs.
- Raw metadata is read-only and does not modify source files.
- Saving non-empty curated fields sends `curated_metadata` to
  `POST /api/runs/start`.
- Completed runs write `metadata/curated_metadata.json` and expose the payload
  in `run.json`.
- Edit-derived runs inherit curated metadata from the parent.
- Non-object API `curated_metadata` is rejected with a clear `400` error.
- DPI, focal length, and filename magnification never auto-fill
  `pixel_size_um`.
- Browser-local session defaults apply only to allowlisted repeated fields.

## Verification

Run:

```bash
python3 -m py_compile apps/ore_pipeline_web.py tests/test_ore_pipeline_web.py
python3 -m unittest discover -s tests -p 'test_ore_pipeline_web.py' -v
git diff --check -- apps/ore_pipeline_web.py tests/test_ore_pipeline_web.py docs/ui/v2/specs/ore-pipeline-metadata-editor-v0.1.md
```
