# Ore UI Artifact Fix and Restart Use Case - 2026-07-05

Live service: `http://127.0.0.1:63589`, restarted after the fix with `/opt/homebrew/opt/python@3.14/bin/python3.14`.

## Scenario

Requested use case:

1. Select an image.
2. Start processing.
3. Open `Fix me`, draw an artefact region, and press `Fix and Restart`.
4. Confirm the edited artefact region is excluded from evaluated results.
5. Confirm a new immutable run is created.
6. Press `Start` again and confirm the assigned artefact region is reused for the new run.

Test image:

```text
dataset/Фото руд по сортам. ч2/тонкие/69 1.jpg
```

Runtime defaults observed from `/api/status`:

```text
sulfide backend = ml
talc backend = ml
grain backend = heuristic
```

The test used the same browser/API path as the UI for image selection: `POST /api/uploads` multipart field `file`.

## Initial Failure

The first live run proved that the post-run artefact edit itself worked, but a later plain `Start` from the same uploaded image did not reuse the artefact mask.

Failed run sequence:

```text
upload_id: 20260705_032504_470162000_e169c1215e
parent run: run_20260705_032504_519850000_14815679
edit run: edit_20260705_032528_241575000_490b5195
repeat start run: run_20260705_032528_358119000_14815679
```

Edited rectangle `(y1, y2, x1, x2) = (142, 192, 260, 330)`.

Before edit, the rectangle contained analyzed and sulfide pixels:

```json
{
  "masks/analyzed_mask.png": 888420,
  "masks/sulfide_mask.png": 734910,
  "masks/talc_mask.png": 0,
  "masks/final_mask.png": 5756,
  "masks/artifact_mask.png": null
}
```

After `Fix and Restart`, the derived run correctly excluded the rectangle:

```json
{
  "masks/analyzed_mask.png": 0,
  "masks/sulfide_mask.png": 0,
  "masks/talc_mask.png": 0,
  "masks/final_mask.png": 0,
  "masks/artifact_mask.png": 892500
}
```

But the following plain `Start` lost the artifact mask:

```json
{
  "masks/analyzed_mask.png": 888420,
  "masks/sulfide_mask.png": 734910,
  "masks/talc_mask.png": 0,
  "masks/final_mask.png": 5756,
  "masks/artifact_mask.png": null
}
```

Root cause: `create_edit_run(..., edit_layer="artifact")` created an immutable derived run with `input/artifact_mask.png`, but did not persist that mask back onto the upload-level artifact-mask state used by the next `/api/runs/start`.

## Fix

`OrePipelineStore` now writes upload-level artifact-mask metadata through one helper for both paths:

- pre-run `POST /api/uploads/{upload_id}/artifact-mask`;
- post-run `POST /api/runs/{run_id}/fix` with `edit_layer="artifact"`.

Existing runs remain immutable. The mutable upload now remembers the latest assigned artifact region, so later starts from the same selected image inherit the exclusion mask.

Regression coverage:

```text
tests/test_ore_pipeline_web.py::OrePipelineWebTest.test_artifact_edit_creates_derived_run_and_excludes_pixels
```

The test now verifies:

- `Fix and Restart` creates a new derived run;
- the edited artifact region is excluded from analyzed/sulfide/talc/final masks;
- the upload payload receives `artifact_mask`;
- a later `start_run(upload_id, ...)` uses that artifact mask.

## Passing Retest

Retest sequence after service restart:

```text
upload_id: 20260705_032854_821246000_e169c1215e
parent run: run_20260705_032854_858818000_e4b44730
edit run: edit_20260705_032921_619460000_c4b1c249
repeat start run: run_20260705_032921_741475000_e4b44730
```

The upload payload after `Fix and Restart` reported:

```text
artifact_mask = present
source_run_id = edit_20260705_032921_619460000_c4b1c249
```

The same rectangle was excluded in both the edited run and the later plain start:

```json
{
  "masks/analyzed_mask.png": 0,
  "masks/sulfide_mask.png": 0,
  "masks/talc_mask.png": 0,
  "masks/final_mask.png": 0,
  "masks/artifact_mask.png": 892500
}
```

Final live checks:

```json
{
  "new_run_created_by_fix": true,
  "edited_region_excluded_analyzed": true,
  "edited_region_excluded_sulfide": true,
  "edited_region_excluded_talc": true,
  "edited_region_excluded_final": true,
  "upload_remembers_artifact_after_fix": true,
  "repeat_start_has_artifact_mask": true,
  "repeat_start_region_excluded_analyzed": true,
  "repeat_start_region_excluded_sulfide": true,
  "repeat_start_region_excluded_talc": true,
  "repeat_start_region_excluded_final": true
}
```

## Verification

```bash
/opt/homebrew/opt/python@3.14/bin/python3.14 -m py_compile apps/ore_pipeline_web.py
/opt/homebrew/opt/python@3.14/bin/python3.14 -m unittest discover -s tests -p 'test_ore_pipeline_web.py' -v
```

Result:

```text
60 tests passed, 1 optional OpenAPI validator test skipped.
```
