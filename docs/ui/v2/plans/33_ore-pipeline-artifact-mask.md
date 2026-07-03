# Plan 33: Ore Pipeline Artifact Mask

Date: 2026-07-03

## Objective

Implement an `Artefacts` layer in the v2 `Edit & Recalculate` dialog so users can mark grinding/polishing artefacts before a run starts and recalculate immutable runs after a run completes.

## Steps

1. Add upload-level artefact mask persistence.
   - Add a store method and API endpoint for `POST /api/uploads/{upload_id}/artifact-mask`.
   - Ensure upload preprocessing keeps existing artefact masks dimension-aligned with the current analysis image.
   - Return artefact mask URLs from upload payloads.

2. Propagate artefact masks into immutable runs.
   - Copy upload masks to `input/artifact_mask.png` during run initialization.
   - Copy parent run masks into edit-derived runs.
   - Finalize run metadata with `masks.artifact` when present.

3. Exclude artefacts from processing and metrics.
   - Apply artefact masks in heuristic and ML backends before component/talc/final metric computation.
   - Preserve exclusion for sulfide and final edit-derived runs.
   - Add artifact edit-derived runs with `operation = recalculate_from_artifact_edit`.

4. Update the browser UI.
   - Add `Artefacts` as a layer tab.
   - Enable `Fix me` after image upload, before `Start`.
   - Disable segmentation tabs until a completed run exists.
   - Use red overlay color for artefact brush marks.
   - Save upload artefact masks pre-run; create derived runs post-run.

5. Verify.
   - Add backend tests for upload artefact masks and derived artefact edit runs.
   - Extend HTML contract tests for the new tab, API endpoint, and pre-run editor flow.
   - Run focused unit tests.
