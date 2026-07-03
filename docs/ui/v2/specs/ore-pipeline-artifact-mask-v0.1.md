# Ore Pipeline Artifact Mask v0.1

Date: 2026-07-03

## Scope

The v2 ore pipeline UI must let a user mark grinding and polishing artefacts before or after a run. Marked artefact pixels are excluded from sulfide/non-sulfide segmentation outputs, talc detection outputs, final segmentation masks, and quantitative metrics.

## User Contract

- `Fix me` is available after an input image is uploaded, even before `Start`.
- `Edit & Recalculate` has three layer tabs:
  - `Artefacts`
  - `sulfide/non-sulfide`
  - `final segmentation`
- Before a completed run exists:
  - only `Artefacts` editing is enabled;
  - `sulfide/non-sulfide` and `final segmentation` tabs are disabled;
  - saving stores the artefact mask on the upload so the next `Start` uses it.
- After a completed run exists:
  - all three tabs are available;
  - editing `Artefacts` creates a new immutable derived run;
  - the derived run records the parent run, edited layer, comment, and recalculated outputs.
- The artefact brush draws in red. Left mouse draws, right mouse erases. Existing zoom, pan, fit view, undo, redo, and brush size controls apply to the artefact layer.

## Data Contract

- Upload-level artefact mask:
  - stored as `uploads/<upload_id>/artifacts/artifact_mask.png`;
  - dimensions match the upload analysis/preprocessed image dimensions;
  - if preprocessing changes analysis dimensions, the saved mask is resized with nearest-neighbor sampling.
- Run-level artefact mask:
  - copied to `input/artifact_mask.png` when a run starts;
  - copied from parent run inputs when edit-derived runs are created;
  - finalized as `masks/artifact_mask.png`;
  - exposed in run metadata as `masks.artifact`.
- If no artefact mask exists, the run behaves as before using an all-zero artefact mask.

## Processing Contract

- Heuristic and ML backends must apply artefact exclusion before final metrics are computed.
- For artefact pixels:
  - `sulfide_mask = 0`;
  - `talc_mask = 0`;
  - `final_mask = 0`;
  - `analyzed_mask = 0`.
- `analyzed_mask = 0` is the denominator guard that removes artefact pixels from sulfide, intergrowth, talc, and analyzed-area fractions.
- Sulfide and final edit-derived runs preserve the current artefact mask and continue excluding those pixels.

## API Contract

- `POST /api/uploads/{upload_id}/artifact-mask`
  - Body: `{ "mask_png": "data:image/png;base64,...", "comment": "..." }`
  - Saves upload artefacts and returns the refreshed upload payload.
- Existing `POST /api/runs/{run_id}/fix` accepts `edit_layer: "artifact"` in addition to `sulfide` and `final`.

## Non-Goals

- Full-resolution 64K pixel manual editing is not implemented in v0.1. Editing remains on the generated analysis/display-scale image used by the current v2 UI.
- Artefact masks are not a replacement for augmentation settings that simulate surface defects; they are an exclusion mask for real defects visible in the input image.
