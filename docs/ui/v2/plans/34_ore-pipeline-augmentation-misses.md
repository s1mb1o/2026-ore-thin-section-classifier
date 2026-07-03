# Plan 34: V2 UI Ore Pipeline Augmentation Misses

Date: 2026-07-03

## Scope

This plan is for the v2 OM-only ore pipeline UI and training stack. The primary
implementation target is `apps/ore_pipeline_web.py`; training parity touches
`src/ore_classifier/augmentation.py`, `src/ore_classifier/datasets.py`, and the
dataset/training manifest outputs.

The runtime v2 UI already has `[ ] Augmentation [Edit]`, grouped color/tone,
acquisition-noise, and grinding/polishing artifact settings, local persistence,
and an `augmented` viewer layer before preprocessing. The misses below are the
remaining gaps for model training, debug trust, and review evidence.

## Main Misses

1. Training augmentation parity is missing.
   - `src/ore_classifier/augmentation.py` is wired into the v2 UI runtime path,
     but `src/ore_classifier/datasets.py` still uses its older hard-coded
     training augmentation: hflip, vflip, 90-degree rotation, brightness,
     contrast, and saturation jitter.
   - The training path does not use the same named profiles/settings as the UI,
     so a reviewer cannot prove that visual debug augmentation and training
     augmentation are compatible.
   - Missing training variants include magnification/scale jitter, JPEG/TIFF
     acquisition degradation, sharpen/blur ranges, gamma/CLAHE or equivalent
     contrast control, illumination/vignetting/flat-field effects, and the
     grinding/polishing scratch, haze, pit, and dust effects already expected by
     the UI.

2. Artifact augmentation and artifact exclusion are not yet one reviewed loop.
   - Runtime augmentation can synthesize scratches, polishing haze, pits, and
     dust, while the artifact-mask path can exclude user-marked real artifacts.
   - The remaining miss is the explicit loop: visualize synthetic artifact
     presets, review/accept realistic ranges, mark real artifacts in the v2 UI,
     and export accepted artifact masks as ignore regions for training.
   - This must avoid turning synthetic artifacts into false positive mineral
     labels.

3. Reviewed talc mask attachment is missing from the v2 UI run path.
   - `scripts/run_ore_pipeline.py` accepts `--talc-mask`, but the v2 UI still
     depends on the automatic talc candidate unless the backend is extended.
   - The UI should allow a reviewed talc mask from the talc review workflow to
     be attached by upload or same-stem lookup, with dimension checks and
     provenance stored in the run.
   - This is critical for the official talc-content claim because the automatic
     candidate is not expert ground truth.

4. Calibration and rule visibility are still too easy to hide.
   - CLI paths support `--rule-config-json`; the v2 UI should select, display,
     and store the chosen calibration artifact and applied thresholds.
   - The result panel should show near-threshold margins and rule warnings so a
     demo cannot silently disagree with the selected calibration.

5. Warning and robustness visibility is incomplete.
   - Pipeline summaries already carry warnings/margins, and the gallery can
     expose augmentation sensitivity visually.
   - The v2 UI still needs a result warning strip and an optional robustness
     check that reruns selected image-only augmentations and reports class
     stability, fraction drift, and talc drift.

6. Training feedback export is missing.
   - Manual sulfide/final edits, artifact masks, reviewed talc masks,
     augmentation settings, and curated metadata should be exportable into a
     split-safe training manifest.
   - The export must store source hashes, parent run IDs, mask provenance,
     ignore masks, accepted artifact masks, augmentation profile IDs, seeds, and
     whether a sample is excluded from training/validation.

7. Augmentation review evidence is too informal.
   - `scripts/generate_augmentation_review_gallery.py` creates a static HTML
     gallery, but there is not yet a durable accept/reject record for presets or
     a final evidence bundle tied to selected training/UI defaults.
   - The selected preset ranges should be reviewable and reproducible before
     they become defaults.

## Resolution Plan

### P0 Phase 1: Canonical Augmentation Profiles

1. Split the augmentation schema into explicit runtime and training sections.
   - Runtime image-only transforms remain geometry-preserving and apply before
     preprocessing in the v2 UI.
   - Training transforms may include paired image/mask geometry transforms, but
     must keep masks and ignore masks aligned.
   - Each transform declares whether it is `image_only` or `paired_image_mask`.

2. Add profile files under `configs/augmentation/`.
   - `runtime_default.json`: conservative v2 UI default.
   - `training_default.json`: mask-safe training profile.
   - `stress_review.json`: stronger gallery-only review profile.

3. Extend `src/ore_classifier/augmentation.py`.
   - Preserve current runtime behavior.
   - Add profile loading and normalization.
   - Add deterministic paired image/mask geometry helpers for training.
   - Add missing image-only transforms where practical: scale/magnification
     jitter, JPEG-like re-encode, sharpen, illumination/vignetting/flat-field,
     and optional CLAHE-style contrast when dependencies allow it.

4. Wire `src/ore_classifier/datasets.py` to the shared profile.
   - Keep the old `augment=True` behavior as a compatibility fallback.
   - Add an optional profile path or settings object for training scripts.
   - Ensure ignore masks stay aligned with geometric transforms.

5. Record augmentation provenance.
   - Training logs and dataset manifests record profile path, normalized
     settings, schema version, and seed policy.
   - UI runs already store settings; confirm the schema version matches the new
     runtime profile.

### P0 Phase 2: Artifact Review And Exclusion Loop

1. Verify the current v2 `Artefacts` editor tab end to end.
   - Confirm pre-run `Fix me` opens the artifact layer after image upload.
   - Confirm `POST /api/uploads/{upload_id}/artifact-mask` persists the upload
     mask and later copies it into immutable runs.
   - Confirm completed-run artifact edits create derived runs and exclude
     marked pixels from sulfide, talc, final masks, and metrics.

2. Tighten UI semantics.
   - Use red artifact overlay consistently.
   - Disable sulfide/final tabs until a complete run exists.
   - Make the save action read `Save Artefacts` before a run and
     `Fix and Restart` after a complete run.

3. Export accepted artifact masks.
   - Treat user-marked real artifacts as ignore masks for training export.
   - Keep synthetic artifact augmentation settings separate from real artifact
     masks.

4. Add regression coverage.
   - Upload-level artifact mask save.
   - Run-level copy and exclusion.
   - Derived artifact edit run.
   - HTML contract for artifact tab and endpoint.

### P0 Phase 3: Reviewed Talc Mask Attachment

1. Add optional talc mask attachment to the v2 UI.
   - Allow manual mask upload for the selected image.
   - Add same-stem lookup from `outputs/talc_blue_line_conversion` or a
     configured talc review workspace.
   - Validate size, resample only with explicit provenance, and warn on
     mismatch.

2. Propagate to run execution.
   - Store attached talc mask under the upload and immutable run artifacts.
   - Pass the mask to the existing pipeline path instead of relying on the
     automatic candidate.
   - Record provenance: source path, review status, timestamp, dimensions, and
     whether resizing was applied.

3. Surface result trust.
   - Result panel labels talc source as `reviewed`, `automatic candidate`, or
     `missing`.
   - Reports include the same talc-mask provenance.

### P1 Phase 4: Calibration, Warnings, And Robustness

1. Add rule-config selector to the v2 UI.
   - Choose a calibration JSON.
   - Display applied thresholds and calibration provenance.
   - Store the exact config in `run.json`.

2. Add result warning strip.
   - Show `summary.warnings`, `needs_expert_review`, talc/intergrowth margins,
     zero-sulfide cases, missing scale, and automatic-talc warnings.

3. Add optional robustness check.
   - Run a small deterministic augmentation set after a completed run.
   - Report ore-class stability, sulfide fraction drift, talc fraction drift,
     and warning changes.
   - Store results under `reports/robustness.json` and display a compact
     scorecard.

### P1 Phase 5: Training Feedback Export

1. Add `Export Training Patch` or batch export action.
   - Include original/augmented/preprocessed provenance, curated metadata,
     manual masks, artifact ignore masks, talc mask provenance, and class label
     source.

2. Make export split-safe.
   - Use content hash and parent-source hash to prevent duplicate leakage.
   - Carry `exclude_from_training` from curated metadata.
   - Keep train/validation eligibility explicit.

3. Add tests.
   - Manifest contains all required paths/provenance.
   - Excluded samples stay excluded.
   - Artifact masks become ignore masks, not positive labels.

### P2 Phase 6: Review Gallery Hardening

1. Extend the static gallery output.
   - Add generated `review_template.csv` or JSONL with preset ID, source image,
     settings hash, reviewer decision, reason, and notes.
   - Add a compact final summary page for accepted/rejected presets.

2. Generate an evidence bundle.
   - Include accepted profile JSONs, gallery HTML, preview images, decision
     file, and command manifest.
   - Link the bundle from `docs/session-sync.md` after review.

## Acceptance Criteria

- The v2 UI and training path share documented compatible augmentation profiles.
- Runtime augmentation remains before preprocessing and geometry-preserving.
- Training augmentation supports paired image/mask transforms without mask drift.
- Every runtime and training artifact records exact augmentation settings and
  seed/provenance.
- Real grinding/polishing artifacts can be marked before and after a run, are
  excluded from masks/metrics, and export as ignore masks.
- Reviewed talc masks can be attached to v2 UI runs and their provenance is
  visible in UI/report artifacts.
- Calibration config, applied thresholds, warnings, and near-threshold margins
  are visible in the v2 UI.
- A reproducible augmentation gallery/evidence bundle records which presets are
  accepted for default training and UI use.
- Focused unit tests and browser smoke checks cover the new contracts.

## Risks And Guardrails

- Do not use synthetic artifact augmentation as geological ground truth.
- Keep runtime transforms image-only and geometry-preserving unless the UI also
  transforms masks and coordinates.
- Keep training geometric transforms paired across image, mask, and ignore mask.
- Do not hide automatic talc-candidate uncertainty behind reviewed-mask wording.
- Treat official image-folder labels as image-level labels, not pixel-level
  mineral truth.
- Keep this in the v2 OM-only pipeline; do not reopen SEM/XRD or the old broad
  product UI surface.
