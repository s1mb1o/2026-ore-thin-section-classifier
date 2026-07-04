# Spec — Preprocessing-aware training for the grade CNN branch

- Status: implementing (v0.1)
- Date: 2026-07-04
- Owner: `scripts/train_grade_classifier.py`
- Plan: `docs/plans/40_preprocessing-aware-grade-training.md`
- Motivating finding: `docs/notes/2026-07-04-grade-cnn-robustness.md`

## 1. Problem

The grade CNN branch (efficientnet_b3, ordinary↔fine) was trained on **raw**
dataset images. Robustness testing showed its single largest sensitivity is to
the pipeline's **own preprocessing** (illumination normalization + denoise +
CLAHE contrast): macro-F1 drops from **0.9303 → 0.8688 (−0.062)** when the eval
images are preprocessed, far more than any acquisition perturbation (−0.02…−0.03).
This is a **train/serve mismatch**: a user who enables preprocessing in the UI
feeds the classifier a distribution it never saw in training.

## 2. Goal

Make the grade branch robust to preprocessing by exposing the model to
preprocessed images during training, **without regressing** raw-image accuracy.
Target: preprocessed-input macro-F1 recovers toward the raw baseline (≈0.90+),
raw baseline stays ≈0.93.

## 3. Approach

Add preprocessing as a **stochastic train-time augmentation**. For each training
sample, with probability `p` apply `ore_classifier.preprocessing.apply_preprocessing`
(the exact same transform as the UI and the robustness harness — parity by shared
code) to the (already geometrically-cropped) PIL image, then continue with the
existing color/blur augmentations and normalization.

- **Stochastic, not always-on:** at `p≈0.5` the model sees a mix of raw and
  preprocessed crops each epoch, so it stays accurate on both raw and
  preprocessed inputs (rather than shifting entirely to the preprocessed domain).
- **Applied after `RandomResizedCrop`** (on the 384² crop, not the full image) so
  the slow `cv2.fastNlMeansDenoisingColored` stays cheap (~tens of ms) and epoch
  time stays reasonable. This is an augmentation, not a parity requirement, so
  the sigma/tile scale difference vs full-res preprocessing is acceptable and
  even adds useful variety.
- **Preset:** default = the UI default preprocess (`illumination_normalization`,
  `denoise`, `contrast_correction` all on), overridable via `--preprocess-json`
  (same schema as the harness).

## 4. Interface (new flags on `train_grade_classifier.py`)

- `--preprocess-aug-prob FLOAT` (default `0.0` → current behavior unchanged).
- `--preprocess-json PATH|JSON` (optional; default preset when prob>0 and omitted).

Recorded in `metrics.json` and the checkpoint metadata (`preprocess_aug_prob`,
`preprocess_preset`) for provenance.

## 5. Validation

1. Train a preprocessing-aware checkpoint (`p=0.5`) on gx10.
2. Re-run the robustness sweep (`scripts/evaluate_grade_branch.py` baseline + the
   4 profiles) on the new checkpoint.
3. Success = preprocessing-profile macro-F1 rises from 0.869 toward ≥0.90 while
   the raw baseline stays ≈0.93 (±0.01) and the other profiles don't regress.

## 6. Non-goals / notes

- Scope is preprocessing only; folding the acquisition augmentations
  (`ore_classifier.augmentation`) into training is a separate follow-up lever.
- No change to the eval split, GT, or the deferred talcose branch.
- Alternative fix (serve grade branch on raw input, bypassing preprocessing)
  remains valid and cheaper; this spec pursues the model-hardening route the user
  chose.
