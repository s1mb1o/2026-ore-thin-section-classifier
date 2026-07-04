# Plan 40 — Preprocessing-aware training for the grade CNN branch

- Date: 2026-07-04
- Spec: `docs/specs/preprocessing-aware-grade-training.md`
- Fixes the −0.062 preprocessing sensitivity from `docs/notes/2026-07-04-grade-cnn-robustness.md`.

## Approach

Stochastic preprocessing augmentation in `train_grade_classifier.py`: with prob
`p`, apply the shared `apply_preprocessing` to each training crop so the model
sees both raw and preprocessed inputs.

## Steps

1. [ ] Spec (done) + plan (this doc).
2. [ ] Code: add `RandomPreprocess` transform + `--preprocess-aug-prob` /
   `--preprocess-json` flags; insert after `RandomResizedCrop`, before ColorJitter;
   record in metrics/checkpoint. Default prob 0.0 = unchanged behavior.
3. [ ] Local MPS smoke (1 epoch, tiny, prob 0.5) — confirms the transform runs and
   nothing crashes.
4. [ ] Sync `src/ore_classifier` + updated script to gx10 (tracking.py is missing
   there; cv2 4.13 confirmed present in `train-models/.venv`).
5. [ ] Train on gx10: `--preprocess-aug-prob 0.5`, out-dir
   `models/grade_classifier/effb3_ordfine_ppaug_20260704`.
6. [ ] Re-run robustness sweep on the new checkpoint (baseline + 4 profiles) into
   `outputs/evaluations/grade_robustness_ppaug_20260704/`.
7. [ ] Compare vs the raw-trained model; update the robustness note, comparison
   note, ChangeLog, session-sync.

## Success criteria

- Preprocessing-profile macro-F1: 0.869 → ≥0.90.
- Raw baseline stays ≈0.93 (no regression > ~0.01).
- Other acquisition profiles do not regress materially.

## Risks

- Denoise cost per crop adds epoch time (mitigated by applying on the 384² crop,
  not full-res, and by prob<1).
- If preprocessing-aware training slightly lowers the raw baseline, keep BOTH
  checkpoints and pick per deployment (raw-served vs preprocessing-served).
