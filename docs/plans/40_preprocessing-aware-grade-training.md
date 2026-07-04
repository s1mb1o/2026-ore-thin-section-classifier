# Plan 40 — Preprocessing-aware training for the grade CNN branch

- Date: 2026-07-04
- Spec: `docs/specs/preprocessing-aware-grade-training.md`
- Fixes the −0.062 preprocessing sensitivity from `docs/notes/2026-07-04-grade-cnn-robustness.md`.

## Approach

Stochastic preprocessing augmentation in `train_grade_classifier.py`: with prob
`p`, apply the shared `apply_preprocessing` to each training crop so the model
sees both raw and preprocessed inputs.

## Steps — all done (2026-07-04)

1. [done] Spec + plan.
2. [done] `RandomPreprocess` + `--preprocess-aug-prob`/`--preprocess-json`
   (after `RandomResizedCrop`, on 384² crop; recorded in metrics/checkpoint; default 0.0).
3. [done] Local MPS smoke (prob 1.0 forced path) — runs, preset recorded.
4. [done] Synced `src/ore_classifier` + script to gx10.
5. [done] Trained on gx10 `--preprocess-aug-prob 0.5` →
   `models/grade_classifier/effb3_ordfine_ppaug_20260704` (internal-val best 0.9636).
6. [done] Robustness sweep → `outputs/evaluations/grade_robustness_ppaug_20260704/`.
7. [done] Docs updated (robustness note, comparison note, ChangeLog, session-sync).

## Result — strict win (all met)

| Profile | raw | pp-aware | Δ |
| --- | ---: | ---: | ---: |
| baseline | 0.9303 | **0.9391** | +0.009 |
| blur+noise | 0.9087 | 0.9435 | +0.035 |
| color shift | 0.9000 | 0.9174 | +0.017 |
| acquisition artifacts | 0.8996 | 0.9079 | +0.008 |
| **preprocess** | 0.8688 | **0.9174** | **+0.049** |

- Preprocessing profile 0.869 → 0.917 (gap to baseline −0.062 → −0.022). ✅
- Raw baseline improved 0.930 → 0.939 (no regression). ✅
- Every profile improved. ✅ → pp-aware is the preferred grade-branch checkpoint.

## Risks

- Denoise cost per crop adds epoch time (mitigated by applying on the 384² crop,
  not full-res, and by prob<1).
- If preprocessing-aware training slightly lowers the raw baseline, keep BOTH
  checkpoints and pick per deployment (raw-served vs preprocessing-served).
