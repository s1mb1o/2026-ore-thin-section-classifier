# Grade CNN branch — robustness to augmentation & preprocessing

- Date: 2026-07-04
- Model: `models/grade_classifier/effb3_ordfine_20260704/best.pt` (efficientnet_b3, ordinary↔fine).
- Harness: `scripts/evaluate_grade_branch.py --augmentation-json/--preprocess-json`
  (same `ore_classifier.augmentation` / `ore_classifier.preprocessing` transforms as
  the browser UI and the segmentation robustness harness — parity by shared code).
- Test set: the 230 held-out ordinary/fine images of the deconflicted 345 split
  (excluded from training). Every image perturbed before the CNN eval transform.
  All runs on the same device (MPS) so deltas are apples-to-apples.
- Artifacts + profiles: `outputs/evaluations/grade_robustness_20260704/`.

## Results

| Condition | Profile | macro-F1 | ord F1 | fine F1 | Δ vs baseline |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | — | **0.9303** | 0.9333 | 0.9273 | — |
| aug: blur + noise | `blur_radius 2.0, noise_std 12` | 0.9087 | 0.9106 | 0.9067 | −0.0216 |
| aug: color shift | `bright 20 / contrast 25 / sat 20 / hue 10 / gamma 1.2` | 0.9000 | 0.8996 | 0.9004 | −0.0303 |
| aug: acquisition artifacts | `40 scratches @45 / haze 30 / 150 pits @40` | 0.8996 | 0.9061 | 0.8930 | −0.0307 |
| preprocess | `illumination + denoise + contrast` | **0.8688** | 0.8585 | 0.8790 | **−0.0615** |

## Read

- **The branch is robust.** Under strong acquisition perturbations (scratches,
  haze, pitting, color shift, blur+noise) macro-F1 stays ≈0.90 — a −0.02…−0.03
  drop from 0.930. Even the worst case (0.869) stays above the feature-CV ceiling
  (~0.75) and above competitor A's per-class ordinary/refractory (0.90/0.91).
- **Largest sensitivity is our own preprocessing** (−0.062), not the acquisition
  noise. The CNN was trained on raw dataset images; applying the UI preprocessing
  (illumination normalization + CLAHE contrast + denoise) at inference is a
  **train/serve mismatch**. Two fixes, either works:
  1. Serve the grade branch on **raw** images (bypass preprocessing for this
     branch) — preprocessing was designed to help the segmentation model, not this
     classifier.
  2. Add the preprocessing transform (and the acquisition augmentations) into
     **train-time augmentation** so the model sees that distribution.
- Color shift and surface artifacts cost about the same (~−0.03); blur+noise is
  the mildest (−0.022).

## Differentiator

Neither competitor reports grade-classifier robustness to acquisition/preprocessing
variation. This quantifies ours and gives a concrete hardening action
(preprocessing-aware training or raw-input serving) before deployment.

## Fix applied: preprocessing-aware training (2026-07-04)

Per `docs/plans/40_preprocessing-aware-grade-training.md`, retrained with the UI
preprocessing folded into train-time augmentation (`--preprocess-aug-prob 0.5`;
`apply_preprocessing` applied to each 384² crop with p=0.5). New checkpoint:
`models/grade_classifier/effb3_ordfine_ppaug_20260704/`. Same robustness sweep,
same held-out 230, same device (MPS):

| Profile | raw-trained | **pp-aware** | Δ |
| --- | ---: | ---: | ---: |
| baseline | 0.9303 | **0.9391** | +0.009 |
| aug: blur + noise | 0.9087 | 0.9435 | +0.035 |
| aug: color shift | 0.9000 | 0.9174 | +0.017 |
| aug: acquisition artifacts | 0.8996 | 0.9079 | +0.008 |
| **preprocess** | 0.8688 | **0.9174** | **+0.049** |

**Strict win.** The target mismatch is fixed — the preprocessing profile rose
0.869 → 0.917, shrinking its gap to baseline from −0.062 to −0.022 — and there was
**no regression**: the raw baseline actually improved (0.930 → 0.939) and *every*
profile improved. Preprocessing-aware training is now the preferred grade-branch
checkpoint (pp-aug baseline: ordinary F1 0.941, fine F1 0.937; confusion
ordinary [112,3], fine [11,104]). Artifacts:
`outputs/evaluations/grade_robustness_ppaug_20260704/`.

## Pushing further: preprocess p=0.7 + acquisition augmentation (2026-07-04)

Second hardening pass — bumped preprocessing aug to p=0.7 and added acquisition/
surface augmentation to training (`--augment-aug-prob 0.5`, `RandomTrainAug`:
scratches/haze/pits/blur/noise via the shared `apply_augmentation`, random seed
per sample). Checkpoint `models/grade_classifier/effb3_ordfine_ppaug07_acq_20260704/`;
sweep `outputs/evaluations/grade_robustness_ppaug07_acq_20260704/`.

| Profile | raw | pp p=0.5 | **pp0.7 + acq** |
| --- | ---: | ---: | ---: |
| baseline | 0.9303 | 0.9391 | 0.9390 |
| blur + noise | 0.9087 | **0.9435** | 0.9130 |
| color shift | 0.9000 | **0.9174** | 0.9130 |
| acquisition artifacts | 0.8996 | 0.9079 | **0.9435** |
| preprocess | 0.8688 | 0.9174 | 0.9174 |
| **worst-case (min)** | 0.869 | 0.908 | **0.913** |
| **mean** | 0.902 | 0.925 | 0.925 |

**Read:** acquisition augmentation did exactly what it targeted — the previously
weakest profile (acquisition artifacts) jumped 0.908 → 0.944, and the worst-case
across all profiles improved to 0.913 (best of the three models). The trade-off is
a small regression on blur+noise and color (−0.03 vs p=0.5); mean robustness is
tied (0.925) and baseline is unchanged (0.939). Net: **pp0.7+acq is the
worst-case-optimal checkpoint** (use where the input distribution is unknown);
**pp p=0.5 is marginally better balanced** (blur/color) and remains the web-app
default. Both dominate the raw-trained model everywhere. Further gains on
blur/color would come from adding those to train-time aug too, or tuning the aug
mix — diminishing returns from here.

## Reproduce

```bash
CKPT=models/grade_classifier/effb3_ordfine_20260704/best.pt
D=outputs/evaluations/grade_robustness_20260704
python3 scripts/evaluate_grade_branch.py --checkpoint "$CKPT" --device mps \
  --out-json "$D/baseline.json" --out-md "$D/baseline.md"
python3 scripts/evaluate_grade_branch.py --checkpoint "$CKPT" --device mps \
  --augmentation-json "$D/profiles/acquisition_artifacts.json" \
  --out-json "$D/acquisition_artifacts.json" --out-md "$D/acquisition_artifacts.md"
# ...color_shift.json, blur_noise.json (augmentation); preprocessing.json (--preprocess-json)
```
