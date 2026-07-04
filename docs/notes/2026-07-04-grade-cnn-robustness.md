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
