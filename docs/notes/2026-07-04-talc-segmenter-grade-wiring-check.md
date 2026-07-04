# CHECK — wiring the trained talc segmenter into the grade decision

- Date: 2026-07-04
- Check script: `scripts/check_talc_model_discriminativeness.py`
- Model checked: `models/talc_segmentation/resunet_non_sulfide_20260703_local/best.pt`
  (ResUNet, val talc IoU 0.53; trained **only** on the 42 talcose blue-contour masks)
- Verdict: **DO NOT wire as-is** — the model is not image-level discriminative.

## Why we checked first

The talcose grade rule is `talc_area / analyzed_area > 0.10`. Today it consumes
the colour auto-candidate (`talc_fraction` ≈ 0), so talcose F1 = 0 everywhere.
The proposed fix was to feed the trained talc segmenter's mask instead. But that
model saw **only positives** (talcose images), never ordinary/fine, so it could
fire on any dark non-sulfide silicate. That must be verified before wiring.

## Result (8 images/class from the baseline, same masks the rule uses)

| class | MODEL talc_fraction (mean / median / max) | > 0.10 rule | AUTO (colour) median |
| --- | --- | ---: | ---: |
| ordinary_intergrowth | 0.137 / **0.068** / 0.362 | **4/8** | 0.0000 |
| fine_intergrowth | 0.106 / **0.063** / 0.294 | **3/8** | 0.0000 |
| talcose | 0.101 / **0.009** / 0.705 | **1/8** | 0.0013 |

## Read — this is the opposite of discriminative

- By median, the model predicts **more** "talc" on ordinary (0.068) and fine
  (0.063) than on talcose (**0.009**) — talcose is the *lowest*.
- It crosses the 0.10 talcose rule on **4/8 ordinary and 3/8 fine** but only
  **1/8 talcose**. Wiring it in would convert real ordinary/fine images into
  false talcose (hurting two classes) while barely recovering talcose — net
  macro-F1 would very likely **drop**, not rise.
- Mechanism: positives-only training. The model has no concept of "not talc" on
  non-talcose silicate/dark textures, so it hallucinates talc there. Higher pixel
  IoU on talcose val tiles (e.g. B0 folds 0.64) does not fix this — that metric
  never measures the false-positive rate on ordinary/fine.

Caveat: n=8/class (24 images). The direction is strong and consistent with the
training regime; a larger sample would tighten the numbers but not the sign.

## Conclusion & recommendation

- **Do not wire the current talc segmenter into the deterministic talcose rule.**
  It is not safe at the image level.
- The real fix is **negative supervision**: retrain the talc segmenter with
  ordinary/fine images as explicit negatives (empty talc masks) so it learns to
  stay silent on non-talcose silicates, then re-run this check before wiring.
- Meanwhile, note the **learned feature-CV path already handles talcose well**
  (talcose F1 ≈ 0.80): it uses `talc_candidate_fraction` as one feature among
  many and lets the classifier weight it, rather than thresholding a
  non-discriminative signal. The deterministic-rule talcose=0 is a limitation of
  the *rule*, not of the whole system.
- Optional interim: keep the colour auto-candidate for the rule but add a
  talcose fallback via the feature classifier's talcose probability, instead of
  the raw segmenter mask.
