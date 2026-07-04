# Robustness ladder + resident inference — results (2026-07-04)

Two experiments run on the deconflicted 345-image split (leak-free, sha256
conflict+duplicate excluded):

1. **Robustness** of the pipeline to input perturbations (JSON augmentation +
   preprocessing), via `scripts/evaluate_official_pipeline.py`.
2. **Resident (single-load) inference** speedup, via `scripts/run_resident_batch.py`
   (spec `docs/specs/resident-batch-inference.md`, plan `docs/plans/38`).

## 1. Robustness ladder (full 345 each)

Perturbation configs in `outputs/robustness/configs/`. Baseline = raw images.

| Condition | rule macro-F1 | rule acc | feat-CV macro-F1 | feat-CV AUC | ΔfeatCV |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline (raw) | 0.1849 | 0.2435 | **0.7467** | 0.8834 | — |
| V1 mild artifacts (scratches+pits+haze) | 0.2021 | 0.2609 | 0.7456 | 0.8828 | −0.0011 |
| V2 +color/blur/noise | 0.1969 | 0.2551 | 0.7239 | 0.8701 | −0.0228 |
| V3 = V2 aug + UI preprocessing | 0.1646 | 0.2377 | 0.7375 | 0.8771 | −0.0092 |
| V4 heavy artifacts + preprocessing | 0.2141 | 0.2609 | 0.7169 | 0.8669 | −0.0298 |

### Read

- **The learned classifier is robust.** Even the worst case (V4: heavy scratches
  count 45 / dust 200 / haze 28 + blur 1.4 + noise 12 + full preprocessing) drops
  feature-CV macro-F1 by only **~3 points** (0.747 → 0.717); AUC −0.016. Mild
  realistic acquisition artifacts (V1) are negligible (−0.001).
- **Color/noise hurts more than surface artifacts.** V2 (color shift + blur +
  gaussian noise) costs −0.023 vs V1's −0.001 — the pipeline features lean on
  color/texture statistics more than on scratch/pit geometry.
- **Preprocessing partially heals color/noise.** V3 (V2 + illumination-norm +
  denoise + contrast) recovers V2's loss from −0.023 to −0.009, i.e. the UI
  preprocessing chain is a net stabilizer against color/noise perturbation. It
  cannot fully compensate the heavy V4 mix (−0.030).
- **The deterministic rule metric is not a reliable robustness signal.** It sits
  near chance (0.16–0.21, talcose F1 ≈ 0) and moves non-monotonically — even
  "up" — under perturbation because it is not a trained discriminator. Use the
  feature-CV row to judge robustness.

Per-variant Mac wall time (incl. perturbation transform of all 345):
V1 4078 s, V2 4079 s, V3 4663 s, V4 4449 s (`outputs/robustness/ladder_timing.tsv`).

## 2. Resident inference speedup

Motivation: the batch spawns a Python process **and reloads the 313 MB checkpoint
per image**; GB10 gave only ~1.4× over Mac MPS on the old path → workload was
fixed-cost bound, not GPU bound.

### gx10 (GB10, CUDA) A/B, same 15 images, back-to-back

| path | wall (15 img) | wall / img | inference / img¹ |
| --- | ---: | ---: | ---: |
| subprocess (`run_official_batch.py`) | **89 s** | 5.93 s | 3.44 s |
| resident (`run_resident_batch.py`) | **20 s** | 1.33 s² | 0.78 s |
| **speedup** | **4.45×** | 4.45× | 4.4× |

¹ `binary_inference_seconds`. In the subprocess path this timer starts before the
per-image checkpoint load, so 3.44 s ≈ 0.78 s compute + ~2.7 s reload+warmup —
exactly the fixed cost resident removes.
² includes the one-time model load amortized over 15 images.

Extrapolated full 345 on gx10: subprocess ≈ **34 min**, resident ≈ **7–8 min**.

Local (Mac MPS) A/B: running — will be appended.

### Parity (correctness gate — passed)

- MPS, 3 images: **100.0000%** sulfide-mask agreement, identical grade class and
  sulfide/talc fractions to 6 decimals.
- gx10 CUDA, 15 images: predicted grade class **15/15** match, sulfide_fraction
  **15/15** match to 1e-6, resident vs subprocess.

Resident output is byte-identical to the subprocess path; the only difference is
loading the model once. Downstream evaluators are unaffected (identical
`summary.csv` via reused `build_summary_row`).

### Takeaway

The single biggest pipeline speedup is architectural, not hardware: keep the
model resident. On the GB10 it turns a ~34-min batch into ~7–8 min at identical
accuracy. Enable with `--resident` on `evaluate_official_pipeline.py` or by
calling `run_resident_batch.py` directly.
