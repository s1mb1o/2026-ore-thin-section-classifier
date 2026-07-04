# Decision gate: does the TRAINED talc segmentation give a signal for the talcose grade?

- Date: 2026-07-04
- Script: `scripts/analyze_talc_grade_signal.py`
- Output: `outputs/evaluations/talc_grade_signal_20260704.{json,md}`
- Model tested: trained talc SegFormer-B0, `outputs/talc_segformer_folds/segformer_b0_full_20260703/fold_00/segformer_b0/best.pt`

## Verdict: YES — strong signal. Invest in a talc branch for the 3-class verdict.

On a balanced subset of the deconflicted 345 split (30 images/class), with the
**42-annotated talcose leak-set excluded** (0/30 talcose were leak), the trained
talc segmenter's predicted `talc_fraction_analyzed` cleanly separates the
talcose grade:

| Grade | n | median talc_fraction | mean | p90 |
| --- | ---: | ---: | ---: | ---: |
| row_ore (рядовая) | 30 | 0.019 | 0.048 | 0.118 |
| hard_to_process (труднообогатимая) | 30 | 0.014 | 0.059 | 0.118 |
| **talcose (оталькованная)** | 30 | **0.685** | **0.652** | 0.802 |

- **talcose-vs-rest ROC-AUC = 0.994** (threshold-free; generalization, leak excluded).
- Best talcose one-vs-rest F1 = **0.968** at `talc_fraction ≥ 0.25` (tp 30, fp 2, fn 0).

## Why this reverses the earlier "talc can't do the grade" conclusion

The earlier finding (`docs/notes/2026-07-04-grade-cnn-robustness.md` and the rule
metrics: talcose F1 = 0, `talc_fraction` ≈ 0 on talcose) was measured on the
**color-heuristic talc candidate** (`talc_candidate_fraction`) that the current
deterministic pipeline feeds into the rule — NOT on the trained talc segmentation
model. The heuristic is nearly blind to the talcose grade; the **trained
SegFormer-B0** (val talc IoU 0.64 on the annotated set) generalizes well to real
talcose-grade images and is highly discriminative. The fix is simply to wire the
**trained** talc segmenter (not the color candidate) into the talcose decision.

## Implication for a full 3-class verdict

- **talcose** ← trained talc segmenter `talc_fraction ≥ τ` (AUC 0.994; τ≈0.25 here,
  or the geological >10%-by-area proxy). Near-perfect on this subset.
- **ordinary ↔ fine** ← the grade CNN branch (held-out macro-F1 0.930; pp-aware 0.939).
- Fuse: talcose gate first, else CNN. Rough projection puts a 3-class macro-F1 in
  the ~0.90+ range — plausibly at/above competitor A's 0.880 (on our 345, not
  their 218).

## Caveats before quoting a final 3-class number

1. Subset of 30/class, not the full 345 — validate on the full split.
2. τ=0.25 was read off this subset; for an honest F1, calibrate τ on a train
   portion (or use the >10% geological proxy) and evaluate held-out. The AUC 0.994
   is threshold-free and already decisive.
3. Leak controlled at basename level (42-annotated excluded). Residual
   specimen-level overlap (same аншлиф, different photo) not fully ruled out, but
   the separation is far too large to be explained by that alone.

## Recommended next step

Build the 3-class fusion (`talc-seg gate ⊕ ordinary/fine CNN`) and evaluate on the
full 345 leak-aware, calibrating τ honestly. This is the concrete path to the
3-class verdict comparable to nail's 0.880.
