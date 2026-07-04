# Competitor metrics comparison — grade classification & talc segmentation

- Date: 2026-07-04
- Our numbers: `scripts/evaluate_official_pipeline.py` on the deconflicted 345-image
  split (`outputs/evaluations/harness_baseline_20260704/metrics_summary.md`),
  confirmed by a fresh full 345/345 run 2026-07-04. Rule metrics are
  deterministic (macro-F1 0.1849, identical to the prior sharded batch);
  feature-CV macro-F1 is 0.7467 here vs 0.7439 on the prior batch — the ~0.003
  delta is MPS inference run-to-run non-determinism in the sulfide fractions that
  feed the features, not a methodology change.
- Competitors: two public hackathon repos the user shared.
  - **nail** = `github.com/nail-rinatovich/hackathon` (branch `dev`)
  - **opium** = `github.com/OpiumProger/Nornikel` (branch `main`)

## TL;DR

All three teams derive grade ground truth the same way — the **official grade
folder** of each аншлиф, propagated to its photos. So the metrics are comparable
in *kind*, but not in *split* (val sizes and class definitions differ). The
honest read:

**Naming:** our nail-inspired supervised classifier is called **Grade-CNN (path A)**
throughout; the competitors are referred to by name (**nail** = nail-rinatovich,
**opium** = OpiumProger) to avoid colliding with our path A / path B tracks.

- nail's headline **F1-macro 0.880** is a **directly-trained supervised
  CNN** (`efficientnet_b3`) for the grade task — the right tool for grade
  classification; our **Grade-CNN (path A)** is the analog of it.
- Our comparable **learned** number is **F1-macro 0.744** (feature-classifier CV
  over pipeline features); our **deterministic shipping** number is **F1-macro
  0.185** (rules over segmentation stats, talcose F1 = 0.0).
- opium's 0.55 is a **1-epoch smoke** (not converged) — not a real
  comparison point.
- On **segmentation** the picture flips: our sulfide model is far stronger
  (IoU 0.97) and our talc IoU (0.53–0.64) beats nail's talc (IoU 0.12 / Dice 0.19).

We win on segmentation; nail wins on grade classification because they trained a
classifier and we ship interpretable segmentation + rules (and our Grade-CNN
matches the approach).

## Grade classification (рядовая / труднообогатимая / оталькованная)

| Team | Method | Val split | F1-macro | Per-class F1 |
| --- | --- | --- | --- | --- |
| **Ours (rule pipeline)** | seg masks → deterministic geological rules | 345, deconflicted (sha256 conflict+dup excl.) | **0.185** | row 0.168 / fine 0.387 / talc **0.000** |
| **Ours (feature CV)** | ExtraTrees 5-fold over pipeline features | 345, deconflicted | **0.747** | row 0.719 / fine 0.722 / talc 0.800 |
| **Ours (path B grains, bootstrap)** | grain classifier → area-weighted fine-fraction ⊕ talc, leak-free grouped CV | 345, deconflicted | **0.190** | row 0.086 / fine 0.483 / talc **0.000** |
| **Ours (path B grains + trained-talc, bootstrap)** | as above but talcose from the trained B0 talc model (`--talc-checkpoint`) | 345, deconflicted (3-class) | **0.513** | row 0.143 / fine 0.575 / talc **0.821** |
| **Ours (path B + trained-talc + variant A, bootstrap)** ⭐ | + replacement-gate heuristic (`--fine-dark-inside-floor 0.08`) — the two levers additive | 345, deconflicted (3-class) | **0.612** | row **0.386** / fine 0.609 / talc **0.841** |
| **Ours — Grade-CNN (path A), ordinary/fine only** ⭐ | `efficientnet_b3` @384, class-weighted CE, cosine+warmup; eval split held out of training | 230 held-out ord/fine of the 345 | **0.930** (2-class) | ord **0.933** / fine **0.927** / talc — deferred |
| **Ours — Grade-CNN (path A), preprocessing-aware** ⭐ | + UI preprocessing folded into train-time aug (p=0.5); preferred checkpoint | 230 held-out ord/fine of the 345 | **0.939** (2-class) | ord **0.941** / fine **0.937** / talc — deferred |
| nail-rinatovich | `efficientnet_b3` @384, supervised | 218, grouped-by-аншлиф + dedup | **0.880** | ord 0.91 / refr 0.90 / talc 0.83 |
| nail-rinatovich (4-class) | `efficientnet_b3` (intermediate) | 218 | 0.791 | ord 0.92 / thin 0.87 / talc 0.91 / **refr 0.47** |
| opium (OpiumProger) | ResNet18, **1-epoch smoke** | `splits.csv` (707 imgs) | ~0.55 | (not reported) |

Path B's 0.190 is the **bootstrap floor** (grain classifier trained on heuristic
pre-labels ≈ re-learns the rule; talcose = 0 because the auto-candidate talc
signal is ≈0 — see `docs/plans/39`). Feeding the **trained talc model** into the
talcose branch (the first lever) lifts it to **0.513 3-class**, with talcose F1
jumping **0.000 → 0.821** — talcose is effectively solved. Adding the variant-A
replacement-gate heuristic (`--fine-dark-inside-floor 0.08`) on top is **additive**
→ **0.612 3-class** (row_ore F1 0.143 → 0.386, talcose held at 0.841). The
remaining loss is still the ordinary↔fine axis (row recall 0.28), which **human
grain labels** address. Path B's value is interpretability
(per-grain, explainable verdict), complementary to **Grade-CNN (path A)**'s raw F1;
Grade-CNN already reaches 0.93 on the ordinary/fine 2-class it targets.

**Grade-CNN (path A) — grade branch (trained 2026-07-04 on gx10, GB10; `docs/plans/37`).**
An analog of nail-rinatovich's approach: a supervised `efficientnet_b3` classifier
trained end-to-end on the grade-folder labels, added as a parallel branch. Because
the fixed 345 eval split consumes all deconflicted talcose (0 left to train on)
and the talc segmentation does not yet identify the оталькованная grade, this
branch classifies **ordinary ↔ fine only**; talcose is deferred to the talc-seg
branch. On the **230 ordinary/fine images held out of training** (from the 345
split): macro-F1 **0.930**, ordinary F1 **0.933** (P 0.896 / R 0.974), fine F1
**0.927** (P 0.971 / R 0.887); confusion ordinary [112, 3], fine [13, 102]. This
**beats competitor A on the same two classes** (their ordinary 0.91 / refractory
0.90) and lifts our learned ordinary/fine from the feature-CV's ~0.72 to ~0.93.
It is a **2-class** number — not directly comparable to A's 3-class 0.880 until
the talcose branch lands. Artifacts: `models/grade_classifier/effb3_ordfine_20260704/`
(`best.pt`, `heldout_eval.{json,md}`), internal-val best macro-F1 0.955.

### Why the gap, honestly

- **Different tool for the task.** A trains a CNN end-to-end on grade labels —
  the natural high-accuracy path. Our pipeline is **segmentation-first**
  (sulfide/talc masks → interpretable rules). Mapping mask statistics to grade
  with hand rules is hard; talcose collapses to 0.0 F1 under the current rules.
- **The signal is there.** Our feature-CV (0.744) shows the pipeline-derived
  features already carry ~0.74 macro-F1 of grade signal — a learned head (like
  A's) is what extracts it. We have not yet trained an end-to-end grade
  classifier; that is the clearest path to closing the gap.
- **Not the same ruler.** A scores on 218 val (grouped-by-аншлиф + dedup); we
  score on 345 (sha256 conflict+dup excluded). A's "3-class" is collapsed from a
  4-class intermediate; note their 4-class **refractory F1 is only 0.47** — the
  hard-to-process class is fragile for them too. Treat 0.880 vs 0.744 as
  indicative, not a controlled head-to-head.
- **Leakage discipline is comparable.** Both A and our split are leak-free by
  аншлиф/content; B's `splits.csv` leakage discipline is unclear and its number
  is a non-converged smoke anyway.

## Talc segmentation (области оталькования)

| Team | Model | Metric | Value |
| --- | --- | --- | --- |
| **Ours** (local ResUNet, non-sulfide) | ResUNet | val talc IoU | **0.527** |
| **Ours** (SegFormer-B0, 5-fold) | SegFormer-B0 | mean talc IoU / F1 | **0.644 / 0.782** |
| nail-rinatovich | U-Net (efficientnet-b0) | IoU / Dice | 0.12 / 0.19 |
| opium (OpiumProger) | U-Net | val Dice | 0.492 |

All talc GT is the **same 42 blue-contour expert images**, auto-converted to
masks — noisy, non-pixel-perfect GT for everyone. Our talc IoU leads; opium's Dice
0.49 is close to our ResUNet; nail's talc segmentation is the weakest.

## Binary sulfide segmentation (our core strength, no competitor equivalent)

- SegFormer-B1 best val: **IoU 0.9715, F1 0.9856, AUC 0.9985, HD95 ~26 px**.
- Neither competitor reports a sulfide-vs-non-sulfide segmentation model; this is
  the backbone our whole pipeline and the interpretable reports are built on.

## Differentiator: robustness harness

Neither competitor reports robustness to acquisition/preprocessing changes. Our
`scripts/evaluate_official_pipeline.py` re-runs the exact same measurement under
JSON-configurable augmentation (scratches, haze, dust) and preprocessing
(illumination/denoise/contrast), using the same transforms as the browser UI, to
quantify metric stability. Planned as the next set of runs.

## What to do next (to actually beat A on grade)

1. Train an end-to-end grade classifier (the feature-CV 0.744 is the floor of
   what a learned head can reach; a CNN on raw tiles/aggregates should exceed it).
2. Fix the deterministic talcose rule (currently 0.0 F1) using the talc
   segmentation output rather than the color heuristic alone.
3. Report on a common split definition if a controlled comparison with A is
   needed (match val size and class collapse).
