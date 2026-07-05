# SOTA note — improving per-sulfide-region classification (обычные vs тонкие срастания)

- Date: 2026-07-04
- Method: multi-agent research workflow (2 code-grounding readers + 8 web-research angles
  → adversarial verification of 12 candidate methods → architect synthesis).
  23 agents, 0 errors. Question: how to improve per-sulfide-region ordinary-vs-fine
  intergrowth classification, searching SOTA.
- Baseline under test: `outputs/evaluations/harness_baseline_20260704/`.

## Current per-region method (what we're improving)

`src/ore_classifier/component_analysis.py`: binary sulfide mask (SegFormer, IoU 0.97) →
`cv2.connectedComponentsWithStats` → per component a hand OR-rule:

```
is_fine = (dark_inside_ratio >= 0.18) OR (solidity <= 0.62) OR (compactness <= 0.12)
```

- Features are **pure binary-mask morphology** (dark_inside_ratio, solidity, compactness,
  boundary_complexity — the last is computed but **unused**). No pixel texture, no grain
  scale.
- Image grade = sulfide-**area** dominance (`ordinary_area >= fine_area`), gated by
  `talc_fraction > 0.10`.
- Shipped-rule result (345 imgs): macro-F1 **0.185**, row recall 0.174, fine recall 0.556,
  **talcose F1 0.000**. The OR over-fires "fine"; talcose never fires.
- The learned aggregate-feature classifier already reaches macro-F1 **0.747** on the *same*
  labels (`evaluate_ore_feature_classifier.py`).

## The reframe (dominates every design choice)

This is **weakly-supervised region classification with image-level-only labels** — the
whole-slide-image / MIL setting from computational pathology (bag = аншлиф, instance =
sulfide region, bag label = grade folder). **But two facts break the naive MIL framing:**

1. **The grade is a PROPORTION / area-dominance label, not a presence label.** Both grades
   contain both region types; only the fraction differs. This violates the standard MIL
   positive-bag axiom → **attention-max MIL / CLAM instance-clustering pseudo-labels are
   systematically wrong here** (they find bag-discriminative instances, not "fine" regions).
   The correct aggregators are **proportion / area-weighted / additive** pooling — which is
   exactly what the incumbent tabular classifier already does.
2. **There is ZERO per-region ground truth.** Per-region accuracy is *literally
   unmeasurable* until a small expert-labeled region set is bought. Every method can
   otherwise be judged only at image level.

Decompose into: **Layer 1** = label-free richer per-region descriptors (real texture +
scale/granulometry, which we completely lack); **Layer 2** = a weak-label bridge to the
image grade (proportion-aware aggregation, not attention-max).

## ⚠️ Measurement integrity comes first (Tier 0, non-negotiable)

The 0.747 baseline is measured under **leaky random `StratifiedKFold`**
(`evaluate_ore_feature_classifier.py:207`) over a set with documented multi-view
same-specimen groups and mixed 5×/10×/20× magnification. **Until CV is specimen-grouped,
no reported gain — texture, granulometry, embeddings, or MIL — can be trusted.** Swap to
specimen-grouped CV (reuse `specimen_group()`/`grouped_split()` already in the newly
scaffolded `scripts/train_grade_classifier.py`) before believing any number.

## Tiered plan (mapped to our files)

### Tier 1 — Quick win (low effort): ship the learned classifier, honestly measured
Replace the OR-rule + area-vote with the already-winning aggregate-feature classifier
(BASE_FEATURES + component aggregates, **which already include `talc_fraction`**).
- Files: `scripts/evaluate_ore_feature_classifier.py` (grouped-CV swap), new
  `scripts/train_region_classifier.py` (fit+persist joblib),
  `src/ore_classifier/component_analysis.py` (ML-mode flag in `ComponentRuleConfig`,
  deterministic path as fallback), the 3 `analyze_components` call sites +
  `scripts/run_ore_pipeline.py`.
- Gain: image macro-F1 **0.185 → ~0.72–0.75** (honest, a few pts below the leaky 0.747);
  **talcose F1 0.000 → ~0.80 for free** (classifier consumes `talc_fraction`).
- This is the single largest, already-demonstrated gain. Everything else is incremental
  and should only be pursued *after* this honest baseline exists.

### Tier 2 — Medium: add the missing signal (real texture + SCALE), strictly gated
Every current region feature is binary-mask morphology; **zero pixel intensity, zero grain
size** is measured. Thread the grayscale crop into `analyze_components` and add a tiny
(<15-value) per-region vector:
- **Scale-NORMALIZED morphological granulometry / pattern-spectrum** on the *dark-inside*
  (intra-sulfide silicate) mask — SE radii as fractions of `r_eq = sqrt(area/π)`. This is
  the genuinely new axis: `dark_inside_ratio` captures the *amount* of fine silicate but
  **zero of its size**.
- **Multi-scale Gabor energy** (`cv2.getGaborKernel`, no new dep) + small **GLCM /
  LBP-riu2** on sulfide-pixels-only, contrast-normalized per region.
- **GATE:** keep a feature only if row-vs-fine macro-F1 improves >~0.02 under
  specimen-grouped CV **AND** the gain survives dropping all absolute-scale variants.
- Files: `component_analysis.py` (+`ComponentFeatures` dataclass, helpers),
  `evaluate_ore_feature_classifier.py` (extend `COMPONENT_FEATURES`),
  `analyze_ore_from_masks.py` + web call sites (pass grayscale).
- Gain: realistically **+0 to +0.03** macro-F1, concentrated on the row/fine pair;
  dark-inclusion granulometry is the most likely non-redundant contributor.
- **Dominant trap = SCALE LEAKAGE:** ~94% of images lack magnification metadata and
  5×/10×/20× are mixed within folders, so any absolute-pixel granulometry/GLCM-distance
  encodes zoom, not intergrowth, and inflates leaky-CV wins that fail on held-out data.
  Mandatory `r_eq` normalization + grouped CV.

### Tier 3 — Ambitious (high value): the only VALIDATABLE per-region classifier
Buy a small expert-labeled region set — the only route that turns "per-region ordinary/fine"
from an unfalsifiable overlay into a measured metric.
- Frozen **DINOv2** embeddings (cached, offline, MPS) of masked region crops at **2–3
  scales** (preserve grain size); **kNN separability sanity-check FIRST** (reflected-light
  OM is OOD). Concatenate DINOv2 + morphology + Tier-2 texture + **explicit scale**
  (area_px, bbox) — **never DINOv2 alone on resized crops** (resize destroys grain size,
  the #1 coarse-vs-fine cue).
- **TypiClust cold-start active labeling** of ~300–400 region crops (oversample
  fine-candidates), **reserve ~80–120 as a frozen region-level val set**. Light head
  (logistic/RF), expand via embedding label-propagation + confidence-thresholded
  self-training.
- If a learned weak-label aggregator is wanted with **no** expert labels, use
  **proportion-aware / mean-pool / ADDITIVE-MIL** (readable signed per-region contribution)
  — explicitly **not** attention-max or CLAM clustering.
- Files: new `src/ore_classifier/region_embeddings.py`,
  `scripts/extract_region_embeddings.py` (reuse bbox from `component_features.csv`), reuse
  `review_queue.py` + `curation.py` for the labeling loop, `scripts/train_region_classifier.py`.
- Gain: first honest per-region accuracy/AUC; modest image lift (0 to +0.05) but a real,
  defensible per-region overlay + a calibrated fine-share that replaces the OR-rule.

### Parallel headline (medium): whole-image EfficientNet-B3 for the judged metric
Already scaffolded: `scripts/train_grade_classifier.py` (EfficientNet-B3, ImageNet init,
specimen-grouped split, sha256 dedup, inverse-freq class weights), `scripts/evaluate_grade_branch.py`.
Sidesteps the no-region-GT problem entirely; image macro-F1 toward **~0.85+**.
**No region attribution** — keep it separate from the
interpretable region pipeline; optionally ensemble its probability with the Tier-1 tabular
classifier. Trap: shortcut/confound learning (magnification/illumination/JPEG/folder) —
grouped CV + a ч1→ч2 cross-batch check mandatory.

## What to avoid (verified)

- **Attention-MAX MIL / CLAM as the primary path** — 2 independent verifiers returned
  **REJECT**: the proportion/area-dominance label violates the presence axiom, attention
  magnitude is not a calibrated fine/ordinary axis, and with ~230 bags + no region GT the
  result is overfit-prone and unvalidatable.
- **Frozen DINOv2 (or any encoder) alone on resized crops** — resizing destroys absolute
  grain size, the single most discriminative cue.
- **Absolute-pixel granulometry / fixed-distance GLCM / non-scale-invariant LBP** under
  mixed magnification — encodes zoom, not intergrowth → leaky wins.
- **Trusting ANY metric (incl. 0.747 and any feature delta) under random StratifiedKFold.**
- **Fixing talcose with more region-morphology tuning** — talcose is a silicate-coverage
  class invisible to sulfide-region morphology; let the feature classifier (sees
  `talc_fraction`) or the talc segmenter own it.
- **Presenting per-region model scores to a jury as validated region truth** when no region
  GT exists — label them model scores, back with the Tier-3 expert val set first.

## SOTA fields surveyed (references live in the workflow journal)

Automated process mineralogy / DL ore microscopy; MIL & weakly-supervised WSI
classification (ABMIL, CLAM, TransMIL, DSMIL, DTFD-MIL, additive-MIL); foundation features
(DINOv2/v3, pathology UNI/CONCH/Prov-GigaPath); deep texture (DeepTEN, FV-CNN, bilinear,
scattering, Gabor, GLCM/Haralick, LBP-riu2); fine-grained visual classification; region/cell
graph GNNs; granulometry / grain-size / watershed sub-grain; weak-label generation
(pseudo-labels, CAM, SAM-assisted, TypiClust active learning).

Verdict tally across 12 verified candidates: 0 unconditional "recommend", 10 "conditional"
(useful only after Tier-0 grouped CV + scale normalization), 2 "reject" (attention-MIL/CLAM).
The consistent adversarial finding: the incumbent aggregate-feature classifier already
extracts most of the available signal; the real bottlenecks are **measurement integrity**
and **absent per-region labels**, not model sophistication.
