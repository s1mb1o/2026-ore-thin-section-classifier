# Talc Pixel Detection and Ground Truth Strategy

Date: 2026-07-03

Status: proposal / discussion note. No implementation decision has been made yet.

## Problem

Talc pixel detection is the weakest link in the current pipeline. Sulfide
segmentation is at IoU ~0.97, while talc is a hand-tuned color heuristic
(`src/ore_classifier/talc_candidate.py`, green-gray HSV bands).

Critically, the 42 blue-line annotations in `Области оталькования` are **not
pixel ground truth**. They are freehand region annotations meaning "talc is
somewhere in this area":

- inside a blue region there can be sulfide grains, other silicates, cracks;
- the boundary itself is imprecise freehand drawing;
- the converted masks (`outputs/talc_blue_line_conversion`) inherit this noise
  even after manual review, because a non-expert reviewer cannot decide the
  true talc boundary either.

There is no external replacement: the targeted dataset review
(`docs/notes/2026-07-02-targeted-om-datasets-models.md`) found no public
polished-section OM talc segmentation dataset and no pretrained talc detector.
No geologist is available. Therefore gold ground truth cannot be downloaded or
delegated; it must be manufactured, with explicit confidence tiers.

## Supervision Inventory

- 42 blue-line annotated images -> conservative converted masks
  (31 `candidate_ok`, 11 needing review) — region-level, noisy.
- ~1180 image-level labels (ordinary / fine / talcose folders) — weak but
  plentiful.
- Trained sulfide model (SegFormer-B2): talc is strictly inside the
  non-sulfide region, so sulfide masks hard-constrain any talc predictor.
- Current HSV color heuristic — a baseline / fusion source only.
- Key asymmetry, pending audit: if ordinary/fine ore images contain little to
  no talc, every pixel in ~770 non-talcose images is a free hard negative.
  Caveat: ordinary ores may legally contain talc below the classification
  threshold. Audit first: sample ~20 ordinary/fine images, run the color
  heuristic, check visually whether it fires.

## Empirical Finding: Talc Is the Darkest Phase (2026-07-03)

User observation from the talc review app's luma-threshold slider
(`luma = 0.299*R + 0.587*G + 0.114*B`, brighter-than-threshold pixels painted
white): at some per-image threshold, what remains are dark flakes that
coincide with the blue-line regions. Reference bands from that session:
sulfides ~180–240, gray matrix ~70–130, talc/pores below ~60–90.

Quantified over the blue-line conversion workspace (scripts and CSV in
`outputs/talc_luma_enrichment/`, half-resolution, per-image best threshold by
in-bag vs out-of-bag dark-rate ratio):

- 28 of 42 samples analyzable; 14 skipped because `filled_talc_region` is
  empty or hairline-thin (mostly the known `needs_manual_review` open-stroke
  conversions).
- Direction confirmed: dark pixels are enriched inside bags in 26/28 samples;
  median enrichment 2.15x, max 8.3x. The two flat failures (`DSCN3056`,
  `2550382-1 10x`) are both known-bad conversions from the review list.
- No global threshold exists: per-image best t ranges 30–150, median in-bag
  luma ranges 20–83. Per-image adaptive thresholding (or illumination
  normalization first) is mandatory.
- BUT the bags cannot validate a dark-threshold detector: even after
  removing small components ("flakes only", >=~600 full-res px), a median of
  only 12% of dark-flake area lies inside the bags (best sample 39%).

The decisive ambiguity: out-of-bag dark flakes are either **unannotated talc**
(blue lines mark examples, not exhaustive coverage) or **confounders**
(pores, cracks, inter-grain shadows). A visual comparison gallery was
generated at `outputs/talc_luma_enrichment/flake_gallery/index.html` (green
frames = in-bag flakes, red = out-of-bag). First non-expert impression from
sampled crops: out-of-bag flakes look like the same dark gray-green rough
material as in-bag ones, favoring the non-exhaustive-annotation reading —
needs confirmation on the full gallery and ideally from organizers.

Implications:

- Luma is the first-order talc feature; the current HSV candidate
  (`talc_candidate.py`, mid-bright greenish bands) is looking at the wrong
  luminance range and should be revisited.
- Safe positive core: dark AND inside bag (minus sulfide, minus stroke zone).
- Safe hard negatives: bright non-sulfide matrix pixels (luma well above the
  per-image threshold).
- Out-of-bag dark flakes: ignore for training, or route into a human verdict
  queue; they are NOT safe negatives.
- If annotations are non-exhaustive, image regions outside bags can never be
  used as talc-free negatives in talcose images — this strengthens point
  counting (strategy 1) as the only unbiased fraction GT.
- New organizer question: are the blue-line annotations exhaustive per image,
  or examples only?

## Reviewed Masks: All 42 Samples Now Have Human Pixel GT (2026-07-03)

All 42 blue-line samples were manually reviewed in `apps/talc_review_web.py`
(luma-threshold preview + fill/brush/SAM2 tools); every sample now has
`reviewed/reviewed_talc_mask.png` and `reviewed_ignore_mask.png`. Analysis
in `outputs/talc_luma_enrichment/reviewed_mask_analysis.csv` (script
`reviewed_mask_analysis.py` beside it, half resolution):

- **Exhaustiveness question answered: blue lines were examples only.**
  A median of 83% of reviewed talc area lies OUTSIDE the original blue-line
  bags; in 37/42 samples more than half is outside. Reviewed talc fractions
  are large: median 0.274 of analyzed area (range 0.035–0.596), versus ~0.05
  for the bags. Any training/eval recipe keyed to the raw bags would have
  missed most talc.
- **Luma-threshold baseline vs reviewed GT**: with a per-image ORACLE
  threshold (chosen against the GT), median IoU is 0.502 (min 0.10, max
  0.82), typically recall 0.7–0.95 with precision 0.4–0.7. Luma alone is a
  strong candidate generator but over-fires on dark non-talc; a deployed
  version must also estimate the threshold per image without GT, so 0.50 is
  an upper bound for the pure-luma approach.
- **The production HSV candidate is broken**: `estimate_talc_candidate_mask`
  scores median IoU 0.000 (max 0.028) against the reviewed masks
  (`hsv_heuristic_eval.py`). It detects essentially none of the reviewed
  talc. Every `--auto-talc-candidate` pipeline run and every calibrated
  talc-fraction threshold derived from it must be considered invalid for
  talc semantics and re-run after a replacement detector exists.

Caveats on the reviewed masks as GT: they are non-expert annotations, and the
review workflow itself used the luma preview slider, so luma-vs-reviewed
agreement numbers carry some circularity. They are silver GT — the best
available — and an independent check (point counting, strategy 1, or
organizer confirmation) remains worthwhile before final claims.

Immediate implications:

1. Training a real talc model (approaches A/B below) is now unblocked: 42
   images with reviewed talc + ignore masks, plus hard negatives from
   non-talcose folders (still pending the talc-poor audit).
2. The number to beat is IoU 0.50 (oracle per-image luma threshold). The HSV
   candidate contributes nothing and should be replaced or re-tuned around
   dark-phase logic.
3. Ore-rule calibration artifacts that consumed `talc_fraction` from the HSV
   candidate need regeneration once a new talc detector lands.

## Ground Truth Strategies

These compose rather than compete.

### 1. Point counting — cheapest real ground truth (for fractions)

Classical petrography answer to exactly this problem. Instead of drawing
boundaries, sample N random points per image and have a human classify each
single point at high zoom: `talc / not talc / unsure`. Single-point
classification is a far easier perceptual task than boundary tracing, and a
non-expert is much more reliable at it.

- Statistics: ~400 points per image gives binomial SE ~2pp on a talc fraction
  around 20% — right at the ±3pp tolerance for the talc-fraction rule.
- Produces an **unbiased talc-fraction ground truth with a confidence
  interval** for a held-out eval set, independent of any mask heuristic.
  Downstream ore classification uses a talc-fraction threshold, so this
  validates exactly what matters.
- Presentation-defensible: "standard mineralogical point-count protocol".
- UI cost is small on top of `apps/talc_review_web.py`: show a crosshair at a
  random point, three hotkeys.
- Bonus: sparse point labels are also usable as training supervision
  (point-supervised segmentation), not just eval.
- Limitation: no boundary-quality GT for IoU/Hausdorff.

### 2. Statistically derived pixel labels (PU learning over blue regions)

Stop treating blue regions as masks; treat them as **positive bags**
(positive-unlabeled learning). Talc is whatever visual appearance is enriched
inside blue regions and rare in ordinary/fine images.

Method sketch: extract per-pixel features (DINOv2 dense features +
color/texture) across all 42 talcose images plus a pool of non-talcose images;
cluster; score each cluster by enrichment ratio inside-blue vs non-talcose.
Clusters strongly enriched inside blue regions define the talc signature.
Build a tri-map from it:

- **positive core** — high-enrichment pixels inside blue regions, away from
  boundaries, not sulfide;
- **negative** — pixels in non-talcose images (pending the audit above) and
  low-enrichment pixels far outside blue regions;
- **ignore** — everything else, including all blue-line boundaries.

This replaces both the hand-tuned HSV bands and the naive "blue region = talc"
assumption with a data-driven label with explicit confidence tiers.

Failure mode: if talc is visually inseparable from another silicate in RGB,
the cluster merges them — point counting (strategy 1) would then surface this
as a systematic fraction bias.

### 3. Small dual-annotator silver set (for boundary metrics)

If pixel-level IoU/Hausdorff scoring is required, some dense masks are needed,
but only a few: ~10–15 diverse crops annotated carefully at high zoom with
SAM2 assist in `apps/talc_review_web.py`, **independently by two people**.
Agreement pixels become eval GT; disagreement becomes ignore; the
inter-annotator agreement score itself is the honest performance ceiling to
report ("humans agree at IoU X, model reaches Y"). Without an expert this is
as gold as it gets, and reporting the measured noise floor turns a weakness
into a methodological strength.

### 4. Ask the organizers (parallel, zero cost)

Two questions for the captains' chat:

1. Is the talc mask scored per-pixel (IoU/Hausdorff) or only via talc fraction
   at image level? If fraction only, strategy 1 alone nearly solves the GT
   problem.
2. Can they share even one pixel-precise talc example, or a verbal definition
   of where the talc boundary should run?

## Detection Approaches (once labels exist)

- **A. Small neural talc segmentation model (target).** Reuse the binary
  sulfide stack (tiling, ignore masks, SegFormer-B0/B1, eval scripts) with a
  new dataset builder: positives from the tri-map positive core, ignore from
  tri-map ignore, negatives from non-talcose tiles (capped) plus non-talc
  regions of talcose images. Sulfide mask as input prior or post-hoc
  exclusion. Risk: small-data overfitting to talcose-folder acquisition
  conditions; mitigated by cross-class negatives and the existing augmentation
  profile.
- **B. Frozen foundation features + light pixel head (fast baseline).**
  DINOv2 dense features + linear/MLP or ExtraTrees head. Trains in minutes,
  good few-shot behavior, plugs into `source_fusion` as a probability source.
  Weakness: ~14 px patch granularity -> coarse boundaries, worse Hausdorff.
- **C. Texture+color classical pixel classifier (cheapest, demo-safe).** Talc
  regions are optically soft: low local contrast, smooth, low edge density.
  Random Forest over Lab/HSV + local std/entropy/Gabor/LBP at 2–3 scales
  strictly dominates the HSV bands and stays CPU-only; could replace the talc
  candidate inside the heuristic backend. `scribble_classifier.py` is a
  starting skeleton.
- **D. Weak supervision from image labels (MIL/CAM).** Later self-training
  booster only; localization too coarse for segmentation metrics.

## Recommended Path

1. Fire off organizer questions (strategy 4) immediately.
2. Audit the "non-talcose images are talc-poor" assumption (~20 images).
3. Build point-counting eval GT (strategy 1) on a held-out set.
4. Run the PU/enrichment analysis (strategy 2) to check whether a clean talc
   cluster exists in feature space; if yes, generate the tri-map training
   labels.
5. Train detection approach B as a one-day baseline, then A as the target;
   keep the HSV heuristic as a third `source_fusion` source and route
   disagreement areas into `talc_review_web` review queues.
6. Add the dual-annotator silver set (strategy 3) only if organizers confirm
   pixel-level scoring.

## Open Questions

- Official scoring of the talc mask: per-pixel or fraction-level?
- Are the blue-line annotations exhaustive per image, or example regions only?
  (Luma analysis suggests most dark-flake area lies outside the bags; review
  `outputs/talc_luma_enrichment/flake_gallery/index.html` and ask organizers.)
- Does talc have a separable RGB appearance from other soft silicates in these
  photos? (Partially answered: luma alone separates talc from matrix/sulfide
  per image, but pores/cracks/shadows share the dark band.)
- How much talc do ordinary/fine ore images actually contain? (Audit; the
  same per-image luma thresholding gives the candidate flakes to inspect.)
- Verbal/visual definition of the talc boundary from the organizers, if any.
