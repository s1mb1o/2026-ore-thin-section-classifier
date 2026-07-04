# How we separate ORE (sulfide) from GANGUE (non-ore) — feature extraction

- Date: 2026-07-04
- Status: documents the current implementation + physical rationale.
- Source note: requested from a shared ChatGPT thread ("Ore with Gangue
  Inclusions", `chatgpt.com/share/6a48bdd0-…`). **That transcript could not be
  retrieved** — the share URL returns only the ChatGPT SPA shell (login/nav), not
  the conversation. The sections below are grounded in **our codebase** and
  well-established reflected-light ore-microscopy background, NOT in that thread.
  §5 is a placeholder to fold in the thread's specific ideas once pasted.

## 1. Physical basis (reflected-light OM background)

In reflected polarized light on a polished section, the ore/gangue split is
primarily a **reflectance (brightness)** contrast:

- **Ore = sulfides** (here the Norilsk assemblage: pyrrhotite, pentlandite,
  chalcopyrite, …) — **high reflectance**, bright metallic grey/yellow/cream.
- **Gangue = silicates/oxides** (talc, serpentine, pyroxene, olivine, magnetite
  edge cases) — **low reflectance**, dark grey; silicates can show internal
  reflections.

Our task labels are grade-level, and the pipeline is deliberately **binary at the
phase level**: sulfide vs not-sulfide. We do **not** separate individual sulfide
species (that would need colour tint / bireflectance / anisotropy under crossed
polars, which the dataset's single-image reflected-light captures don't reliably
support).

## 2. Primary ore/gangue separator — learned sulfide segmentation

The ore vs gangue decision is made by the **binary sulfide segmentation model**
(SegFormer-B2, weak-label trained; val IoU ≈ 0.97), not by a fixed threshold:

- `sulfide_mask.png` (1 = ore/sulfide, 0 = gangue/matrix) is the ground of every
  downstream fraction. Produced by `scripts/infer_binary_sulfide.py` /
  `src/ore_classifier/resident_pipeline.py`.
- It is physically rooted in the reflectance contrast above (sulfides are the
  bright phase), but learned end-to-end so it tolerates lighting/acquisition
  variation that fixed brightness thresholds break on.

## 3. Analyzed-area mask — what counts as "real specimen"

`src/ore_classifier/analyzed_area.py::build_analyzed_mask` decides which pixels
are valid specimen (so ore/gangue fractions are not diluted by non-rock pixels):

- **Excludes dark mount/border**: HSV `value < 18` (black surround, epoxy,
  vignette) — not gangue, just non-specimen.
- **Excludes blue annotation strokes**: `blue_bias = B − max(R,G) > 45` and
  `saturation > 80` (`blue_annotation_like`) — expert pen marks, not minerals.
- Morphological open (radius 1) to despeckle.

Everything inside the analyzed mask that is **not** sulfide is treated as gangue
matrix.

## 4. Gangue sub-features we actually compute

### 4a. Talc (a specific soft silicate gangue) — colour/brightness heuristic
`src/ore_classifier/talc_candidate.py::estimate_talc_candidate_mask` finds talc in
the non-sulfide matrix by an HSV **green-grey** gate (`TalcCandidateConfig`):
`hue ∈ [35, 95]`, `saturation ∈ [12, 145]`, `value ∈ [55, 238]`, then subtract
sulfide, subtract blue annotation, drop components `< 320 px`. This colour-only
candidate is weak (near-zero recall on the official talcose grade — see
`docs/plans/39`); the **trained talc segmentation model** is the reliable talc
detector and is what feeds the talcose grade decision in v0.2.

### 4b. Ore↔gangue intergrowth texture — `dark_inside_ratio`
`src/ore_classifier/component_analysis.py::component_features` measures, per
sulfide grain, how much **gangue is enclosed within the ore grain outline**:

- `footprint` = morphologically closed + hole-filled grain outline.
- `dark_inside = footprint ∧ ¬sulfide` → gangue inclusions/replacement inside the
  grain footprint.
- `dark_inside_ratio = dark_inside_area / footprint_area`.
- Heuristic: `is_fine = dark_inside_ratio ≥ 0.18 OR solidity ≤ 0.62 OR
  compactness ≤ 0.12`. High replacement / ragged shape → **fine intergrowth**
  (труднообогатимая); clean compact grains → **ordinary** (рядовая).

So "ore vs gangue" is exploited at three scales: **phase** (sulfide mask),
**region** (talc in matrix), and **intragrain texture** (`dark_inside_ratio`,
solidity, compactness, boundary_complexity — the per-grain features surfaced in
`apps/grain_review_web.py` and used by the path-B grain classifier).

## 4c. Known heuristic weakness — jagged contour → false "fine" (труднообогатимое)

The per-grain ordinary/fine **heuristic pre-label** is an OR of three conditions
(`component_analysis.py:213`):

```
is_fine = dark_inside_ratio ≥ 0.18   # gangue replacement INSIDE the grain
       OR solidity        ≤ 0.62     # area / convex-hull area (concavities)
       OR compactness      ≤ 0.12     # 4π·area / perimeter² (boundary length)
```

**Failure case:** a *massive, homogeneous-core* sulfide grain (low
`dark_inside_ratio` → no internal replacement) but with a **strongly jagged /
serrated boundary**. The jaggedness inflates the perimeter and the convex hull,
so `solidity` and/or `compactness` drop below their thresholds and the OR fires →
the grain is pre-labelled **`fine_intergrowth` (труднообогатимое) purely because
of its boundary, at zero internal replacement**.

- Only **strong** irregularity trips it: `solidity ≤ 0.62` means concavities eat
  ≳38% of the convex hull; `compactness ≤ 0.12` is very low (circle = 1.0, square
  ≈ 0.79). Mild serration on a compact grain stays "ordinary".
- `boundary_complexity` is computed but **not** in the rule; jaggedness enters via
  `solidity`/`compactness`.

**Why it's geologically questionable:** труднообогатимость is about *fine
intergrowth / replacement* of sulfide by non-ore phase (dispersed, hard to
liberate). A jagged edge often *co-occurs* with fine intergrowth (so the
heuristic isn't absurd), but a massive grain with merely a ragged contour (e.g.
grinding pluck-out, or a coarse grain with a toothy-but-clean contact) is **not**
hard-to-process. This is a real false-positive mode.

**Mitigation (path B):** the grain-review app shows the feature report with the
triggering metric highlighted, so the annotator sees "low replacement but low
solidity/compactness → heuristic says fine" and can override to ordinary /
uncertain. A grain classifier trained on those human labels should learn to
separate "ragged boundary" from "true fine intergrowth" — which the OR heuristic
cannot. This is a primary reason the ordinary/fine axis is weak in v0.2
(row_ore F1 0.14) and why human grain labels are the lever (`docs/plans/39`).

**UI aid (built):** `apps/grain_review_web.py` now flags grains where `is_fine`
is driven ONLY by `solidity`/`compactness` while `dark_inside_ratio < 0.18`
(`boundary_only_fine`) with a "⚠ край" badge + a warning in the report — the exact
ambiguous case for the annotator. (31/200 fine-grade grains hit this.)

**Heuristic fix that helps (variant A), validated on the 345 split:** gate the
boundary "fine" signal on a small replacement floor —
`is_fine = dark_inside_ratio ≥ 0.18 OR (dark_inside_ratio ≥ floor AND (solidity ≤
0.62 OR compactness ≤ 0.12))`. Implemented as `recompute_fine_label`
(`src/ore_classifier/grain_features.py`) and exposed opt-in via
`aggregate_grade_from_grains.py --fine-dark-inside-floor` (default 0.0 = current).
A/B on the baseline batch (no talc model, so talcose = 0; ordinary/fine axis
only): floor 0.0 → 0.08 lifts **row_ore recall 0.087 → 0.304, row_ore F1 0.104 →
0.289**, hard_to_process F1 0.475 → 0.515 (not hurt); 2-class macro 0.193 → 0.268.
It recovers 25/115 ordinary аншлифы baseline mislabelled hard-to-process. Still
partial (row recall 0.30) — the rest of the ordinary↔fine axis needs human grain
labels; the floor is un-tuned (0.08 first pick, sweepable). The heuristic default
is unchanged; this is opt-in, so the deterministic-rule branch and other sessions
are unaffected.

**Variant B (boundary smoothing) — tested, NOT a grade lever.** Added
`ComponentRuleConfig.boundary_smooth_px` (default 0) + `smoothed_grain()` (OPEN
before solidity/compactness) and `scripts/regenerate_component_features.py`. At
the *grain* level smoothing flips ~40% of boundary-only-fine to ordinary while
preserving 100% of replacement-fine (see `scripts/prototype_boundary_smoothing.py`).
But at the *grade* level (345 split, bootstrap, no talc) it does NOT help:
macro-F1 0.193→0.198, row_ore F1 0.104→0.087; A+B (0.256) is slightly *below* A
alone (0.268). Reason: the grade discriminator is the *relative* fine-fraction
between ordinary and fine аншлифы — A's replacement gate removes fine-calls
**asymmetrically** (from massive-grain ordinary images), while B smooths
boundaries **symmetrically** across all images, so calibrated τ_fine adapts it
away. Decision: **adopt A, leave B off** (kept behind the default-0 flag).

## 5. To integrate from the shared ChatGPT thread (pending paste)

The thread ("Ore with Gangue Inclusions") likely proposes additional ore-vs-gangue
features. Paste its content and we will fold the concrete suggestions in here,
e.g. candidates worth evaluating against our data:

- reflectance/brightness statistics per region (mean/percentiles of V, contrast
  to local matrix);
- colour-tint separation of sulfide species (cream vs yellow vs pink) if we ever
  go beyond binary sulfide;
- internal-reflection / low-reflectance cues to positively identify silicate
  gangue vs merely "not sulfide";
- texture descriptors (GLCM/LBP) on the matrix for talc vs other gangue;
- edge/contact features between ore and gangue (liberation proxies — partly
  already in `src/ore_classifier/component_reports.py::component_liberation_proxies`).

Each should be validated on the deconflicted 345 split before adoption.
