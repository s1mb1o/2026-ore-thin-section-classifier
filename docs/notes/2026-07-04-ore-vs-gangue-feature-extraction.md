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
