# Spec — Phase-aware intergrowth features (магнетит / нерудная фаза)

- Status: draft (v0.1, not implemented)
- Date: 2026-07-04
- Decision (user, 2026-07-04): phase context comes from **our own multi-class phase
  segmenter** trained on LumenStone S2 + weak labels (plan 26 path), not petroscope
  inference and not an intensity-only cue.
- Builds on (does NOT replace):
  - `docs/specs/grain-human-in-the-loop-classifier.md` (path B grain classifier)
  - `docs/plans/26_weak-supervision-sulfide-binary-model.md` (weak-supervision training)
  - `src/ore_classifier/component_analysis.py` (`ComponentFeatures`, `analyze_components`)
  - `src/ore_classifier/grain_features.py` (`FEATURE_NAMES`, grain model)
- Related: `docs/notes/2026-07-04-external-datasets-models-sulfide-intergrowth.md`

## 1. Problem / the exact gap

The task defines the two grade-driving intergrowth types by **which phase replaces the
sulfide**:

- **обычные срастания** — large, isolated sulfides, *minimal* replacement by a gray/dark
  phase (e.g. **magnetite**) → рядовая ore marker.
- **тонкие срастания** — sulfides *significantly replaced by a non-ore phase* → 
  труднообогатимая ore marker.

Our current replacement proxy is **phase-blind**. In
`component_analysis.component_features`:

```python
dark_inside = ((footprint > 0) & (sulfide == 0))     # everything non-sulfide in the footprint
dark_inside_ratio = dark_inside_area / footprint_area
```

`dark_inside` lumps **magnetite, silicate gangue, resin, voids, cracks and glare** into
one bucket. So the deterministic rule and the grain classifier cannot distinguish:

- a sulfide genuinely *replaced by magnetite* (a true fine-intergrowth signal), from
- a sulfide with an enclosed **crack / resin gap / polishing void** (an artifact that
  should NOT push the grade toward труднообогатимая).

This is the single feature-level reason the ordinary↔fine axis is hard (baseline rule
per-class F1: fine 0.39, ordinary 0.17; feature-CV 0.747). Adding **phase identity** to
the replacement measurement is the lever the external-data scan surfaced, and LumenStone
S2 supports it directly (see §3).

## 2. Approach in one line

Add a second segmentation output — a **multi-class phase mask** — alongside the existing
binary sulfide mask, then replace the phase-blind `dark_inside_*` features with
**phase-resolved replacement and contact features** per sulfide grain. Feed the enriched
features into the *existing* grain classifier (path B) and re-evaluate. Everything is
additive and backward compatible: with no phase mask, the new fields default to 0 and the
current pipeline is unchanged.

```text
image
  ├─ binary sulfide model      → sulfide_mask         (exists, val IoU ≈0.97)
  ├─ talc detector             → talc_mask            (exists)
  └─ NEW multi-class phase seg → phase_mask {non-ore, sulfide, magnetite}
        │
        ▼
  connected components on sulfide_mask (exists)
        │
        ▼
  PHASE-AWARE component features  (this spec)
        │
        ▼
  grain classifier (path B, exists) → ordinary/fine per grain
        │
        ▼
  area-weighted aggregation ⊕ talc branch → ore grade (exists)
```

## 3. Phase segmenter (multi-class, LumenStone S2 + weak labels)

### Classes (v0.1)

LumenStone class ids are already defined in `src/ore_classifier/pseudo_labels.py`:
sulfides = {1 chalcopyrite, 5 pyrrhotite, 7 pentlandite, …}, **magnetite = 3**,
background/non-metallic = 0. Collapse to the phase ontology the features need:

| phase id | name | LumenStone source | role in intergrowth logic |
| --- | --- | --- | --- |
| 0 | `non_ore` | class 0 (background / non-metallic / gangue / resin) | "нерудная фаза" replacement |
| 1 | `sulfide` | ids {1,2,4,5,6,7,8,9,11,12,13} | the grain itself |
| 2 | `magnetite` | class 3 | "серая/тёмная фаза (магнетит)" replacement |

Explicit v0.1 limitation (be honest in outputs): LumenStone class 0 does **not**
separate silicate gangue from resin / void / crack — all are "non-metallic". So
`non_ore` still mixes genuine gangue replacement with artifacts. This is *strictly better*
than today (magnetite is now separated), and the artifact confusion is bounded by adding
the existing brightness/void heuristic as an `ignore`/`void` cue (see §4, `void` handling).
Splitting gangue vs resin/void is deferred to v0.2 (needs extra weak labels).

### Training (weak-supervision, per plan 26)

1. **Supervised pretraining on LumenStone S2** (`.../lumenstone/full/S2_v2/S2_v2`,
   Norilsk Cu-Ni: pyrrhotite/pentlandite/chalcopyrite + magnetite + non-metallic).
   Reuse the mask decoding in `pseudo_labels.py` and the tiling/dataset builder in
   `scripts/build_binary_sulfide_dataset.py` (add a 3-class label-map variant instead of
   the binary collapse). Optionally warm-start from the existing binary sulfide encoder.
2. **Teacher pseudo-labels on official images**: run the pretrained phase model + the
   brightness/morphology baseline; keep agreement zones, mark disagreement as `ignore`
   (loss masks `ignore`, same contract as plan 26). The binary sulfide model already
   fixes the sulfide channel with high confidence — reuse it to constrain the sulfide
   class so the multi-class model only has to learn the magnetite ↔ non-ore split.
3. **Student fine-tuning** on fused labels. Architecture: same family as the binary model
   (ResUNet/SegFormer-B0), 3-class head, high-res tiled inference (tile 1536/2048,
   overlap ≥192) — the tiling path already exists.

Compute: gx10 (GB10, aarch64) per plan 37; smoke on MPS locally first.

### Phase-seg deliverables

```text
models/phase_segmenter/<arch>_s2phase_<date>/best.pt (+ classes, img_size, normalize)
metrics_lumenstone_phase.json      # per-class IoU on held-out S2 (magnetite IoU is the one to watch)
official_phase_pseudo/<id>/phase_mask.png, ignore_mask.png, qa_overlay.png
```

## 4. Phase-aware component features (the core deliverable)

Extend `ComponentFeatures` (dataclass in `component_analysis.py`) with the fields below.
Inputs per sulfide component: reconstructed footprint `F` (existing
`reconstructed_footprint`), sulfide pixels `S`, and the phase mask `P` cropped to the same
window. Let `boundary(S)` = `dilate(S,1) & ~S` (1–2 px ring just outside the grain).

### Replacement-inside features (what fills the grain's footprint)

| field | definition | reads as |
| --- | --- | --- |
| `magnetite_inside_ratio` | `|F ∩ (P==magnetite)| / |F|` | replacement by gray/dark oxide |
| `nonore_inside_ratio` | `|F ∩ (P==non_ore)| / |F|` | replacement by non-ore / gangue |
| `void_inside_ratio` | `|F ∩ (P==non_ore) ∩ artifact_cue| / |F|` | crack/resin/glare (NOT replacement) |
| `replacement_inside_ratio` | `magnetite_inside_ratio + nonore_inside_ratio − void_inside_ratio` | phase-true "dark inside", artifact-discounted |
| `dark_inside_purity` | `(magnetite+nonore inside) / max(dark_inside_area,1)` | how much of today's phase-blind `dark_inside` is real replacement |

`artifact_cue` = the near-threshold/void mask from
`pseudo_labels.brightness_sulfide_pseudo_mask` (already implemented), used only to
*discount* voids, not as a hard class.

### Contact-perimeter features (what the grain touches — captures external replacement)

Replacement often eats a grain from its rim; morphological closing does not see that.
Measure the composition of the grain boundary:

| field | definition | reads as |
| --- | --- | --- |
| `contact_magnetite_frac` | `|boundary(S) ∩ magnetite| / |boundary(S)|` | magnetite embayment / rim replacement |
| `contact_nonore_frac` | `|boundary(S) ∩ non_ore| / |boundary(S)|` | gangue contact |
| `contact_sulfide_frac` | `|boundary(S) ∩ other-sulfide| / |boundary(S)|` | part of a larger sulfide aggregate |

### Combined index (optional, interpretable headline)

`replacement_index = clip(0.6·replacement_inside_ratio + 0.4·(contact_magnetite_frac +
contact_nonore_frac), 0, 1)` — one number per grain; high → тонкое срастание, low →
обычное. Weights are placeholders; calibrate on train folds. Kept explicit so the report
can say "grain replaced 47% by magnetite".

### Mapping back to the task definitions

- **обычное срастание**: large `area_px`, high `solidity`, **low** `magnetite_inside_ratio`
  + **low** `contact_magnetite_frac` + low `contact_sulfide_frac` (isolated) → `ordinary`.
- **тонкое срастание**: **high** `replacement_index` (magnetite/non-ore replacement),
  low `solidity`/`compactness`, high `boundary_complexity` (fragmented) → `fine`.

## 5. Integration points (exact code changes)

1. `component_analysis.analyze_components(...)` — add optional `phase_mask: np.ndarray |
   None = None`. When `None`, all new fields = 0.0 (backward compatible; existing tests and
   the binary-only path unchanged). When present, compute the §4 fields.
2. `ComponentFeatures` — add the new numeric fields (CSV auto-widens via `asdict`).
3. `grain_features.py` — append the new fields to `_RAW_FEATURES` / `FEATURE_NAMES`,
   guarded through `_to_float` (missing → 0.0) so old manifests still load and the current
   model keeps training. No change to the estimator factory.
4. `resident_pipeline.py` — after the sulfide/talc masks, run the phase segmenter, pass
   `phase_mask` into `analyze_components`; persist `phase_mask.png` next to the others.
5. Deterministic rule (`ComponentRuleConfig` / `component_features`) — optionally add a
   `fine_magnetite_replacement_min` term to `is_fine`; but the *primary* win is retraining
   the grain classifier on the richer vector, not hand-tuning the rule.

## 6. Metrics / how we know it worked

- **Phase seg**: per-class IoU on held-out S2, **magnetite IoU is the key number** (and
  HD95 per organizer guidance). Report weak-label agreement on official images separately
  (not as ground truth).
- **Feature ablation (headline)**: grain-level macro-F1 and image-level grade macro-F1
  under the *same* GroupKFold as path B, **with vs without** the phase-aware features.
  The claim is only credible if it beats the phase-blind feature-CV 0.747. Report both.
- Interpretability artifact: per-grain "replaced X% by magnetite / Y% by non-ore" in the
  overlay/report — a differentiator no competitor offers.

## 7. Non-goals (v0.1)

- No gangue-vs-resin-vs-void separation (LumenStone class 0 is one bucket; artifacts only
  *discounted* via the brightness cue). v0.2.
- No new pixel-level GT on official images; grade GT stays folder-derived; phase labels on
  official images are weak/pseudo, explicitly flagged.
- Does not touch the talcose branch (talc handled separately, per plan 37 constraints).
- Not a replacement for path A (whole-image CNN) or path B; it upgrades path B's features.

## 8. Open questions

- Is magnetite the only relevant "gray/dark phase", or do other oxides/silicates matter
  enough to warrant more phase classes for S2? (LumenStone only labels magnetite + hematite
  as oxides; hematite ≈absent in Norilsk Cu-Ni — verify on S2 masks.)
- Contact ring width (1 vs 2–3 px) and whether to weight contact by local curvature.
- Aggregation: does `replacement_index` belong in per-grain features only, or also as an
  image-level summary term feeding the grade rule?
- Domain gap: how well does an S2-trained phase model transfer to official images before
  weak-label fine-tuning? (Run the §3.2 teacher step and inspect agreement first.)

## 9. Risks

- **Magnetite transfer risk**: magnetite reflectivity/appearance differs between S2 optics
  and official optics; magnetite IoU on official images is unvalidated until the teacher
  step runs. Do not ship a magnetite-replacement claim without that check.
- **Artifact leakage**: if the brightness `void` cue is weak, cracks/resin still inflate
  `nonore_inside_ratio`. Mitigate by reporting `void_inside_ratio` and letting the tree
  model down-weight non-ore-only replacement relative to magnetite replacement.
- **Label scarcity**: S2 is 39 images; heavy augmentation + warm-start from the binary
  encoder is required. Magnetite may be under-represented — check class balance before
  training and weight the loss.
