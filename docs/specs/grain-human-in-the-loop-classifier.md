# Spec — Grain-level human-in-the-loop classifier (path B)

- Status: implemented (v0.1, heuristic-bootstrap; awaiting human grain labels)
- Date: 2026-07-04
- Plan: `docs/plans/39_grain-human-in-the-loop-classifier.md`
- Related: `docs/notes/2026-07-04-competitor-metrics-comparison.md`, `docs/specs/ore-pipeline-eval-harness.md`
- Sibling track: **path A** (`scripts/train_grade_classifier.py`, a whole-image efficientnet_b3 grade classifier) is built in a separate session. Path B here is the interpretable, segmentation-first track.

## 1. Idea and why it should work

Reuse our strongest asset — the binary sulfide segmentation model (val IoU ≈ 0.97)
— to extract every sulfide grain (connected component), let a human classify a
sample of grains (ordinary vs fine intergrowth), train a grain classifier on
those labels, then aggregate per-grain predictions into the image grade. The
ordinary↔fine axis IS a per-grain morphology question, so grain labels give
clean supervision exactly where the deterministic rule is weakest
(baseline rule per-class F1: fine 0.39, ordinary 0.17, talcose 0.00).

Interpretability is the differentiator: the verdict is explainable as "N% of
grains are fine intergrowths, here they are highlighted", which no competitor
offers.

## 2. Scope and the talc caveat (critical)

The three grades are NOT all grain-driven:

| Grade | Driver | Covered by grain classifier? |
| --- | --- | --- |
| рядовая (`row_ore`) | coarse sulfide intergrowths | yes (grain morphology) |
| труднообогатимая (`hard_to_process_ore`) | fine / replaced sulfide intergrowths | yes (grain morphology) |
| оталькованная (`talcose_ore`) | talc (silicate gangue) content > ~10% | **no — talc is not a sulfide grain** |

Therefore path B covers 2 of 3 grades via grains and **fuses the existing talc
branch** (`talc_fraction` from `ore_summary.json`) for оталькованная. Final grade
= grain-aggregation for ordinary/fine ⊕ talc-fraction rule for talcose.

## 3. Ground truth and label provenance

- **Grade GT (image level):** the official grade folder, via
  `LABEL_TO_ORE_CLASS` (`ordinary_intergrowth→row_ore`,
  `fine_intergrowth→hard_to_process_ore`, `talcose→talcose_ore`). Same GT as the
  harness; evaluated on the leak-free deconflicted 345-image split.
- **Grain GT (new, created by this work):** a human assigns each shown grain a
  class in {`ordinary_intergrowth`, `fine_intergrowth`, `uncertain`}. This GT does
  not exist a priori — we create it. It is therefore **not independently
  validatable**; grain-level accuracy is an intermediate number and the *reported*
  metric stays image-level grade F1.
- **Heuristic pre-labels:** each grain arrives pre-labelled by the existing
  `component_features` rule (`is_fine = dark_inside_ratio≥0.18 OR solidity≤0.62 OR
  compactness≤0.12`). The human corrects rather than labels from scratch. Until
  human labels exist, the classifier is trained on these pre-labels (a
  weak-supervision **bootstrap**, explicitly flagged in outputs).

## 4. Data model

### Grain record (from `build_grain_dataset.py`, one row per exported grain)
`grain_uid` (= `<run_id>__c<component_id>`), `run_id`, `grade_label`,
`expected_ore_class`, `image_rel_path`, `source_dataset_path`, `specimen_group`,
`component_id`, `heuristic_label`, `crop_path`, and the 13 numeric
`component_features` fields (area_px … centroid_y).

- Grains are read from a completed official batch's
  `runs/<label>/<run_id>/ore_analysis/component_features.csv`; bbox/centroid are
  in full-image pixel coords (verified 1:1 with the source image), so crops come
  straight from the original image.
- Selection: `area_px ≥ --min-grain-area-px` then top `--max-grains-per-image` by
  area (the batch has ~69k grains total, median 147/image — labeling all is
  infeasible; the largest grains carry the grade signal).

### Annotation store (`grain_review_web.py`)
`annotations.json`: `{ schema_version, updated_at, labels: { <grain_uid>:
{ "label": "ordinary_intergrowth"|"fine_intergrowth"|"uncertain", "at": iso } } }`.
Keyed by `grain_uid` so labels survive dataset rebuilds and both a tabular and a
future CNN trainer can consume them.

### Specimen grouping
`src/ore_classifier/specimen.py::specimen_group(rel_path)` — leading ≥3-digit run
→ `spec:<n>`, else `file:<stem>`, scoped by parent folder (fixes ч2 counter
collisions). Mirrors path A's convention. Coverage is partial (documented); used
to keep all photos of one аншлиф on one side of every CV fold.

## 5. Stages

1. **`scripts/build_grain_dataset.py`** — batch → grain crops + `grains_manifest.csv`.
2. **`apps/grain_review_web.py`** — browser grid of crops; click/keyboard to
   assign class; persists `annotations.json`. Stdlib `http.server`, cloned from
   `talc_review_web.py` architecture (server holds a `GrainReviewStore`; JSON
   API; artifact sandbox for crops; port 0). **Decision support:** each grain
   surfaces the full morphology feature report (the same numbers the v2 pipeline
   grain report shows) — a per-card `d/s/c` strip and a side panel with all
   features plus the heuristic verdict and its reasons (which of
   `dark_inside_ratio ≥ 0.18` / `solidity ≤ 0.62` / `compactness ≤ 0.12` tripped,
   mirroring `ComponentRuleConfig` defaults), with the triggering metrics
   highlighted, so the annotator sees *why* a grain reads ordinary vs fine.
3. **`scripts/train_grain_classifier.py`** — tabular classifier (sklearn
   ExtraTrees/GradientBoosting) over per-grain features; labels = human
   annotations if present else heuristic bootstrap; **GroupKFold by
   `specimen_group`**; reports grain-level macro-F1; saves `model.joblib` +
   `metadata.json` (class_order, feature_names, sklearn version, cv metrics).
4. **`scripts/aggregate_grade_from_grains.py`** — apply the grain classifier to
   ALL grains of each image (from `component_features.csv`), compute area-weighted
   `fine_fraction`; read `talc_fraction` from `ore_summary.json`; grade rule:
   `talc_fraction ≥ τ_talc → talcose_ore; elif fine_fraction ≥ τ_fine →
   hard_to_process_ore; else row_ore`. Thresholds calibrated on train folds under
   the same GroupKFold; reports **image-level grade F1** on the 345 split,
   directly comparable to the harness (rule 0.185 / feature-CV 0.747) and to
   competitor A (0.88).

## 6. Metrics reported

- Grain-level macro-F1 (ordinary/fine), grouped CV — intermediate.
- **Image-level grade macro-F1 (3-class), grouped CV — headline**, plus
  per-class F1 and confusion, in the same format as the harness.
- Bootstrap vs human-labelled runs reported separately and labelled as such.

## 7. Non-goals (v0.1)

- No CNN over crops yet (tabular first; crops are stored so a CNN is a v0.2 drop-in).
- No new pixel-level GT; grade GT stays folder-derived.
- Not a replacement for path A; a complementary interpretable track.
- Talcose is handled by the existing talc branch, not by grains.

## 8. Open questions

- Label budget: how many grains per image / total should a human label before
  retraining? (v0.1 exports top-48/image; revisit after a first labeling pass.)
- Should aggregation use grain COUNT fraction, AREA-weighted fraction, or both?
  (v0.1 = area-weighted; count fraction logged for comparison.)
- Merge with path A (ensemble whole-image CNN + grain aggregation)?
