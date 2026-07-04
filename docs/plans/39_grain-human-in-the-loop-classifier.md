# Plan 39 — Grain-level human-in-the-loop classifier (path B)

- Date: 2026-07-04
- Spec: `docs/specs/grain-human-in-the-loop-classifier.md`
- Sibling: path A (whole-image `efficientnet_b3`) in a separate session — do NOT edit `scripts/train_grade_classifier.py`.

## Goal

Segment sulfide grains → human-classify grains → train grain classifier →
aggregate to image grade (⊕ talc branch) → evaluate image-level grade F1 on the
leak-free 345 split. Interpretable, segmentation-first alternative to path A.

## Deliverables & steps

1. **[done]** `src/ore_classifier/specimen.py` — shared аншлиф grouping, aligned
   with path A convention + parent-folder scoping (fixes ч2 counter collisions).
2. **[done]** `scripts/build_grain_dataset.py` — completed batch → per-grain crops
   + `grains_manifest.csv` (features, heuristic pre-label, specimen group, grade).
   Selection: `min_grain_area_px` floor + top-`max_grains_per_image` by area.
3. **[done]** `apps/grain_review_web.py` — stdlib browser app: paginated grid of
   grain crops, class buttons + keyboard (o/f/u), pre-label shown, persists
   `annotations.json` keyed by `grain_uid`; artifact sandbox; port 0.
4. **[done]** `scripts/train_grain_classifier.py` — tabular ExtraTrees/GBM over
   per-grain features; labels = annotations if present else heuristic bootstrap;
   GroupKFold by specimen; `model.joblib` + `metadata.json` + `metrics.md`.
5. **[done]** `scripts/aggregate_grade_from_grains.py` — grain classifier over all
   grains → area-weighted fine fraction; fuse `talc_fraction`; calibrate
   τ_fine/τ_talc on train folds; report image-level grade F1 (grouped CV).
6. **[done]** Tests `tests/test_grain_pipeline.py` (specimen grouping, dataset
   build, aggregation rule, threshold calibration).
7. **[done]** Presentation entries (`presentation_ru.md` slide + `features_ru.md`),
   re-render `presentation.html` / `features.html`.
8. **[done]** `COMMANDS.md`, `ChangeLog.md`, `docs/session-sync.md`.

## Execution model

- v0.1 runs end-to-end on **heuristic pre-labels** (bootstrap) so the whole chain
  is verifiable now; when human labels arrive via the app, retrain for the real
  gain. Bootstrap numbers are labelled as such.
- Grain source = the completed baseline batch
  `outputs/evaluations/harness_baseline_20260704/` (345 images, ~69k grains).

## Design decisions

- **Tabular first, CNN later.** Reuses the rich `component_features`; trains in
  seconds; robust with few labels. Crops are stored so a crop-CNN is a v0.2 drop-in.
- **Area-weighted fine fraction** for aggregation (count fraction logged too).
- **Talc stays separate** — grains cannot represent оталькование.
- **Reported metric = image-level grade F1 on the 345 split**, grouped CV, so it
  is directly comparable to the harness and competitor A. Grain accuracy is
  intermediate and not independently validatable (we author its GT).

## Results — v0.1 bootstrap (heuristic grain labels, no human labels yet)

Ran end-to-end on `outputs/evaluations/harness_baseline_20260704` (345 images):

- `build_grain_dataset.py`: 14,443 grains exported (top-48/image, area ≥ 300 px),
  334 specimen groups, 0 images skipped.
- `train_grain_classifier.py`: grain-level grouped-CV macro-F1 **0.998**
  (random_forest) — expected and **tautological**: bootstrap labels + the same
  morphology features the heuristic uses, so the model just re-learns the rule.
  Meaningful only once human labels diverge from the heuristic.
- `aggregate_grade_from_grains.py`: leak-free grouped-CV image-grade macro-F1
  **0.1895** (row 0.086 / fine 0.483 / **talcose 0.000**), ≈ the deterministic
  rule (0.185). Accuracy 0.29.

**Key finding — talcose is unrecoverable from the current inputs.** The
auto-candidate `talc_fraction` fed into the batch is ≈0 for talcose images
(median 0.0000, max 0.006; 0 images above 0.02), so no τ_talc can separate the
оталькованная class — exactly why both the harness rule and path B bootstrap get
talcose F1 = 0. The talcose branch must consume the **trained talc segmentation
model** (ResUNet/SegFormer, talc IoU 0.53–0.64), not the colour auto-candidate.
This is the highest-value v0.2 fix, alongside real human grain labels.

## Adversarial review & fixes (2026-07-04)

A find→verify review workflow (10 agents) confirmed 5 defects, all fixed +
regression-tested (`tests/test_grain_pipeline.py`, now 14 tests; full suite 151 OK):

1–2. `grouped_n_splits` (trainer + aggregator) floored `n_splits` at 2, so a
   class confined to one specimen group produced an empty-train fold (crash) or a
   single-class train fold (garbage macro-F1). Now fails loudly (`SystemExit`)
   when the smallest class has < 2 groups.
3. `grain_review_web.py` crop sandbox used a bare `str.startswith` → a sibling
   dir sharing the `crops` prefix (`crops_backup/`) could escape. Now
   `Path.is_relative_to`.
4. `build_grain_dataset.py` crop loop was unguarded → one non-numeric bbox cell
   aborted the whole build. Now skips the bad grain with a warning.
5. `specimen.py` docstring overclaimed that DSCN singletons "cannot leak" —
   corrected; multi-frame DSCN specimens can still leak (documented residual).

## Risks

- Label quality is the make-or-break (needs geological expertise); mitigated by
  pre-labels + `uncertain` option, but noise is expected.
- Specimen grouping is low-coverage (~22% of images); documented. Singletons
  can't leak, so the main risk is unmodelled multi-photo specimens — the
  folder-scoped key keeps cross-grade specimens apart.
- Bootstrap grade F1 gains over the deterministic rule come mostly from the
  calibrated aggregation threshold; the grain-classifier gain needs human labels.
