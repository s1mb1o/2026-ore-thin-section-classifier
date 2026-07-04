# Spec — Official Ore-Pipeline Evaluation & Robustness Harness

- Status: implemented (v0.1)
- Date: 2026-07-04
- Owner script: `scripts/evaluate_official_pipeline.py`
- Related: `docs/plans/36_official-pipeline-eval-harness.md`, `docs/plans/25_standalone-ore-classifier-project.md`, `COMMANDS.md`

## 1. Purpose

Provide a single reproducible command that runs **our** ore pipeline over the
official optical-microscopy dataset and prints **our** classification metrics on
a leak-free evaluation split, and that can re-run the same measurement under
JSON-configurable augmentation and preprocessing perturbations so we can quantify
the model's robustness to input changes.

## 2. Ground truth (what the metrics are measured against)

There is no per-image or pixel-level grade annotation in this dataset. The grade
label is the **official grade folder** each аншлиф (polished section) was filed
into by the organizers' geologists, propagated to every photo of that аншлиф:

| Folder (ч1 / ч2) | `label_hint` | ore class |
| --- | --- | --- |
| Рядовые руды / рядовые | `ordinary_intergrowth` | `row_ore` |
| Труднообогатимые руды / тонкие | `fine_intergrowth` | `hard_to_process_ore` |
| Оталькованные руды / оталькованные | `talcose` | `talcose_ore` |
| Панорамы | `panorama` | (unlabelled, excluded) |

Provenance chain: `scripts/build_official_manifest.py` (path → `label_hint`) →
`scripts/audit_official_labels.py` (sha256 dedup + label-conflict detection) →
`scripts/build_official_balanced_eval_split.py` (`--exclude-conflicts
--dedupe-sha256`, class-balanced, seed 20260703).

## 3. "Exclude images present in multiple variants" — deconfliction

Requirement: images that appear in multiple variants must not be scored. Two
distinct cases, both excluded by the audit + deconflicted split:

1. **Label conflicts** — identical image content (same sha256) filed under two
   different grade folders. Ambiguous GT; excluded entirely (`--exclude-conflicts`).
2. **Exact duplicates** — identical content repeated within/across folders; only
   the ambiguous-label copies are dropped, and remaining duplicate content is
   collapsed to one representative (`--dedupe-sha256`).

Current dataset audit: 48 conflict paths + 32 duplicate paths removed, leaving a
class-balanced **345-image** split (115 ordinary / 115 fine / 115 talcose). The
raw (non-deconflicted) split is 387 images and is kept only as a baseline.

Non-goal: near-duplicate detection (perceptual hashing). Only exact-content
(sha256) variants are excluded in v0.1.

## 4. Pipeline under test

Per image: `run_ore_pipeline.py` → binary sulfide segmentation
(SegFormer-B2 checkpoint) → auto talc candidate mask → deterministic ore
analysis (component features + rule classification). `run_official_batch.py`
aggregates per-image outputs into `summary.csv`.

## 5. Metrics (both reported, never conflated)

1. **Deterministic rule pipeline** (`evaluate_ore_classification.py`):
   image-level accuracy, per-class precision/recall/F1, macro/weighted F1,
   confusion matrix, one-vs-rest AUC from rule scores. This is the pure pipeline
   output with no learning on labels.
2. **Feature-classifier CV** (`evaluate_ore_feature_classifier.py`): 5-fold
   stratified CV of ExtraTrees/RandomForest/Logistic over pipeline-derived
   features (fractions + component aggregates). This is the **learnable ceiling**
   from the same features, not a pixel-level geological score.

Both are written to `metrics_summary.json` / `metrics_summary.md`. Reporting only
one is misleading: the rule pipeline shows what ships deterministically; the CV
shows how separable the classes are given the extracted features.

## 6. Robustness parameterization (requirement #5)

Two optional inputs, each a **file path or inline JSON string**:

- `--augmentation-json` — normalized by `ore_classifier.augmentation.normalize_augmentation_settings`.
  Schema: `{enabled, color:{brightness_pct, contrast_pct, saturation_pct, hue_degrees, gamma},
  acquisition:{blur_radius, gaussian_noise_std},
  surface_artifacts:{scratch_count, scratch_intensity_pct, polishing_haze_pct, pit_count, pit_intensity_pct},
  runtime:{random_seed}}`. `enabled` must be `true` for any effect.
- `--preprocess-json` — normalized by `ore_classifier.preprocessing.normalize_preprocess_settings`.
  Schema: `{preprocessing_enabled, illumination_normalization, denoise, contrast_correction, panorama_*}`.

When either is enabled, every split image is transformed (augmentation →
preprocessing, same order as the UI) into `<out-dir>/transformed_dataset/`
mirroring the split's relative paths, and the batch runs against that root.
Transforms are **geometry-preserving** (verified), so tiling and per-tile
inference stay aligned with the baseline. Baseline = both omitted.

**Parity requirement:** the harness must apply the exact same transforms as the
browser UI. This is guaranteed by sharing code: preprocessing was extracted to
`src/ore_classifier/preprocessing.py` and `apps/ore_pipeline_web.py` now imports
`apply_preprocessing` / `normalize_preprocess_settings` from it; augmentation was
already a shared module. Panorama scaling (a resize) is intentionally not applied
in the batch path, which tiles at native resolution.

## 7. Outputs (under `--out-dir`)

- `runs/<label>/<run_id>/…` — immutable per-image pipeline artifacts
- `summary.csv` / `summary.json` / `failures.json`
- `ore_classification_metrics.{json,md}` — rule metrics
- `ore_feature_classifier_cv.{json,md}` — feature-CV metrics
- `metrics_summary.{json,md}` — combined report incl. the resolved perturbation
- `transformed_dataset/…` — only when perturbed

## 8. Non-goals

- No retraining; evaluation only.
- No pixel-level segmentation GT (does not exist for grades).
- No near-duplicate (perceptual) exclusion in v0.1.
- Panorama images stay unlabelled and out of scored metrics.

## 9. Open questions

- Should robustness runs report per-image class-flip rate (baseline→perturbed
  prediction changes), not just aggregate F1 deltas? (proposed v0.2)
- Should we add a seed sweep for augmentation to average over stochastic
  artifacts? Augmentation is currently deterministic per settings.
