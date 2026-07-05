# Plan 36 — Official Pipeline Evaluation & Robustness Harness

- Date: 2026-07-04
- Spec: `docs/specs/ore-pipeline-eval-harness.md`
- Deliverable: `scripts/evaluate_official_pipeline.py` (+ `src/ore_classifier/preprocessing.py`)

## Goal

One command: dataset → exclude multi-variant images → our pipeline → our
metrics, with optional JSON augmentation/preprocessing perturbation for
robustness testing. Metrics measured against official grade-folder GT on the
leak-free deconflicted 345-image split.

## Approach

Thin orchestrator that shells out to the existing, tested step scripts (no logic
duplication), plus a shared preprocessing module so the harness and the browser
UI perturb pixels identically.

## Steps

1. **[done]** Extract preprocessing into `src/ore_classifier/preprocessing.py`
   (`apply_preprocessing`, `normalize_preprocess_settings`, `preprocess_image`,
   panorama constants). Update `apps/ore_pipeline_web.py` to import from it;
   `normalize_settings_preprocess` becomes a thin `ApiError` wrapper.
   Regression check: `tests/test_ore_pipeline_web.py` + `tests/test_augmentation.py`
   → 39/40 pass; the 1 failure (`test_page_exposes_required_controls`) is
   pre-existing (asserted JS string absent at HEAD too, unrelated to this change).
2. **[done]** Write `scripts/evaluate_official_pipeline.py`:
   manifest → audit → deconflicted split (reused if present) → optional
   perturbation → `run_official_batch.py` → `evaluate_ore_classification.py` +
   `evaluate_ore_feature_classifier.py` → combined `metrics_summary.{json,md}`.
   Flags: `--augmentation-json`, `--preprocess-json` (path or inline JSON),
   `--skip-inference`, `--rebuild-*`, subset (`--per-label`/`--max-total`),
   pipeline knobs pass-through.
3. **[done]** Smoke: `--skip-inference` reproduces the reference metrics
   (rule macro-F1 0.1849, feature-CV macro-F1 0.7439); transform path verified
   geometry-preserving on 3 images.
4. **[done]** Fresh baseline run over 345/345 images →
   `outputs/evaluations/harness_baseline_20260704/`. Result: rule macro-F1 0.1849
   (deterministic), feature-CV (ExtraTrees) macro-F1 0.7467 / AUC 0.8834
   (~0.003 above the prior batch due to MPS inference non-determinism).
5. **[done]** `COMMANDS.md`: run instructions + timing.
6. **[done]** ChangeLog + session-sync update.

## Timing (this Mac, MPS, batch-size 1, tile 1024/768)

- ~10 s/image end-to-end (sulfide inference + talc candidate + ore analysis).
- Full 345-image baseline: ~50–60 min wall.
- Evaluation (both metric scripts) on a ready `summary.csv`: < 1 min.
- Each perturbation variant repeats the full batch (~50–60 min) since inference
  re-runs on transformed pixels.

## Design decisions

- **Preprocessing location:** extracted to `src/` for exact UI/harness parity
  (chosen over importing from `apps/` or augmentation-only) — user decision
  2026-07-04.
- **Baseline execution:** fresh full inference (chosen over reusing the existing
  deconflicted batch) to prove the harness end-to-end — user decision 2026-07-04.
- **Multi-variant exclusion** = sha256 label-conflict + duplicate removal via the
  existing deconflicted split, not perceptual near-dup detection.

## Risks / notes

- Robustness variants are expensive (full re-inference each). Consider a
  `--max-total` subset for quick directional checks before a full variant run.
- Transformed images are written as lossless PNG content under the original
  suffix to avoid JPEG recompression confounds.
