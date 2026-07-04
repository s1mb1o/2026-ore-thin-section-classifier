# Plan 38 — Resident (single-load) Batch Inference

- Date: 2026-07-04
- Spec: `docs/specs/resident-batch-inference.md`
- Motivation: gx10 GB10 gave only ~1.4× over Mac MPS → batch is fixed-cost bound
  (per-image 313 MB checkpoint reload + process spawn), not GPU bound.

## Approach

Additive, no edits to the per-image scripts (web app + running robustness ladder
depend on them). New module + new batch script + one opt-in harness flag.

## Steps

1. **[done]** Spec + this plan.
2. **[done]** `src/ore_classifier/resident_pipeline.py` — `ResidentSulfidePipeline`
   loads the model once; `run_image()` reproduces `run_ore_pipeline.py` in-process
   (tiled sulfide inference → analyzed area → auto talc candidate → ore analysis)
   with identical artifact layout + `pipeline_summary.json` schema.
3. **[done]** `scripts/run_resident_batch.py` — same CLI as `run_official_batch.py`,
   one model load, reuses `build_summary_row`/`write_batch_outputs`/`select_items`
   for an identical `summary.csv`. Resumable + `--keep-going`.
4. **[done]** `scripts/evaluate_official_pipeline.py` `--resident` flag (default off)
   routes the batch step to the resident script.
5. **[done]** Parity check: resident vs subprocess `sulfide_mask.png` pixel
   agreement + `ore_summary.json` grade/fractions on sample images.
6. **[done]** Speedup measurement (gx10 GB10 A/B, same 15 imgs, back-to-back):
   subprocess **89 s** vs resident **20 s** → **4.45× wall**; inference-only
   3.44 s → 0.78 s/img. Parity: CUDA 15/15 predicted-class + sulfide_fraction
   match; MPS 100% mask agreement. Extrapolated full 345 on gx10: ~34 min → ~7–8
   min. Results in `docs/notes/2026-07-04-robustness-and-resident-inference-results.md`.
   Local (MPS) A/B running.
7. **[pending]** Cleanup (after the ladder finishes): dedupe the inference inner
   loop by extracting a shared helper used by both `infer_binary_sulfide.py` and
   `resident_pipeline.py`. Deferred to avoid modifying a script the ladder runs.

## Verification gate

Ship only if parity holds (≥99.9% mask agreement, identical grade class on
samples) AND resident wall time < subprocess wall time on the same subset.

## Risks / notes

- Duplicated inference loop until step 7 — keep in sync with
  `infer_binary_sulfide.py` meanwhile.
- CUDA nondeterminism → a few boundary pixels may differ; acceptable.
- Do not run `--resident` through the same out-dir as a subprocess run mid-ladder;
  use distinct out-dirs.
