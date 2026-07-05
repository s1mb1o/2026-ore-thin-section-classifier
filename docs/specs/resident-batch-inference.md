# Spec — Resident (single-load) Batch Inference

- Status: implemented (v0.1)
- Date: 2026-07-04
- Owner: `src/ore_classifier/resident_pipeline.py`, `scripts/run_resident_batch.py`
- Related: `docs/plans/38_resident-batch-inference.md`,
  `scripts/evaluate_official_pipeline.py`

## 1. Problem

The batch path spawns a fresh OS process **per image**:
`run_official_batch.py` → `subprocess run_ore_pipeline.py` →
`subprocess infer_binary_sulfide.py` (+ `analyze_ore_from_masks.py`). Each image
therefore pays: a Python interpreter start, a **313 MB checkpoint load**
(`torch.load` + SegFormer build + `load_state_dict` + move-to-device), and CUDA
context / kernel warmup — none of which depend on the image.

Measured 2026-07-04: sulfide inference is ~5.0 s/img on Mac MPS and ~3.5 s/img on
the gx10 GB10 (Blackwell). A top-tier GPU gave only ~1.4× — the classic signature
of a workload dominated by per-image fixed cost (model reload + CPU cv2 stages),
not GPU compute. Resident inference removes the per-image model reload and
process-spawn, which is the largest fixed cost.

## 2. Goal

Load the model **once** and process the whole batch in a single process, while
producing **byte-identical downstream artifacts** so existing evaluators
(`evaluate_ore_classification.py`, `evaluate_ore_feature_classifier.py`) and the
harness (`evaluate_official_pipeline.py`) work unchanged.

Non-goals: changing model math, batching multiple *images'* tiles together
(kept per-image for identical stitching), GPU-side talc/ore analysis (those stay
CPU/cv2), or touching the single-image scripts used by the web UI.

## 3. Design

Additive, zero-modification to the existing per-image scripts (they remain in use
by the web app and by the currently-running robustness ladder):

- `src/ore_classifier/resident_pipeline.py` — `ResidentSulfidePipeline`:
  - `__init__` loads the checkpoint once (`resolve_device` +
    `load_binary_segmentation_checkpoint`), precomputes the Hann tile weight.
  - `run_image(image_path, out_dir, ...)` reproduces `run_ore_pipeline.py`
    end-to-end **in-process**: tiled sulfide inference (reusing `forward_logits`,
    `iter_tiles`, ImageNet normalization) → `build_analyzed_mask` → optional auto
    talc candidate (`estimate_talc_candidate_mask` + `save_talc_candidate_outputs`)
    → ore analysis (`analyze_components` + `save_component_outputs`) → writes the
    same `binary_sulfide/`, `talc_candidate/`, `ore_analysis/` files and a
    `pipeline_summary.json` with the identical `paths`/`talc_source`/`rule_config`
    schema.
- `scripts/run_resident_batch.py` — drop-in replacement for
  `run_official_batch.py`. Same CLI. Loads one `ResidentSulfidePipeline`, loops
  the split, and reuses `run_official_batch.build_summary_row` /
  `write_batch_outputs` / `select_items` / `safe_run_id` so `summary.csv/json`
  are schema-identical. Resumable (skips images whose `pipeline_summary.json`
  exists unless `--overwrite`); `--keep-going` records per-image failures.
- `scripts/evaluate_official_pipeline.py` gains an opt-in `--resident` flag that
  routes the batch step to `run_resident_batch.py`. Default (flag absent) is the
  unchanged subprocess path.

The tiled-inference inner loop and a few save helpers are duplicated from
`infer_binary_sulfide.py` rather than extracted, deliberately, so no
currently-running script is modified. A later cleanup can dedupe once the ladder
finishes (tracked in the plan).

## 4. Correctness / parity requirement

The resident path must produce the same masks and metrics as the subprocess path.
Verified by:
1. Pixel agreement of `sulfide_mask.png` (resident vs subprocess) on sample
   images — expect ≥ 99.9% (only float/CUDA-nondeterminism differences).
2. Identical `ore_summary.json` grade class + fractions on those images.
3. `summary.csv` columns identical in name/order (guaranteed by reusing
   `build_summary_row`).

## 5. Expected benefit

Eliminates one interpreter start + one 313 MB checkpoint load + CUDA warmup per
image. Largest win where fixed cost dominates (GPU hosts). Actual speedup to be
recorded from a before/after run on the 345-image split (local + gx10).

## 6. Risks

- CUDA run-to-run nondeterminism can make a handful of boundary pixels differ;
  acceptable and documented.
- Long-lived process holds the model in memory for the whole batch (fine: one
  model, ~0.3–1 GB).
- If a future change edits `infer_binary_sulfide.py`, the duplicated loop here
  must be kept in sync until the dedupe cleanup lands.
