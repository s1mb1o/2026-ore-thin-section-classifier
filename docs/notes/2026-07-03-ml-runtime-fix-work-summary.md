# ML Runtime Fix Work Summary

Date: 2026-07-03

## Scope

This note summarizes the work completed after the v2 ore pipeline UI failed to
run the ML backend with the local SegFormer-B2 checkpoint.

The failing command was launched by the UI run worker:

```text
scripts/run_ore_pipeline.py
  --image outputs/ore_pipeline_ui/runs/run_20260703_190937_076430000_75c57d1b/input/preprocessed.png
  --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt
  --auto-talc-candidate
  --preview-max-side 4096
```

The original failure happened in `load_state_dict`, before binary sulfide
inference started.

## Work Completed

- Confirmed the active v2 workspace and project handoff requirements.
- Read the failed run log under `outputs/ore_pipeline_ui/runs/run_20260703_190937_076430000_75c57d1b/ml_pipeline.log`.
- Inspected the B2 checkpoint and local Transformers-created model state dict.
- Identified the root cause as SegFormer namespace drift:
  - checkpoint keys use `segformer.stages.*` and `decode_head.linear_projections.*`;
  - local Transformers creates `segformer.encoder.*` and `decode_head.linear_c.*`;
  - both sides contain `380` tensors for the same model shape.
- Updated `src/ore_classifier/model_io.py`:
  - added a strict SegFormer namespace remap between the two known Transformers layouts;
  - remap is accepted only when every key and tensor shape matches;
  - no partial load and no `strict=False` path is used;
  - checkpoint metadata now records `state_dict_compatibility`.
- Updated `apps/ore_pipeline_web.py` Runtime `Test`:
  - ML probe now uses `resolve_device("auto")`, matching real `infer_binary_sulfide.py` behavior;
  - on this Mac, `auto` resolves to `mps`.
- Added `tests/test_model_io.py` for the namespace remap in both directions plus shape-mismatch rejection.
- Restarted the local UI service on `http://127.0.0.1:63589/workspace`.
- Wrote focused documentation in `docs/notes/2026-07-03-segformer-transformers-namespace-compatibility.md`.

## Verification Completed

- `python3 -m py_compile src/ore_classifier/model_io.py apps/ore_pipeline_web.py tests/test_model_io.py`
- `python3 -m unittest discover -s tests -p 'test_model_io.py' -v`
- Focused `/api/runtime/test` web regression:

```text
python3 -m unittest discover -s tests -p 'test_ore_pipeline_web.py' \
  -k runtime_test_endpoint_checks_heuristic_and_ml_probe -v
```

- B2 checkpoint load on CPU:
  - model: `segformer_b2`;
  - epoch: `20`;
  - compatibility: `segformer_transformers_namespace_remap`;
  - parameters: `27,348,162`.
- B2 checkpoint load on `auto` / `mps`.
- B1 and B0 checkpoint load on CPU with the same compatibility remap.
- Live Runtime Test through the running UI service:
  - URL: `POST http://127.0.0.1:63589/api/runtime/test`;
  - backend: `ml`;
  - checkpoint: `models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt`;
  - result: `ok=true`;
  - device: `mps`;
  - torch: `2.11.0`;
  - transformers: `5.5.4`;
  - best sulfide IoU from checkpoint metadata: `0.9743806926422606`;
  - parameter count: `27,348,162`.
- Real `scripts/infer_binary_sulfide.py` succeeded on the same failed run input.
- Full `scripts/run_ore_pipeline.py` succeeded on the same failed run input into:

```text
outputs/runtime_probe_b2_failed_input_pipeline
```

- `git diff --check` passed.
- `http://127.0.0.1:63589/workspace` returned `200 text/html`.

## Current State

- The running local UI service is available at `http://127.0.0.1:63589`.
- Settings currently point to:

```text
backend = ml
checkpoint = /Volumes/T7_2TB/Projects-T7_2TB/2026_Nornikel_Hackaton_v2/models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt
```

- New ML runs should pass the previous checkpoint-load failure.
- The old failed immutable run remains failed by design; rerun from the UI to
  create a new immutable run with the fixed loader.

## Files Changed For This Work

- `src/ore_classifier/model_io.py`
- `apps/ore_pipeline_web.py`
- `tests/test_model_io.py`
- `docs/notes/2026-07-03-segformer-transformers-namespace-compatibility.md`
- `docs/notes/2026-07-03-ml-runtime-fix-work-summary.md`
- `ChangeLog.md`
- `ResearchLog.md`
- `docs/session-sync.md`
- `SMOKE_TESTS.md`
- `docs/ui/v2/specs/ore-pipeline-system-settings-v0.1.md`
- `docs/ui/v2/plans/31_ore-pipeline-system-settings.md`

## Follow-Up

- Keep using Settings -> Runtime -> Test after dependency upgrades.
- If the Runtime Test fails again after a Transformers upgrade, inspect whether
  a new explicit namespace remap is needed before falling back to zelda.
