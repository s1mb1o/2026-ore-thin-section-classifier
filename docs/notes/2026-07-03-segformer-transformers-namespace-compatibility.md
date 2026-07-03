# SegFormer Transformers Namespace Compatibility

Date: 2026-07-03

## Context

The v2 UI failed when running the ML backend with the local SegFormer-B2 checkpoint:

```text
scripts/run_ore_pipeline.py --image .../input/preprocessed.png --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt
```

The failure happened before inference, inside `load_state_dict`.

## Finding

The zelda-trained checkpoint and the local macOS Transformers runtime describe the same SegFormer architecture with the same `380` tensors, but use different module key namespaces:

- checkpoint: `segformer.stages.*`, `decode_head.linear_projections.*`;
- local model: `segformer.encoder.*`, `decode_head.linear_c.*`.

Only the unchanged decode-head/common keys matched exactly before remapping.

## Fix

`src/ore_classifier/model_io.py` now applies an explicit SegFormer namespace remap before strict `load_state_dict`:

- `segformer.stages.{stage}.patch_embeddings.*` -> `segformer.encoder.patch_embeddings.{stage}.*`;
- `segformer.stages.{stage}.blocks.{block}.*` -> `segformer.encoder.block.{stage}.{block}.*`;
- attention, MLP, norm, and decode-head submodule names are remapped one by one;
- the remap is accepted only when every key and every tensor shape matches the created model state dict.

The loader records `state_dict_compatibility` in checkpoint metadata. No partial or `strict=False` model load is used.

The Settings Runtime `Test` probe now resolves `device=auto`, matching real `infer_binary_sulfide.py` execution.

## Verification

Local verification completed on 2026-07-03:

- B2 checkpoint loads on CPU with `state_dict_compatibility=segformer_transformers_namespace_remap`;
- B2 checkpoint loads on local `auto` device, currently `mps`;
- local B1 and B0 mirrors also load on CPU with the same strict namespace remap;
- `POST /api/runtime/test` on `http://127.0.0.1:63589` returns `ok=true`, `model=segformer_b2`, `device=mps`;
- `scripts/infer_binary_sulfide.py` succeeds on failed run input `outputs/ore_pipeline_ui/runs/run_20260703_190937_076430000_75c57d1b/input/preprocessed.png`;
- `scripts/run_ore_pipeline.py` succeeds on the same input into `outputs/runtime_probe_b2_failed_input_pipeline`;
- `python3 -m unittest discover -s tests -p 'test_model_io.py' -v` passes;
- focused `/api/runtime/test` web regression passes.
