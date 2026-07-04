# Ore Pipeline CLI Batch Contract

Date: 2026-07-04

## Summary

The v2 ore pipeline can already run as a CLI batch tool. The recommended path is
`scripts/run_resident_batch.py`, which loads the sulfide model once and processes
the selected image split in a single Python process. This is the right interface
for reproducible model batches, evaluation runs, and judged evidence generation.

The browser UI and REST API remain useful for upload-driven review, immutable
run history, Series galleries, PDF/ZIP downloads, and live demos. They are not
required for offline batch inference.

## Recommended Batch Command

```bash
cd /Volumes/T7_2TB/Projects-T7_2TB/2026_Nornikel_Hackaton_v2

python3 scripts/run_resident_batch.py \
  --split-json outputs/official_balanced_eval_split_deconflicted.json \
  --dataset-root dataset \
  --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt \
  --out-dir outputs/evaluations/b2_cli_batch \
  --tile-size 1024 \
  --stride 768 \
  --batch-size 1 \
  --device auto \
  --keep-going
```

Use `scripts/run_official_batch.py` only when process-per-image parity with the
older runner is needed. It has the same output schema, but reloads the model per
image and is slower.

## One-Image CLI

```bash
python3 scripts/run_ore_pipeline.py \
  --image "dataset/Фото руд по сортам. ч1/Рядовые руды/2539589-1.JPG" \
  --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt \
  --out-dir outputs/demo_ore_pipeline_cli \
  --tile-size 1024 \
  --stride 768 \
  --batch-size 1 \
  --auto-talc-candidate
```

When an accepted talc mask exists, prefer `--talc-mask path/to/mask.png` over
`--auto-talc-candidate`. The automatic talc candidate is weak review evidence,
not expert talc ground truth.

## Outputs

Each image run writes the same core pipeline artifacts under its run directory:

- `pipeline_summary.json`
- `binary_sulfide/sulfide_mask.png`
- `binary_sulfide/confidence.png`
- `binary_sulfide/analyzed_mask.png`
- `binary_sulfide/overlay_preview.jpg`
- optional `talc_candidate/` or `talc_model/` artifacts
- `ore_analysis/ore_summary.json`
- `ore_analysis/component_features.csv`
- `ore_analysis/intergrowth_overlay_preview.jpg`

Batch runners also write:

- `summary.csv`
- `summary.json`
- `failures.json`
- per-image run folders under `runs/<source_label>/<run_id>/`

The CSV output is the expected input for:

```bash
python3 scripts/evaluate_ore_classification.py \
  --summary-csv outputs/evaluations/b2_cli_batch/summary.csv \
  --out-json outputs/evaluations/b2_cli_batch/ore_classification_metrics.json \
  --out-md outputs/evaluations/b2_cli_batch/ore_classification_metrics.md
```

## CLI Versus UI/API

Use the direct CLI when the goal is:

- reproducible offline batch inference;
- model or rule evaluation;
- applying `--rule-config-json`;
- applying an accepted `--talc-mask`;
- avoiding browser upload/history overhead;
- running on a GPU host or in a scheduled job.

Use the REST/UI path when the goal is:

- interactive upload and review;
- immutable browser run history;
- Series gallery workflow;
- metadata editing before a run;
- result inspection, `View files`, PDF report, and ZIP export;
- API sandbox demos.

The current UI path still has two important differences from the direct CLI:

- it does not yet expose `--rule-config-json`;
- it does not yet accept a reviewed talc mask as a first-class run input.

Therefore, the CLI can be stronger than the browser service for judged batch
evidence where calibrated rules or reviewed talc masks are required.

## API-Client CLI Option

If exact browser-service behavior is needed from the terminal, add a small
API-client CLI instead of replacing the existing direct batch scripts. That
client should:

1. start or connect to `apps/ore_pipeline_web.py`;
2. upload images through `POST /api/uploads`;
3. create a Series with `POST /api/batches`;
4. add uploaded images with `POST /api/batches/{batch_id}/items`;
5. run the Series with `POST /api/batches/{batch_id}/run`;
6. poll `GET /api/batches/{batch_id}`;
7. download `GET /api/batches/{batch_id}/results.csv` and per-run artifacts.

This would preserve UI/REST semantics, but it would not replace
`scripts/run_resident_batch.py` as the fastest core inference path.

## Performance Note

`scripts/run_resident_batch.py` is preferred for ML batches because it keeps the
model resident. The current resident-inference note reports a gx10 A/B result of
`89 s` for the subprocess path versus `20 s` for the resident path on the same
15 images, with matching predicted classes and fractions. That is about a
`4.45x` wall-time speedup before any deeper model or GPU optimization.
