# Alize Dataset Pipeline Performance Benchmark

Date: 2026-07-04

## Status

Running on Selectel Alize (`root@111.88.124.80`) in tmux session
`dataset_perf_20260704_2154`.

Remote output directory:

```text
/opt/nornickel-ai-hackathon-v2/outputs/benchmarks/alize_dataset_pipeline_perf_20260704_2154
```

Local split artifact:

```text
outputs/benchmarks/alize_dataset_pipeline_perf_20260704_2154/split.json
```

## Dataset Scope

Source dataset:

```text
/Volumes/T7_2TB/Projects-T7_2TB/2026_Nornikel_Hackaton_v2/dataset
```

The benchmark uses the class-folder split from `Фото руд по сортам. ч1` and
`Фото руд по сортам. ч2`, with cross-label duplicate hashes excluded.

Split:

| Label | Images |
|---|---:|
| `ordinary_intergrowth` | 543 |
| `fine_intergrowth` | 471 |
| `talcose` | 118 |
| **Total selected** | **1132** |

Skipped as cross-label conflicts: `48` images across `24` conflict groups.

The `14` panorama images under `dataset/Панорамы/` are unlabelled stress images
and are not part of this labelled class-folder run.

## Runner

The benchmark uses the resident batch path:

```text
scripts/run_resident_batch.py
```

This loads the sulfide model once and processes all selected images in one
Python process inside the production Docker image:

```text
nornickel-ore-pipeline-ui:v2-ml
sha256:f72410291546f2250f0a7608070312703cadc82b60d8a319960f714325076118
```

Runtime host:

| Field | Value |
|---|---|
| Host | Alize |
| GPU | NVIDIA L4 24 GB |
| Device argument | `cuda` |
| Tile size | `1024` |
| Stride | `768` |
| Batch size | `2` |
| Minimum free disk guard | `20000 MB` |

## Models Used

Core benchmark pipeline:

| Stage | Model | Checkpoint | Used in run |
|---|---|---|---|
| Binary sulfide segmentation | SegFormer-B2 | `/app/models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt` | yes |
| Talc segmentation | SegFormer-B0 | `/app/outputs/talc_segformer_folds/segformer_b0_full_20260703/fold_00/segformer_b0/best.pt` | yes |
| Ore classification summary | deterministic component rules | `talc_fraction_threshold=0.10` plus default fine-grain thresholds | yes |
| Grade CNN branch | EfficientNet-B3 ordinary-vs-fine | `/app/models/grade_classifier/effb3_ordfine_ppaug_20260704/best.pt` | no |

The EfficientNet-B3 grade checkpoint is part of the production UI deployment,
but `scripts/run_resident_batch.py` does not execute that auxiliary branch. This
benchmark therefore measures the resident segmentation/component-rule pipeline,
not the UI's extra grade-CNN opinion.

## Launch Command

The remote tmux session runs `/tmp/alize_dataset_perf_20260704_2154.sh`, whose
main resident command is:

```bash
docker run --rm \
  --name nornickel-dataset-perf-20260704-2154 \
  --gpus all \
  --ipc=host \
  -v /opt/nornickel-ai-hackathon-v2/dataset:/app/dataset:ro \
  -v /opt/nornickel-ai-hackathon-v2/models:/app/models:ro \
  -v /opt/nornickel-ai-hackathon-v2/outputs/talc_segformer_folds:/app/outputs/talc_segformer_folds:ro \
  -v /opt/nornickel-ai-hackathon-v2/outputs/benchmarks:/app/outputs/benchmarks \
  nornickel-ore-pipeline-ui:v2-ml \
  python scripts/run_resident_batch.py \
    --split-json /app/outputs/benchmarks/alize_dataset_pipeline_perf_20260704_2154/split.json \
    --dataset-root /app/dataset \
    --checkpoint /app/models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt \
    --out-dir /app/outputs/benchmarks/alize_dataset_pipeline_perf_20260704_2154 \
    --talc-checkpoint /app/outputs/talc_segformer_folds/segformer_b0_full_20260703/fold_00/segformer_b0/best.pt \
    --talc-threshold 0.50 \
    --talc-fraction-threshold 0.10 \
    --device cuda \
    --tile-size 1024 \
    --stride 768 \
    --batch-size 2 \
    --keep-going \
    --overwrite \
    --min-free-disk-mb 20000
```

## Monitoring

```bash
ssh root@111.88.124.80 'tmux attach -t dataset_perf_20260704_2154'
ssh root@111.88.124.80 'tail -n 40 /opt/nornickel-ai-hackathon-v2/outputs/benchmarks/alize_dataset_pipeline_perf_20260704_2154/alize_run.log'
ssh root@111.88.124.80 'find /opt/nornickel-ai-hackathon-v2/outputs/benchmarks/alize_dataset_pipeline_perf_20260704_2154/runs -name pipeline_summary.json | wc -l'
```

Expected post-run artifacts:

- `summary.csv`
- `summary.json`
- `failures.json`
- `ore_classification_metrics.json`
- `ore_classification_metrics.md`
- `performance_summary.json`
- `alize_run.log`

## Initial Evidence

The job started successfully:

```text
start_iso=2026-07-04T19:00:29+00:00
host=alize
image=nornickel-ore-pipeline-ui:v2-ml
image_id=sha256:f72410291546f2250f0a7608070312703cadc82b60d8a319960f714325076118
NVIDIA L4, 375 MiB, 23034 MiB, 0 %
[resident] model loaded once on cuda: sulfide=segformer_b2 talc=segformer_b0
[1/1132] fine_intergrowth: Фото руд по сортам. ч1/Труднообогатимые руды/2539439-3.JPG
```
