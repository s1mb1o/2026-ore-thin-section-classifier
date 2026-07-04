# Codex gx10 Ch1/Ch2 Class-Folder E2E Evaluation

Date: 2026-07-04

## Scope

This is an independent Codex rerun of the ch1/ch2 class-folder E2E batch
evaluation on `gx10`, isolated from the earlier Claude output directory.

Dataset folders:

- `dataset/Фото руд по сортам. ч1`
- `dataset/Фото руд по сортам. ч2`

Duplicate policy: skip every image whose SHA-256 content hash appears under more
than one source class label. Same-label duplicates are listed in the audit CSVs
but are not removed.

This benchmark uses the documented resident batch path
(`scripts/run_resident_batch.py`): sulfide segmentation, talc segmentation,
component analysis, deterministic ore rule, and post-run metrics over
`summary.csv`. The optional EfficientNet grade-CNN branch is not part of this
resident batch interface; its evaluator is a separate ordinary-vs-fine held-out
test and is not mixed into these full ch1/ch2 metrics.

## Artifacts

Remote full artifact tree, including per-image `runs/`:

```text
gx10:~/Projects/nornikel_codex_e2e_eval_20260704_1333/repo/outputs/evaluations/codex_ch1_ch2_class_folder_eval_20260704_1333/
```

Local top-level evidence copy, excluding heavy per-image `runs/`:

```text
outputs/evaluations/codex_ch1_ch2_class_folder_eval_20260704_1333/
```

Key files:

- `split.json`, `split.csv`
- `class_folder_label_conflicts.csv`, `class_folder_duplicate_groups.csv`
- `summary.csv`, `summary.json`, `failures.json`
- `classification_metrics.{json,md}`
- `ore_feature_classifier_cv.{json,md}`
- `talc_gate_metrics.{json,md}`
- `metrics_summary.{json,md}`
- `gx10_run_codex.log`

## Environment

- host: `gx10-fb56` (`ashmelev@192.168.86.14`)
- isolated repo copy: `~/Projects/nornikel_codex_e2e_eval_20260704_1333/repo`
- Python env: `~/Projects/benchmark_venv`
- torch: `2.11.0+cu130`
- torchvision: `0.26.0+cu130`
- transformers: `5.4.0`
- scikit-learn: `1.8.0`
- CUDA available: `true`

Synced code hashes for the relevant runner files:

| File | SHA-256 |
| --- | --- |
| `scripts/run_resident_batch.py` | `c9b163dade12456739807c00f553dccd90a8e00bb7953300d64e105bc192fdb6` |
| `src/ore_classifier/resident_pipeline.py` | `d1cc7fcbbaa41fde8ff2222c41528e15f557e0831a6fdcf30c6ec969d854b486` |
| `scripts/evaluate_ore_feature_classifier.py` | `245ed7e0f5302db2ab8109054c05a9019b9c6c2c7da97c785204befe47d8d286` |

## Exact Steps

GPU preflight:

```bash
ssh 192.168.86.14 'hostname; date; free -h; nvidia-smi; ps aux --sort=-%mem | head -15; docker ps --format "{{.Names}}\t{{.Status}}\t{{.Ports}}"; docker stats --no-stream --format "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" 2>/dev/null || true'
```

Create isolated gx10 workspace and sync current local code, excluding large
datasets/models/outputs:

```bash
ssh 192.168.86.14 'mkdir -p ~/Projects/nornikel_codex_e2e_eval_20260704_1333/repo'

rsync -a --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude 'dataset/' \
  --exclude 'models/' \
  --exclude 'outputs/' \
  --exclude 'data/external/' \
  ./ 192.168.86.14:~/Projects/nornikel_codex_e2e_eval_20260704_1333/repo/
```

Wire the isolated code copy to the already-synced gx10 dataset/checkpoints:

```bash
ssh 192.168.86.14 'cd ~/Projects/nornikel_codex_e2e_eval_20260704_1333/repo && \
  mkdir -p outputs && \
  ln -sfn ~/Projects/2026_Nornikel_Hackaton_v2/dataset dataset && \
  ln -sfn ~/Projects/2026_Nornikel_Hackaton_v2/models models && \
  ln -sfn ~/Projects/2026_Nornikel_Hackaton_v2/outputs/talc_segformer_folds outputs/talc_segformer_folds'
```

Compile/help smoke:

```bash
ssh 192.168.86.14 'cd ~/Projects/nornikel_codex_e2e_eval_20260704_1333/repo && \
  source ~/Projects/benchmark_venv/bin/activate && \
  python -m py_compile \
    scripts/run_resident_batch.py \
    scripts/build_class_folder_eval_split.py \
    scripts/evaluate_ore_classification.py \
    scripts/evaluate_ore_feature_classifier.py \
    src/ore_classifier/resident_pipeline.py && \
  python scripts/run_resident_batch.py --help >/tmp/codex_run_resident_help.txt && \
  python scripts/evaluate_ore_feature_classifier.py --help >/tmp/codex_feature_help.txt'
```

Build the duplicate-conflict-skipping split:

```bash
ssh 192.168.86.14 'cd ~/Projects/nornikel_codex_e2e_eval_20260704_1333/repo && \
  source ~/Projects/benchmark_venv/bin/activate && \
  OUT=outputs/evaluations/codex_ch1_ch2_class_folder_eval_20260704_1333 && \
  mkdir -p "$OUT" && \
  python scripts/build_class_folder_eval_split.py \
    --dataset-root dataset \
    --out-json "$OUT/split.json" \
    --out-csv "$OUT/split.csv" \
    --conflicts-csv "$OUT/class_folder_label_conflicts.csv" \
    --duplicates-csv "$OUT/class_folder_duplicate_groups.csv" | tee "$OUT/build_split.log"'
```

Run resident E2E inference in a detached tmux session:

```bash
ssh 192.168.86.14 'tmux new-session -d -s codex_e2e_20260704_1333 bash -lc '\''cd ~/Projects/nornikel_codex_e2e_eval_20260704_1333/repo && source ~/Projects/benchmark_venv/bin/activate && OUT=outputs/evaluations/codex_ch1_ch2_class_folder_eval_20260704_1333 && { echo "start: $(date -Is)"; free -h; nvidia-smi; python scripts/run_resident_batch.py --split-json "$OUT/split.json" --dataset-root dataset --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt --talc-checkpoint outputs/talc_segformer_folds/segformer_b0_full_20260703/fold_00/segformer_b0/best.pt --talc-threshold 0.50 --talc-fraction-threshold 0.10 --out-dir "$OUT" --device cuda --tile-size 1024 --stride 768 --batch-size 2 --keep-going --min-free-disk-mb 5000; rc=$?; echo "finish: $(date -Is) rc=$rc"; exit $rc; } 2>&1 | tee "$OUT/gx10_run_codex.log"'\'''
```

Metric passes over the completed `summary.csv`:

```bash
ssh 192.168.86.14 'cd ~/Projects/nornikel_codex_e2e_eval_20260704_1333/repo && \
  source ~/Projects/benchmark_venv/bin/activate && \
  OUT=outputs/evaluations/codex_ch1_ch2_class_folder_eval_20260704_1333 && \
  python scripts/evaluate_ore_classification.py \
    --summary-csv "$OUT/summary.csv" \
    --out-json "$OUT/classification_metrics.json" \
    --out-md "$OUT/classification_metrics.md" | tee "$OUT/evaluate_ore_classification.log" && \
  python scripts/evaluate_ore_feature_classifier.py \
    --summary-csv "$OUT/summary.csv" \
    --out-json "$OUT/ore_feature_classifier_cv.json" \
    --out-md "$OUT/ore_feature_classifier_cv.md" \
    --folds 5 | tee "$OUT/evaluate_ore_feature_classifier.log"'
```

`talc_gate_metrics.{json,md}` and `metrics_summary.{json,md}` were generated
from the same `summary.csv`: fixed gate `talc_fraction > 0.10`, threshold grid
`0.00..0.60` step `0.01`, source-label talc-fraction statistics, and combined
rule/feature/gate summary.

Top-level artifacts were copied back locally without per-image `runs/`:

```bash
mkdir -p outputs/evaluations/codex_ch1_ch2_class_folder_eval_20260704_1333

rsync -a \
  --exclude 'runs/' \
  --exclude 'transformed_dataset/' \
  192.168.86.14:~/Projects/nornikel_codex_e2e_eval_20260704_1333/repo/outputs/evaluations/codex_ch1_ch2_class_folder_eval_20260704_1333/ \
  outputs/evaluations/codex_ch1_ch2_class_folder_eval_20260704_1333/
```

## Split

| Source label | Source images | Selected after conflict skip |
| --- | ---: | ---: |
| `fine_intergrowth` | 486 | 471 |
| `ordinary_intergrowth` | 565 | 543 |
| `talcose` | 129 | 118 |
| **Total** | **1180** | **1132** |

Duplicate/conflict audit:

- duplicate content groups: `56`
- duplicate items in duplicate groups: `112`
- cross-label conflict groups: `24`
- skipped cross-label duplicate paths: `48`

## Pipeline

- runner: `scripts/run_resident_batch.py`
- binary sulfide checkpoint: `models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt`
- talc checkpoint: `outputs/talc_segformer_folds/segformer_b0_full_20260703/fold_00/segformer_b0/best.pt`
- talc model threshold: `0.50`
- deterministic talc gate threshold: `talc_fraction > 0.10`
- tile/stride: `1024 / 768`
- batch size: `2`
- device: `cuda`
- run start: `2026-07-04T13:37:43+03:00`
- run finish: `2026-07-04T14:46:50+03:00`
- inference wall time: `1:09:07`
- rows: `1132`
- failures: `0`

## Deterministic Rule Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | `0.4293` |
| Macro F1 | `0.4642` |
| Weighted F1 | `0.3992` |
| Macro AUC OVR | `0.6266` |

Per-class metrics:

| Class | Support | Precision | Recall | F1 | AUC OVR |
| --- | ---: | ---: | ---: | ---: | ---: |
| `row_ore` | 543 | `0.4188` | `0.2136` | `0.2829` | `0.4973` |
| `hard_to_process_ore` | 471 | `0.4143` | `0.5541` | `0.4741` | `0.4226` |
| `talcose_ore` | 118 | `0.4844` | `0.9237` | `0.6356` | `0.9599` |

Confusion matrix:

| True \ Pred | `row_ore` | `hard_to_process_ore` | `talcose_ore` |
| --- | ---: | ---: | ---: |
| `row_ore` | 116 | 361 | 66 |
| `hard_to_process_ore` | 160 | 261 | 50 |
| `talcose_ore` | 1 | 8 | 109 |

## Feature-Classifier CV Metrics

These are image-level 5-fold CV metrics over features extracted by the same
segmentation pipeline (`summary.csv` plus component aggregate features). They are
reported separately from deterministic-rule metrics.

Best model: `random_forest`.

| Metric | Value |
| --- | ---: |
| Accuracy | `0.7800` |
| Macro F1 | `0.7699` |
| Weighted F1 | `0.7771` |
| Macro AUC OVR | `0.9025` |

Model comparison:

| Model | Accuracy | Macro F1 | Weighted F1 | Macro AUC OVR |
| --- | ---: | ---: | ---: | ---: |
| `random_forest` | `0.7800` | `0.7699` | `0.7771` | `0.9025` |
| `extra_trees` | `0.7774` | `0.7690` | `0.7750` | `0.9051` |
| `logistic` | `0.7394` | `0.7372` | `0.7393` | `0.8714` |

Best-model per-class metrics:

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| `row_ore` | 543 | `0.7419` | `0.8895` | `0.8090` |
| `hard_to_process_ore` | 471 | `0.8269` | `0.6794` | `0.7459` |
| `talcose_ore` | 118 | `0.8511` | `0.6780` | `0.7547` |

Best-model confusion matrix:

| True \ Pred | `row_ore` | `hard_to_process_ore` | `talcose_ore` |
| --- | ---: | ---: | ---: |
| `row_ore` | 483 | 55 | 5 |
| `hard_to_process_ore` | 142 | 320 | 9 |
| `talcose_ore` | 26 | 12 | 80 |

## Talc Gate Metrics

Fixed first gate, matching the ore rule:

| Metric | Value |
| --- | ---: |
| Threshold | `talc_fraction > 0.10` |
| TP / FP / FN / TN | `109 / 116 / 9 / 898` |
| Precision | `0.4844` |
| Recall | `0.9237` |
| Specificity | `0.8856` |
| F1 | `0.6356` |
| Binary accuracy | `0.8896` |

Descriptive threshold grid (`0.00..0.60`, step `0.01`) gives best F1 at
threshold `0.23`:

| Metric | Value |
| --- | ---: |
| TP / FP / FN / TN | `84 / 47 / 34 / 967` |
| Precision | `0.6412` |
| Recall | `0.7119` |
| Specificity | `0.9536` |
| F1 | `0.6747` |
| Binary accuracy | `0.9284` |

The grid result is not a replacement for the fixed 10% gate; it trades recall
for precision.

Talc fractions by source label:

| Source label | Count | Mean | Median | P10 | P90 | Max | `>10%` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fine_intergrowth` | 471 | `0.0403` | `0.0040` | `0.0001` | `0.1044` | `0.8200` | 50 |
| `ordinary_intergrowth` | 543 | `0.0433` | `0.0079` | `0.0004` | `0.1170` | `0.5811` | 66 |
| `talcose` | 118 | `0.4066` | `0.4056` | `0.1201` | `0.7440` | `0.8619` | 109 |

## Conclusion

The independent Codex gx10 rerun reproduced the earlier deterministic rule and
talc-gate numbers on the same deconflicted ch1/ch2 class-folder split:
`1132` selected images, `0` failures, rule macro-F1 `0.4642`, and fixed talc
gate F1 `0.6356`.

The additional feature-classifier CV pass on the same completed run gives a
stronger image-level learned benchmark: best macro-F1 `0.7699`, weighted-F1
`0.7771`, macro AUC OVR `0.9025`. Report it separately from the deterministic
rule because it is a CV benchmark over extracted features, not the current
production rule output.

