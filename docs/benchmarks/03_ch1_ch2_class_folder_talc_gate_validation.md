# Ch1/Ch2 Class-Folder Talc-Gate Validation

Date: 2026-07-04

## Scope

This run validates the current ore pipeline on all labelled images from:

- `dataset/Фото руд по сортам. ч1`
- `dataset/Фото руд по сортам. ч2`

The split is unbalanced by design. It keeps every image from the requested class
folders except content hashes that appear under more than one class label.

Artifacts:

- split: `outputs/evaluations/ch1_ch2_class_folder_talc_gate_20260704/split.json`
- batch summary: `outputs/evaluations/ch1_ch2_class_folder_talc_gate_20260704/summary.csv`
- classification metrics: `outputs/evaluations/ch1_ch2_class_folder_talc_gate_20260704/classification_metrics.md`
- talc gate metrics: `outputs/evaluations/ch1_ch2_class_folder_talc_gate_20260704/talc_gate_metrics.md`
- gx10 log: `outputs/evaluations/ch1_ch2_class_folder_talc_gate_20260704/gx10_run.log`

## Split

| Source label | Source images | Selected after conflict skip |
| --- | ---: | ---: |
| `fine_intergrowth` | 486 | 471 |
| `ordinary_intergrowth` | 565 | 543 |
| `talcose` | 129 | 118 |
| **Total** | **1180** | **1132** |

Duplicate/conflict policy:

- duplicate content groups: `56`
- cross-label conflict groups: `24`
- skipped cross-label duplicate images: `48`

## Pipeline

Runtime:

- host: `gx10` (`NVIDIA GB10`)
- runner: `scripts/run_resident_batch.py`
- binary sulfide checkpoint: `models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt`
- talc checkpoint: `outputs/talc_segformer_folds/segformer_b0_full_20260703/fold_00/segformer_b0/best.pt`
- talc threshold: `0.50`
- talc gate rule in ore analysis: `talc_fraction > 0.10 => talcose_ore`
- tile/stride: `1024 / 768`
- batch size: `2`

Run result:

- rows: `1132`
- failures: `0`

## Overall Classification

| Metric | Value |
| --- | ---: |
| Accuracy | `0.4293` |
| Macro F1 | `0.4642` |
| Weighted F1 | `0.3992` |
| Macro AUC OVR | `0.6266` |

Per-class F1:

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| `row_ore` | 543 | `0.4188` | `0.2136` | `0.2829` |
| `hard_to_process_ore` | 471 | `0.4143` | `0.5541` | `0.4741` |
| `talcose_ore` | 118 | `0.4844` | `0.9237` | `0.6356` |

Confusion matrix:

| True \ Pred | `row_ore` | `hard_to_process_ore` | `talcose_ore` |
| --- | ---: | ---: | ---: |
| `row_ore` | 116 | 361 | 66 |
| `hard_to_process_ore` | 160 | 261 | 50 |
| `talcose_ore` | 1 | 8 | 109 |

## Talc Gate

At the requested first gate `talc_fraction > 10%`:

| Metric | Value |
| --- | ---: |
| TP / FP / FN / TN | `109 / 116 / 9 / 898` |
| Precision | `0.4844` |
| Recall | `0.9237` |
| Specificity | `0.8856` |
| F1 | `0.6356` |
| Binary gate accuracy | `0.8896` |

Interpretation: the current talc model plus 10% threshold is a useful
high-recall first gate for `Оталькованные руды`, but it is noisy: `116`
ordinary/fine images cross the gate. It should not be treated as a final grade
classifier without calibration and the ordinary/fine branch.

The threshold grid `0.00..0.60` gives the best talc-gate F1 at `0.23`:

- F1: `0.6747`
- precision: `0.6412`
- recall: `0.7119`

That threshold improves precision/F1 but loses talcose recall, so it is a
product decision rather than a silent replacement for the requested `10%` gate.

## Talc Fractions By Source Label

| Source label | Count | Mean | Median | P10 | P90 | Max | `>10%` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fine_intergrowth` | 471 | `0.0403` | `0.0040` | `0.0001` | `0.1044` | `0.8200` | 50 |
| `ordinary_intergrowth` | 543 | `0.0433` | `0.0079` | `0.0004` | `0.1170` | `0.5811` | 66 |
| `talcose` | 118 | `0.4066` | `0.4056` | `0.1201` | `0.7440` | `0.8619` | 109 |

## Conclusion

The trained talc model now separates talcose folders much better than the old
auto-candidate talc heuristic: only `9 / 118` talcose samples miss the 10% gate.
The immediate weakness is false positives from ordinary/fine folders, including
some very high predicted talc fractions. Next steps:

- inspect the highest non-talc false positives from `talc_gate_metrics.md`;
- decide whether the first gate should optimize recall (`10%`) or F1/precision
  (around `23%` on this split);
- fuse this talc gate with the trained ordinary/fine CNN branch for the final
  3-class classifier instead of relying on the deterministic ordinary/fine rule.

## Reproduction

```bash
python3 scripts/build_class_folder_eval_split.py \
  --dataset-root dataset \
  --out-json outputs/evaluations/ch1_ch2_class_folder_talc_gate_20260704/split.json \
  --out-csv outputs/evaluations/ch1_ch2_class_folder_talc_gate_20260704/split.csv

python scripts/run_resident_batch.py \
  --split-json outputs/evaluations/ch1_ch2_class_folder_talc_gate_20260704/split.json \
  --dataset-root dataset \
  --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt \
  --talc-checkpoint outputs/talc_segformer_folds/segformer_b0_full_20260703/fold_00/segformer_b0/best.pt \
  --talc-threshold 0.50 \
  --out-dir outputs/evaluations/ch1_ch2_class_folder_talc_gate_20260704 \
  --device cuda \
  --tile-size 1024 \
  --stride 768 \
  --batch-size 2 \
  --keep-going

python3 scripts/evaluate_ore_classification.py \
  --summary-csv outputs/evaluations/ch1_ch2_class_folder_talc_gate_20260704/summary.csv \
  --out-json outputs/evaluations/ch1_ch2_class_folder_talc_gate_20260704/classification_metrics.json \
  --out-md outputs/evaluations/ch1_ch2_class_folder_talc_gate_20260704/classification_metrics.md
```
