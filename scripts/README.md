# Scripts

Utility CLIs for dataset manifests, pseudo-label generation, training launchers, and evaluation should live here.

Keep heavy GPU training jobs outside Streamlit. Streamlit may emit the exact command, but training should run as a separate script on the selected GPU host.

## Manual Review Pack

Prepare a balanced B2 review pack with review panels, probability heatmaps,
uncertainty crops, and spreadsheet feedback templates:

```bash
python3 scripts/prepare_manual_review_pack.py \
  --per-label 3 \
  --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt \
  --out-dir outputs/manual_review/b2_balanced_review_pack \
  --device auto \
  --batch-size 1 \
  --overwrite
```

If the local Python environment cannot load a SegFormer checkpoint because of a
`transformers` namespace mismatch, run the same command on the training host
that produced the checkpoint and rsync `outputs/manual_review/b2_balanced_review_pack`
back locally.

## Talc Blue-Line Conversion

```bash
python3 scripts/convert_talc_blue_lines.py \
  --input "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
  --output-dir outputs/talc_blue_line_conversion \
  --summary-json outputs/talc_blue_line_conversion_summary.json
```

Use `--sulfide-mask-dir path/to/binary_sulfide_masks` when the binary sulfide
detector produces masks named by image stem. Without that directory, the
converter uses the conservative bright-phase sulfide heuristic.

## Full Ore Pipeline

Run one image through sulfide segmentation, automatic talc candidate extraction,
component analysis, and deterministic ore-class rules:

```bash
python3 scripts/run_ore_pipeline.py \
  --image "dataset/Фото руд по сортам. ч1/Рядовые руды/2539589-1.JPG" \
  --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt \
  --out-dir outputs/demo_ore_pipeline \
  --tile-size 1024 \
  --stride 768 \
  --batch-size 1 \
  --auto-talc-candidate
```

Use `--talc-mask path/to/final_talc_mask.png` instead of
`--auto-talc-candidate` when an accepted or manually corrected talc mask is
available. The automatic talc path is a conservative runtime candidate, not
expert ground truth.

## Official Batch And Image-Level Metrics

```bash
python3 scripts/run_official_batch.py \
  --split-json outputs/official_balanced_eval_split.json \
  --dataset-root dataset \
  --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt \
  --out-dir outputs/evaluations/b2_official_balanced_auto_talc \
  --tile-size 1024 \
  --stride 768 \
  --batch-size 1 \
  --device auto \
  --overwrite
```

```bash
python3 scripts/evaluate_ore_classification.py \
  --summary-csv outputs/evaluations/b2_official_balanced_auto_talc/summary.csv \
  --out-json outputs/evaluations/b2_official_balanced_auto_talc/ore_classification_metrics.json \
  --out-md outputs/evaluations/b2_official_balanced_auto_talc/ore_classification_metrics.md
```

The evaluator reports image-level accuracy, per-class precision/recall/F1,
macro/weighted F1, one-vs-rest AUC, and a confusion matrix. These are separate
from pixel-level sulfide segmentation IoU/Hausdorff metrics.
