# Scripts

Utility CLIs for dataset manifests, pseudo-label generation, training launchers, and evaluation should live here.

Keep heavy GPU training jobs outside Streamlit. Streamlit may emit the exact command, but training should run as a separate script on the selected GPU host.

## Talc Training And Inference

Build the tiled talc/not-talc dataset from the human-reviewed masks in the
blue-line conversion workspace. By default, sample `sulfide_mask.png` pixels
are marked ignored so the model learns talc only over non-sulfide pixels:

```bash
python3 scripts/build_talc_dataset.py \
  --out-dir outputs/talc_non_sulfide_dataset_v0 \
  --overwrite
```

Defaults: `outputs/talc_blue_line_conversion` reviewed masks + clean originals
from `dataset/Фото руд по сортам. ч1/Оталькованные руды`, 512 px tiles with
stride 384, per-image stratified train/val split (`--val-samples` forces an
explicit held-out list for k-fold reruns). Pure-negative tiles from
ordinary/fine folders stay disabled (`--max-negative-images 0`) until the
talc-poor audit passes; negative selection dedupes by SHA-256. Use
`--include-sulfide-pixels` only to reproduce the old sulfide-as-negative
behavior.

Train a local ResUNet baseline:

```bash
python3 scripts/train_talc_segmentation.py \
  --dataset-manifest outputs/talc_non_sulfide_dataset_v0/manifest.json \
  --out-dir models/talc_segmentation/resunet_non_sulfide_20260703_local \
  --model resunet \
  --base-channels 16 \
  --epochs 3 \
  --batch-size 4 \
  --num-workers 0 \
  --device auto \
  --max-steps-per-epoch 80
```

Run image-level SegFormer folds with threshold calibration. This command is the
short local smoke; remove `--folds-to-run 0` and the step cap for the full
zelda/gx10 run:

```bash
python3 scripts/run_talc_segformer_folds.py \
  --out-dir outputs/talc_segformer_folds/segformer_b0_smoke_20260703 \
  --model segformer_b0 \
  --folds 2 \
  --folds-to-run 0 \
  --epochs 1 \
  --batch-size 1 \
  --max-steps-per-epoch 10 \
  --thresholds 0.30,0.40,0.50,0.60,0.70 \
  --overwrite
```

Full zelda SegFormer-B0 run used for the current quality reference:

```bash
python scripts/run_talc_segformer_folds.py \
  --conversion-dir outputs/talc_blue_line_conversion \
  --clean-image-dir "dataset/Фото руд по сортам. ч1/Оталькованные руды" \
  --out-dir outputs/talc_segformer_folds/segformer_b0_full_20260703 \
  --model segformer_b0 \
  --folds 5 \
  --folds-to-run all \
  --tile-size 384 \
  --stride 288 \
  --max-tiles-per-source 36 \
  --epochs 20 \
  --batch-size 8 \
  --calibration-batch-size 8 \
  --num-workers 4 \
  --lr 0.00006 \
  --weight-decay 0.0001 \
  --device cuda \
  --amp \
  --thresholds 0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80 \
  --seed 20260703 \
  --overwrite
```

Run tiled talc inference clipped to non-sulfide pixels:

```bash
python3 scripts/infer_talc_segmentation.py \
  --image outputs/talc_blue_line_conversion/samples/DSCN4714/DSCN4714.JPG \
  --sulfide-mask outputs/talc_blue_line_conversion/samples/DSCN4714/sulfide_mask.png \
  --checkpoint models/talc_segmentation/resunet_non_sulfide_20260703_local/best.pt \
  --out-dir outputs/talc_segmentation_predictions/resunet_non_sulfide_20260703_local_DSCN4714 \
  --tile-size 384 \
  --stride 288 \
  --batch-size 4 \
  --device auto
```

See `docs/plans/35_talc-detector-training.md` and
`docs/notes/2026-07-03-talc-non-sulfide-segmentation-training.md`.

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

## Augmentation Review Gallery

Generate a static HTML gallery for visual review of the v2 runtime augmentation
settings:

```bash
python3 scripts/generate_augmentation_review_gallery.py \
  --per-label 1 \
  --max-side 720 \
  --overwrite
```

The default output is `outputs/augmentation_review/index.html`. The gallery uses
the deconflicted balanced official split, chooses one source image per class,
renders the original plus deterministic color/tone, acquisition-noise, and
grinding/polishing artifact variants, and writes the exact settings JSON into
each review card.

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
