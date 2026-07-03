# Talc Non-Sulfide Segmentation Training Run

Date: 2026-07-03

Status: local baseline and full zelda SegFormer-B0 fold run completed.

## Goal

Train a binary talc segmentation model from the reviewed talc annotations, but
only learn and emit talc on non-sulfide pixels. Sulfide pixels are ignored in
training and clipped out during inference.

## Implemented Scripts

- `scripts/build_talc_dataset.py`: now marks each sample's `sulfide_mask.png`
  pixels as ignore by default. Use `--include-sulfide-pixels` only to reproduce
  the older behavior.
- `scripts/train_talc_segmentation.py`: talc-named binary segmentation trainer
  over the existing ResUNet/SegFormer model stack.
- `scripts/infer_talc_segmentation.py`: tiled talc inference that writes
  `talc_mask.png`, `confidence.png`, `confidence_non_sulfide.png`,
  `non_sulfide_mask.png`, and an overlay, with final talc clipped to
  `analyzed_area & ~sulfide_mask`.
- `scripts/run_talc_segformer_folds.py`: image-level fold runner for
  pretrained SegFormer talc training. For each fold it rebuilds a dataset with
  held-out image ids, trains via `scripts/train_talc_segmentation.py`, and
  calibrates a probability threshold on validation tiles.

## Dataset Build

Command:

```bash
python3 scripts/build_talc_dataset.py \
  --conversion-dir outputs/talc_blue_line_conversion \
  --clean-image-dir "dataset/Фото руд по сортам. ч1/Оталькованные руды" \
  --out-dir outputs/talc_non_sulfide_dataset_v0 \
  --tile-size 384 \
  --stride 288 \
  --max-tiles-per-source 36 \
  --min-positive-fraction 0.001 \
  --min-valid-fraction 0.30 \
  --negative-keep-fraction 0.20 \
  --seed 20260703 \
  --overwrite
```

Result:

- Manifest: `outputs/talc_non_sulfide_dataset_v0/manifest.json`.
- Reviewed samples: `42`.
- Tiles: `1510` total, `1150` train, `360` val.
- Tiles with talc positives: `1499`.
- Sulfide masks loaded: `42`.
- Sulfide pixels marked ignored: `25,036,407`.
- Reviewed talc/sulfide overlap in this workspace: `0` pixels.

## Local Training Run

Command:

```bash
python3 scripts/train_talc_segmentation.py \
  --dataset-manifest outputs/talc_non_sulfide_dataset_v0/manifest.json \
  --out-dir models/talc_segmentation/resunet_non_sulfide_20260703_local \
  --model resunet \
  --base-channels 16 \
  --epochs 3 \
  --batch-size 4 \
  --num-workers 0 \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --device auto \
  --max-steps-per-epoch 80 \
  --seed 20260703
```

Result:

- Checkpoint: `models/talc_segmentation/resunet_non_sulfide_20260703_local/best.pt`.
- Device: local MPS.
- Best checkpoint epoch: `1`.
- Best validation talc IoU: `0.526502`.
- `train_log.csv`:

| epoch | train_loss | val_loss | val_iou_talc | val_iou_not_talc | val_pixel_acc |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.646255 | 0.558129 | 0.526502 | 0.645541 | 0.745750 |
| 2 | 0.586491 | 0.579430 | 0.377868 | 0.631251 | 0.698727 |
| 3 | 0.538150 | 0.564225 | 0.407014 | 0.642344 | 0.712839 |

The checkpoint metadata still contains `best_iou_sulfide` because the shared
binary checkpoint loader uses that key. In this talc run, it means positive
class talc IoU.

## Inference Smoke

Command:

```bash
python3 scripts/infer_talc_segmentation.py \
  --image outputs/talc_blue_line_conversion/samples/DSCN4714/DSCN4714.JPG \
  --sulfide-mask outputs/talc_blue_line_conversion/samples/DSCN4714/sulfide_mask.png \
  --checkpoint models/talc_segmentation/resunet_non_sulfide_20260703_local/best.pt \
  --out-dir outputs/talc_segmentation_predictions/resunet_non_sulfide_20260703_local_DSCN4714 \
  --tile-size 384 \
  --stride 288 \
  --batch-size 4 \
  --device auto \
  --threshold 0.5
```

Result:

- Prediction directory:
  `outputs/talc_segmentation_predictions/resunet_non_sulfide_20260703_local_DSCN4714`.
- Predicted talc on sulfide pixels: `0`.
- Reviewed talc fraction over non-sulfide pixels: `0.606860`.
- Predicted talc fraction over non-sulfide pixels: `0.732165`.
- Non-sulfide IoU vs reviewed mask for `DSCN4714`: `0.693980`.
- Output files include `talc_mask.png`, `confidence_non_sulfide.png`,
  `non_sulfide_mask.png`, `sulfide_mask_aligned.png`, `overlay_preview.jpg`,
  and `summary.json`.

## Interpretation

The quick local baseline proved the data path and the key contract: no talc is
emitted on sulfide pixels, and the held-out sample smoke is plausible. The
local validation IoU is just above the previous oracle per-image luma-threshold
median (`0.502`), but that run is capped and uses a small ResUNet from random
initialization.

The full zelda SegFormer-B0 fold run is now the better quality reference:
mean calibrated talc IoU `0.644191`, mean calibrated F1 `0.782301` across 5
image-level folds.

## SegFormer Fold Runner Smoke

Command:

```bash
python3 scripts/run_talc_segformer_folds.py \
  --conversion-dir outputs/talc_blue_line_conversion \
  --clean-image-dir "dataset/Фото руд по сортам. ч1/Оталькованные руды" \
  --out-dir outputs/talc_segformer_folds/segformer_b0_smoke_20260703 \
  --model segformer_b0 \
  --folds 2 \
  --folds-to-run 0 \
  --tile-size 384 \
  --stride 288 \
  --max-tiles-per-source 12 \
  --epochs 1 \
  --batch-size 1 \
  --calibration-batch-size 1 \
  --num-workers 0 \
  --lr 0.00006 \
  --device auto \
  --max-steps-per-epoch 10 \
  --thresholds 0.30,0.40,0.50,0.60,0.70 \
  --seed 20260703 \
  --overwrite
```

Result:

- Output: `outputs/talc_segformer_folds/segformer_b0_smoke_20260703`.
- Pretrained encoder: `nvidia/mit-b0` loaded; segmentation head initialized for
  two talc classes.
- Fold `0` validation samples: `21` source images.
- Fold dataset tiles: `504`.
- Training: `1` epoch, capped at `10` train steps.
- Training-time val talc IoU: `0.223376`.
- Best calibrated threshold: `0.40`.
- Best calibrated validation tile talc IoU: `0.373902`.
- Best calibrated validation tile talc F1: `0.544292`.

This smoke proves the fold/train/calibration script and pretrained loading
path. It is intentionally under-trained; use the same script with all folds,
more epochs, and no `--max-steps-per-epoch` cap on zelda/gx10 for quality
numbers.

## Full SegFormer-B0 Five-Fold Run

Remote launcher copied locally:

```bash
outputs/logs/run_talc_segformer_b0_full_20260703.sh
```

Core command:

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

Result:

- Remote host: zelda `root@161.104.48.181`, RTX 4090.
- Output copied locally: `outputs/talc_segformer_folds/segformer_b0_full_20260703`.
- Log copied locally: `outputs/logs/talc_segformer_b0_full_20260703.log`.
- Mean calibrated talc IoU across 5 image-level folds: `0.644191`.
- Mean calibrated talc F1 across 5 image-level folds: `0.782301`.

| fold | threshold | talc IoU | talc F1 | precision | recall |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.50 | 0.630514 | 0.773393 | 0.743669 | 0.805593 |
| 1 | 0.50 | 0.552224 | 0.711526 | 0.621009 | 0.832932 |
| 2 | 0.40 | 0.654700 | 0.791322 | 0.747161 | 0.841030 |
| 3 | 0.35 | 0.711828 | 0.831658 | 0.796350 | 0.870243 |
| 4 | 0.55 | 0.671690 | 0.803606 | 0.780119 | 0.828551 |
