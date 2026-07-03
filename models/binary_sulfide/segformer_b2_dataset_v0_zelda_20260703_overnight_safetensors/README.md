# SegFormer-B2 Binary Sulfide Checkpoint

Date: 2026-07-03

This local mirror contains the current default binary `sulfide / not_sulfide` segmentation checkpoint.

## Source Run

- Host: zelda `root@161.104.48.181`
- Remote workspace: `/root/2026_Nornikel_Hackaton_v2`
- Remote output dir: `outputs/train_segformer_b2_zelda_20260703_overnight_safetensors`
- Dataset manifest: `outputs/binary_sulfide_dataset_v0/manifest.json`
- Model: `segformer_b2`
- Epochs: `30`
- Best epoch: `20`

## Metrics

- Best val sulfide IoU: `0.974381`
- Val background IoU at best: `0.970874`
- Val pixel accuracy at best: `0.986181`
- Sulfide F1: `0.987024`
- Sulfide AUC: `0.998811`
- Hausdorff mean on 512 sampled val tiles: `73.32 px`
- HD95 mean on 512 sampled val tiles: `23.57 px`

Extended metrics are saved at `outputs/evaluations/segformer_b2_best_eval_metrics.json`.

## Files

- `best.pt`: best validation IoU checkpoint.
- `last.pt`: epoch 30 checkpoint.
- `metrics.json`: training best metric summary.
- `train_log.csv`: per-epoch training/validation metrics.

## Checksums

- `best.pt`: `55c31ef645cfb5c9b0b8fd91f4b9d2070e425b32ed60e23b3c15b292546b910f`
- `last.pt`: `40cc2fa920282964d70588a9815a94915611a22f0182e97327c629220119f00c`

## Caveat

Metrics are against weak/pseudo labels, not expert pixel ground truth. Use this checkpoint as the best current engineering artifact, not as a final geological validation claim.
