# SegFormer-B1 Binary Sulfide Checkpoint

Date: 2026-07-03

This local mirror contains the current default binary `sulfide / not_sulfide` segmentation checkpoint.

## Source Run

- Host: zelda `root@161.104.48.181`
- Remote workspace: `/root/2026_Nornikel_Hackaton_v2`
- Remote output dir: `outputs/train_segformer_b1_zelda_20260703_overnight_safetensors`
- Dataset manifest: `outputs/binary_sulfide_dataset_v0/manifest.json`
- Model: `segformer_b1`
- Epochs: `30`
- Best epoch: `16`

## Metrics

- Best val sulfide IoU: `0.971548`
- Val background IoU at best: `0.967670`
- Val pixel accuracy at best: `0.984634`
- Sulfide F1: `0.985569`
- Sulfide AUC: `0.998522`
- Hausdorff mean on 512 sampled val tiles: `76.81 px`
- HD95 mean on 512 sampled val tiles: `26.25 px`

Extended metrics are saved at `outputs/evaluations/segformer_b1_best_eval_metrics.json`.

## Files

- `best.pt`: best validation IoU checkpoint.
- `last.pt`: epoch 30 checkpoint.
- `metrics.json`: training best metric summary.
- `train_log.csv`: per-epoch training/validation metrics.

## Checksums

- `best.pt`: `e71ceb0d3df88b8f24473c5fb4b82678303d854a2f8b15ad1af66022dea11908`
- `last.pt`: `03db84dbce6395cd381c2be568d9a366aeaf94cfab573ce80c34566d7a435d11`

## Caveat

Metrics are against weak/pseudo labels, not expert pixel ground truth. Use this checkpoint as the best current engineering artifact, not as a final geological validation claim.
