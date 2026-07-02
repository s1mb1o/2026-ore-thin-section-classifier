# Model Card: Binary Sulfide Segmentation

Date: 2026-07-03

## Intended Use

Binary `sulfide / not_sulfide` segmentation for reflected-light optical microscopy images in the official Nornickel ore-classification task. The mask is an upstream artifact for connected-component ordinary/fine intergrowth rules and visual QA overlays.

## Current Checkpoints

- Current default: SegFormer-B1, zelda run `outputs/train_segformer_b1_zelda_20260703_overnight_safetensors/best.pt`
- Local B1 mirror: `models/binary_sulfide/segformer_b1_dataset_v0_zelda_20260703_overnight_safetensors/`
- Stable fallback: SegFormer-B0, zelda run `outputs/train_segformer_b0_zelda_20260702_220225/best.pt`
- Local B0 mirror: `models/binary_sulfide/segformer_b0_dataset_v0_zelda_20260702_220225/`
- ResUNet sanity check: gx10 run `outputs/train_resunet_gx10_20260703_004425/best.pt`

## Training Data

- Dataset manifest: `outputs/binary_sulfide_dataset_v0/manifest.json`
- Total tiles: `8536`
- Split: `6948` train / `1588` val
- Sources: LumenStone pixel masks plus official-image heuristic pseudo masks
- Supervision type: weak/pseudo labels, not expert geological ground truth

## Metrics

Current SegFormer-B1 best checkpoint:

- val sulfide IoU: `0.971548`
- val background IoU: `0.967670`
- val pixel accuracy: `0.984634`
- sulfide F1: `0.985569`
- sulfide AUC: `0.998522`
- Hausdorff mean on 512 sampled val tiles: `76.81 px`
- HD95 mean on 512 sampled val tiles: `26.25 px`

Fallback SegFormer-B0 best checkpoint:

- val sulfide IoU: `0.953371`
- val background IoU: `0.947638`
- val pixel accuracy: `0.974712`
- sulfide F1: `0.976129`
- sulfide AUC: `0.996154`
- Hausdorff mean on 512 sampled val tiles: `86.23 px`
- HD95 mean on 512 sampled val tiles: `33.92 px`

## Limitations

- Weak-label metrics can overestimate real geological performance.
- Official panoramas are unannotated and unclassified; use them for stress testing and visual QA, not labelled accuracy.
- The model only detects sulfide pixels. It does not directly classify talc or ore type.
- Ordinary/fine intergrowth is a downstream component rule, not a learned pixel class yet.
- Checkpoints trained on zelda currently require a Transformers version compatible with the saved SegFormer key namespace; project requirements pin `transformers>=5.12.1`.

## Recommended Use In Demo

Use the final mirrored SegFormer-B1 checkpoint for sulfide masks and confidence heatmaps. Keep SegFormer-B0 as the smaller fallback if B1 loading or memory becomes a blocker during the live demo.
