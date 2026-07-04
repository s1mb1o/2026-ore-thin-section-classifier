# Model Card: Binary Sulfide Segmentation

Date: 2026-07-03

Updated: 2026-07-04 with Mask2Former-Swin-Tiny comparison.

## Intended Use

Binary `sulfide / not_sulfide` segmentation for reflected-light optical microscopy images in the official Nornickel ore-classification task. The mask is an upstream artifact for connected-component ordinary/fine intergrowth rules and visual QA overlays.

## Current Checkpoints

- Current default: SegFormer-B2, zelda run `outputs/train_segformer_b2_zelda_20260703_overnight_safetensors/best.pt`
- Local B2 mirror: `models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/`
- Fast fallback: SegFormer-B1, zelda run `outputs/train_segformer_b1_zelda_20260703_overnight_safetensors/best.pt`
- Local B1 mirror: `models/binary_sulfide/segformer_b1_dataset_v0_zelda_20260703_overnight_safetensors/`
- Small fallback: SegFormer-B0, zelda run `outputs/train_segformer_b0_zelda_20260702_220225/best.pt`
- Local B0 mirror: `models/binary_sulfide/segformer_b0_dataset_v0_zelda_20260702_220225/`
- Non-default comparison: Mask2Former-Swin-Tiny, zelda run `outputs/train_mask2former_zelda_20260704_1553/best.pt`
- Local Mask2Former mirror: `models/binary_sulfide/mask2former_swin_tiny_dataset_v0_zelda_20260704_1553/`
- ResUNet sanity check: gx10 run `outputs/train_resunet_gx10_20260703_004425/best.pt`

## Training Data

- Dataset manifest: `outputs/binary_sulfide_dataset_v0/manifest.json`
- Total tiles: `8536`
- Split: `6948` train / `1588` val
- Sources: LumenStone pixel masks plus official-image heuristic pseudo masks
- Supervision type: weak/pseudo labels, not expert geological ground truth

## Metrics

Current SegFormer-B2 best checkpoint:

- val sulfide IoU: `0.974381`
- val background IoU: `0.970874`
- val pixel accuracy: `0.986181`
- sulfide F1: `0.987024`
- sulfide AUC: `0.998811`
- Hausdorff mean on 512 sampled val tiles: `73.32 px`
- HD95 mean on 512 sampled val tiles: `23.57 px`

Fallback SegFormer-B1 best checkpoint:

- val sulfide IoU: `0.971548`
- val background IoU: `0.967670`
- val pixel accuracy: `0.984634`
- sulfide F1: `0.985569`
- sulfide AUC: `0.998522`
- Hausdorff mean on 512 sampled val tiles: `76.81 px`
- HD95 mean on 512 sampled val tiles: `26.25 px`

Small fallback SegFormer-B0 best checkpoint:

- val sulfide IoU: `0.953371`
- val background IoU: `0.947638`
- val pixel accuracy: `0.974712`
- sulfide F1: `0.976129`
- sulfide AUC: `0.996154`
- Hausdorff mean on 512 sampled val tiles: `86.23 px`
- HD95 mean on 512 sampled val tiles: `33.92 px`

Non-default Mask2Former-Swin-Tiny comparison:

- val sulfide IoU: `0.968313`
- val background IoU: `0.963827`
- val pixel accuracy: `0.982819`
- sulfide F1: `0.983901`
- sulfide AUC: `0.998492`
- Hausdorff mean on 512 sampled val tiles: `83.24 px`
- HD95 mean on 512 sampled val tiles: `29.55 px`
- average training time: `226.39 s/epoch`

## Limitations

- Weak-label metrics can overestimate real geological performance.
- Official panoramas are unannotated and unclassified; use them for stress testing and visual QA, not labelled accuracy.
- The model only detects sulfide pixels. It does not directly classify talc or ore type.
- Ordinary/fine intergrowth is a downstream component rule, not a learned pixel class yet.
- Checkpoints trained on zelda currently require a Transformers version compatible with the saved SegFormer key namespace; project requirements pin `transformers>=5.12.1`.

## Recommended Use In Demo

Use the final mirrored SegFormer-B2 checkpoint for sulfide masks and confidence heatmaps. Keep SegFormer-B1 as the faster fallback if B2 loading or memory becomes a blocker during the live demo. Do not switch to Mask2Former-Swin-Tiny for the demo: it is slower and below B2 on this weak-label benchmark.
