# Binary Sulfide Model Benchmark

Date: 2026-07-03

## Scope

This benchmark compares the first binary `sulfide / not_sulfide` segmentation models for the official optical-microscopy pipeline.

Dataset:

- Manifest: `outputs/binary_sulfide_dataset_v0/manifest.json`
- Total tiles: `8536`
- Split: `6948` train / `1588` val
- Sources: `2976` LumenStone tiles and `5560` official-image heuristic pseudo-label tiles
- Tile size / stride: `512 / 384`

Important caveat: this is a weak-supervision benchmark against pseudo-labels, not an expert geology ground-truth benchmark. The numbers are useful for choosing a first checkpoint and for finding obviously bad model families, but they must not be presented as final geological accuracy.

Organizer metric clarification from 2026-07-03: production-solution metrics to track are IoU and Hausdorff distance for segmentation, and F1 and AUC for classification. The table below gives model-selection metrics from training; the extended evaluator JSONs include all four requested metric families.

Panorama clarification: official panoramas may be used as a test/stress set, but they are unannotated and unclassified in the provided dataset. The recommended evaluation set should be balanced from several labelled classes where labels exist.

## Results

| Model | Host | Status | Best val sulfide IoU | Best epoch | Val bg IoU at best | Val pixel acc at best | Avg sec/epoch | Checkpoint size |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SegFormer-B2 | zelda `root@161.104.48.181` | completed 30/30 | `0.974381` | 20 | `0.970874` | `0.986181` | `78.40` | `320M` |
| SegFormer-B1 | zelda `root@161.104.48.181` | completed 30/30 | `0.971548` | 16 | `0.967670` | `0.984634` | `36.59` | `160M` |
| ResUNet `base_channels=32` | gx10 `ashmelev@192.168.86.14` | completed 30/30 | `0.956436` | 26 | `0.950908` | `0.976373` | `241.41` | `96M` |
| SegFormer-B0 | zelda `root@161.104.48.181` | completed 30/30 | `0.953371` | 13 | `0.947638` | `0.974712` | `27.49` | `43M` |

SegFormer-B2 extended metrics from `scripts/evaluate_binary_sulfide.py`:

- F1 sulfide: `0.987024`
- AUC sulfide: `0.998811`
- Hausdorff mean on 512 sampled val tiles: `73.32 px`
- HD95 mean on 512 sampled val tiles: `23.57 px`
- Output: `outputs/evaluations/segformer_b2_best_eval_metrics.json`

SegFormer-B1 extended metrics from `scripts/evaluate_binary_sulfide.py`:

- F1 sulfide: `0.985569`
- AUC sulfide: `0.998522`
- Hausdorff mean on 512 sampled val tiles: `76.81 px`
- HD95 mean on 512 sampled val tiles: `26.25 px`
- Output: `outputs/evaluations/segformer_b1_best_eval_metrics.json`

SegFormer-B0 extended metrics from `scripts/evaluate_binary_sulfide.py`:

- F1 sulfide: `0.976129`
- AUC sulfide: `0.996154`
- Hausdorff mean on 512 sampled val tiles: `86.23 px`
- HD95 mean on 512 sampled val tiles: `33.92 px`
- Output: `outputs/evaluations/segformer_b0_best_eval_metrics.json`

ResUNet extended metrics from `scripts/evaluate_binary_sulfide.py`:

- F1 sulfide: `0.977733`
- AUC sulfide: `0.996942`
- Hausdorff mean on 512 sampled val tiles: `92.30 px`
- HD95 mean on 512 sampled val tiles: `37.37 px`
- Output: `outputs/evaluations/resunet_best_eval_metrics.json`

SegFormer-B0 final epoch metrics:

- epoch: `30`
- train loss: `0.045051`
- val loss: `0.071400`
- val sulfide IoU: `0.951119`
- val bg IoU: `0.945790`
- val pixel accuracy: `0.973618`

SegFormer-B1 final epoch metrics:

- epoch: `30`
- train loss: `0.027995`
- val loss: `0.055467`
- val sulfide IoU: `0.964032`
- val bg IoU: `0.959587`
- val pixel accuracy: `0.980600`

ResUNet final epoch metrics:

- epoch: `30`
- train loss: `0.044054`
- val loss: `0.065122`
- val sulfide IoU: `0.953216`
- val bg IoU: `0.946566`
- val pixel accuracy: `0.974418`

SegFormer-B2 final epoch metrics:

- epoch: `30`
- train loss: `0.021654`
- val loss: `0.043971`
- val sulfide IoU: `0.969119`
- val bg IoU: `0.965199`
- val pixel accuracy: `0.983366`

## Checkpoints

- Local SegFormer-B2 mirror: `models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/`
- Current best binary sulfide checkpoint: `/root/2026_Nornikel_Hackaton_v2/outputs/train_segformer_b2_zelda_20260703_overnight_safetensors/best.pt`
- SegFormer-B2 last checkpoint: `/root/2026_Nornikel_Hackaton_v2/outputs/train_segformer_b2_zelda_20260703_overnight_safetensors/last.pt`
- Local SegFormer-B1 fallback mirror: `models/binary_sulfide/segformer_b1_dataset_v0_zelda_20260703_overnight_safetensors/`
- SegFormer-B1 last checkpoint: `/root/2026_Nornikel_Hackaton_v2/outputs/train_segformer_b1_zelda_20260703_overnight_safetensors/last.pt`
- Local SegFormer-B0 fallback mirror: `models/binary_sulfide/segformer_b0_dataset_v0_zelda_20260702_220225/`
- SegFormer-B0 last checkpoint: `/root/2026_Nornikel_Hackaton_v2/outputs/train_segformer_b0_zelda_20260702_220225/last.pt`
- ResUNet checkpoint: `/home/ashmelev/Projects/2026_Nornikel_Hackaton_v2/outputs/train_resunet_gx10_20260703_004425/best.pt`

Local SegFormer-B2 checksums:

- `best.pt`: `55c31ef645cfb5c9b0b8fd91f4b9d2070e425b32ed60e23b3c15b292546b910f`
- `last.pt`: `40cc2fa920282964d70588a9815a94915611a22f0182e97327c629220119f00c`

Local SegFormer-B1 checksums:

- `best.pt`: `e71ceb0d3df88b8f24473c5fb4b82678303d854a2f8b15ad1af66022dea11908`
- `last.pt`: `03db84dbce6395cd381c2be568d9a366aeaf94cfab573ce80c34566d7a435d11`

Local SegFormer-B0 checksums:

- `best.pt`: `6133984ab605424ef9a42a4486857ba1872fae87fa2a1fa63ebe9b49a6368162`
- `last.pt`: `fa64b00fe460ad67c4b150622d51ef33cbbe5aeadbbd82fb3952282498263cce`

## Recommendation

Use SegFormer-B2 as the current default binary sulfide checkpoint. It beats SegFormer-B1 on the same weak-label validation split across IoU, F1, AUC, Hausdorff mean, and HD95 mean. Keep SegFormer-B1 as the faster fallback and SegFormer-B0 as the smallest fallback.

ResUNet is useful as an architecture-diversity sanity check and beats SegFormer-B0 on IoU, but it is slower and below B1/B2 on the extended metrics.

## Next Benchmark Actions

1. Run B2/B1/heuristic disagreement sampling for the Streamlit sulfide QA queue.
2. Use `outputs/official_balanced_eval_split.json` for balanced image-level class validation; keep unlabelled panoramas separate for performance and visual stress tests.
3. Calibrate ordinary/fine component thresholds with image-level F1/AUC, not just weak-label pixel IoU.
4. After non-expert QA produces corrected masks, repeat the benchmark with a new dataset version and keep the current numbers as `binary_sulfide_dataset_v0` baseline only.
