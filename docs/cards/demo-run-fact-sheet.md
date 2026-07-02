# Run Fact Sheet: B2 Demo Ore Pipeline

Date: 2026-07-03

## Input

- Image: `outputs/inference_demo/source_images/row_2539589_1.JPG`
- Original official path: `dataset/Фото руд по сортам. ч1/Рядовые руды/2539589-1.JPG`
- Size: `2272 x 1704`

## Checkpoint

- Checkpoint: `models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt`
- Remote run checkpoint: `outputs/train_segformer_b2_zelda_20260703_overnight_safetensors/best.pt`
- Training run: `outputs/train_segformer_b2_zelda_20260703_overnight_safetensors`
- Best epoch: `20`
- Best val sulfide IoU: `0.974381`

## Parameters

- Tile size: `1024`
- Stride: `768`
- Batch size: `8`
- Device: `cuda`
- Threshold: `0.5`
- Component min area: `128 px`
- Component closing kernel: `21 px`

## Outputs

- Binary sulfide mask: `outputs/inference_demo/b2_final_row_2539589_1/binary_sulfide/sulfide_mask.png`
- Confidence heatmap: `outputs/inference_demo/b2_final_row_2539589_1/binary_sulfide/confidence.png`
- Sulfide overlay: `outputs/inference_demo/b2_final_row_2539589_1/binary_sulfide/overlay_preview.jpg`
- Ore summary: `outputs/inference_demo/b2_final_row_2539589_1/ore_analysis/ore_summary.json`
- Component features: `outputs/inference_demo/b2_final_row_2539589_1/ore_analysis/component_features.csv`
- Intergrowth overlay: `outputs/inference_demo/b2_final_row_2539589_1/ore_analysis/intergrowth_overlay_preview.jpg`
- Pipeline summary: `outputs/inference_demo/b2_final_row_2539589_1/pipeline_summary.json`

## Result

- Inference time: `3.536 s`
- Sulfide fraction: `29.626%`
- Component count: `154`
- Ordinary sulfide fraction: `19.9%`
- Fine sulfide fraction: `80.1%`
- Talc fraction: `0.0%`
- Deterministic class: `hard_to_process_ore`
- Russian report text: `Руда классифицирована как труднообогатимая руда: тальк 0.0%, обычные срастания 19.9% сульфидной площади, тонкие срастания 80.1% сульфидной площади.`

## Caveat

This result is a deterministic engineering demo on one official image using weakly supervised sulfide masks and rule-based ordinary/fine component classification. It is not expert geological ground truth.
