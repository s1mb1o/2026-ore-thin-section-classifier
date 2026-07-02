# B1 Visual Validation Pack

Date: 2026-07-03

## Scope

Small visual sanity pack for the final SegFormer-B1 sulfide checkpoint:

- checkpoint: `models/binary_sulfide/segformer_b1_dataset_v0_zelda_20260703_overnight_safetensors/best.pt`
- source split: `outputs/official_balanced_eval_split.json`
- selected images: `2` ordinary, `2` fine/hard-to-process, `2` talcose
- output directory: `outputs/visual_validation_b1_final/`
- runtime host: zelda `root@161.104.48.181`

This is not a statistical benchmark. It is a visual/debug artifact for overlays, confidence maps, and immediate calibration risks.

## Summary

| Source label | Image id | Predicted class | Sulfide fraction | Ordinary sulfide | Fine sulfide |
| --- | --- | --- | ---: | ---: | ---: |
| ordinary_intergrowth | `ordinary_intergrowth_1_2539589-2` | hard_to_process_ore | `0.4274` | `0.1563` | `0.8368` |
| ordinary_intergrowth | `ordinary_intergrowth_2_2544791-1_10x_аншлиф` | row_ore | `0.5737` | `0.9993` | `0.0001` |
| fine_intergrowth | `fine_intergrowth_1_2539444-1` | hard_to_process_ore | `0.2180` | `0.2982` | `0.6914` |
| fine_intergrowth | `fine_intergrowth_2_2539446-2` | row_ore | `0.1580` | `0.6267` | `0.3121` |
| talcose | `talcose_1_2550374-2_10х` | hard_to_process_ore | `0.2247` | `0.1487` | `0.8437` |
| talcose | `talcose_2_2550375-1_10х` | hard_to_process_ore | `0.5472` | `0.1144` | `0.8833` |

Prediction counts:

- ordinary_intergrowth: `1` row_ore, `1` hard_to_process_ore
- fine_intergrowth: `1` row_ore, `1` hard_to_process_ore
- talcose: `2` hard_to_process_ore

## Interpretation

The final B1 sulfide segmentation produces usable masks and overlays on all six sampled images, but the downstream deterministic ordinary/fine rule is not calibrated enough to claim image-level class accuracy.

Specific risks:

- Some labelled ordinary images contain enough dark inclusions/replacement inside large sulfide masks that the current component rule flips them to hard-to-process.
- Some labelled fine/hard-to-process images contain many small ordinary components, so the current area-weighted rule can flip them to row ore.
- Talcose images are not yet using the talc mask in this pack, so they are classified only by sulfide ordinary/fine morphology. That is expected and should be fixed by wiring accepted talc masks into the final pipeline.

## Next Actions

1. Use this pack as the first QA/disagreement queue, not as final accuracy.
2. Calibrate component thresholds on the balanced split and report image-level F1/AUC separately from weak-label pixel metrics.
3. Wire talc masks into `run_ore_pipeline.py` for talcose examples before presenting talc classification quality.
4. Add a few false-positive/false-negative crops to the Streamlit QA app so non-expert reviewers can mark obvious rule errors.
