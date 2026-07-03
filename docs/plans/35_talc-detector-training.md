# Talc Detector Training Plan

Date: 2026-07-03

Status: approved; implementation started with the dataset builder.

Context: `docs/notes/2026-07-03-talc-ground-truth-strategy.md`. All 42
blue-line samples now have human-reviewed pixel masks
(`reviewed/reviewed_talc_mask.png` + `reviewed_ignore_mask.png`). Reviewed
talc is median 27% of analyzed area, 83% of it outside the original blue-line
bags. Baselines against reviewed GT: oracle per-image luma threshold IoU
0.502; production HSV candidate IoU 0.000 (to be replaced).

## Goal

Train a binary talc / not-talc pixel segmentation model that beats the
oracle-luma baseline (IoU 0.50) against reviewed masks and produces a
calibrated `talc_fraction` for the deterministic ore rule.

## Training Data Recipe

Per reviewed talcose sample (42 images, clean originals from
`dataset/Фото руд по сортам. ч1/Оталькованные руды`, masks from
`outputs/talc_blue_line_conversion/samples/*/reviewed/`):

- positive: `reviewed_talc_mask.png` pixels (authoritative, kept even where
  darker than the analyzed-area floor);
- ignore: `reviewed_ignore_mask.png` plus non-analyzed pixels (black borders)
  outside the reviewed talc;
- negative: everything else, including sulfide pixels (sulfide is a
  guaranteed non-talc phase).

Additional pure-negative tiles from ordinary/fine folders are supported by
the builder but disabled by default (`--max-negative-images 0`) until the
"ordinary ores are talc-poor" audit passes (~20 images eyeballed with the
luma preview). Negative images are SHA-256 deduplicated before selection.

Tiling mirrors the binary sulfide dataset: 512 px tiles, stride 384, tile
filters on valid fraction and positive fraction with a negative-keep
fraction, per-source tile caps. Output manifest uses the same schema as
`outputs/binary_sulfide_dataset_v0`, so `BinarySulfideTileDataset` and
`scripts/train_binary_sulfide.py` consume it unchanged.

Split discipline: assignment is per source image, never per tile, stratified
by series (`DSCN` vs scanner `25503xx`) and magnification (5x/10x) groups.
Default val fraction 0.2 (~8 images). `--val-samples` allows explicit
held-out lists for cross-validation reruns. With 42 images, final claims
should use image-level k-fold (rerun builder per fold) rather than one split.

## Model Ladder

1. Fast baseline: frozen DINOv2 dense features + light pixel head; probability
   maps feed `source_fusion`. Coarse boundaries expected.
2. Target: SegFormer-B0/B1 via the existing `train_binary_sulfide.py` stack
   (ignore-pixel loss, AMP, IoU logging) on the talc manifest, pretrained
   init mandatory at this data size.

## Augmentation Caution

Geometric flips/rotations and grinding/polishing artifact augmentation are
safe. Brightness jitter must stay moderate: luminance is the primary talc
signal (per-image thresholds ranged 30-150), and aggressive brightness
augmentation destroys it. The existing dataset-class jitter (0.85-1.15) is
acceptable.

## Evaluation Protocol

- Pixel: IoU/F1 (+ HD95 where meaningful) vs reviewed masks on held-out
  images, mean±std across folds.
- Baselines to report alongside: oracle per-image luma threshold (0.502
  median IoU), blind auto-threshold luma variant, HSV candidate (0.000).
- Fraction: `talc_fraction` absolute error vs reviewed masks, target ±3pp.
- Downstream: regenerate ore-rule calibration with the new talc source and
  re-run the deconflicted balanced batch -> image-level macro F1.

## Known Risks

- Reviewed masks are non-expert and were luma-slider-assisted: agreement
  numbers between luma-based detectors and these masks carry circularity;
  point counting on a few held-out images remains the independent check.
- 42 images share acquisition conditions; cross-class negatives and careful
  augmentation mitigate but do not remove domain overfit.
- Official dataset has 56 duplicate-content groups; any negative-image
  selection must dedupe by SHA-256 (the builder does) and future folds must
  keep duplicates on one side of the split.
- Ore-rule calibrations that consumed HSV `talc_fraction` are invalid for
  talc semantics and must be regenerated after the detector lands.

## Execution Order

1. `scripts/build_talc_dataset.py`: tiled dataset from reviewed masks (this
   plan's first implementation step).
2. Ordinary/fine talc-poor audit; then enable negative tiles.
3. DINOv2 + pixel-head baseline, report IoU vs oracle-luma.
4. SegFormer-B0/B1 training run (zelda/gx10), k-fold image-level eval.
5. Wire the winner into the pipeline as the talc source (replacing the HSV
   candidate), regenerate rule calibration, re-run official batch eval.
6. Optional: self-training over the remaining talcose folder images with
   `talc_review_web` spot review.
