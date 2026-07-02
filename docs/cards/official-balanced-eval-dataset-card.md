# Dataset Card: Official Balanced Image-Level Evaluation Split

Date: 2026-07-03

## Source

- Official manifest: `outputs/official_manifest.json`
- Split JSON: `outputs/official_balanced_eval_split.json`
- Split CSV: `outputs/official_balanced_eval_split.csv`

## Composition

- Total selected labelled images: `387`
- Ordinary / row ore: `129`
- Fine / hard-to-process ore: `129`
- Talcose ore: `129`
- Unlabelled panoramas kept separately: `14`

## Labels

Labels come from official class folders, not per-pixel expert masks. They are suitable for image-level ore classification checks and threshold calibration, but not for direct segmentation accuracy.

## Recommended Use

- Use for balanced image-level F1/AUC and threshold calibration.
- Keep source folders/stems leakage-safe when creating train/validation/test variants.
- Keep panoramas separate for performance and visual stress tests unless they receive explicit labels.

## Caveats

- Class folders may contain near-duplicates or related fields of view.
- Talcose class labels do not provide dense talc masks for all images.
- Balanced sampling improves class fairness, but it does not replace expert validation.
