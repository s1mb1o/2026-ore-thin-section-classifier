# Heuristic Segmentation Subproject

Date: 2026-07-03

## Purpose

Add a separate non-neural segmentation path beside the SegFormer/ResUNet work.
This is not intended to replace the binary sulfide model. Its value is a fast,
explainable baseline and an independent disagreement source for QA.

## Location

```text
heuristic_segmentation/
  README.md
  run_heuristic_segmentation.py
  src/heuristic_segmentation/
  tests/
```

Generated artifacts stay under ignored `outputs/`.

## Current Algorithm

1. Build an analyzed-area mask and suppress black borders plus blue annotation
   strokes.
2. Normalize illumination on the HSV value channel.
3. Detect bright, non-green metallic regions with Otsu thresholding and
   morphology.
4. Split sulfide pixels into connected components.
5. Classify components as `ordinary_intergrowth` or `fine_intergrowth` using
   area, solidity, compactness, and internal-dark replacement ratio.
6. Optionally mark green-gray non-sulfide regions as `talc_candidate`.

The output `class_mask.png` uses:

| id | label |
| ---: | --- |
| 0 | background_matrix |
| 1 | ordinary_intergrowth |
| 2 | fine_intergrowth |
| 3 | talc_candidate |

`talc_candidate` is deliberately conservative wording. The stronger talc path
remains the blue-line conversion and QA workflow.

## Smoke Result

Command:

```bash
python3 heuristic_segmentation/run_heuristic_segmentation.py \
  --image "dataset/Фото руд по сортам. ч1/Рядовые руды/DSCN2176.JPG" \
  --output-dir outputs/heuristic_segmentation_smoke \
  --max-side 900 \
  --overwrite
```

Result:

```text
ore_class_candidate: fine_intergrowth_candidate
sulfide_fraction: 0.164864
talc_candidate_fraction: 0.000708
component_count: 70
ordinary_component_count: 9
fine_component_count: 61
```

Artifacts written:

```text
analysis_image.jpg
class_mask.png
sulfide_mask.png
talc_candidate_mask.png
analyzed_mask.png
overlay.png
components.csv
metrics.json
run_summary.json
batch_summary.json
```

The overlay was visually checked for a nonblank render and correct class colors.

## Known Limits

- The rule can over-classify ragged large sulfide regions as `fine_intergrowth`
  because it has no learned mineral texture context.
- Masks are emitted at analysis size by default; use `--max-side 0` only when
  full-resolution processing fits memory.
- Talc remains a candidate signal, not expert-confirmed talc.

## Next Use

Use this subproject as the heuristic source in the weak-supervision plan:

```text
neural mask vs heuristic mask -> agreement / disagreement layer -> QA priority
```
