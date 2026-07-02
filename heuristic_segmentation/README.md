# Heuristic Segmentation Subproject

This is a separate non-neural baseline for the official OM-only ore-classifier
path. It does not train a model and does not change the SegFormer/ResUNet
pipeline. The goal is a fast, explainable segmentation candidate that can be
used as:

- an independent baseline for sulfide segmentation;
- a disagreement layer against neural predictions;
- a quick demo path when GPU checkpoints are unavailable;
- a source of component features for ordinary/fine intergrowth rules.

## Output Labels

The generated `class_mask.png` uses the official task-facing labels:

| id | label |
| ---: | --- |
| 0 | background_matrix |
| 1 | ordinary_intergrowth |
| 2 | fine_intergrowth |
| 3 | talc_candidate |

`talc_candidate` is intentionally named as a candidate signal. The stronger
talc path remains the blue-line conversion and QA workflow in the main v2
project.

## Run One Image

From the repository root:

```bash
python3 heuristic_segmentation/run_heuristic_segmentation.py \
  --image "dataset/Фото руд по сортам. ч1/Рядовые руды/DSCN2176.JPG" \
  --output-dir outputs/heuristic_segmentation_sample \
  --max-side 1600 \
  --overwrite
```

Artifacts:

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

Masks are emitted at the analysis size. Use `--max-side 0` only when the full
image fits comfortably in memory.

## Run A Folder

```bash
python3 heuristic_segmentation/run_heuristic_segmentation.py \
  --input-dir "dataset/Фото руд по сортам. ч1/Рядовые руды" \
  --output-dir outputs/heuristic_segmentation_row_smoke \
  --max-images 3 \
  --max-side 1200 \
  --overwrite
```

Folder mode writes per-sample artifacts under `samples/` plus `summary.csv` and
`batch_summary.json`.

## Method

1. Build an analyzed-area mask and suppress black borders plus blue annotation
   strokes.
2. Normalize illumination on the value channel.
3. Detect bright, non-green metallic regions with Otsu thresholding and
   morphology.
4. Split sulfide pixels into connected components.
5. Classify each component as ordinary or fine using area, solidity,
   compactness, and internal-dark replacement ratio.
6. Optionally mark green-gray non-sulfide regions as talc candidates.

This is a heuristic candidate, not expert geological ground truth. Its most
useful role is comparison: stable agreement with the neural mask is reassuring,
and disagreement is valuable QA signal.

## Tests

```bash
python3 -m unittest discover -s heuristic_segmentation/tests -p 'test_*.py'
```
