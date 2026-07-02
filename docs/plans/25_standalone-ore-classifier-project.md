# Standalone Ore Classifier Project Plan

Date: 2026-07-02

## Purpose

Create a separate, narrow source-code project for the official `Скажи мне кто твой шлиф` task instead of trying to refactor the full multimodal QC assistant before the submission deadline.

The official judged path is now optical microscopy only:

- input: panoramic OM images of polished sections;
- output: three-class ore mask plus metrics and deterministic ore class;
- visible classes: ordinary sulfide intergrowths, fine sulfide intergrowths, talc;
- final rule: talc `> 10%` means talcose ore, otherwise ordinary-vs-fine predominance decides the ore type.

The existing repository remains valuable as a source of utilities, UI/report examples, and backup material, but the final P0 code should be easy to read, run, and explain without SEM/XRD/product-platform noise.

## Recommendation

Use a standalone source tree first:

```text
submissions/ore_classifier/
```

This is faster than creating a new repository immediately, keeps the code versioned in the current project, and makes later extraction to a clean public/private VCS repository straightforward.

If the submission needs a very clean VCS link, extract only this folder into a separate repository after the pipeline is runnable and documented.

## Why Not Rewire the Full Project

The current project already has broad capabilities: upload UI, OM/SEM/XRD paths, reports, batch mode, correction UI, robustness checks, and previous runner templates.

For this task, that breadth is also a risk:

- current code is centered on a phase/mineral ontology, not the official ore classes;
- many modules mention SEM, XRD, defects, generic QC, or LumenStone labels;
- adapting the whole UI and report stack can consume time without improving the judged core;
- the jury needs to see `image -> mask -> metrics -> ore class`, not a broad research platform.

A standalone P0 code path makes the official logic explicit and auditable.

## Required Output Contract

For one image:

```bash
python run_image.py \
  --image path/to/panorama.jpg \
  --output-dir outputs/sample_001 \
  --scale-um-per-pixel 0.5
```

Required artifacts:

```text
outputs/sample_001/
  mask.png
  overlay.png
  confidence_heatmap.png          # optional in first version
  metrics.csv
  ore_classification.json
  report.pdf
  run_summary.json
```

For a folder:

```bash
python run_batch.py \
  --input-dir data/official/panoramas \
  --output-dir outputs/batch_001 \
  --tile-size 2048 \
  --overlap 256
```

Batch artifacts:

```text
outputs/batch_001/
  summary.csv
  batch_summary.json
  index.html                      # optional lightweight report
  samples/<sample_id>/...
```

## Official Class Ontology

Use the task labels directly:

| id | label | RU label | color |
| ---: | --- | --- | --- |
| 0 | background_matrix | Матрица / фон | transparent or gray |
| 1 | ordinary_intergrowth | Обычные срастания | green |
| 2 | fine_intergrowth | Тонкие срастания | red |
| 3 | talc | Тальк | blue |

`background_matrix` is not the main scientific output. It is the residual analyzed area after sulfides, talc, and excluded artifacts are removed.

## Proposed File Layout

```text
submissions/ore_classifier/
  README.md
  requirements.txt
  run_image.py
  run_batch.py
  src/
    __init__.py
    config.py
    io.py
    preprocessing.py
    tiling.py
    sulfides.py
    talc.py
    intergrowths.py
    masks.py
    metrics.py
    ore_rules.py
    visualization.py
    report.py
    logging_utils.py
  tests/
    test_ore_rules.py
    test_metrics.py
    test_masks.py
  examples/
    README.md
```

## Pipeline

### 1. Image Loading

Implementation:

- support JPEG/PNG through Pillow or OpenCV;
- support TIFF through `tifffile` or Pillow, with a fallback note for very large pyramidal TIFFs;
- preserve original dimensions;
- compute SHA-256 of the input file;
- collect EXIF/DPI/scale metadata when available;
- create a browser-friendly preview if the image is too large.

Output:

- `ImageData` object with `rgb`, `width`, `height`, `metadata`, and `source_hash`.

### 2. Preprocessing

Implementation:

- detect the analyzed region and remove black borders/background;
- normalize illumination with large-kernel background subtraction or rolling-ball style correction;
- apply CLAHE to the luminance channel;
- optionally apply median/bilateral denoising;
- log every preprocessing setting.

Important constraint:

- do not downscale the final analysis image just to fit a model;
- downscale only previews and explicitly marked fallback runs.

### 3. Tiled Processing

Implementation:

- split large images into overlapping tiles;
- process each tile independently;
- stitch masks or probabilities back to the original coordinate system;
- record tile size, overlap, runtime, and device.

First version can stitch class masks by vote. If model probabilities are available, stitch logits/probabilities instead.

### 4. Sulfide Detection

P0 implementation:

- sulfides are the bright phases against darker silicate/oxide matrix;
- detect bright regions on the normalized image with adaptive thresholding;
- clean with morphology: remove small objects, close gaps, fill holes;
- split into connected components.

Output:

- `sulfide_mask`;
- connected-component table with area, bounding box, mean brightness, solidity, perimeter, and compactness.

Upgrade path:

- replace the threshold detector with SegFormer/ResUNet/YOLO while keeping the same `sulfide_mask` contract.

### 5. Intergrowth Classification

For each sulfide component:

```text
footprint = closed component or local convex hull
internal_dark = footprint - sulfide_pixels
replacement_ratio = area(internal_dark) / area(footprint)
```

Use features:

- component area;
- replacement ratio;
- solidity;
- compactness;
- boundary complexity;
- fragmentation/local component density.

P0 rule:

```text
large/compact component with low replacement_ratio -> ordinary_intergrowth
high replacement_ratio or fragmented component -> fine_intergrowth
```

Calibrate thresholds using the official class folders:

- `рядовые` / `Рядовые руды`;
- `тонкие` / `Труднообогатимые руды`.

Output:

- mask pixels for each sulfide component are assigned to `ordinary_intergrowth` or `fine_intergrowth`;
- per-component CSV can be saved as debug evidence.

### 6. Talc Detection

Talc should be searched inside non-ore matrix context:

```text
matrix_candidate = analyzed_mask - sulfide_mask - artifact_mask
```

P0 implementation:

- inspect `Области оталькования`;
- if annotations are blue lines, convert them with:
  - color threshold;
  - line cleanup;
  - contour closing;
  - region fill;
  - visual QA overlay;
- train or calibrate a dark-texture detector from these regions;
- detect dark dispersed matrix texture, not all dark pixels.

Features:

- local mean/standard deviation;
- dark contrast against nearby matrix;
- connected-component size distribution;
- LBP or simple texture filters if time allows.

Output:

- `talc_mask`;
- `talc_fraction`;
- warning if no usable talc annotation/calibration is available.

### 7. Final Mask Assembly

Priority order:

```text
artifact/excluded area -> ignored
talc -> class 3
ordinary sulfides -> class 1
fine sulfides -> class 2
remaining analyzed pixels -> background_matrix
```

If masks overlap, resolve deterministically and log overlap counts.

### 8. Quantitative Metrics

Use analyzed non-excluded pixels as the default denominator unless organizers clarify otherwise.

Compute:

```text
analysis_area_px
ordinary_area_px
fine_area_px
talc_area_px
total_sulfide_area_px = ordinary + fine

ordinary_fraction = ordinary / analysis_area
fine_fraction = fine / analysis_area
talc_fraction = talc / analysis_area
total_sulfide_fraction = total_sulfide / analysis_area

ordinary_share_among_sulfides = ordinary / total_sulfide
fine_share_among_sulfides = fine / total_sulfide
```

If scale is known:

```text
area_um2 = pixel_count * um_per_pixel_x * um_per_pixel_y
```

### 9. Ore Classification Rule

Implement deterministic logic:

```python
if talc_fraction > 0.10:
    ore_class = "talcose_ore"
elif ordinary_area_px >= fine_area_px:
    ore_class = "ordinary_ore"
else:
    ore_class = "hard_to_process_ore"
```

RU labels:

```text
talcose_ore -> оталькованная руда
ordinary_ore -> рядовая руда
hard_to_process_ore -> труднообогатимая руда
```

Boundary behavior:

- exactly `10%` talc is not talcose because the task says `> 10%`;
- ordinary/fine tie defaults to ordinary and emits a review warning;
- zero sulfides emits a warning and still reports talc fraction.

### 10. Visualization

Generate:

- `mask.png`: indexed class mask;
- `overlay.png`: source image plus transparent color mask;
- optional `confidence_heatmap.png`;
- optional tiled preview for large panoramas.

Colors:

```text
ordinary_intergrowth: green
fine_intergrowth: red
talc: blue
background_matrix: transparent/gray
```

### 11. CSV, JSON, PDF

`metrics.csv` columns:

```text
class_id,class_label,class_label_ru,pixels,area_um2,fraction_of_analysis_area,share_among_sulfides
```

`ore_classification.json`:

```json
{
  "ore_class": "talcose_ore",
  "ore_class_ru": "оталькованная руда",
  "talc_fraction": 0.14,
  "total_sulfide_fraction": 0.08,
  "ordinary_share_among_sulfides": 0.38,
  "fine_share_among_sulfides": 0.62,
  "conclusion_ru": "Руда классифицирована как оталькованная: содержание талька — 14.0%, преобладание тонких срастаний — 62.0%."
}
```

PDF should contain:

- image preview;
- overlay;
- fraction table;
- deterministic conclusion;
- run parameters;
- warnings and limitations.

### 12. Web Interface

P0 can be CLI-only if time is short.

If a web interface is needed, build a minimal Streamlit/Gradio app:

- upload image;
- optional scale input;
- run analysis;
- show overlay and table;
- download mask, overlay, CSV, JSON, PDF.

Do not port SEM/XRD/generic QC controls into this interface.

## Implementation Phases

### Phase 1. Runnable Skeleton

Deliver:

- folder structure;
- `run_image.py`;
- image loading;
- dummy or simple heuristic mask;
- overlay;
- metrics;
- ore rule;
- JSON/CSV output.

Acceptance:

- one command produces all core artifacts for a sample image.

### Phase 2. Official Heuristic Core

Deliver:

- preprocessing;
- sulfide bright-phase detection;
- connected-component intergrowth classification;
- talc detector stub or calibrated detector if annotations are ready;
- report PDF.

Acceptance:

- output mask uses all official classes where present;
- conclusion follows the task rule;
- `run_summary.json` records parameters.

### Phase 3. Official Data Calibration

Deliver:

- manifest for official data used by this standalone project;
- talc blue-line conversion;
- threshold calibration for ordinary/fine folders;
- validation split;
- first metrics: talc fraction error and image-level ordinary/fine/talcose F1.

Acceptance:

- measured metrics are saved, not claimed from intuition;
- failures and unsupported cases are visible in the report.

### Phase 4. Batch and Submission Package

Deliver:

- `run_batch.py`;
- batch `summary.csv`;
- README with exact commands;
- final example outputs;
- source zip contents list.

Acceptance:

- raw image folder to final artifacts in one command;
- all final demo/presentation screenshots come from this command.

### Phase 5. Optional Model Upgrade

Only after Phases 1-4 work:

- train/fine-tune a dense 4-class model from pseudo-labels or converted annotation masks;
- keep the heuristic pipeline as fallback and explanation baseline;
- record model license, weights hash, and validation metrics.

## Reuse From Existing Repository

Allowed reuse:

- report/overlay patterns from `experiments/qc_pipeline/create_sample_report.py`;
- tiled inference ideas from `experiments/qc_pipeline/run_segmentation_inference.py`;
- existing batch/report artifact conventions;
- project docs and official requirement sources.

Avoid direct dependencies on:

- SEM/XRD modules;
- generic phase/mineral label maps;
- upload UI-specific state;
- LLM narration;
- old runner variants unless the final submission format requires them.

## README Outline

The standalone README should answer:

1. What the project solves.
2. How to install.
3. How to run one image.
4. How to run a batch.
5. What files are produced.
6. What the four mask classes mean.
7. How ore class is computed.
8. What metrics are reported.
9. Known limitations.
10. License/provenance of models and data.

## Risks

| Risk | Mitigation |
| --- | --- |
| Heuristics are weaker than a trained model | Make the outputs interpretable, calibrate thresholds on official folders, keep dense model as optional upgrade. |
| Blue-line talc annotations are not directly fillable | Save QA overlays, allow manual correction for failed conversions, report talc metric only where ground truth is usable. |
| Large panoramas exceed memory | Tile processing; do not allocate full float32 logits for all classes unless needed. |
| Scale metadata is absent | Report pixel fractions and add a warning that absolute `um2` areas require scale. |
| Separate project duplicates some utilities | Accept duplication for deadline clarity; extract shared code only after submission. |

## Done Definition

This plan is complete when:

- `submissions/ore_classifier/README.md` exists;
- `run_image.py` and `run_batch.py` run from a clean checkout;
- one sample produces mask, overlay, metrics CSV, JSON conclusion, PDF, and run summary;
- batch mode produces `summary.csv`;
- the final presentation can point to this standalone path as the official solution core.

## Review (Claude Cross-Check, 2026-07-02)

Verdict: the standalone-tree strategy is endorsed. Given the 2026-07-04 deadline, a narrow auditable `image -> mask -> metrics -> ore class` tree beats rewiring the 13+ S1-hardcoded platform files, and the pipeline design (two-stage heuristic core, blue-line talc conversion, threshold calibration on class folders, deterministic rule with tie/10% semantics, gated model upgrade) is consistent with plans 23/24 and the captains-chat facts. No errors found in the ontology, rule, or metric definitions; the example JSON matches the official conclusion wording.

Gaps to close during implementation (ordered by severity):

1. Image decode limits are a Phase 1 blocker, not a footnote. Official panoramas average ~96 MB JPEG (~200-500 Mpx decoded at typical microscopy compression). Pillow's default `MAX_IMAGE_PIXELS` (~179 Mpx) raises decompression-bomb warnings/errors, and OpenCV's `imread` has its own pixel cap (~2^30, env `CV_IO_MAX_IMAGE_PIXELS`). The loader must deliberately raise limits, decode once, and hand bands to the tiler. The plan mentions pyramidal TIFFs but the actual official inputs are large JPEGs.
2. `artifact_mask` is consumed (§6 talc context, §7 assembly) but no stage produces it. Either add a minimal exclusion step (black borders, saturated glare, out-of-focus margin — heuristics can be borrowed from `detect_defect_candidates.py`) or remove it from the formulas until it exists; otherwise the denominator definition silently diverges between code and docs.
3. The Phase 3 calibration loop is underspecified. Class folders provide image-level labels only, so per-component thresholds must be fitted against image-level outcomes: predict each image's class via the expert rule applied to component stats, then grid-search thresholds maximizing image-level macro-F1 (ordinary/fine/talcose). State this explicitly so the metric matches what is optimized.
4. The 5-minute performance bar has no deliverable. Add runtime/memory measurement on a real official panorama to Phase 3/4 acceptance, and note the T4 (~16 GB) budget for the Phase 5 model path (one T4 per team, captains chat).
5. Detection vs CLAHE: run sulfide thresholding on illumination-corrected, non-CLAHE luminance; CLAHE amplifies matrix noise into bright false positives. Keep CLAHE for visualization and talc texture features only.
6. Footprint choice for `replacement_ratio`: prefer morphological closing over convex hull for elongated vein-like sulfides — a convex hull inflates the ratio for solid elongated grains and misclassifies them as fine intergrowths. Keep convex hull only for compact grains.
7. Web interface: do not build a third UI. Either stay CLI-only (the TZ marks the web UI as optional) or wire this engine into the existing portal as one new model family (`ore_classifier_official`) reusing upload/preview/PDF/correction — that preserves already-implemented TZ wishes (manual mask correction, expert-review mode, metadata entry) without porting SEM/XRD noise. Keep the dependency one-way: portal -> engine, never engine -> portal. This is a user decision because it changes the demo video script.
8. Blue-line pairing: first check whether `Области оталькования` files are annotated duplicates of clean parents (pair by filename stem). If yes, train on the clean parent plus the extracted mask; if the lines exist only on the analysis image, the planned line-pixel exclusion is mandatory so models do not learn the markup.
9. Minor: specify `ore_class` and `talc_fraction` columns in batch `summary.csv`; rename `src/io.py` to `image_io.py` to avoid confusion with the stdlib module; GeoJSON export stays P2 and can reuse `petroscope/analysis/geometry.py` if requested.

Relation to plan 24: phases map cleanly onto workstreams (Phase 1-2 = W1/W2/W3-core, Phase 3 = W3-calibration/W6, Phase 4 = W5/W7, Phase 5 = W3-upgrade); the standalone tree replaces W1/W5's "adapt existing report/UI" with "build clean, reuse patterns", which is accepted. `docs/specs/official-tz-solution-map.ru.md` rows that answer TZ wishes via the existing portal remain valid only if item 7's portal integration happens — otherwise their execution vehicle changes to this standalone tree and the CLI/README.
