# Nornickel Hackathon v2: Official Ore Classifier

Clean workspace for the official `Скажи мне, кто твой шлиф` task.

The goal is a narrow optical-microscopy pipeline:

```text
panorama image
-> binary sulfide segmentation
-> component features
-> ordinary_intergrowth / fine_intergrowth classification
-> talc detection
-> official ore-class rule and report artifacts
```

This v2 directory intentionally avoids the old broad QC assistant surface. The old repository remains the source for archived plans, prior experiments, and reusable snippets, but new P0 implementation should live here.

## Layout

```text
AGENTS.md / CLAUDE.md
ChangeLog.md
ResearchLog.md
SMOKE_TESTS.md
docs/
  official/   # official task page copy
  plans/      # selected implementation plans
  specs/      # official requirement mapping
  notes/      # selected source/research notes
apps/         # Streamlit QA tools
scripts/      # dataset and training utilities
src/ore_classifier/
heuristic_segmentation/  # separate non-neural segmentation baseline
outputs/      # generated artifacts, ignored by git
models/       # local pointers/config only; HF cache stays outside repo
dataset -> ../2026_Nornikel_Hackaton/dataset
```

## Current Data Source

`dataset` is a relative symlink to the verified dataset in the original project:

```text
../2026_Nornikel_Hackaton/dataset
```

The source manifest in the old repository verified `1236/1236` files and about `3.0 GB` of official data. Keep the symlink unless there is a concrete reason to copy the dataset.

## Core Docs

- `docs/plans/25_standalone-ore-classifier-project.md`
- `docs/plans/26_weak-supervision-sulfide-binary-model.md`
- `docs/notes/talc-blue-line-conversion.md`
- `docs/specs/official-tz-solution-map.ru.md`
- `docs/official/Скажи мне кто твой шлиф.md`
- `docs/cards/binary-sulfide-model-card.md`
- `docs/cards/official-balanced-eval-dataset-card.md`
- `docs/cards/demo-run-fact-sheet.md`
- `SMOKE_TESTS.md`

## Implemented Blocks

### Talc Blue-Line Conversion And QA

The talc annotation path is implemented in the v2 layout:

```bash
python3 scripts/convert_talc_blue_lines.py \
  --input "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
  --output-dir outputs/talc_blue_line_conversion \
  --summary-json outputs/talc_blue_line_conversion_summary.json
```

Review UI:

```bash
streamlit run apps/talc_review_streamlit.py -- \
  --conversion-dir outputs/talc_blue_line_conversion
```

The current full run contains `42` samples with status counts:
`31 candidate_ok`, `9 needs_manual_review`, and
`2 sulfide_overlap_review_required`.

### Heuristic Segmentation Baseline

The separate `heuristic_segmentation/` subproject provides a non-neural
baseline for sulfide/intergrowth segmentation and disagreement analysis:

```bash
python3 heuristic_segmentation/run_heuristic_segmentation.py \
  --image "dataset/Фото руд по сортам. ч1/Рядовые руды/DSCN2176.JPG" \
  --output-dir outputs/heuristic_segmentation_smoke \
  --max-side 900 \
  --overwrite
```

It writes a four-label `class_mask.png`, binary sulfide/talc-candidate masks,
an overlay, component CSV, and JSON metrics. Treat `talc_candidate` and the
ordinary/fine decision as heuristic QA signals, not expert ground truth.

### Neural Binary Sulfide Pipeline

Build a balanced official image-level evaluation split:

```bash
python3 scripts/build_official_balanced_eval_split.py \
  --official-manifest outputs/official_manifest.json \
  --out-json outputs/official_balanced_eval_split.json \
  --out-csv outputs/official_balanced_eval_split.csv
```

Current split: `129` ordinary, `129` fine, `129` talcose images; panoramas are
kept separately as `14` unlabelled stress/performance images.

Evaluate a binary sulfide checkpoint with organizer-relevant segmentation
metrics:

```bash
python3 scripts/evaluate_binary_sulfide.py \
  --dataset-manifest outputs/binary_sulfide_dataset_v0/manifest.json \
  --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt \
  --split val \
  --batch-size 16 \
  --hausdorff-max-items 512 \
  --out-json outputs/evaluations/segformer_b0_best_eval_metrics.json
```

Run one image through the current end-to-end path:

```bash
python3 scripts/run_ore_pipeline.py \
  --image "dataset/Фото руд по сортам. ч1/Рядовые руды/2539589-1.JPG" \
  --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt \
  --out-dir outputs/demo_ore_pipeline \
  --tile-size 1024 \
  --stride 768 \
  --batch-size 4 \
  --auto-talc-candidate
```

The pipeline writes:

- binary sulfide mask;
- confidence heatmap;
- sulfide overlay preview;
- automatic talc candidate mask/overlay/summary, or a provided `--talc-mask`;
- component ordinary/fine CSV;
- intergrowth overlay preview;
- deterministic ore summary JSON.

Run the full image-level balanced split and compute organizer-facing
classification metrics:

```bash
python3 scripts/run_official_batch.py \
  --split-json outputs/official_balanced_eval_split.json \
  --dataset-root dataset \
  --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt \
  --out-dir outputs/evaluations/b2_official_balanced_auto_talc \
  --tile-size 1024 \
  --stride 768 \
  --batch-size 1 \
  --device auto \
  --overwrite
```

```bash
python3 scripts/evaluate_ore_classification.py \
  --summary-csv outputs/evaluations/b2_official_balanced_auto_talc/summary.csv \
  --out-json outputs/evaluations/b2_official_balanced_auto_talc/ore_classification_metrics.json \
  --out-md outputs/evaluations/b2_official_balanced_auto_talc/ore_classification_metrics.md
```

## Next Implementation Steps

1. Run the B2 official balanced batch and inspect `ore_classification_metrics.md`.
2. Calibrate component-level ordinary/fine thresholds against the balanced labelled split.
3. Compare SegFormer-B2/B1/B0 and heuristic segmentation outputs to build the first disagreement queue if time permits.
4. Use accepted talc masks from `outputs/talc_blue_line_conversion` via `--talc-mask` when stronger talc claims are needed; otherwise keep `--auto-talc-candidate` framed as a conservative candidate.
