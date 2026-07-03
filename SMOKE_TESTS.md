# Smoke Tests

## Current Structural Checks

Run from the v2 root:

```bash
test -L dataset
python3 - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("dataset/_download_manifest.json").read_text())
assert manifest["download_status"] == "complete"
assert manifest["file_count"] == 1236
assert manifest["local_verified_count"] == 1236
assert manifest["local_verified_size_bytes"] == 3018194503
print("dataset manifest ok")
PY
```

## Talc Blue-Line Converter Smoke

Run from the v2 root:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

```bash
python3 scripts/convert_talc_blue_lines.py \
  --input "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
  --output-dir outputs/talc_blue_line_conversion \
  --summary-json outputs/talc_blue_line_conversion_summary.json
```

Optional Streamlit review after installing `requirements.txt`:

```bash
streamlit run apps/talc_review_streamlit.py -- \
  --conversion-dir outputs/talc_blue_line_conversion
```

Expected:

- Unit tests pass.
- `outputs/talc_blue_line_conversion/manifest.json` exists with `42` samples.
- Current status counts are `31` `candidate_ok`, `9` `needs_manual_review`, and `2` `sulfide_overlap_review_required`.
- Each sample directory contains source image copy, blue stroke masks, talc candidates, sulfide/overlap/ignore masks, optional-silicate derived masks, final talc mask, QA overlay, and `conversion_summary.json`.
- The talc converter accepts `--silicate-mask-dir` and writes `silicate_support_mask.png`, `silicate_supported_talc_mask.png`, `silicate_unsupported_talc_mask.png`, `talc_positive_core_mask.png`, and `silicate_hard_negative_mask.png`; synthetic unit tests verify supported talc, unsupported uncertain talc, and silicate hard negatives.
- Streamlit review shows original blue annotation lines explicitly, uses the stateful `Workspace` segmented control, defaults `Review canvas` to the current mask, exposes `Brush`, `Erase`, `Filled polygon`, `Filled box`, and `SAM2 assist`, keeps stroke width only for brush/erase, supports polygon vertex drag/add/delete and box corner/edge drag in the filled-area tools, keeps coordinate fallback editors under `Advanced`, updates local edit metrics after apply, and saves reviewed outputs under each sample's `reviewed/` directory.
- The `SAM2 assist` canvas tool shows model/device controls, `Load/check SAM2`, draggable point/box prompts, and `Run SAM2`; without local `torch` and `sam2`, it should report missing optional dependencies rather than blocking the rest of the app.

## Heuristic Segmentation Smoke

Run from the v2 root:

```bash
python3 -m unittest discover -s heuristic_segmentation/tests -p 'test_*.py' -v
```

```bash
python3 heuristic_segmentation/run_heuristic_segmentation.py \
  --image "dataset/Фото руд по сортам. ч1/Рядовые руды/DSCN2176.JPG" \
  --output-dir outputs/heuristic_segmentation_smoke \
  --max-side 900 \
  --overwrite
```

Expected:

- Unit tests pass.
- The CLI writes `class_mask.png`, `sulfide_mask.png`, `talc_candidate_mask.png`, `analyzed_mask.png`, `overlay.png`, `components.csv`, `metrics.json`, `run_summary.json`, and `batch_summary.json`.
- Smoke metrics currently include `sulfide_fraction 0.164864`, `talc_candidate_fraction 0.000708`, and `70` sulfide components for `DSCN2176.JPG` at `--max-side 900`.
- The overlay is nonblank and displays ordinary intergrowth in green, fine intergrowth in red/orange, and talc candidates in blue.

## Reusable Demo Libraries Smoke

Run from the v2 root:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Expected:

- Unit tests pass; current local result is `31` tests.
- Coverage includes `source_fusion`, `review_queue`, `curation`, `component_reports`, `report_cards`, and `scribble_classifier`.
- These tests use synthetic inputs and do not require GPU, Streamlit, SAM2, or external datasets.

## Binary Sulfide Training Smoke

Run from the v2 root:

```bash
python3 scripts/build_official_manifest.py \
  --dataset-root dataset \
  --out outputs/commit_smoke_official_manifest.json
```

```bash
python3 scripts/build_binary_sulfide_dataset.py \
  --out-dir outputs/commit_smoke_binary_sulfide_dataset \
  --tile-size 128 \
  --stride 128 \
  --max-lumenstone-images 1 \
  --max-official-images-per-label 1 \
  --max-tiles-per-source 2 \
  --max-total-tiles 12 \
  --downscale-max-side 512 \
  --overwrite
```

```bash
python3 scripts/train_binary_sulfide.py \
  --dataset-manifest outputs/commit_smoke_binary_sulfide_dataset/manifest.json \
  --model resunet \
  --out-dir outputs/commit_smoke_train_resunet \
  --epochs 1 \
  --batch-size 2 \
  --num-workers 0 \
  --base-channels 8 \
  --device cpu \
  --max-steps-per-epoch 1
```

```bash
python3 scripts/train_binary_sulfide.py \
  --dataset-manifest outputs/commit_smoke_binary_sulfide_dataset/manifest.json \
  --model segformer_b0 \
  --pretrained-model random \
  --allow-random-init \
  --out-dir outputs/commit_smoke_train_segformer_b0 \
  --epochs 1 \
  --batch-size 1 \
  --num-workers 0 \
  --device cpu \
  --max-steps-per-epoch 1
```

Expected:

- Official manifest command reports `1236` images.
- Binary sulfide smoke dataset writes a manifest and at least one train and one val tile.
- ResUNet and SegFormer smoke commands each write `train_log.csv`, `last.pt`, and `best.pt`.

## Binary Sulfide Evaluation And Pipeline Smoke

Run from the v2 root after the binary smoke commands above:

```bash
python3 scripts/evaluate_binary_sulfide.py \
  --dataset-manifest outputs/commit_smoke_binary_sulfide_dataset/manifest.json \
  --checkpoint outputs/commit_smoke_train_resunet/best.pt \
  --split val \
  --batch-size 2 \
  --num-workers 0 \
  --hausdorff-max-items 4 \
  --out-json outputs/commit_smoke_train_resunet/eval_metrics.json
```

```bash
python3 scripts/run_ore_pipeline.py \
  --image outputs/commit_smoke_binary_sulfide_dataset/tiles/train/images/official_heuristic_014709cbad4a_384_0.jpg \
  --checkpoint outputs/commit_smoke_train_resunet/best.pt \
  --out-dir outputs/commit_smoke_ore_pipeline \
  --tile-size 64 \
  --stride 32 \
  --batch-size 2 \
  --device cpu \
  --preview-max-side 256
```

```bash
python3 scripts/build_official_balanced_eval_split.py \
  --official-manifest outputs/official_manifest.json \
  --out-json outputs/official_balanced_eval_split.json \
  --out-csv outputs/official_balanced_eval_split.csv
```

Expected:

- Evaluation JSON includes `iou_sulfide`, `f1_sulfide`, `auc_sulfide`, `hausdorff_px_mean`, and `hd95_px_mean`.
- The pipeline writes `binary_sulfide/sulfide_mask.png`, `binary_sulfide/confidence.png`, `binary_sulfide/overlay_preview.jpg`, `ore_analysis/ore_summary.json`, `ore_analysis/component_features.csv`, and `ore_analysis/intergrowth_overlay_preview.jpg`.
- Balanced official split contains `387` labelled images: `129` ordinary, `129` fine, `129` talcose; the `14` panoramas remain listed separately as unlabelled stress-test images.

## Manual Review Pack Smoke

Run from the v2 root after a compatible B2 checkpoint is available:

```bash
python3 scripts/prepare_manual_review_pack.py \
  --per-label 1 \
  --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt \
  --out-dir outputs/manual_review/smoke_b2_review_pack \
  --device auto \
  --batch-size 1 \
  --overwrite
```

Expected:

- `review_manifest.csv`, `review_manifest.json`, `feedback_template.csv`, and `review_candidates.csv` exist.
- Each `runs/*` directory contains `pipeline_summary.json`, `review_panel.jpg`, `source_preview.jpg`, `binary_sulfide/confidence_heatmap.jpg`, sulfide mask/overlay, and ordinary/fine overlay.
- The generated pack can be opened with:

```bash
streamlit run apps/sulfide_qa_streamlit.py -- \
  --runs-dir outputs/manual_review/smoke_b2_review_pack/runs \
  --review-dir outputs/manual_review/smoke_b2_review_pack/reviews
```

Note: on macOS, local `transformers` may not load the zelda-trained SegFormer-B2 checkpoint because of module namespace drift. In that case, run the same command on zelda and rsync the generated pack back.

## Planned Pipeline Checks

- Streamlit QA app opens the pseudo-label manifest and saves a JSON patch.
