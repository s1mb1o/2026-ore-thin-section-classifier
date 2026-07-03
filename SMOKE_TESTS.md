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

## Ore Pipeline UI Smoke

Run from the v2 root:

```bash
python3 -m py_compile apps/ore_pipeline_web.py
python3 -m unittest discover -s tests -p 'test_ore_pipeline_web.py' -v
```

```bash
python3 apps/ore_pipeline_web.py \
  --host 127.0.0.1 \
  --port 0
```

Expected:

- Focused tests pass for upload, preprocessing, complete run artifacts, metrics CSV, PDF report, sulfide edit rerun, final segmentation edit recalculation, history, and required UI controls.
- The app prints a local URL such as `http://127.0.0.1:<port>/`.
- `/` redirects to `/workspace`; `/workspace` and `/history` can be opened directly, tab clicks update the URL, and browser back/forward restores the matching page.
- The `Theme` selector supports `System`, `Light`, and `Dark`; explicit dark/light choices update panels, controls, toolbar, modal, history rows, and viewer backgrounds and persist across reloads.
- The `Language` selector defaults to `Русский`; the Russian UI labels are visible on first load, and switching to English updates controls, preprocessing labels, result/history summaries, metrics, and editor labels without clearing the current run/history state.
- The drop zone accepts click-to-upload and drag/drop for PNG, JPEG, TIFF, and RAW-extension files.
- Dropping or selecting an unsupported file extension shows an inline warning in the upload panel and does not start an upload; supported extensions remain PNG, JPEG, TIFF, and RAW-camera variants.
- Large uploads show progress in the input panel: byte-transfer progress during upload, then a preview-preparation phase while the server decodes and scales display pyramids; manual preprocessing preview updates also show preparation progress.
- After image selection, the drop zone shows a thumbnail, filename, dimensions, and an `×` clear button; clearing resets upload/run/result/progress/view state and returns to `Waiting for image`.
- Preview supports primary `original`, `preprocessed`, `sulfide`, and `final` layers with pan/zoom. `preprocessed`, `sulfide`, and `final` stay greyed/disabled until the corresponding artifacts are ready.
- The `show tiling` checkbox is available when the current upload/run has a tile manifest and overlays the tile grid used for scaled/large-image analysis on top of original, preprocessed, sulfide, final, and side-by-side views.
- Side-by-side is a separate `none/preprocessed/sulfide/final` selector; choosing a ready comparison layer shows it on the right side with a draggable left/right splitter.
- Preprocessing controls include `нормализация освещения`, `шумоподавление`, `коррекция контраста`, and `масштабирование для панорамных снимков`; all four are enabled by default and the user's chosen preset persists across reloads and upload clears.
- `Start` creates an immutable run under `outputs/ore_pipeline_ui/runs/` with original input, preprocess preset, preprocessed image, masks, metrics, and report artifacts.
- While a run is queued/running, the `Start` control is replaced by a red `Stop` button. Pressing `Stop` disables that button, shows stopping status, cancels the active run through `/api/runs/{run_id}/cancel`, and leaves the run in terminal `canceled` state without enabling result editing.
- Result view shows class toggles for background, ordinary intergrowth, fine intergrowth, and talc.
- `Fix me` opens the editor; sulfide/non-sulfide and final segmentation are selected with tabs, the top toolbar contains Brush/Pan, Undo/Redo, Zoom in/out, Fit view, and Brush size, Brush left-draws and right-erases, and the right panel shows live pixel/% statistics for sulfide, non-sulfide, ordinary intergrowth, fine intergrowth, and talc.
- The edit dialog refreshes the selected run before loading masks so image and segmentation layers appear even after a server restart or stale browser state; missing prerequisites show a localized editor error rather than a blank canvas.
- `Fix and Restart` creates a new derived run instead of mutating the parent.
- The History page shows a table with thumbnail, filename, date, ore classification, sulfides, non-sulfides, ordinary intergrowth, fine intergrowth, talc, and Actions; clicking a thumbnail opens a preview popup, `Load` restores the original upload/preprocessing preset for tuning, and `Remove` deletes the selected completed/failed run artifact from history.

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

Preferred browser/canvas review:

```bash
python3 apps/talc_review_web.py \
  --conversion-dir outputs/talc_blue_line_conversion \
  --host 127.0.0.1 \
  --port 0
```

Legacy Streamlit review after installing `requirements-ui.txt`:

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
- Preferred browser review shows original blue annotation lines explicitly, edits the current talc mask directly, supports brush left-draw/right-erase, direct editable polygon/rectangle regions, optional SAM2 prompts, undo, autosave, default-on sulfide protection, manual sulfide subtraction, top-right `Save` and `Save & Next`, and saves reviewed outputs under each sample's `reviewed/` directory. Polygon/rectangle regions stay editable while the current image is open and are flattened on save. The legacy Streamlit review remains available only as a fallback.
- The `SAM2 assist` canvas tool shows model/device controls, `Load/check SAM2`, draggable point/box prompts, and `Run SAM2`; without local `torch` and `sam2`, it should report missing optional dependencies rather than blocking the rest of the app.

## Talc Browser Review App Smoke

Run from the v2 root:

```bash
python3 -m py_compile apps/talc_review_web.py
python3 -m unittest discover -s tests -p 'test_talc_review_web.py' -v
```

```bash
python3 apps/talc_review_web.py \
  --conversion-dir outputs/talc_blue_line_conversion \
  --host 127.0.0.1 \
  --port 0
```

Expected:

- Focused tests pass and cover same-filename annotated/original pairing, first-open `current_talc_mask.png` creation, reviewed save artifacts, reset behavior, and basic HTTP endpoints.
- The app prints a local URL such as `http://127.0.0.1:<port>/`.
- `/api/manifest` reports `42` samples for the full conversion workspace.
- The `Theme` selector supports `System`, `Light`, and `Dark`; explicit dark/light choices update the UI immediately and persist across reloads.
- Opening a sample auto-creates or reuses `current_talc_mask.png`.
- The top toolbar is ordered `Brush`, `Rectangle`, `Polygon`, `SAM2`, `Undo`, `Zoom In`, `Zoom Out`, `Fit`, followed by active-tool parameters, with `Save` and `Save & Next` pinned at the top right; Brush shows brush size, SAM2 shows prompt mode and `Check SAM2`.
- Mouse wheel over the canvas zooms in/out without changing mask geometry.
- The canvas edits the talc mask itself with brush left-draw/right-erase, polygon, rectangle, and optional SAM2 prompt tools.
- `Base image` includes `Sulfide mask (sulfide/non-sulfide mask segmentation)` and selecting it shows the binary sulfide mask as the canvas background while talc overlays remain editable.
- In Brush mode, left mouse draws talc and right mouse erases without opening the browser context menu.
- In Brush mode, hovering over the image shows a circle matching the current brush draw/erase area; changing brush size updates the circle.
- Polygon drawing supports click-to-add points and closes by clicking the first point; right-click on a polygon point removes it, while right-click elsewhere cancels the current draft without opening the browser context menu.
- Completed polygon regions fill immediately, stay editable while the current image is open, and support point drag, edge-click-to-insert, whole-shape move, and live mask updates.
- Rectangle drawing fills on drag or on the second corner click; right-click cancels the current draft, and completed rectangle regions stay editable while the current image is open with corner/edge drag, whole-shape move, and live mask updates.
- Pressing Delete/Backspace removes the selected completed polygon or rectangle, while text fields keep normal delete behavior.
- No `Apply Polygon`, `Cancel Polygon`, `Apply Rectangle`, or `Cancel Rectangle` controls are shown; `Save` flattens live polygon/rectangle regions into the reviewed mask.
- `Protect sulfides while drawing` is enabled by default; add tools cannot add new talc pixels on the sulfide mask, `Current on sulfide px` is shown for existing overlap, and `Subtract sulfides from mask` removes overlap and autosaves.
- `Save` writes `reviewed/reviewed_talc_mask.png`, `reviewed/reviewed_ignore_mask.png`, `reviewed/reviewed_overlay.png`, `reviewed/review_patch.json`, and `reviewed/review_summary.json`.
- `SAM2 box` is the default SAM2 canvas prompt; point mode remains available. SAM2 box/polygon prompt results are clipped to the prompt bounds, and SAM2 masks covering more than half the image are rejected with a clear status message instead of filling the whole canvas.
- In SAM2 mode, hovering over the image shows a dashed orange preview of the proposed box/point area; dragging a SAM2 box keeps the dashed preview visible.
- In `SAM2 point` mode, keeping the cursor still over the image for about two seconds requests a SAM2 mask as an orange preview overlay without changing the talc mask; `Apply SAM2` applies that preview, or runs and applies the current hover point if no preview is ready yet.
- If SAM2 dependencies are missing, `Check SAM2` and the SAM2 canvas tool report the missing optional dependency without blocking manual editing.

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

- Unit tests pass; current local result is `55` tests.
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
  --out-dir outputs/commit_smoke_ore_pipeline_auto_talc \
  --tile-size 64 \
  --stride 32 \
  --batch-size 2 \
  --device cpu \
  --preview-max-side 256 \
  --auto-talc-candidate
```

```bash
python3 scripts/run_official_batch.py \
  --split-json outputs/official_balanced_eval_split.json \
  --dataset-root dataset \
  --checkpoint outputs/commit_smoke_train_resunet/best.pt \
  --out-dir outputs/commit_smoke_official_batch \
  --max-total 1 \
  --tile-size 512 \
  --stride 512 \
  --batch-size 2 \
  --device cpu \
  --preview-max-side 256 \
  --overwrite
```

```bash
python3 scripts/evaluate_ore_classification.py \
  --summary-csv outputs/commit_smoke_official_batch/summary.csv \
  --out-json outputs/commit_smoke_official_batch/ore_classification_metrics.json \
  --out-md outputs/commit_smoke_official_batch/ore_classification_metrics.md
```

```bash
python3 scripts/build_official_balanced_eval_split.py \
  --official-manifest outputs/official_manifest.json \
  --out-json outputs/official_balanced_eval_split.json \
  --out-csv outputs/official_balanced_eval_split.csv
```

Preferred deconflicted split:

```bash
python3 scripts/audit_official_labels.py \
  --official-manifest outputs/official_manifest.json \
  --dataset-root dataset \
  --out-dir outputs/official_label_audit

python3 scripts/build_official_balanced_eval_split.py \
  --official-manifest outputs/official_manifest.json \
  --label-audit-json outputs/official_label_audit/summary.json \
  --exclude-conflicts \
  --dedupe-sha256 \
  --out-json outputs/official_balanced_eval_split_deconflicted.json \
  --out-csv outputs/official_balanced_eval_split_deconflicted.csv
```

Expected:

- Evaluation JSON includes `iou_sulfide`, `f1_sulfide`, `auc_sulfide`, `hausdorff_px_mean`, and `hd95_px_mean`.
- The pipeline writes `binary_sulfide/sulfide_mask.png`, `binary_sulfide/confidence.png`, `binary_sulfide/analyzed_mask.png`, `binary_sulfide/overlay_preview.jpg`, `talc_candidate/talc_candidate_mask.png`, `talc_candidate/talc_candidate_summary.json`, `ore_analysis/ore_summary.json`, `ore_analysis/component_features.csv`, `ore_analysis/analyzed_mask.png`, and `ore_analysis/intergrowth_overlay_preview.jpg`.
- `ore_summary.json` reports analyzed-denominator `sulfide_fraction` / `talc_fraction`, full-image `*_fraction_image`, `analyzed_fraction`, `talc_margin`, `intergrowth_margin`, `needs_expert_review`, and `warnings`.
- Raw balanced official split contains `387` labelled images: `129` ordinary, `129` fine, `129` talcose. Preferred deconflicted split contains `345` labelled images: `115` per class after excluding label-conflict hashes and duplicate hashes. The `14` panoramas remain listed separately as unlabelled stress-test images.
- The one-image official batch smoke writes `summary.csv`, `summary.json`, `failures.json`, and image-level classification metrics JSON/Markdown. On a one-class smoke sample, AUC may be `null`; full balanced evaluation is needed for meaningful F1/AUC.
- `scripts/merge_official_batch_shards.py` combines class-sharded `run_official_batch.py` outputs, rejects duplicate `run_id` values, and writes combined `summary.csv`, `summary.json`, and `failures.json`.
- `scripts/evaluate_ore_feature_classifier.py` cross-validates image-level classifiers from `summary.csv` plus per-run `component_features.csv`; use the full balanced split, not the one-image smoke, for meaningful F1/AUC.

## Ore Rule Calibration Smoke

Run from the v2 root after an official batch has written `summary.csv` and
per-run `ore_analysis/component_features.csv` files:

```bash
python3 scripts/calibrate_ore_rules.py \
  --summary-csv outputs/commit_smoke_official_batch/summary.csv \
  --out-json outputs/commit_smoke_official_batch/ore_rule_calibration.json \
  --out-md outputs/commit_smoke_official_batch/ore_rule_calibration.md
```

Expected:

- The command writes `ore_rule_calibration.json` and optional Markdown.
- Output includes `best_config`, `best_metrics`, `top_results`, and an explicit note that calibration uses image-level labels, not pixel-level geological ground truth.
- `scripts/analyze_ore_from_masks.py`, `scripts/run_ore_pipeline.py`, and `scripts/run_official_batch.py` accept `--rule-config-json` with either this calibration artifact or a direct config object; explicit CLI flags still override the file.
- On one-class smoke data, AUC may be `null`; use the full balanced batch for meaningful F1/AUC calibration.

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
