# Commands

Run from the repository root:

```bash
cd /Volumes/T7_2TB/Projects-T7_2TB/2026_Nornikel_Hackaton_v2
```

## Setup Python Tools

If `streamlit: command not found`, use the existing temporary venv from this
workspace session:

```bash
/private/tmp/nornikel_ore_classifier_streamlit_venv/bin/python -m streamlit run apps/talc_review_streamlit.py -- \
  --conversion-dir outputs/talc_blue_line_conversion
```

For a persistent repo-local UI venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ui.txt
```

After activation, use `python -m streamlit ...` in the UI commands below.

For the full ML pipeline, recreate `.venv` with Python >=3.10. If `python -V`
prints `3.9.x`, remove and recreate it:

```bash
deactivate 2>/dev/null || true
rm -rf .venv
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate
python -V
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Ore Pipeline UI

Default local demo/smoke mode uses the heuristic backend, creates immutable run
artifacts under `outputs/ore_pipeline_ui/`, and prints the selected local URL:

```bash
python3 apps/ore_pipeline_web.py \
  --host 127.0.0.1 \
  --port 0
```

Use the ML sulfide checkpoint instead of the heuristic backend when the full ML
environment is active:

```bash
python3 apps/ore_pipeline_web.py \
  --backend ml \
  --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt \
  --host 127.0.0.1 \
  --port 0
```

## Ore Pipeline UI Docker Runtime

This Docker path is for running the browser GUI on the Nornickel VM. It defaults
to the heuristic backend and does not copy the dataset, generated outputs, or
model checkpoints into the image.

```bash
docker compose -f docker-compose.ore-pipeline-ui.yml up --build
```

On the organizer VM, use `sudo docker compose ...` if the `team123` user still
does not have Docker socket access:

```bash
sudo docker compose -f docker-compose.ore-pipeline-ui.yml up --build
```

Open:

```text
http://<vm-host>:8080/workspace
```

Use a different host port if `8080` is already occupied:

```bash
ORE_UI_PUBLIC_PORT=18080 docker compose -f docker-compose.ore-pipeline-ui.yml up --build
```

The persistent UI workspace is bind-mounted at `./outputs/ore_pipeline_ui`.
If a checkpoint-capable derived image is used later, mount the existing
`./models` directory and set:

```bash
ORE_UI_BACKEND=ml \
ORE_UI_CHECKPOINT=/app/models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt \
docker compose -f docker-compose.ore-pipeline-ui.yml up --build
```

## Talc Review UI (Preferred Browser/Canvas App)

Directly from the official annotated folder:

```bash
python3 apps/talc_review_web.py \
  --annotated-dir "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
  --workspace-dir outputs/talc_blue_line_conversion \
  --host 127.0.0.1 \
  --port 0
```

Prepared conversion workspace mode:

```bash
python3 apps/talc_review_web.py \
  --conversion-dir outputs/talc_blue_line_conversion \
  --host 127.0.0.1 \
  --port 0
```

The app prints the selected local URL. It edits `current_talc_mask.png` directly
and saves final reviewed artifacts under each sample's `reviewed/` directory.

## Legacy Streamlit Talc Review UI

```bash
python -m streamlit run apps/talc_review_streamlit.py -- \
  --conversion-dir outputs/talc_blue_line_conversion
```

## Convert Talc Blue Lines

```bash
python3 scripts/convert_talc_blue_lines.py \
  --input "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
  --output-dir outputs/talc_blue_line_conversion \
  --summary-json outputs/talc_blue_line_conversion_summary.json
```

With sulfide and silicon/silicate support masks:

```bash
python3 scripts/convert_talc_blue_lines.py \
  --input "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
  --output-dir outputs/talc_blue_line_conversion \
  --sulfide-mask-dir path/to/binary_sulfide_masks \
  --silicate-mask-dir path/to/silicate_support_masks
```

## Run Ore Pipeline

```bash
python3 scripts/run_ore_pipeline.py \
  --image "dataset/Фото руд по сортам. ч1/Рядовые руды/DSCN2176.JPG" \
  --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt \
  --out-dir outputs/inference_demo/local_dscn2176_b2 \
  --device auto \
  --auto-talc-candidate
```

Use an accepted talc mask instead of the automatic candidate when available:

```bash
python3 scripts/run_ore_pipeline.py \
  --image "dataset/Фото руд по сортам. ч1/Оталькованные руды/example.JPG" \
  --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt \
  --out-dir outputs/inference_demo/example_with_accepted_talc \
  --device auto \
  --talc-mask outputs/talc_blue_line_conversion/example/final_talc_mask.png
```

## Official Batch Metrics

Audit duplicate-content label conflicts and build the preferred deconflicted
balanced split:

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

```bash
python3 scripts/run_official_batch.py \
  --split-json outputs/official_balanced_eval_split_deconflicted.json \
  --dataset-root dataset \
  --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt \
  --out-dir outputs/evaluations/b2_official_deconflicted_auto_talc_analyzed \
  --tile-size 1024 \
  --stride 768 \
  --batch-size 1 \
  --device auto \
  --overwrite
```

```bash
python3 scripts/evaluate_ore_classification.py \
  --summary-csv outputs/evaluations/b2_official_deconflicted_auto_talc_analyzed/summary.csv \
  --out-json outputs/evaluations/b2_official_deconflicted_auto_talc_analyzed/ore_classification_metrics.json \
  --out-md outputs/evaluations/b2_official_deconflicted_auto_talc_analyzed/ore_classification_metrics.md
```

Cross-validate a tabular image-level ore classifier from the same batch outputs:

```bash
python3 scripts/evaluate_ore_feature_classifier.py \
  --summary-csv outputs/evaluations/b2_official_deconflicted_auto_talc_analyzed/summary.csv \
  --out-json outputs/evaluations/b2_official_deconflicted_auto_talc_analyzed/ore_feature_classifier_cv.json \
  --out-md outputs/evaluations/b2_official_deconflicted_auto_talc_analyzed/ore_feature_classifier_cv.md
```

Merge a class-sharded official batch before evaluation:

```bash
python3 scripts/merge_official_batch_shards.py \
  --shard-dirs \
    outputs/evaluations/b2_official_deconflicted_auto_talc_analyzed_sharded_20260703_1000/fine_intergrowth \
    outputs/evaluations/b2_official_deconflicted_auto_talc_analyzed_sharded_20260703_1000/ordinary_intergrowth \
    outputs/evaluations/b2_official_deconflicted_auto_talc_analyzed_sharded_20260703_1000/talcose \
  --out-dir outputs/evaluations/b2_official_deconflicted_auto_talc_analyzed_sharded_20260703_1000
```

Calibrate deterministic talc/ordinary/fine rule thresholds from the completed
batch without rerunning B2 inference:

```bash
python3 scripts/calibrate_ore_rules.py \
  --summary-csv outputs/evaluations/b2_official_deconflicted_auto_talc_analyzed/summary.csv \
  --out-json outputs/evaluations/b2_official_deconflicted_auto_talc_analyzed/ore_rule_calibration.json \
  --out-md outputs/evaluations/b2_official_deconflicted_auto_talc_analyzed/ore_rule_calibration.md
```

Apply a calibration artifact to a rerun or a single-image demo:

```bash
python3 scripts/run_official_batch.py \
  --split-json outputs/official_balanced_eval_split_deconflicted.json \
  --dataset-root dataset \
  --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt \
  --out-dir outputs/evaluations/b2_official_deconflicted_auto_talc_calibrated \
  --rule-config-json outputs/evaluations/b2_official_deconflicted_auto_talc_analyzed/ore_rule_calibration.json \
  --tile-size 1024 \
  --stride 768 \
  --batch-size 1 \
  --device auto \
  --overwrite
```

```bash
python3 scripts/run_ore_pipeline.py \
  --image "dataset/Фото руд по сортам. ч1/Рядовые руды/DSCN2176.JPG" \
  --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt \
  --out-dir outputs/inference_demo/local_dscn2176_b2_calibrated \
  --rule-config-json outputs/evaluations/b2_official_deconflicted_auto_talc_analyzed/ore_rule_calibration.json \
  --device auto \
  --auto-talc-candidate
```

## Sulfide QA UI

```bash
python -m streamlit run apps/sulfide_qa_streamlit.py -- \
  --runs-dir outputs/inference_demo \
  --review-dir outputs/sulfide_qa_reviews
```

## Manual Review Pack

```bash
python3 scripts/prepare_manual_review_pack.py \
  --out-dir outputs/manual_review/local_review_pack \
  --per-label 1 \
  --device auto \
  --overwrite
```

## Heuristic Baseline

```bash
python3 heuristic_segmentation/run_heuristic_segmentation.py \
  --image "dataset/Фото руд по сортам. ч1/Рядовые руды/DSCN2176.JPG" \
  --output-dir outputs/heuristic_segmentation_sample \
  --max-side 1600 \
  --overwrite
```

## Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m unittest discover -s heuristic_segmentation/tests -p 'test_*.py'
```
