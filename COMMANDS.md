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

The quickest launch is the repo-root wrapper `run_main_app.sh`. It prefers the
repo-local `.venv`, `cd`s to the repo root, defaults to the heuristic backend on
an OS-assigned port, and passes any extra arguments straight through:

```bash
./run_main_app.sh                     # heuristic backend, OS-assigned port
./run_main_app.sh --port 8080         # fixed port
./run_main_app.sh --backend ml        # ML sulfide backend (B2 checkpoint default)
./run_main_app.sh --help              # app help
```

Host/port can also be set via the `ORE_HOST` / `ORE_PORT` environment variables.

The equivalent explicit invocation uses the heuristic backend, creates immutable
run artifacts under `outputs/ore_pipeline_ui/`, and prints the selected local URL:

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

The quickest launch is the repo-root wrapper `run_talc_app.sh`. It prefers the
repo-local `.venv`, `cd`s to the repo root, defaults to the prepared
`outputs/talc_blue_line_conversion` workspace on an OS-assigned port, and passes
extra arguments through. It skips its default `--conversion-dir` when the caller
supplies `--conversion-dir` or `--annotated-dir`:

```bash
./run_talc_app.sh                     # prepared-workspace mode, OS-assigned port
./run_talc_app.sh --port 8081         # fixed port
./run_talc_app.sh --reconvert         # regenerate the conversion workspace
./run_talc_app.sh --annotated-dir "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования"
./run_talc_app.sh --help              # app help
```

Host/port can also be set via the `TALC_HOST` / `TALC_PORT` environment
variables, and the default workspace via `TALC_CONVERSION_DIR`.

The equivalent explicit invocations follow. Directly from the official annotated
folder:

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

## Train Non-Sulfide Talc Segmentation

Build a talc/not-talc dataset from reviewed masks. `sulfide_mask.png` pixels are
ignored by default, so the model learns only on non-sulfide pixels:

```bash
python3 scripts/build_talc_dataset.py \
  --conversion-dir outputs/talc_blue_line_conversion \
  --clean-image-dir "dataset/Фото руд по сортам. ч1/Оталькованные руды" \
  --out-dir outputs/talc_non_sulfide_dataset_v0 \
  --overwrite
```

Train the local baseline:

```bash
python3 scripts/train_talc_segmentation.py \
  --dataset-manifest outputs/talc_non_sulfide_dataset_v0/manifest.json \
  --out-dir models/talc_segmentation/resunet_non_sulfide_20260703_local \
  --model resunet \
  --base-channels 16 \
  --epochs 3 \
  --batch-size 4 \
  --num-workers 0 \
  --device auto \
  --max-steps-per-epoch 80
```

Run a pretrained SegFormer fold smoke with threshold calibration:

```bash
python3 scripts/run_talc_segformer_folds.py \
  --conversion-dir outputs/talc_blue_line_conversion \
  --clean-image-dir "dataset/Фото руд по сортам. ч1/Оталькованные руды" \
  --out-dir outputs/talc_segformer_folds/segformer_b0_smoke_20260703 \
  --model segformer_b0 \
  --folds 2 \
  --folds-to-run 0 \
  --tile-size 384 \
  --stride 288 \
  --max-tiles-per-source 12 \
  --epochs 1 \
  --batch-size 1 \
  --calibration-batch-size 1 \
  --num-workers 0 \
  --lr 0.00006 \
  --device auto \
  --max-steps-per-epoch 10 \
  --thresholds 0.30,0.40,0.50,0.60,0.70 \
  --overwrite
```

Run the full SegFormer-B0 5-fold talc evaluation on a CUDA host:

```bash
python scripts/run_talc_segformer_folds.py \
  --conversion-dir outputs/talc_blue_line_conversion \
  --clean-image-dir "dataset/Фото руд по сортам. ч1/Оталькованные руды" \
  --out-dir outputs/talc_segformer_folds/segformer_b0_full_20260703 \
  --model segformer_b0 \
  --folds 5 \
  --folds-to-run all \
  --tile-size 384 \
  --stride 288 \
  --max-tiles-per-source 36 \
  --epochs 20 \
  --batch-size 8 \
  --calibration-batch-size 8 \
  --num-workers 4 \
  --lr 0.00006 \
  --weight-decay 0.0001 \
  --device cuda \
  --amp \
  --thresholds 0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80 \
  --seed 20260703 \
  --overwrite
```

Segment talc on one reviewed sample, clipping final output to non-sulfide
pixels:

```bash
python3 scripts/infer_talc_segmentation.py \
  --image outputs/talc_blue_line_conversion/samples/DSCN4714/DSCN4714.JPG \
  --sulfide-mask outputs/talc_blue_line_conversion/samples/DSCN4714/sulfide_mask.png \
  --checkpoint models/talc_segmentation/resunet_non_sulfide_20260703_local/best.pt \
  --out-dir outputs/talc_segmentation_predictions/resunet_non_sulfide_20260703_local_DSCN4714 \
  --tile-size 384 \
  --stride 288 \
  --batch-size 4 \
  --device auto
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
