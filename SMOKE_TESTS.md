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
- Each sample directory contains source image copy, blue stroke masks, talc candidates, sulfide/overlap/ignore masks, final talc mask, QA overlay, and `conversion_summary.json`.
- Streamlit review shows original blue annotation lines explicitly, uses the stateful `Editor` segmented control, defaults canvas editing to the current mask, exposes filled polygon/box area edits, keeps stroke width only for pen/eraser, shows `Move/resize`, includes the `Geometry` component for polygon vertex drag/add/delete and box corner/edge drag, includes coordinate fallback editors, updates local edit metrics after apply, and saves reviewed outputs under each sample's `reviewed/` directory.
- The `SAM2` editor shows model/device controls and `Load/check SAM2`; without local `torch` and `sam2`, it should report missing optional dependencies rather than blocking the rest of the app.

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

## Planned Pipeline Checks

- Streamlit QA app opens the pseudo-label manifest and saves a JSON patch.
