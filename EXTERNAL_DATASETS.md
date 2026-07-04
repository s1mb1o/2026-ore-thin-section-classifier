# External Datasets

Date: 2026-07-03

This file lists the external data needed to reproduce the current v2 pipeline. Raw datasets are intentionally not committed to this repository.

## Quick Setup

Minimum for running inference, talc review, and official-image evaluation:

1. Download or copy the official Nornickel hackathon package into `dataset/`.
2. Keep `dataset/` as either a local directory or a symlink. The current local setup uses a local copy:

```bash
test -d dataset
```

Extra for rebuilding the binary sulfide training dataset or retraining checkpoints:

1. Download LumenStone `S1_v1` and `S2_v2`.
2. Pass their extracted roots explicitly to `scripts/build_binary_sulfide_dataset.py` with `--lumenstone-root`.

## Required: Official Nornickel Task Package

Purpose:

- Primary target-domain data for the official `Скажи мне, кто твой шлиф` task.
- Used by almost every pipeline command through `dataset/`.
- Provides image-level ore class folders, unlabelled panoramas, and blue-line talc annotation images.

Source:

- Official task data URL: `https://disk.yandex.ru/d/Fo5eIM984glHaA`
- Local source note: [docs/notes/2026-07-02-domain-datasets-search.md](docs/notes/2026-07-02-domain-datasets-search.md)

Expected local path:

```text
dataset/
```

Current verified local status:

```text
download_status: complete
file_count: 1236
local_verified_count: 1236
local_verified_size_bytes: 3018194503
```

Important subfolders:

```text
dataset/Панорамы/
dataset/Фото руд по сортам. ч1/Оталькованные руды/
dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования/
dataset/Фото руд по сортам. ч1/Рядовые руды/
dataset/Фото руд по сортам. ч1/Труднообогатимые руды/
dataset/Фото руд по сортам. ч2/оталькованные/
dataset/Фото руд по сортам. ч2/рядовые/
dataset/Фото руд по сортам. ч2/тонкие/
```

Verification after download:

```bash
python3 scripts/build_official_manifest.py \
  --dataset-root dataset \
  --out outputs/official_manifest.json

python3 scripts/build_official_balanced_eval_split.py \
  --official-manifest outputs/official_manifest.json \
  --out-json outputs/official_balanced_eval_split.json \
  --out-csv outputs/official_balanced_eval_split.csv
```

If the copied package includes `dataset/_download_manifest.json`, this quick check should pass:

```bash
python3 - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("dataset/_download_manifest.json").read_text())
assert manifest["download_status"] == "complete"
assert manifest["file_count"] == 1236
assert manifest["local_verified_count"] == 1236
assert manifest["local_verified_size_bytes"] == 3018194503
print("official dataset manifest ok")
PY
```

License/provenance note:

- Treat this as hackathon-provided task data. Do not redistribute it publicly unless the organizer rules explicitly allow that.

## Required For Retraining: LumenStone S1_v1 and S2_v2

Purpose:

- Pixel-mask proxy supervision for the binary `sulfide / not_sulfide` model.
- `scripts/build_binary_sulfide_dataset.py` combines LumenStone masks with heuristic pseudo-labels from the official dataset.
- The current `binary_sulfide_dataset_v0` benchmark used LumenStone plus official-image pseudo masks.

Source page:

- `https://imaging.cs.msu.ru/en/research/geology/lumenstone`

Archives currently used by the project:

| Subset | Download URL | Archive | Size bytes | MD5 |
| --- | --- | --- | ---: | --- |
| `S1_v1` | `https://disk.yandex.ru/d/aiWh3rBEwdJ2_g` | `S1_v1.zip` | `534897733` | `4da00e8dc59ea5e840967e4a9fd736f8` |
| `S2_v2` | `https://disk.yandex.ru/d/wYK_5JyQy0pIcg` | `S2_v2.zip` | `418742024` | `c739c4fb57f684a7e01b9608c80d1635` |

Expected extracted roots:

```text
S1_v1/S1_v1/
  imgs/train/
  imgs/test/
  masks/train/
  masks/test/

S2_v2/S2_v2/
  imgs/train/
  imgs/test/
  masks/train/
  masks/test/
```

Current local reference paths:

```text
data/external/lumenstone/full/S1_v1/S1_v1
data/external/lumenstone/full/S2_v2/S2_v2
```

Current local counts:

| Subset | Train images | Train masks | Test images | Test masks |
| --- | ---: | ---: | ---: | ---: |
| `S1_v1` | `59` | `59` | `16` | `16` |
| `S2_v2` | `37` | `37` | `12` | `12` |

Build command with explicit roots:

```bash
python3 scripts/build_binary_sulfide_dataset.py \
  --official-root dataset \
  --lumenstone-root data/external/lumenstone/full/S1_v1/S1_v1 \
  --lumenstone-root data/external/lumenstone/full/S2_v2/S2_v2 \
  --out-dir outputs/binary_sulfide_dataset_v0 \
  --tile-size 512 \
  --stride 384 \
  --overwrite
```

Notes:

- The script defaults to the v2-local `data/external/lumenstone/full/...` paths. Passing `--lumenstone-root` remains clearer for fresh clones or alternative storage.
- The helper downloader is available as `scripts/download_lumenstone_all.sh`; it can fetch the full public LumenStone mirror, but current training defaults use only `S1_v1` and `S2_v2`.
- LumenStone labels are mineral classes, not official `ordinary_intergrowth`, `fine_intergrowth`, or `talc` labels.
- The current sulfide mapping uses class ids for sulfide minerals and excludes background, magnetite, hematite, and native gold. See `src/ore_classifier/pseudo_labels.py`.
- Do not commit or bundle raw LumenStone data without checking its data-use and citation requirements.

## Not Needed For The Current V2 Run

These datasets are documented research candidates, but they are not required to run the current v2 pipeline and should not be downloaded by default:

| Dataset | Source | Current status |
| --- | --- | --- |
| Cu RLM/SEM ore-resin dataset | `https://zenodo.org/records/5020566` | Research/support candidate only; not used by current scripts. |
| FeM iron ore RLM/SEM dataset | `https://zenodo.org/records/5014700` | Research/support candidate only; not used by current scripts. |
| Iron Ore Image Data | `https://data.mendeley.com/datasets/6hp82tsb8g/2` | Research/support candidate only; not used by current scripts. |
| USGS Wet Mountains thin sections | `https://www.usgs.gov/data/thin-section-images-automated-mineralogy-scans-lithogeochemistry-and-nd-sr-pb-isotopic` | Research/support candidate only; not used by current scripts. |
| SEM/materials datasets such as cigRockSEM, Ni-WC, OD_MetalDAM | See `docs/notes/2026-07-02-training-datasets-and-models-search.md` | Out of current OM-only scope. |

## Dataset Hygiene

- Keep raw downloads out of git. `dataset`, `outputs/*`, and `models/*` are ignored intentionally.
- Prefer symlinks to local shared dataset folders when multiple clones need the same data.
- Record any new dataset actually used for training or evaluation in this file before teammates depend on it.
