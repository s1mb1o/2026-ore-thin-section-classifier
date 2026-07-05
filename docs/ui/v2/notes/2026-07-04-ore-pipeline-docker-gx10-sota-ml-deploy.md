# GX10 SOTA ML Docker Deployment — Ore Pipeline UI

Date: 2026-07-04

## Target

- Host: `ashmelev@192.168.86.14`
- URL: `http://192.168.86.14:8210/workspace`
- Container: `nornikel-ore-pipeline-ui-v2`
- Image: `nornikel/ore-pipeline-ui:v2-gx10-ml`
- Workspace mount: `~/nornikel-ore-pipeline-ui/outputs/ore_pipeline_ui:/data/ore_pipeline_ui`

## Deployed Models

- Binary sulfide segmentation: `/app/models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt`
- Talc segmentation: `/app/outputs/talc_segformer_folds/segformer_b0_full_20260703/fold_00/segformer_b0/best.pt`
- Talc threshold: `0.50`
- Grade branch: `/app/models/grade_classifier/effb3_ordfine_ppaug_20260704/best.pt`

The model mounts are read-only:

```bash
-v "$REPO/models:/app/models:ro"
-v "$REPO/outputs/talc_segformer_folds:/app/outputs/talc_segformer_folds:ro"
```

## Deployment Command

The deployed container was started with:

```bash
docker run -d \
  --name nornikel-ore-pipeline-ui-v2 \
  --restart unless-stopped \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --pull=never \
  -p 8210:8080 \
  -e ORE_UI_HOST=0.0.0.0 \
  -e ORE_UI_PORT=8080 \
  -e ORE_UI_WORKSPACE=/data/ore_pipeline_ui \
  -e ORE_UI_BACKEND=ml \
  -e ORE_UI_CHECKPOINT=/app/models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt \
  -e ORE_UI_TALC_BACKEND=ml \
  -e ORE_UI_TALC_CHECKPOINT=/app/outputs/talc_segformer_folds/segformer_b0_full_20260703/fold_00/segformer_b0/best.pt \
  -e ORE_UI_TALC_THRESHOLD=0.50 \
  -e ORE_UI_GRADE_CHECKPOINT=/app/models/grade_classifier/effb3_ordfine_ppaug_20260704/best.pt \
  -e ORE_UI_PROCESSING_MAX_SIDE=2600 \
  -e ORE_UI_PANORAMA_MAX_SIDE=1800 \
  -e ORE_UI_PREVIEW_MAX_SIDES=1024,2048,4096 \
  -v "$RUNTIME/outputs/ore_pipeline_ui:/data/ore_pipeline_ui" \
  -v "$REPO/models:/app/models:ro" \
  -v "$REPO/outputs/talc_segformer_folds:/app/outputs/talc_segformer_folds:ro" \
  nornikel/ore-pipeline-ui:v2-gx10-ml
```

## Verification

Image/dependency smoke:

- `docker image inspect nornikel/ore-pipeline-ui:v2-gx10-ml`: `arch=arm64`, `os=linux`, `id=sha256:3406de0bdbc3a3a7c3e528b53f0fbe19315bb188d96d4d73903fb7c0c7e0ad7b`
- Image import smoke: `torch 2.10.0a0+b558c986e8.nv25.11`, `torchvision 0.25.0a0+7a13ad0f`, `transformers 5.13.0`, `cv2 4.13.0`, `cuda True NVIDIA GB10`

Service smoke:

- `GET /workspace`: `200` in `0.053817` seconds.
- `GET /api/status`: `backend=ml`, B2 checkpoint exists, `talc_backend=ml`, B0 talc checkpoint exists, `gpu.available=true`, device `NVIDIA GB10`.
- Health status is `warning` only because gx10 flash free space is about `8.4%`.

Runtime test:

- `POST /api/runtime/test`: `ok=true`, `status=ok`, total `8.996` seconds.
- Binary sulfide model loaded on `cuda`: `segformer_b2`, `27,348,162` parameters, best weak-label sulfide IoU `0.9743806926422606`.
- Talc model loaded on `cuda`: `segformer_b0`, `3,714,658` parameters, task `binary_talc_non_sulfide`.

Functional ML smoke:

- Sample: `dataset/Фото руд по сортам. ч1/Рядовые руды/DSCN2176.JPG`
- Completed run: `run_20260704_110152_278228522_afef7b24`
- Runtime elapsed: `17.143` seconds, `6/6` ML tiles processed.
- Final rule output: `hard_to_process_ore` / `труднообогатимая руда`
- Fractions: `sulfide_fraction=0.3486527660682404`, `talc_fraction=0.00024047601335713814`
- Grade branch executed with checkpoint `effb3_ordfine_ppaug_20260704/best.pt`: predicted `ordinary_intergrowth` / `рядовая руда`, confidence `0.9997366070747375`.

## 2026-07-05 Redeploy

The gx10 backup was rebuilt from the current v2 tree and redeployed through the
repo-root Compose GPU profile.

- Host preflight: gx10 reported `115Gi` RAM available, `222G` root free space,
  low load, and no heavy training process on the GB10.
- Staged build tree: `gx10:~/Projects/nornikel-v2-ore-pipeline-ui-build`,
  with `.git`, virtualenvs, `dataset/`, `data/external/`, `outputs/`, model
  weights, archives, and `presentation/videos/` excluded.
- Docker build context: `1.89MB`.
- New image: `nornikel/ore-pipeline-ui:v2-gx10-ml`
  `sha256:4e08a307651453d86f33c10e4e4ae3a9fe3614181016fc6c3c28ea932bcbcc9d`,
  created `2026-07-05T04:58:41+03:00`, size about `20.0GB`.
- Container: `nornikel-ore-pipeline-ui-v2`, id `5a9c800fc126`, published as
  `0.0.0.0:8210->8080/tcp`.
- The previous container had been created outside the current Compose project,
  so it was removed by exact name before `docker compose --profile gpu up -d
  --no-build ore-pipeline-ui-gpu`.

Verification:

- Direct `GET http://192.168.86.14:8210/workspace`: `200` in `0.075449s`.
- Public unauthenticated
  `GET https://nornickel-ai-hackathon.my.3simbio.ru/workspace`: `401`.
- Public authenticated `/workspace`: `200` in `0.074314s`.
- Direct and public authenticated `/api/status`: `health.overall=ok`,
  `app.version=v2`, `app.backend=ml`, `app.talc_backend=ml`, all mounted
  checkpoint paths present, GPU `NVIDIA GB10`, `history.runs_total=6`.
- `POST /api/runtime/test`: `ok=true`, `status=ok`, total `6.653s`, device
  `cuda`; loaded SegFormer-B2 sulfide and SegFormer-B0 talc checkpoints.

Public functional smoke:

- Sample: `dataset/Фото руд по сортам. ч2/тонкие/69 1.jpg`.
- Upload: `20260705_020001_794549784_e169c1215e`.
- Completed run: `run_20260705_020001_861893916_b8b9425d`.
- Result: `status=complete`, `progress=100`, `backend=ml`,
  `talc_backend=ml`, `ore_class=hard_to_process_ore`,
  `sulfide_fraction=0.33442079670457303`,
  `talc_cluster_fraction=0.0`.
- Files endpoint returned `47` artifacts and an artifacts ZIP link.

## Caveats

- gx10 port `8210` is LAN-only; public access uses the protected MikroTik Caddy
  route `https://nornickel-ai-hackathon.my.3simbio.ru/workspace`.
- During the original 2026-07-04 deployment, a separate isolated class-folder
  evaluation was running in `tmux codex_e2e_20260704_1333`; the 2026-07-05
  redeploy preflight showed no heavy gx10 training process.
- The grade branch is an auxiliary ordinary/fine opinion. The final UI verdict still comes from the current segmentation/rule path unless the application logic is changed.
