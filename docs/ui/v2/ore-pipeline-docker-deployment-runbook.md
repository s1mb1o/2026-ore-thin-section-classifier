# Ore Pipeline Docker Deployment Runbook

Date: 2026-07-04

This runbook deploys the v2 ore pipeline browser UI (`apps/ore_pipeline_web.py`) as a Docker service on verified targets:

- Alize: `root@111.88.124.80`, public URL `https://nornickel-ai-hackathon.alola.ru/workspace`, `linux/amd64`, current public production target behind Caddy.
- Nornickel VM: `team123@111.88.145.15`, public URL `http://111.88.145.15:8080/workspace`, `linux/amd64`.
- gx10: `ashmelev@192.168.86.14`, LAN URL `http://192.168.86.14:8210/workspace`, `linux/arm64`.

The root `compose.yaml` is the primary deployment file. Its default service is heuristic-first and does not bundle the official dataset, generated outputs, or model checkpoints. The opt-in `gpu` profile targets gx10/ML deployments with a verified image based on `nvcr.io/nvidia/pytorch:25.11-py3` for the current SOTA checkpoints. `docker-compose.ore-pipeline-ui.yml` remains as a compatibility file for older scripted commands.

## Preflight

Run from the canonical v2 checkout on the Mac:

```bash
cd /Volumes/T7_2TB/Projects-T7_2TB/2026_Nornikel_Hackaton_v2
git diff --check
python3 -m unittest discover -s tests -p 'test_ore_pipeline_docker.py' -v
docker compose config >/tmp/nornikel-ore-pipeline-ui.compose.yaml
```

Check that the Docker context stays small:

```bash
docker buildx build --dry-run -f docker/ore-pipeline-ui/Dockerfile . 2>/dev/null || true
```

If local Docker is unavailable, skip the dry-run and rely on the remote build
context check. `.dockerignore` excludes `.git`, `dataset`, `outputs`, model
checkpoints, and common archives. The Alize ML build context was about
`386 MB` because the repo still contains presentation/static assets; this is
acceptable for the current VM, but future production syncs should keep large
media out of the remote build tree.

## Alize Public Production

Use this path for the reviewer-facing public endpoint
`https://nornickel-ai-hackathon.alola.ru/`. Caddy and wildcard TLS already live
on Alize, enforce reviewer Basic Auth, and proxy to private backend
`127.0.0.1:8765`. The same backend is also available over plain HTTP by IP at
`http://111.88.124.80/` for reviewers who use the host IP directly. Prefer the
HTTPS hostname when entering credentials; Basic Auth over plain HTTP is not
encrypted on the network.

Deployment evidence is recorded in
`docs/ui/v2/notes/2026-07-04-alize-v2-production-deploy.md`.

### 1. Sync v2 Runtime Tree

```bash
REPO=/Volumes/T7_2TB/Projects-T7_2TB/2026_Nornikel_Hackaton_v2
ALIZE=root@111.88.124.80
REMOTE=/opt/nornickel-ai-hackathon-v2
SSH_OPTS="-o UserKnownHostsFile=/tmp/alize_known_hosts -o StrictHostKeyChecking=yes"

cd "$REPO"
rsync -az --delete --delete-excluded \
  -e "ssh $SSH_OPTS" \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.mypy_cache/' \
  --exclude='dataset/' \
  --exclude='data/' \
  --exclude='presentation/videos/' \
  --exclude='outputs/' \
  ./ "$ALIZE:$REMOTE/"

ssh $SSH_OPTS "$ALIZE" \
  "mkdir -p '$REMOTE/outputs/talc_segformer_folds/segformer_b0_full_20260703/fold_00/segformer_b0'"

rsync -az --delete \
  -e "ssh $SSH_OPTS" \
  "$REPO/outputs/talc_segformer_folds/segformer_b0_full_20260703/fold_00/segformer_b0/" \
  "$ALIZE:$REMOTE/outputs/talc_segformer_folds/segformer_b0_full_20260703/fold_00/segformer_b0/"
```

The Docker image does not bake model checkpoints. The production run mounts:

```text
$REMOTE/models:/app/models:ro
$REMOTE/outputs/talc_segformer_folds:/app/outputs/talc_segformer_folds:ro
$REMOTE/runtime/outputs/ore_pipeline_ui:/data/ore_pipeline_ui
```

### 2. Build ML Image On Alize

```bash
ssh $SSH_OPTS "$ALIZE" "
  set -e
  cd '$REMOTE'
  DOCKER_BUILDKIT=0 docker build \
    -f docker/ore-pipeline-ui/Dockerfile.gx10-ml \
    -t nornickel-ore-pipeline-ui:v2-ml .
"
```

Verified production image:

```text
nornickel-ore-pipeline-ui:v2-ml
linux/amd64
sha256:f72410291546f2250f0a7608070312703cadc82b60d8a319960f714325076118
```

### 3. Run On Alize

```bash
ssh $SSH_OPTS "$ALIZE" '
set -e
REMOTE=/opt/nornickel-ai-hackathon-v2
CONTAINER=nornickel-ore-pipeline-ui-v2
BINARY=/app/models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt
TALC=/app/outputs/talc_segformer_folds/segformer_b0_full_20260703/fold_00/segformer_b0/best.pt
GRADE=/app/models/grade_classifier/effb3_ordfine_ppaug_20260704/best.pt
mkdir -p "$REMOTE/runtime/outputs/ore_pipeline_ui"
docker rm -f nornickel-qc-ui "$CONTAINER" 2>/dev/null || true
docker run -d \
  --name "$CONTAINER" \
  --restart unless-stopped \
  --gpus all \
  --ipc=host \
  -p 127.0.0.1:8765:8080 \
  -e ORE_UI_HOST=0.0.0.0 \
  -e ORE_UI_PORT=8080 \
  -e ORE_UI_WORKSPACE=/data/ore_pipeline_ui \
  -e ORE_UI_BACKEND=ml \
  -e ORE_UI_CHECKPOINT="$BINARY" \
  -e ORE_UI_TALC_BACKEND=ml \
  -e ORE_UI_TALC_CHECKPOINT="$TALC" \
  -e ORE_UI_TALC_THRESHOLD=0.50 \
  -e ORE_UI_GRADE_CHECKPOINT="$GRADE" \
  -e ORE_UI_PROCESSING_MAX_SIDE=2600 \
  -e ORE_UI_PANORAMA_MAX_SIDE=1800 \
  -e ORE_UI_PREVIEW_MAX_SIDES=1024,2048,4096 \
  -v "$REMOTE/runtime/outputs/ore_pipeline_ui:/data/ore_pipeline_ui" \
  -v "$REMOTE/models:/app/models:ro" \
  -v "$REMOTE/outputs/talc_segformer_folds:/app/outputs/talc_segformer_folds:ro" \
  nornickel-ore-pipeline-ui:v2-ml
docker ps --filter name="$CONTAINER" --format "{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
'
```

### 4. Smoke Alize

```bash
BASE=https://nornickel-ai-hackathon.alola.ru
ALIZE_REVIEWER_PASS="$(security find-generic-password -a reviewer -s nornickel-ai-hackathon-basic-auth -w)"
curl -sS -o /dev/null -w '%{http_code}\n' "$BASE/workspace"  # 401
curl -fsS -u "reviewer:$ALIZE_REVIEWER_PASS" "$BASE/workspace" -o /tmp/alize-workspace.html
curl -fsS -u "reviewer:$ALIZE_REVIEWER_PASS" "$BASE/api/status" | jq '{overall:.health.overall, backend:.app.backend, talc_backend:.app.talc_backend, gpu:.gpu.devices[0].name}'
curl -fsS -u "reviewer:$ALIZE_REVIEWER_PASS" -H 'Content-Type: application/json' -d '{}' "$BASE/api/runtime/test" | jq '{ok,status,backend,talc_backend,device:.details.device}'
curl -sS -o /dev/null -w '%{http_code}\n' http://111.88.124.80/  # 401
curl -fsS -u "reviewer:$ALIZE_REVIEWER_PASS" -L http://111.88.124.80/ -o /tmp/alize-ip-workspace.html
curl -fsS -u "reviewer:$ALIZE_REVIEWER_PASS" http://111.88.124.80/api/status | jq '{overall:.health.overall, backend:.app.backend, gpu:.gpu.devices[0].name}'
```

Functional smoke:

```bash
BASE=https://nornickel-ai-hackathon.alola.ru
ALIZE_REVIEWER_PASS="$(security find-generic-password -a reviewer -s nornickel-ai-hackathon-basic-auth -w)"
SAMPLE='dataset/Фото руд по сортам. ч2/тонкие/69 1.jpg'
curl -fsS -u "reviewer:$ALIZE_REVIEWER_PASS" -F "file=@${SAMPLE}" "$BASE/api/uploads" -o /tmp/alize-upload.json
UPLOAD_ID="$(jq -r '.upload_id' /tmp/alize-upload.json)"
curl -fsS -u "reviewer:$ALIZE_REVIEWER_PASS" -H 'Content-Type: application/json' \
  -d "{\"upload_id\":\"$UPLOAD_ID\",\"panorama_scaling\":false}" \
  "$BASE/api/runs/start" -o /tmp/alize-run-start.json
RUN_ID="$(jq -r '.run_id' /tmp/alize-run-start.json)"
for _ in $(seq 1 180); do
  curl -fsS -u "reviewer:$ALIZE_REVIEWER_PASS" "$BASE/api/runs/$RUN_ID" -o /tmp/alize-run.json
  STATUS="$(jq -r '.status' /tmp/alize-run.json)"
  [ "$STATUS" = complete ] || [ "$STATUS" = failed ] || [ "$STATUS" = canceled ] && break
  sleep 2
done
jq '{run_id:.run_id,status:.status,progress:.progress}' /tmp/alize-run.json
curl -fsS -u "reviewer:$ALIZE_REVIEWER_PASS" "$BASE/api/runs/$RUN_ID/files" | jq '.files | length'
```

Expected current Alize details:

- Unauthenticated hostname and IP requests return `401`.
- Authenticated `/workspace` returns `200` with the v2 HTML app.
- Authenticated `http://111.88.124.80/` redirects to `/workspace`, and
  authenticated `http://111.88.124.80/workspace` returns `200`.
- Authenticated `/api/status` returns `health.overall=ok`, backend `ml`, talc
  backend `ml`, GPU `NVIDIA L4`, and mounted checkpoint files present.
- Authenticated `/api/runtime/test` returns `ok=true` and loads SegFormer-B2
  sulfide plus SegFormer-B0 talc on `cuda`.
- Direct external `http://111.88.124.80:8765/` remains closed; only Caddy is
  public.

## Nornickel VM

Do not run this path while the 2026-07-04 organizer stop request is active. It is retained only for recovery or an explicit authorized redeploy. Build on `docker-srv` (`root@192.168.86.16`) and stream the built image into the VM, so the VM does not need to build the image itself.

### 1. Prepare SSH Access

Keep the `team123` private key outside the repository. If using the provided key archive:

```bash
KEYDIR="$(mktemp -d /tmp/team123-key.XXXXXX)"
unzip -q /Users/ashmelev/Downloads/team123.zip -d "$KEYDIR"
find "$KEYDIR" -type f -print
```

Set `KEY` to the extracted private key path, not the `.pub` file:

```bash
KEY="$KEYDIR/<private-key-file>"
chmod 600 "$KEY"
VM_SSH=(ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new team123@111.88.145.15)
"${VM_SSH[@]}" 'hostname; uname -m; sudo docker --version'
```

Remove the temporary key directory after deployment:

```bash
rm -rf "$KEYDIR"
```

### 2. Stage And Build On docker-srv

```bash
REPO=/Volumes/T7_2TB/Projects-T7_2TB/2026_Nornikel_Hackaton_v2
BUILD_HOST=root@192.168.86.16
BUILD_DIR=/tmp/nornikel-v2-ore-pipeline-ui-build
IMAGE=nornikel/ore-pipeline-ui:v2

ssh "$BUILD_HOST" "rm -rf '$BUILD_DIR' && mkdir -p '$BUILD_DIR'"
rsync -az --delete \
  --exclude='.git/' --exclude='.venv/' --exclude='venv/' \
  --exclude='dataset/' --exclude='outputs/' \
  --include='models/' --include='models/README.md' --exclude='models/***' \
  --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='*.pyc' --exclude='*.zip' \
  "$REPO/" "$BUILD_HOST:$BUILD_DIR/"

ssh "$BUILD_HOST" "cd '$BUILD_DIR' && docker compose build ore-pipeline-ui"
ssh "$BUILD_HOST" "docker image inspect '$IMAGE' --format 'arch={{.Architecture}} os={{.Os}} size={{.Size}} id={{.Id}}'"
```

Verified image from the first VM deployment:

```text
nornikel/ore-pipeline-ui:v2
linux/amd64
sha256:3e2a83981f56208cfa9804c92ad4dc391a9532c645736057d0f49e00395e6a6c
```

### 3. Upload Image To VM

Stream the image through the Mac so the VM private key is not copied to `docker-srv`:

```bash
ssh "$BUILD_HOST" "docker save '$IMAGE' | gzip -1" | "${VM_SSH[@]}" "gzip -dc | sudo docker load"
```

### 4. Run On VM

```bash
"${VM_SSH[@]}" '
set -e
RUNTIME=$HOME/ore-pipeline-ui
mkdir -p "$RUNTIME/outputs/ore_pipeline_ui" "$RUNTIME/models"
'
scp -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
  compose.yaml team123@111.88.145.15:ore-pipeline-ui/compose.yaml
"${VM_SSH[@]}" '
set -e
CONTAINER=nornikel-ore-pipeline-ui
RUNTIME=$HOME/ore-pipeline-ui
cd "$RUNTIME"
sudo docker rm -f "$CONTAINER" 2>/dev/null || true
sudo env \
  ORE_UI_IMAGE=nornikel/ore-pipeline-ui:v2 \
  ORE_UI_CONTAINER_NAME="$CONTAINER" \
  ORE_UI_PUBLIC_PORT=8080 \
  ORE_UI_BACKEND=heuristic \
  ORE_UI_WORKSPACE_HOST="$RUNTIME/outputs/ore_pipeline_ui" \
  ORE_UI_MODELS_HOST="$RUNTIME/models" \
  docker compose up -d --no-build ore-pipeline-ui
sudo docker ps --filter name="$CONTAINER" --format "{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
'
```

### 5. Smoke VM

```bash
curl -sS -o /dev/null -w '%{http_code} %{time_total}\n' http://111.88.145.15:8080/workspace
curl -sS http://111.88.145.15:8080/api/status | jq '{overall:.health.overall, app:.app, history:.history}'
```

Optional functional smoke from the Mac:

```bash
BASE=http://111.88.145.15:8080
SAMPLE='dataset/Фото руд по сортам. ч2/тонкие/69 1.jpg'
UPLOAD_JSON=/tmp/ore_vm_upload.json
RUN_JSON=/tmp/ore_vm_run.json

curl -sS -F "file=@${SAMPLE}" "$BASE/api/uploads" -o "$UPLOAD_JSON"
UPLOAD_ID="$(jq -r '.upload_id' "$UPLOAD_JSON")"
curl -sS -H 'Content-Type: application/json' \
  -d "{\"upload_id\":\"$UPLOAD_ID\"}" \
  "$BASE/api/runs/start" -o "$RUN_JSON"
RUN_ID="$(jq -r '.run_id' "$RUN_JSON")"
for _ in $(seq 1 60); do
  curl -sS "$BASE/api/runs/$RUN_ID" -o "$RUN_JSON"
  STATUS="$(jq -r '.status' "$RUN_JSON")"
  [ "$STATUS" = complete ] || [ "$STATUS" = failed ] || [ "$STATUS" = canceled ] && break
  sleep 1
done
jq '{run_id:.run_id,status:.status,summary:.summary}' "$RUN_JSON"
```

## gx10 ARM64

Use this path for the ARM64 LAN deployment on `asus_gx10`. It builds natively on gx10 and runs through the Compose GPU profile so `/api/status` can report `NVIDIA GB10`.

### 1. Stage And Build On gx10

```bash
REPO=/Volumes/T7_2TB/Projects-T7_2TB/2026_Nornikel_Hackaton_v2
GX10=ashmelev@192.168.86.14
BUILD_DIR=/home/ashmelev/Projects/nornikel-v2-ore-pipeline-ui-build
IMAGE=nornikel/ore-pipeline-ui:v2-arm64
IMAGE_ML=nornikel/ore-pipeline-ui:v2-gx10-ml

ssh "$GX10" "rm -rf '$BUILD_DIR' && mkdir -p '$BUILD_DIR'"
rsync -az --delete \
  --exclude='.git/' --exclude='.venv/' --exclude='venv/' \
  --exclude='dataset/' --exclude='data/external/' --exclude='outputs/' \
  --include='models/' --include='models/README.md' --exclude='models/***' \
  --exclude='presentation/videos/' \
  --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='*.pyc' \
  --exclude='*.zip' --exclude='*.tar' --exclude='*.tar.gz' --exclude='*.7z' \
  "$REPO/" "$GX10:$BUILD_DIR/"

ssh "$GX10" "cd '$BUILD_DIR' && docker build --platform linux/arm64 --pull=false -f docker/ore-pipeline-ui/Dockerfile -t '$IMAGE' ."
ssh "$GX10" "cd '$BUILD_DIR' && docker build --platform linux/arm64 --pull=false -f docker/ore-pipeline-ui/Dockerfile.gx10-ml -t '$IMAGE_ML' ."
ssh "$GX10" "docker image inspect '$IMAGE' --format 'heuristic arch={{.Architecture}} os={{.Os}} size={{.Size}} id={{.Id}}'"
ssh "$GX10" "docker image inspect '$IMAGE_ML' --format 'ml arch={{.Architecture}} os={{.Os}} size={{.Size}} id={{.Id}}'"
```

Verified image from the first gx10 deployment:

```text
nornikel/ore-pipeline-ui:v2-arm64
linux/arm64
sha256:a55f381707f584b8cbceb94e073603fb2988cdf1a947166b23d5242d4e0c22be
```

Verified current SOTA ML image from the 2026-07-05 gx10 redeploy:

```text
nornikel/ore-pipeline-ui:v2-gx10-ml
linux/arm64
sha256:4e08a307651453d86f33c10e4e4ae3a9fe3614181016fc6c3c28ea932bcbcc9d
```

### 2. Run On gx10

```bash
ssh "$GX10" '
set -e
CONTAINER=nornikel-ore-pipeline-ui-v2
RUNTIME=$HOME/nornikel-ore-pipeline-ui
BUILD_DIR=$HOME/Projects/nornikel-v2-ore-pipeline-ui-build
REPO=$HOME/Projects/2026_Nornikel_Hackaton_v2
BINARY=/app/models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt
TALC=/app/outputs/talc_segformer_folds/segformer_b0_full_20260703/fold_00/segformer_b0/best.pt
GRADE=/app/models/grade_classifier/effb3_ordfine_ppaug_20260704/best.pt
mkdir -p "$RUNTIME/outputs/ore_pipeline_ui" "$RUNTIME/models"
docker rm -f "$CONTAINER" 2>/dev/null || true
cd "$BUILD_DIR"
env \
  ORE_UI_GPU_IMAGE=nornikel/ore-pipeline-ui:v2-gx10-ml \
  ORE_UI_GPU_CONTAINER_NAME="$CONTAINER" \
  ORE_UI_PUBLIC_PORT=8210 \
  ORE_UI_BACKEND=ml \
  ORE_UI_CHECKPOINT="$BINARY" \
  ORE_UI_TALC_BACKEND=ml \
  ORE_UI_TALC_CHECKPOINT="$TALC" \
  ORE_UI_TALC_THRESHOLD=0.50 \
  ORE_UI_GRADE_CHECKPOINT="$GRADE" \
  ORE_UI_WORKSPACE_HOST="$RUNTIME/outputs/ore_pipeline_ui" \
  ORE_UI_MODELS_HOST="$REPO/models" \
  ORE_UI_TALC_OUTPUTS_HOST="$REPO/outputs/talc_segformer_folds" \
  docker compose --profile gpu up -d --no-build ore-pipeline-ui-gpu
docker ps --filter name="$CONTAINER" --format "{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
'
```

To fall back to the lightweight heuristic image, run the default `ore-pipeline-ui` service with `ORE_UI_IMAGE=nornikel/ore-pipeline-ui:v2-arm64`, set `ORE_UI_BACKEND=heuristic`, omit the talc/grade env vars, and mount only the persistent workspace plus any optional model directory.

### 3. Smoke gx10

```bash
curl -sS -o /dev/null -w '%{http_code} %{time_total}\n' http://192.168.86.14:8210/workspace
curl -sS http://192.168.86.14:8210/api/status | jq '{overall:.health.overall, gpu:.gpu, history:.history}'
curl -sS -H 'Content-Type: application/json' -d '{}' http://192.168.86.14:8210/api/runtime/test | jq '{ok, status, backend, talc_backend, models}'
ssh "$GX10" 'docker stats --no-stream --format "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" nornikel-ore-pipeline-ui-v2'
```

Public backup smoke through MikroTik Caddy:

```bash
PUBLIC=https://nornickel-ai-hackathon.my.3simbio.ru
REVIEWER_PASS="$(security find-generic-password -a reviewer -s nornickel-ai-hackathon-basic-auth -w)"
curl -sS -o /dev/null -w '%{http_code}\n' "$PUBLIC/workspace"  # 401
curl -fsS -u "reviewer:$REVIEWER_PASS" "$PUBLIC/workspace" -o /tmp/gx10-backup-workspace.html
curl -fsS -u "reviewer:$REVIEWER_PASS" "$PUBLIC/api/status" | jq '{overall:.health.overall, backend:.app.backend, talc_backend:.app.talc_backend, gpu:.gpu.devices[0].name}'
curl -fsS -u "reviewer:$REVIEWER_PASS" -H 'Content-Type: application/json' -d '{}' "$PUBLIC/api/runtime/test" | jq '{ok,status,backend,talc_backend,device:.details.device}'
```

Expected gx10 status details:

- `/api/status` returns `200`.
- `gpu.available` is `true`, device name is `NVIDIA GB10`.
- For the SOTA ML deployment, `/api/status` reports `app.backend=ml`, the SegFormer-B2 binary checkpoint, `app.talc_backend=ml`, and the SegFormer-B0 talc checkpoint.
- `POST /api/runtime/test` returns `ok=true` and loads both segmentation checkpoints on `cuda`.
- GB10 memory values may be `null` because gx10's `nvidia-smi` reports memory as `[N/A]`.
- Health is `ok` when gx10 flash free space is around `12%`; it may become
  `warning` again if build caches or generated artifacts fill the root disk.

Optional functional smoke from the Mac:

```bash
BASE=http://192.168.86.14:8210
SAMPLE='dataset/Фото руд по сортам. ч2/рядовые/38.jpg'
UPLOAD_JSON=/tmp/ore_gx10_upload.json
RUN_JSON=/tmp/ore_gx10_run.json

curl -sS -F "file=@${SAMPLE}" "$BASE/api/uploads" -o "$UPLOAD_JSON"
UPLOAD_ID="$(jq -r '.upload_id' "$UPLOAD_JSON")"
curl -sS -H 'Content-Type: application/json' \
  -d "{\"upload_id\":\"$UPLOAD_ID\"}" \
  "$BASE/api/runs/start" -o "$RUN_JSON"
RUN_ID="$(jq -r '.run_id' "$RUN_JSON")"
for _ in $(seq 1 120); do
  curl -sS "$BASE/api/runs/$RUN_ID" -o "$RUN_JSON"
  STATUS="$(jq -r '.status' "$RUN_JSON")"
  [ "$STATUS" = complete ] || [ "$STATUS" = failed ] || [ "$STATUS" = canceled ] && break
  sleep 1
done
jq '{run_id:.run_id,status:.status,summary:.summary}' "$RUN_JSON"
```

## Operations

VM:

```bash
"${VM_SSH[@]}" 'sudo docker logs --tail 100 nornikel-ore-pipeline-ui'
"${VM_SSH[@]}" 'sudo docker restart nornikel-ore-pipeline-ui'
"${VM_SSH[@]}" 'sudo docker stats --no-stream nornikel-ore-pipeline-ui'
```

gx10:

```bash
ssh "$GX10" 'docker logs --tail 100 nornikel-ore-pipeline-ui-v2'
ssh "$GX10" 'docker restart nornikel-ore-pipeline-ui-v2'
ssh "$GX10" 'docker stats --no-stream nornikel-ore-pipeline-ui-v2'
```

To stop serving but preserve workspace data:

```bash
"${VM_SSH[@]}" 'sudo docker stop nornikel-ore-pipeline-ui'
ssh "$GX10" 'docker stop nornikel-ore-pipeline-ui-v2'
```

Do not remove mounted workspace directories unless explicitly clearing demo state.

## Known Caveats

- The VM is public on `111.88.145.15:8080`; gx10 port `8210` is LAN-only. The
  current public backup path is the MikroTik Caddy hostname
  `https://nornickel-ai-hackathon.my.3simbio.ru/workspace`, protected by Basic
  Auth and reverse-proxied to `192.168.86.14:8210`.
- Full panorama uploads currently create full-size `preprocessed_full.png` synchronously before the run id is returned. Normal images are fine; panorama live demos should wait for the preparation path optimization.
- Keep the team VM SSH key outside the repo and remove temporary extraction directories after use.
- If gx10 `/api/status` fails with `could not convert string to float: '[N/A]'`, the deployed image predates the GB10 parser fix; rebuild from the current v2 checkout and redeploy.

## Evidence

- VM smoke: `docs/ui/v2/notes/2026-07-03-ore-pipeline-docker-vm-smoke.md`
- gx10 smoke: `docs/ui/v2/notes/2026-07-03-ore-pipeline-docker-gx10-arm64-smoke.md`
- gx10 SOTA ML deploy: `docs/ui/v2/notes/2026-07-04-ore-pipeline-docker-gx10-sota-ml-deploy.md`
