# Alize v2 Production Deploy

Date: 2026-07-04

## Summary

`https://nornickel-ai-hackathon.alola.ru/` now serves the v2 ore pipeline app from:

```text
/Volumes/T7_2TB/Projects-T7_2TB/2026_Nornikel_Hackaton_v2/apps/ore_pipeline_web.py
```

This replaced the earlier same-day legacy deployment from
`/Volumes/T7_2TB/Projects-T7_2TB/2026_Nornikel_Hackaton`. The legacy
`nornickel-qc-ui` container is no longer on the active Caddy backend path.

## Host

| Field | Value |
|---|---|
| Provider host | Selectel Alize |
| Public IP | `111.88.124.80` |
| Zone | Moscow `ru-7a` |
| GPU | `NVIDIA L4 24 GB` |
| Public URL | `https://nornickel-ai-hackathon.alola.ru/` |
| Plain HTTP IP URL | `http://111.88.124.80/` |
| Reverse proxy | Caddy, public `80/443` -> `127.0.0.1:8765` |
| Access control | Caddy `basicauth`, username `reviewer`; password is not stored in this repo |
| TLS | Let's Encrypt wildcard `*.alola.ru` / `alola.ru` |
| Remote checkout | `/opt/nornickel-ai-hackathon-v2` |
| Runtime workspace | `/opt/nornickel-ai-hackathon-v2/runtime/outputs/ore_pipeline_ui` |

## Image And Container

| Field | Value |
|---|---|
| Image | `nornickel-ore-pipeline-ui:v2-ml` |
| Image id | `sha256:f72410291546f2250f0a7608070312703cadc82b60d8a319960f714325076118` |
| Image size | about `21.1 GB` |
| Container | `nornickel-ore-pipeline-ui-v2` |
| Restart policy | `unless-stopped` |
| Port bind | `127.0.0.1:8765:8080` |
| GPU flag | `--gpus all` |

Runtime env:

```text
ORE_UI_BACKEND=ml
ORE_UI_CHECKPOINT=/app/models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt
ORE_UI_TALC_BACKEND=ml
ORE_UI_TALC_CHECKPOINT=/app/outputs/talc_segformer_folds/segformer_b0_full_20260703/fold_00/segformer_b0/best.pt
ORE_UI_TALC_THRESHOLD=0.50
ORE_UI_GRADE_CHECKPOINT=/app/models/grade_classifier/effb3_ordfine_ppaug_20260704/best.pt
ORE_UI_PROCESSING_MAX_SIDE=2600
ORE_UI_PANORAMA_MAX_SIDE=1800
ORE_UI_PREVIEW_MAX_SIDES=1024,2048,4096
```

Mounted read-only assets:

```text
/opt/nornickel-ai-hackathon-v2/models:/app/models:ro
/opt/nornickel-ai-hackathon-v2/outputs/talc_segformer_folds:/app/outputs/talc_segformer_folds:ro
```

## Build And Start

Preflight on the Mac:

```bash
python3 -m py_compile apps/ore_pipeline_web.py
python3 -m unittest discover -s tests -p 'test_ore_pipeline_docker.py' -v
```

Both passed before deployment.

The first rsync copied a broader source mirror than strictly necessary. Future
redeploys should exclude generated presentation media and sync only the runtime
tree plus required model assets.

Build on Alize:

```bash
cd /opt/nornickel-ai-hackathon-v2
DOCKER_BUILDKIT=0 docker build \
  -f docker/ore-pipeline-ui/Dockerfile.gx10-ml \
  -t nornickel-ore-pipeline-ui:v2-ml .
```

The `nvcr.io/nvidia/pytorch:25.11-py3` base pulled successfully and the apt
step accepted `libjpeg-turbo8` on Ubuntu Noble; no Dockerfile patch was needed.

The image was first started on temporary local port `18765` for status/runtime
testing, then promoted to the production Caddy backend port `8765`.

## Verification

Unauthenticated public access is blocked at Caddy:

```text
https://nornickel-ai-hackathon.alola.ru/workspace -> 401
http://111.88.124.80/                             -> 401
```

Authenticated public `GET /workspace`:

```text
HTTP/2 200
content-type: text/html; charset=utf-8
content-length: 423042
server: Caddy
server: BaseHTTP/0.6 Python/3.12.3
```

Authenticated plain HTTP IP access:

```text
http://111.88.124.80/           -> 302 /workspace
http://111.88.124.80/workspace  -> 200
http://111.88.124.80/api/status -> health ok, backend ml, GPU NVIDIA L4
```

Wrong reviewer credentials return `401`.

Note: `HEAD /workspace` returns `501` because the plain Python handler only
implements `GET`; use `GET` probes.

Authenticated public `/api/status` summary:

```json
{
  "status_health": "ok",
  "backend": "ml",
  "talc_backend": "ml",
  "gpu": "NVIDIA L4",
  "checkpoint_exists": true,
  "talc_checkpoint_exists": true
}
```

Authenticated public `POST /api/runtime/test` summary:

```json
{
  "runtime_ok": true,
  "runtime_status": "ok",
  "runtime_device": "cuda",
  "binary_model": "binary_sulfide ML checkpoint loaded: model=segformer_b2, device=cuda",
  "talc_model": "talc ML checkpoint loaded: model=segformer_b0, device=cuda"
}
```

Functional public smoke:

| Field | Value |
|---|---|
| Sample | `dataset/Фото руд по сортам. ч2/тонкие/69 1.jpg` |
| Upload id | `20260704_180452_821455814_e169c1215e` |
| Run id | `run_20260704_180453_030668504_b9327d1a` |
| Final status | `complete` |
| Final progress | `100` |
| Files listed | `49` |

Representative metrics from the completed run:

```text
analyzed_fraction = 98.306607%
sulfide_fraction = 33.443628%
ordinary_sulfide_fraction = 7.180342%
fine_sulfide_fraction = 92.274804%
component_count = 14
```

## Current Operations

Check the container:

```bash
ssh -o UserKnownHostsFile=/tmp/alize_known_hosts \
  -o StrictHostKeyChecking=yes \
  root@111.88.124.80 \
  'docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Ports}}\t{{.Status}}"'
```

Expected active row:

```text
nornickel-ore-pipeline-ui-v2   nornickel-ore-pipeline-ui:v2-ml   127.0.0.1:8765->8080/tcp   Up
```

Direct external `111.88.124.80:8765` should remain closed; only Caddy should be
public.
