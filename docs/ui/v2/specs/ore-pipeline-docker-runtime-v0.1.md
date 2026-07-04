# Ore Pipeline Docker Runtime v0.1

Date: 2026-07-03

## Goal

Provide a Docker image for running the v2 ore pipeline browser GUI on the Nornickel VM. This is an operational demo/runtime path for `apps/ore_pipeline_web.py`, not the main model-training or benchmark path.

## Runtime Contract

- The container starts the web UI on `0.0.0.0:8080` by default.
- The primary deployment file is repo-root `compose.yaml`; `docker-compose.ore-pipeline-ui.yml` remains as a compatibility file for older scripted commands.
- The host maps a public VM port to container port `8080`.
- On the organizer VM, launch may require `sudo docker compose` if Docker socket access is not granted to the default user.
- The app stores uploads, settings, immutable runs, batches, masks, CSV files, and PDF reports in `/data/ore_pipeline_ui`.
- The default backend is `heuristic` so the UI can run without GPU, Torch, Transformers, or checkpoints.
- RAW upload decoding is supported when `rawpy` can decode the camera file; otherwise the existing UI error asks the user to convert to TIFF, PNG, or JPEG.
- The Docker build must not copy the official dataset symlink, generated outputs, or model checkpoints into the image.

## Configuration

Environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ORE_UI_HOST` | `0.0.0.0` | Bind address inside the container |
| `ORE_UI_PORT` | `8080` | Port inside the container |
| `ORE_UI_WORKSPACE` | `/data/ore_pipeline_ui` | Persistent UI workspace |
| `ORE_UI_BACKEND` | `heuristic` | `heuristic` or `ml` |
| `ORE_UI_CHECKPOINT` | empty | Optional binary sulfide checkpoint path for `ml` backend |
| `ORE_UI_TALC_BACKEND` | empty | Optional talc backend override: `heuristic` or `ml` |
| `ORE_UI_TALC_CHECKPOINT` | empty | Optional talc checkpoint path for talc `ml` backend |
| `ORE_UI_TALC_THRESHOLD` | empty | Optional talc probability threshold |
| `ORE_UI_GRADE_CHECKPOINT` | empty | Optional grade-branch CNN checkpoint path |
| `ORE_UI_PROCESSING_MAX_SIDE` | `2600` | Analysis-scale longest-side limit |
| `ORE_UI_PANORAMA_MAX_SIDE` | `1800` | Panorama preprocessing longest-side limit |
| `ORE_UI_PREVIEW_MAX_SIDES` | `1024,2048,4096` | Preview pyramid side sizes |
| `ORE_UI_PUBLIC_HOST` | `0.0.0.0` | Host interface used by root Compose |
| `ORE_UI_PUBLIC_PORT` | `8080` | Host port used by Compose |
| `ORE_UI_IMAGE` | `nornikel/ore-pipeline-ui:v2` | Default root Compose image |
| `ORE_UI_DOCKERFILE` | `docker/ore-pipeline-ui/Dockerfile` | Default root Compose Dockerfile |
| `ORE_UI_WORKSPACE_HOST` | `./outputs/ore_pipeline_ui` | Host workspace bind mount |
| `ORE_UI_MODELS_HOST` | `./models` | Host model-checkpoint bind mount |
| `ORE_UI_TALC_OUTPUTS_HOST` | `./outputs/talc_segformer_folds` | Optional talc-checkpoint bind mount for the GPU profile |

## Volumes

- `./outputs/ore_pipeline_ui:/data/ore_pipeline_ui` persists app state across restarts.
- `./models:/app/models:ro` makes local checkpoints available if an ML-capable derived image is used.
- gx10 SOTA ML deployments also mount `outputs/talc_segformer_folds:/app/outputs/talc_segformer_folds:ro` so the talc SegFormer-B0 fold checkpoint remains outside the image.

## Non-goals

- Do not install the full training stack in this image.
- Do not bundle the official dataset or generated outputs.
- Do not make this the primary judged inference path; it is a deployable GUI convenience for the VM.
