# Ore Pipeline Docker VM Smoke

Date: 2026-07-03

## Target

- VM: `team123@111.88.145.15`.
- Public URL: `http://111.88.145.15:8080/workspace`.
- Container: `nornikel-ore-pipeline-ui`.
- Image: `nornikel/ore-pipeline-ui:v2`.
- Runtime mode: heuristic backend, no GPU dependency.
- Persistent state: `/home/team123/ore-pipeline-ui/outputs/ore_pipeline_ui`.

## Build and Upload

- Build host: `root@192.168.86.16` (`docker-srv`), x86_64 Docker host.
- Build command: `docker compose -f docker-compose.ore-pipeline-ui.yml build` from a staged v2 checkout.
- BuildKit context size: `928.82 kB`, confirming that `dataset`, `outputs`, `.git`, and model checkpoints were not included in the image context.
- Built image: `nornikel/ore-pipeline-ui:v2`, image list size `861 MB`, `amd64/linux`, image id `sha256:3e2a83981f56208cfa9804c92ad4dc391a9532c645736057d0f49e00395e6a6c`.
- Upload path: streamed `docker save nornikel/ore-pipeline-ui:v2 | gzip -1` from `docker-srv` through the local machine into `sudo docker load` on the Nornickel VM.
- Credential handling: used the user-provided `team123` SSH key archive from outside the repository; no private key was copied into either repository or onto the build host.

## Runtime State

Container launch:

```bash
sudo docker run -d \
  --name nornikel-ore-pipeline-ui \
  --restart unless-stopped \
  --pull=never \
  -p 8080:8080 \
  -e ORE_UI_HOST=0.0.0.0 \
  -e ORE_UI_PORT=8080 \
  -e ORE_UI_WORKSPACE=/data/ore_pipeline_ui \
  -e ORE_UI_BACKEND=heuristic \
  -e ORE_UI_PROCESSING_MAX_SIDE=2600 \
  -e ORE_UI_PANORAMA_MAX_SIDE=1800 \
  -e ORE_UI_PREVIEW_MAX_SIDES=1024,2048,4096 \
  -v /home/team123/ore-pipeline-ui/outputs/ore_pipeline_ui:/data/ore_pipeline_ui \
  -v /home/team123/ore-pipeline-ui/models:/app/models:ro \
  nornikel/ore-pipeline-ui:v2
```

Verified container state:

- `0.0.0.0:8080->8080/tcp`.
- restart policy: `unless-stopped`.
- mounts: `/home/team123/ore-pipeline-ui/outputs/ore_pipeline_ui -> /data/ore_pipeline_ui`, `/home/team123/ore-pipeline-ui/models -> /app/models`.
- VM free disk after smoke and cleanup: `/` has `17 GB` free out of `29 GB`.
- Docker disk usage on target: images about `863 MB`; one active container.
- Idle container usage after restart: `0.01%` CPU, `38.51 MiB / 15.61 GiB`, `7` PIDs.
- Persistent UI workspace after completed smokes and panorama-timeout cleanup: `32 MB`.

## Serve Latency

Local from VM, 10 requests each:

| Path | Avg | Min | Max | HTTP |
| --- | ---: | ---: | ---: | --- |
| `/workspace` | `0.0016 s` | `0.0012 s` | `0.0033 s` | `200` |
| `/api/settings` | `0.0010 s` | `0.0008 s` | `0.0011 s` | `200` |
| `/api/runs` | `0.0024 s` | `0.0022 s` | `0.0027 s` | `200` |

Public from the local Mac to `111.88.145.15:8080`, 10 requests each:

| Path | Avg | Min | Max | HTTP |
| --- | ---: | ---: | ---: | --- |
| `/workspace` | `0.0864 s` | `0.0772 s` | `0.1110 s` | `200` |
| `/api/settings` | `0.0244 s` | `0.0227 s` | `0.0257 s` | `200` |
| `/api/runs` | `0.0276 s` | `0.0262 s` | `0.0294 s` | `200` |

## Functional Smokes

Small official sample:

- Source: `dataset/Фото руд по сортам. ч2/тонкие/69 1.jpg`, `37 KB`, uploaded as `smoke_69_1.jpg`.
- Upload: HTTP `200`, `0.049846 s`.
- Start: HTTP `200`, `0.849596 s`.
- Run: `run_20260703_174159_985498854_090de6b5`, completed after two polls.
- Metrics CSV: HTTP `200`, `0.030124 s`, `1108` bytes.
- Result summary: `hard_to_process_ore`, sulfide fraction `0.171146`, talc fraction `0.0`, component count `48`.

Medium official sample:

- Source: `dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования/DSCN3052.JPG`, `2.2 MB`, uploaded as `medium_DSCN3052.JPG`.
- Upload: HTTP `200`, `1.533690 s`.
- Start/preparation: HTTP `200`, `8.091471 s`.
- Run: `run_20260703_174249_228402877_a1ce3711`, complete when polled after start.
- Analysis image: `1800 x 1350`.
- Metrics CSV: HTTP `200`, `0.025521 s`, `1137` bytes.
- Result summary: `hard_to_process_ore`, sulfide fraction `0.344981`, talc fraction `0.006059`, component count `198`.

## Panorama Stress Finding

Smallest official panorama tested:

- Source: `dataset/Панорамы/13.jpg`, `46 MB`, uploaded as `panorama_13.jpg`.
- Upload: HTTP `200`, `8.946612 s`.
- Start/preparation did not return within the smoke window after several minutes.
- Resource usage during the active request: about `100%` of one CPU core, `1.54 GiB` resident memory, `1.84 GiB` high-water memory.
- Observed bottleneck: the server was writing full-size `preprocessed/preprocessed_full.png` for the panorama upload before returning the run ID; the partial PNG reached `66 MB`.
- Action taken: stopped the client request, restarted the container, removed only the incomplete panorama upload directory, and rechecked `/workspace`.

Conclusion: the VM has enough CPU/RAM/disk for the current Docker runtime and normal official images, and the public UI serves quickly. Full panorama uploads expose a real latency bottleneck in the current UI path because full-size preprocessing artifacts are created synchronously before the async run starts. For live judging, use normal images or fix panorama preparation to avoid blocking on full-size PNG writes before claiming live panorama support.
