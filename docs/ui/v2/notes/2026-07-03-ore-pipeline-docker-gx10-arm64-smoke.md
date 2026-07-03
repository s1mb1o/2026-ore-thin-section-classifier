# GX10 ARM64 Docker Smoke — Ore Pipeline UI

Date: 2026-07-03 21:49 MSK

## Deployment

- Host: `gx10-fb56`, `ashmelev@192.168.86.14`, `aarch64`.
- Build dir on gx10: `~/Projects/nornikel-v2-ore-pipeline-ui-build`.
- Image: `nornikel/ore-pipeline-ui:v2-arm64`.
- Image inspect: `linux/arm64`, size `557039931` bytes, image id `sha256:a55f381707f584b8cbceb94e073603fb2988cdf1a947166b23d5242d4e0c22be`.
- Container: `nornikel-ore-pipeline-ui-v2`.
- Runtime: `docker run`, `--restart unless-stopped`, `--gpus all`, `-p 8210:8080`.
- LAN URL: `http://192.168.86.14:8210/workspace`.
- Persistent workspace: `/home/ashmelev/nornikel-ore-pipeline-ui/outputs/ore_pipeline_ui`.
- Models bind: `/home/ashmelev/nornikel-ore-pipeline-ui/models:/app/models:ro`.

The old v1 CUDA upload UI container `nornickel-qc-ui` was observed as `Exited (255) 7 days ago` on port `18765` and was left untouched.

## GX10-Specific Fix

Running the container with `--gpus all` exposed `/usr/bin/nvidia-smi`, but GB10 reports GPU memory fields as `[N/A]`. The first GPU-visible status check failed with:

```text
could not convert string to float: '[N/A]'
```

`apps/ore_pipeline_web.py` now tolerates optional `nvidia-smi` numeric fields and returns `null` for unknown memory values. The UI byte formatter now renders unknown bytes as `—` instead of `0 B`. Regression coverage was added in `tests/test_ore_pipeline_web.py`.

## Smoke

Final image smoke used:

```text
dataset/Фото руд по сортам. ч2/рядовые/38.jpg
498 KiB JPEG, 3120 x 1886
```

Final run:

- Upload id: `20260703_184838_403715564_add7e3dea0`.
- Run id: `run_20260703_184851_271976463_2c96a2ad`.
- Status: `complete`.
- Classification: `рядовая руда`.
- Upload HTTP time: `0.174661 s`.
- Start/process HTTP time: `18.600675 s`.
- Poll tail after start returned complete in `3 s`.
- `/api/runs/.../report.pdf`: `200`, `0.721966 s`, `586619` bytes.

Endpoint timings after the final restart:

| Endpoint | Status | Time | Bytes |
|---|---:|---:|---:|
| `/workspace` | 200 | `0.037569 s` | `302083` |
| `/api/status` | 200 | `0.032978 s` | `7026` |
| `/api/runs/run_20260703_184851_271976463_2c96a2ad/report.pdf` | 200 | `0.721966 s` | `586619` |

## Resources

Final `/api/status`:

- Health: `warning` only because flash free is about `9.0%`.
- GPU: `NVIDIA GB10`, utilization `0.0%`, temperature `42 C`, memory fields unknown from driver (`[N/A]`).
- RAM available: `116.96 GiB`.
- Flash free: `164.97 GiB`.
- History: `2` complete runs, workspace size about `50.7 MB`.

Final idle `docker stats` after the smoke:

```text
nornikel-ore-pipeline-ui-v2  0.01% CPU  300.9MiB / 121.6GiB  0.24%
```

The service has enough RAM/CPU headroom for normal-image demos. Disk is adequate for short demos but should be cleaned or moved before repeated panorama/stress runs because the gx10 root filesystem is already about `91%` used.

## Follow-Up

- No WAN/NAT rule was added; this is a LAN service at `192.168.86.14:8210`.
- The VM finding still applies: large panorama demos remain CPU-bound by synchronous full-size `preprocessed_full.png` generation before returning a run id.
