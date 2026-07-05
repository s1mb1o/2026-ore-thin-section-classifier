# Panorama Performance Benchmark - 2026-07-05

## Requirement

Official performance target:

> Обработка одного панорамного изображения до `10000 x 10000` px - не более 5 минут на рабочей станции с CPU/GPU.

This benchmark uses the closest available official panorama to a `10000 x 10000` image by both pixel area and dimension distance:

| Image | Size | Pixels | Why selected |
| --- | ---: | ---: | --- |
| `dataset/Панорамы/13.jpg` | `13330 x 9489` | `126,488,370` | Closest panorama to `100,000,000` px; next closest is `11.jpg` at `129,357,072` px. |

The run is intentionally full-resolution for the CLI pipelines:

- Heuristic-only uses `heuristic_segmentation/run_heuristic_segmentation.py --max-side 0`.
- ML variants use `scripts/run_ore_pipeline.py` with `tile_size=1024`, `stride=768`, `batch_size=2`, SegFormer-B2 sulfide checkpoint, and either heuristic talc candidate or SegFormer-B0 talc checkpoint.
- Result times are wall-clock seconds measured by a Python wrapper around each command, including model load and artifact writes.

## Result Summary

Target pass/fail is judged against `300 s`. The first table is the original
measurement before the ROI component-classification optimization.

| Machine | Load state | Heuristic sulfides + heuristic talc, no ML | ML sulfides + heuristic talc | ML sulfides + ML talc | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| MacBook Pro 2023, Apple M2 Max | No ore jobs active at local row start; post-run load high from benchmark | `>300.1 s` timeout | `122.9 s` | `186.9 s` | ML paths pass; no-ML full-resolution path fails. |
| gx10, NVIDIA GB10 | Contended during ML runs by another resident batch started at 00:41 MSK | `834.9 s`, killed after fail was clear | `55.1 s` | `99.6 s` | ML paths pass even under contention; no-ML full-resolution path fails. |
| zelda, RTX 4090 | Compute idle; root disk full, so scratch/output used `/dev/shm` | `835.2 s`, killed after fail was clear | `61.6 s` | `104.3 s` | ML paths pass; no-ML full-resolution path fails. |
| VM 102 `ubuntu-dev` | Not runnable from current access path | Not measured | Not measured | Not measured | Blocked: QEMU guest agent not running, IP unknown, only IPv6 link-local neighbor visible. |

## ROI Component Optimization Follow-up

After the original run, `_classify_sulfide_components()` was optimized to process
each connected component inside its padded bounding box instead of rebuilding and
morphologically processing a full-frame boolean mask for every component. The
algorithmic features are unchanged: contour perimeter, convex-hull solidity,
compactness, footprint-closing, internal-dark area, and replacement ratio are
computed on the same component pixels, with a `2 * footprint_close_radius + 2`
ROI halo clipped to image bounds.

Focused validation:

- Local regression test compares the ROI implementation with the old full-frame
  reference on edge and interior components.
- gx10 Docker runtime ran the same focused unit suite: `3` tests passed.
- gx10 full-resolution rerun on the same `dataset/Панорамы/13.jpg` completed:
  `real 4m54.074s`, `user 10m45.239s`, `sys 0m3.375s`.

Additional requested reruns:

| Machine | Optimized heuristic-only wall time | Status | Evidence |
| --- | ---: | --- | --- |
| Mac M2 Max | `130.71 s` | Pass | `outputs/benchmarks/heuristic_roi_component_20260705_mac/` |
| gx10 GB10 | `294.074 s` | Pass, narrow margin | `outputs/benchmarks/heuristic_roi_component_20260705_gx10/` |
| zelda RTX 4090 | `569.43 s` | Fail | `outputs/benchmarks/heuristic_roi_component_20260705_zelda/` |

All three optimized heuristic-only runs produced `4413` components,
`sulfide_fraction=0.027347`, `talc_candidate_fraction=0.0`, and
`ordinary_intergrowth_candidate`. Evidence is under:

```text
outputs/benchmarks/heuristic_roi_component_20260705_{mac,gx10,zelda}/
```

This supersedes the original Mac and gx10 heuristic-only failures for current
code. It does not supersede the zelda heuristic-only failure: zelda completed,
but above the 5-minute target.

## Current Pipeline Refresh

The current full-resolution CLI pipeline was rerun after the component-grade
and magnetite-prep integration. These runs used the same `13.jpg` panorama and
the same tiled SegFormer-B2 sulfide / SegFormer-B0 talc settings, with:

- `--component-model models/component_grade/hgb_weak100_nomag_20260705/model.joblib`
- `--magnetite-prep`
- `--grade-checkpoint models/grade_classifier/effb3_ordfine_ppaug_20260704/best.pt`

On this panorama, magnetite prep was enabled but did not trigger a second
sulfide pass (`no_giant(5%/127026px)` or `127027px` on zelda). The component
model and Grade-CNN branch were loaded and recorded in `pipeline_summary.json`.

| Machine | Load state for refresh | Heuristic sulfides + heuristic talc, no ML (ROI current) | Current ML sulfides + heuristic talc | Current ML sulfides + ML talc | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| MacBook Pro 2023, Apple M2 Max | No ore jobs active; rerun pinned to Homebrew Python 3.14 after Xcode Python 3.9 rejected `zip(..., strict=True)` | `130.71 s` | `88.80 s` | `166.02 s` | Current ML paths pass. |
| gx10, NVIDIA GB10 | Waited for a separate resident batch to finish; benchmark started after GPU utilization returned to `0%` | `294.074 s` | `50.96 s` | `76.36 s` | All current rows pass; heuristic-only has narrow margin. |
| zelda, RTX 4090 | Compute idle; root disk still full, so staged under `/dev/shm`; scratch cleaned after run | `569.43 s` | `73.57 s` | `116.50 s` | Current ML paths pass; heuristic-only still fails. |

Evidence for the refresh is under:

```text
outputs/benchmarks/panorama_performance_20260705_current/{mac,gx10,zelda}/
```

## Detailed Timings

| Machine | Mode | Wall, s | Main sulfide tiled inference, s | Talc model inference, s | Device | Tiles |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Mac M2 Max | heuristic-only | `>300.1` timeout | n/a | n/a | CPU | n/a |
| Mac M2 Max | ML sulfide + heuristic talc | `122.9` | `100.0` | n/a | MPS | `234` |
| Mac M2 Max | ML sulfide + ML talc | `186.9` | `107.9` | `46.2` | MPS | `234` |
| Mac M2 Max | current ML sulfide + heuristic talc + component/grade | `88.80` | `69.293` | n/a | MPS | `234` |
| Mac M2 Max | current ML sulfide + ML talc + component/grade | `166.02` | `110.297` | `37.776` | MPS | `234` |
| gx10 GB10 | heuristic-only | `834.9`, killed | n/a | n/a | CPU | n/a |
| gx10 GB10 | ML sulfide + heuristic talc | `55.1` | `41.8` | n/a | CUDA | `234` |
| gx10 GB10 | ML sulfide + ML talc | `99.6` | `47.9` | `41.1` | CUDA | `234` |
| gx10 GB10 | current ML sulfide + heuristic talc + component/grade | `50.96` | `33.352` | n/a | CUDA | `234` |
| gx10 GB10 | current ML sulfide + ML talc + component/grade | `76.36` | `32.899` | `28.551` | CUDA | `234` |
| zelda RTX 4090 | heuristic-only | `835.2`, killed | n/a | n/a | CPU | n/a |
| zelda RTX 4090 | ML sulfide + heuristic talc | `61.6` | `31.6` | n/a | CUDA | `234` |
| zelda RTX 4090 | ML sulfide + ML talc | `104.3` | `31.8` | `52.3` | CUDA | `234` |
| zelda RTX 4090 | current ML sulfide + heuristic talc + component/grade | `73.57` | `31.105` | n/a | CUDA | `234` |
| zelda RTX 4090 | current ML sulfide + ML talc + component/grade | `116.50` | `31.342` | `52.149` | CUDA | `234` |

Observed classification side effect on this unlabelled panorama:

- `ML sulfide + heuristic talc`: `row_ore`, `talc_fraction=0.0`, `sulfide_fraction ~= 0.018575`.
- `ML sulfide + ML talc`: `talcose_ore`, `talc_fraction ~= 0.973`, `sulfide_fraction ~= 0.018575`.

The classification is not a ground-truth claim for the panorama; this run is performance evidence.

## Machine Configurations

| Machine | CPU | GPU | RAM | Flash / scratch |
| --- | --- | --- | --- | --- |
| MacBook Pro 2023 | Apple M2 Max, `12` physical/logical CPU cores | Apple M2 Max, `38` GPU cores, Metal/MPS | `34,359,738,368` bytes (`32 GiB`) unified | Internal Apple SSD `1 TB`, `104 GB` free after run; project on Samsung T7 APFS, about `899 GiB` free. |
| gx10 | ARM64, `10 x Cortex-X925 + 10 x Cortex-A725`, `20` CPUs | NVIDIA GB10, driver `580.142`, CUDA `13.0` | `121 GiB`, `102 GiB` available in post-run probe | NVMe root `1.8T`, `224 GiB` free. |
| zelda | `8` vCPU, Intel Xeon Gold 6530 host CPU | NVIDIA GeForce RTX 4090, `24564 MiB` VRAM, driver `580.126.09`, CUDA `13.0` | `31 GiB`, `29 GiB` available in post-run probe | Root disk `40G` was full/near-full; benchmark input/checkpoints/outputs were staged under `/dev/shm` (`16G`, ~`455M` used after run). |
| VM 102 `ubuntu-dev` | KVM `host`, `4` cores from Proxmox config | Virtio VGA only; no discrete GPU documented | `8 GB` from VM config | `64 GB` virtual SSD on `local-lvm`; guest IP unavailable. |

## Load Checks And Caveats

- Mac: before the local row, `pgrep` found no active `run_resident_batch.py`, `run_ore_pipeline.py`, or `run_heuristic_segmentation.py` processes. The post-run probe shows high load because it was captured immediately after the benchmark.
- gx10: initial preflight at 00:24 MSK showed low load and idle GPU, but a separate resident batch job appeared at 00:41 MSK and overlapped the ML measurements. Keep gx10 numbers as load-contended pass evidence, not as a clean peak-performance number.
- zelda: compute was idle and GPU had no running processes, but root disk was already almost full. All benchmark scratch data was kept in RAM disk. The root disk should be cleaned before using zelda for more runs.
- VM 102: Proxmox confirms VMID `102` is running with `agent: 1`, but `qm agent 102 ping` returns `QEMU guest agent is not running`. The bridge FDB sees MAC `BC:24:11:FD:4D:BF`, but only a link-local IPv6 neighbor is visible; no usable SSH/IP path was found.
- Current refresh: Mac was rerun with `/opt/homebrew/bin/python3` (`3.14.4`) because `bash -lc python3` resolved to Xcode Python `3.9.6`; the first wrapper failed before any valid timing with `TypeError: zip() takes no keyword arguments`. gx10 was checked repeatedly and the current refresh was delayed until the concurrent `run_resident_batch.py` finished (`345` rows, `0` failures) and GPU utilization returned to `0%`. zelda was staged in `/dev/shm` because root remained `100%` full; the scratch directory was removed after evidence copy-back.

## Artifacts

Raw evidence is under:

```text
outputs/benchmarks/panorama_performance_20260705/raw/
outputs/benchmarks/panorama_performance_20260705_current/
```

Important files:

- `raw/mac/results.json`
- `raw/gx10/results_heuristic_timeout.json`
- `raw/gx10/results_ml.json`
- `raw/zelda/results_heuristic_timeout.json`
- `raw/zelda/results_ml.json`
- `raw/machine_probes/{mac,gx10,zelda,vm102}.txt`
- `panorama_performance_20260705_current/{mac,gx10,zelda}/ml_sulfide_heuristic_talc_current_timing.txt`
- `panorama_performance_20260705_current/{mac,gx10,zelda}/ml_sulfide_ml_talc_current_timing.txt`
- `panorama_performance_20260705_current/{mac,gx10,zelda}/*_pipeline_summary.json`

## Answer To Requirement

The current judged/default ML processing path satisfies the 5-minute panorama-performance target on the measured CPU/GPU workstations for the selected `13330 x 9489` panorama:

- Mac M2 Max: `166.02 s` for the refreshed current ML sulfide + ML talc path with component model, magnetite prep enabled, and Grade-CNN branch.
- gx10 GB10: `76.36 s` for the same refreshed current path after waiting for GPU load to clear.
- zelda RTX 4090: `116.50 s` for the same refreshed current path.

After ROI component classification, the full-resolution heuristic-only CLI also
passes on Mac (`130.71 s`) and gx10 (`294.074 s`), but fails on zelda
(`569.43 s`). Because the heuristic path is machine-sensitive and has a narrow
gx10 margin, the ML path remains the stronger primary performance claim.
