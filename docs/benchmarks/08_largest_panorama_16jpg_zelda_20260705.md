# Largest Panorama End-To-End Benchmark (16.jpg) - 2026-07-05

## Purpose

`docs/benchmarks/07_panorama_performance_20260705.md` proved the 5-minute target
on the panorama closest to the official `10000 x 10000` bar (`13.jpg`,
126 Mpx). This follow-up answers the remaining open question: does the **largest
panorama in the official dataset** run end-to-end at all, and what does it
actually consume? Until this run, "fits in a 16 GB GPU" was a design goal, not
evidence.

## Input

| Image | Size | Pixels | Note |
| --- | ---: | ---: | --- |
| `dataset/Панорамы/16.jpg` | `27025 x 21227` | `573,659,675` (574 Mpx) | Largest official panorama; 211 MB JPEG. |

## Configuration

Same mode as the strongest path in benchmark 07 (`ml_sulfide_ml_talc`):

```bash
/usr/bin/time -v python3 scripts/run_ore_pipeline.py \
  --image "dataset/Панорамы/16.jpg" \
  --checkpoint <segformer_b2_dataset_v0_zelda_20260703_overnight best.pt> \
  --tile-size 1024 --stride 768 --batch-size 2 --device cuda \
  --preview-max-side 1800 \
  --talc-checkpoint <talc segformer_b0_full_20260703 fold_00 best.pt> \
  --talc-threshold 0.50 \
  --out-dir /dev/shm/bench16/out
```

- Machine: zelda, RTX 4090 24 GB, 8 vCPU Xeon Gold 6530, 31 GiB RAM.
- Root disk was 100% full (known issue), so checkpoints, fresh `scripts/`+`src/`
  code from the local repo, and outputs were staged in `/dev/shm/bench16/`;
  the input image was read from the on-disk repo copy.
- GPU was otherwise idle; GPU memory polled via `nvidia-smi` every 2 s.

## Results

| Metric | Value |
| --- | ---: |
| Wall clock end-to-end | **7:07.85 (427.85 s)** |
| Binary sulfide tiled inference | `118.6 s` |
| Talc model inference | `238.6 s` |
| Peak GPU memory (2 s polling, 198 samples) | **4777 MiB (~4.7 GiB)** |
| Peak process RSS (system RAM) | `21,045,836 kB (~20.1 GiB)` |
| CPU utilization of the job | `129 %` |
| Output artifacts size | `309 MB` |
| Exit code | `0` |

Classification side effect (not a ground-truth claim; the panorama is
unlabelled): `row_ore`, `talc_fraction_image ~= 0.0274`,
`sulfide_fraction ~= 0.0925`, `component_count = 13170`.

## Interpretation

- **Throughput is consistent with benchmark 07**: 574 Mpx in 427.9 s is
  ~74.6 s per 100 Mpx, in line with the 104.3 s / 126 Mpx zelda measurement.
  Scaling is roughly linear in pixel area; there is no blow-up on the largest
  input.
- **GPU memory does not grow with panorama size**: peak 4777 MiB on a
  574 Mpx panorama, far below 16 GB. The "fits a T4-class 16 GB GPU" slide
  claim is now measured, not aspirational (for GPU memory specifically).
- **System RAM is the real floor**: peak RSS ~20.1 GiB. A 16 GB-RAM
  workstation would likely swap or fail on this panorama; ~32 GB RAM is the
  honest workstation requirement for the largest official panorama. This is
  full-resolution stitching cost, not GPU cost.
- Relative to the official bar (10k x 10k = 100 Mpx in <= 300 s), the largest
  panorama runs ~5.7x the target area in 427.9 s, i.e. ~4x faster than the
  target rate.

## Evidence

```text
outputs/benchmarks/largest_panorama_16jpg_zelda_20260705/
  run_stdout.log      # pipeline stage logs and summaries
  time_stderr.log     # /usr/bin/time -v output, EXIT_CODE=0
  gpu_mem.log         # nvidia-smi memory.used samples, 2 s interval
  ore_summary.json    # final ore analysis summary
  start_epoch.txt
```

Remote staging (`zelda:/dev/shm/bench16/`) is ephemeral (tmpfs) and can be
reclaimed at any time; local copies above are the durable evidence.
