# Ore Pipeline Large-Image Performance

Date: 2026-07-03

Scope: `apps/ore_pipeline_web.py`, `scripts/infer_binary_sulfide.py`, and `scripts/run_ore_pipeline.py`.

## Bottlenecks Found

- Upload registration decoded full panorama images just to create display previews. Preview pyramids also resized from the full image once per configured preview size.
- Path-based upload registration hashed the same file twice: once for upload id allocation and once for raw metadata.
- Large-image preprocessing synchronously wrote full-size `preprocessed_full.png` before the async run could make useful progress. This was the main Docker VM panorama-demo risk already seen in the VM smoke.
- ML tiled inference already batches tiles on the selected device. Extra CPU-side parallel tile workers would likely increase memory pressure and can contend with GPU/MPS execution unless measured on a specific backend. The immediate UI gap was missing processed-tile telemetry.

## Changes Made

- Display loads now use decoder downsampling where Pillow supports it, bounded by the largest configured preview size.
- Preview pyramids are generated largest-to-smallest from the previous resized preview, and transient preview encoding avoids expensive optimize passes.
- Path upload SHA-1 is computed once and reused in raw metadata.
- Panorama-scale preprocessing now records `full_size_processing_deferred=true`, skips synchronous full-size `preprocessed_full.png`, and applies preprocessing to the analysis-scale image. Normal smaller images keep the previous full-size artifact contract.
- Binary sulfide tiled inference accepts `--progress-json` and writes atomic tile progress after each processed tile batch. The web ML worker polls it and exposes `running ML tiled inference (processed/total tiles)` plus a `tile_progress` payload.

## Smoke Result

Local panorama preparation smoke:

```text
image: dataset/Панорамы/13.jpg
source: 13330 x 9489 px
analysis: 1800 x 1282 px
tile_count: 6
register_seconds: 0.936
prepare_seconds: 1.368
total_seconds: 2.305
full_size_processing_deferred: true
preprocessed_full_path_present: false
```

## Follow-Up

- Do not add generic process/thread parallelism around ML tile inference until device utilization and memory are measured on the target host.
- If live demos need full-resolution preprocessed artifacts later, generate them as an explicit background/export action instead of blocking upload/start.
