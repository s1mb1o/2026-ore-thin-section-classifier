# V2 Ore Pipeline Series Bottleneck Test

Date: 2026-07-04

Scope: test the v2 UI Series/Batch path for bottlenecks when processing several images as one group. The test used three official single-image JPG inputs with shared preprocessing settings and one image per immutable child run.

## Inputs

Source images:

- `dataset/Фото руд по сортам. ч1/Рядовые руды/2539589-1.JPG` - 2272 x 1704, 0.81 MB
- `dataset/Фото руд по сортам. ч1/Труднообогатимые руды/2539439-3.JPG` - 2272 x 1704, 0.83 MB
- `dataset/Фото руд по сортам. ч1/Оталькованные руды/2550374-2 10х.JPG` - 2272 x 1704, 0.78 MB

Shared settings:

- Preprocessing enabled.
- Illumination normalization, denoise, and contrast correction enabled.
- Panorama scaling enabled with max side 1800 px.
- Runtime augmentation disabled.

## Live ML Series Test

Live local app: `http://127.0.0.1:63589`, launched from `apps/ore_pipeline_web.py`.

Runtime status during the run:

- Binary sulfide backend: `ml`
- Sulfide checkpoint: `models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt`
- Talc backend: `ml`
- Talc checkpoint: `outputs/talc_segformer_folds/segformer_b0_full_20260703/fold_00/segformer_b0/best.pt`
- Concurrent resident evaluation process was active: `scripts/run_resident_batch.py ... --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt ... --device auto`

Measured API setup time:

- Upload 1: 1.570 s, upload id `20260704_041859_381501000_3cdae2790b`
- Upload 2: 0.138 s, upload id `20260704_041900_637438000_0139944fd8`
- Upload 3: 0.120 s, upload id `20260704_041900_788119000_06534d21c6`
- Series create: 0.040 s
- Series start: 0.029 s

Persisted Series:

- Batch id: `batch_20260704_041900_865328000`
- Batch path: `outputs/ore_pipeline_ui/batches/batch_20260704_041900_865328000`
- Status: `canceled`
- Started: `2026-07-04T04:19:00.930000+00:00`
- Completed/canceled: `2026-07-04T04:22:49.893492+00:00`

Progress observations:

- T+0.00 s: item 1 running or queued, items 2 and 3 queued.
- T+4.08 s: item 1 `running ML tiled inference`, progress 18, child run id `run_20260704_041903_400496000_141d0e9a`.
- T+46.57 s: item 1 `running ML tiled inference (0/6 tiles)`, progress 18.
- T+70.11 s: item 1 `running ML tiled inference (4/6 tiles)`, progress 55.
- Later the progress file remained stale at 4/6 while the system was under high load and low available RAM.
- The run was canceled at about T+229 s to avoid wedging the local UI.

Persisted child run:

- Run id: `run_20260704_041903_400496000_141d0e9a`
- Run path: `outputs/ore_pipeline_ui/runs/run_20260704_041903_400496000_141d0e9a`
- Status: `canceled`
- Progress: 74
- Elapsed: 225.83 s
- Run artifact size: 16 MB

Outcome:

- The first ML item did not complete inside the bounded local test.
- Items 2 and 3 never started, as expected for strict sequential Series execution.
- The ML child subprocesses were gone after cancellation.
- The separate `run_resident_batch.py` process remained active and should be treated as a contention source for this measurement.

## Heuristic Control Test

Control workspace: `outputs/ore_pipeline_ui_bottleneck_heuristic_20260704`

Backend settings:

- Binary sulfide backend: `heuristic`
- Talc backend: `heuristic`

Measured setup time:

- Upload 1: 0.099 s
- Upload 2: 0.093 s
- Upload 3: 0.084 s
- Series create: 0.005 s

Persisted Series:

- Batch id: `batch_20260704_042328_019943000`
- Created: `2026-07-04T04:23:28.020023+00:00`
- Started: `2026-07-04T04:23:28.025938+00:00`
- Completed: `2026-07-04T04:24:02.856166+00:00`
- Total wall time: 34.841 s

Per-item timings:

| Item | File | Batch wall | Recorded run elapsed | Run size | Predicted class |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | `2539589-1.JPG` | ~9.69 s | 4.827 s | 24.356 MB | `hard_to_process_ore` |
| 2 | `2539439-3.JPG` | ~18.06 s | 13.034 s | 27.205 MB | `row_ore` |
| 3 | `2550374-2 10х.JPG` | ~7.08 s | 4.473 s | 22.433 MB | `hard_to_process_ore` |

Workspace size after the heuristic control:

- Source JPGs total: about 2.4 MB
- Whole control workspace: 103 MB
- Upload artifacts: 32 MB
- Run artifacts: 71 MB
- Batch summaries: 12 KB
- Storage amplification: about 43x versus source JPG bytes.

## Bottleneck Conclusions

1. Series orchestration is not the dominant bottleneck. Upload, Series create, and Series start were sub-second after the first cold upload.
2. The live ML Series bottleneck is per-item full pipeline execution: `run_ore_pipeline.py` launches `infer_binary_sulfide.py` for tiled B2 inference and also runs the talc ML path. Under concurrent resident evaluation load, the first image did not complete before the bounded cancellation.
3. Sequential Series behavior is correct but makes the first slow ML item block all remaining queued images.
4. The heuristic control completed all three images in 34.841 s. The sum of recorded child run elapsed times was about 22.3 s, leaving about 12.5 s in preprocessing, prep/display/reporting, and Series bookkeeping outside the child `elapsed_seconds` field.
5. Polling the Series endpoint was not a bottleneck in the observed test. A full `/api/status` read during ML contention took about 3.1 s, so performance tests should avoid aggressive Status polling.
6. Storage amplification is high for repeated Series rehearsals because each item keeps upload previews, immutable run artifacts, masks, reports, and display layers.

## Recommended Next Fixes

1. Do not demo local ML Series while `run_resident_batch.py` or other MPS/GPU evaluation jobs are active.
2. Add a v2 UI runtime option for resident/single-load sulfide inference, or route Series ML through the existing resident pipeline so the checkpoint is loaded once per Series rather than once per image.
3. Add per-item retry/cancel controls and a `Stop remaining` action so a single slow ML item does not make the whole Series unusable.
4. Add a Series summary timing panel that separates upload/prep, child run elapsed, reporting, and queue wait time.
5. Add a cleanup policy for disposable Series rehearsal artifacts, or a `Remove Series and child runs` workflow in history for demo prep.
