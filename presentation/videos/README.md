# Demo Video Artifacts

Canonical generated video artifacts for the presentation live in this directory.
The old `outputs/demo_video_*` paths are compatibility symlinks only.

## Bundles

- `demo_video_talc_ui_20260704/`
  - `nornikel_talc_ui_demo_1080p_ru.mp4`: 5:00 Talc UI-only Russian demo, male TTS, burned-in subtitles, corner time marks.
  - `gx10_silero/nornikel_talc_ui_demo_1080p_ru.mp4`: gx10-local Silero `aidar` Russian male voice variant.
- `demo_video_v2_ui_only_20260704/`
  - `nornikel_v2_ui_only_demo_1080p_ru.mp4`: 5:00 main v2 ore-pipeline UI-only Russian demo.
- `demo_video_v2_ui_only_20260704_video2/`
  - `script.md`: VIDEO #2 director script for the main v2 ore-pipeline UI,
    using sample `2550382-1-10x`, disabled augmentation/preprocessing, screenshot
    markup, and reference captures under `screenshots/`.
- `demo_video_v2_ui_20260704/`
  - `nornikel_v2_ui_demo_1080p_ru.mp4`: 5:00 broader v2 UI demo with talc-review context.

Each bundle keeps the generation and review materials beside the video:
`build_video.py`, `script_ru.md`, `subtitles_ru.srt`, `timeline.json`,
`build_metadata.json`, `contact_sheet.jpg`, Whisper STT output under `stt/`,
and `stt_verification_report.md`.
The Talc bundle also includes `talc_annotation_script_ru.md`, a sample-specific
director script for `/sample/2550382-1-10x` with screenshot capture points,
required UI state, and Russian narration for re-recording the Talc annotation
walkthrough.
The VIDEO #2 bundle is script-first rather than a rendered MP4 bundle: it stores
the exact UI navigation, screenshot filenames, and run ids needed to re-record
the requested `apps/ore_pipeline_web.py` walkthrough.

`manifest.sha256` records checksums for all MP4 files in this directory tree.
`stt_refresh_verification_20260704.md` summarizes the fresh Whisper transcripts
saved under each bundle's `stt_refresh_20260704/` folder.
`METADATA.md` contains upload-ready titles, descriptions with timecodes,
categories, and access settings for the final videos.

## Regeneration

```bash
scripts/generate_talc_ui_video.sh
CHECK_ONLY=1 RUN_STT=0 scripts/generate_talc_ui_video.sh
PULL_RESULT=1 scripts/generate_talc_ui_video_gx10.sh
```

The non-Talc bundles can be regenerated directly from their local renderers:

```bash
python presentation/videos/demo_video_v2_ui_only_20260704/build_video.py
python presentation/videos/demo_video_v2_ui_20260704/build_video.py
```
