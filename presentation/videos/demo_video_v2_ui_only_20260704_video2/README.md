# VIDEO #2 Ore Pipeline UI Script

This directory contains the director script for a second main v2 ore-pipeline UI
recording based on the previous `demo_video_v2_ui_only_20260704` bundle.

Primary artifact:

- `nornikel_v2_ui_video2_1080p_ru.mp4` - rendered 8:00 Russian HD1080
  walkthrough with burned-in Russian subtitles and `ru-RU-DmitryNeural`
  narration at Edge TTS rate `-12%`.
- `script.md` - step-by-step operator/narration script with screenshot markup.
- `build_video.py` - local renderer for the Russian MP4 from the HD1080
  screenshots and scene timeline. It uses `edge_tts` with
  `ru-RU-DmitryNeural`.

Repeat rendering requires the optional Python package `edge-tts` in the active
virtualenv.

Render artifacts:

- `subtitles_ru.srt` - Russian subtitle track used for the burned-in captions.
- `timeline_ru.json` - scene timing, text, screenshot, and raw TTS duration
  metadata.
- `build_metadata_ru.json` - ffprobe/build metadata for the final MP4.
- `contact_sheet_ru.jpg` - visual contact sheet of the rendered Russian frames.
- `narration_480s_ru.wav` - padded 8:00 Russian narration bed.
- `tts_sentences_ru_dmitry/` - cached raw Edge TTS sentence MP3 files.
- `video_with_subs_ru.mp4` - silent subtitle-burned draft used before muxing
  the final AAC audio.

Reference captures:

- `screenshots/` - actual local browser captures from the verification run,
  one PNG per scene, each exactly `1920x1080`.
- `all_screenshots_review_sheet.png` - single review image with all 15
  screenshots at 50% scale; the source PNG files remain full HD frames.

Verification run:

- workspace: `outputs/ore_pipeline_ui_video2_20260705`
- sample: `dataset/Фото руд по сортам. ч1/Оталькованные руды/2550382-1 10x.JPG`
- parent run: `run_20260705_013329_516251000_1643dd89`
- artifact-edit child run: `edit_20260705_013353_461997000_0088678f`
- series: `batch_20260705_013902_291751000`

The run was recorded with both `Augmentation` and `Preprocessing` disabled.
