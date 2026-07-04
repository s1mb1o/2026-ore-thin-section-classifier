# Talc UI video verification

Date: 2026-07-04

Source script: `docs/ui/v2/notes/2026-07-04-talc-ui-video-script-ru.md`

Final video: `presentation/videos/demo_video_talc_ui_20260704/nornikel_talc_ui_demo_1080p_ru.mp4`

## Render properties

- Duration: `300.000000` seconds.
- Video: H.264, `1920x1080`, `24 fps`.
- Audio: AAC, `48000 Hz`, stereo.
- Voice: `ru-RU-DmitryNeural`, Russian male TTS.
- Subtitles: burned into the MP4 and exported as `subtitles_ru.srt`.
- Synchronization marks: top-right timestamp/title marker on every frame.
- Raw TTS duration before final fit: `289.488` seconds.
- Final audio fit tempo: `0.968187`, so the voice is only slightly slowed to match the 5-minute target.

## Visual review

Reviewed `contact_sheet.jpg` sampled every 30 seconds from the final MP4.

The video is Talc UI-only. It shows the selected Talc sample, original photo, MS Paint annotation, converter QA overlay, segmentation class widget, Brush, Fill, Rectangle, Polygon, Similar preview, SAM2, Dark pixel preview, Talc cluster areas, sulfide protection, Save / Save & Next controls, and the reviewed-mask result slide.

English UI labels remain visible where they are part of the actual interface, but the spoken narration uses Russian equivalents instead of reading those labels aloud.

No other applications are visible in the final sampled frames.

## STT review

STT command:

```bash
/Users/ashmelev/.local/bin/whisper presentation/videos/demo_video_talc_ui_20260704/nornikel_talc_ui_demo_1080p_ru.mp4 --model base --language ru --output_dir presentation/videos/demo_video_talc_ui_20260704/stt --output_format all
```

Transcript: `presentation/videos/demo_video_talc_ui_20260704/stt/nornikel_talc_ui_demo_1080p_ru.txt`

Measured against `timeline.json` narration:

- Script words: `510`.
- STT words: `509`.
- Matched script words: `476`.
- Normalized word similarity: `0.9333`.
- Sequence ratio by words: `0.9342`.
- Normalized character similarity: `0.5481`.
- Latin words in script narration: `0`.
- Latin words in burned subtitles: `0`.
- Latin words in Whisper transcript: `0`.

The lower character score is caused by ordinary Russian ASR spelling differences and inflection mistakes, not by English UI labels. Key concepts are present in the transcript: talc, blue-line draft annotation, mask editing, sulfides, brush, fill, similar-pixel preview, dark-pixel preview, save flow, and training output.

## Checklist

- Blue lines are explained as draft manual annotation, not a finished mask: yes.
- Background modes `Original photo`, `MS Paint annotation`, and `Converter QA overlay` are shown: yes.
- `Positive bag` is described as a probable talc container, not confirmed talc: yes.
- `Talc` is described as confirmed pixels used for training and evaluation: yes.
- `Talc cluster areas` is described as visual-only and not a saved training class: yes.
- Brush left-draw / right-erase behavior is described: yes.
- `Fill` is described as bounded-area filling: yes.
- `Rectangle` and `Polygon` are described as editable filled geometry before save: yes.
- `Similar` is described as preview-only until applied/saved: yes.
- `SAM2` is described as optional assistant behavior: yes.
- `Dark pixel preview threshold` is described as non-destructive: yes.
- Sulfide protection and `Subtract sulfides from mask` are described: yes.
- `Save`, `Save & Next`, `Next`, and reviewed-mask outputs are described: yes.

## Verdict

Pass. The video content, visible UI states, Russian-only narration, burned subtitles, and STT transcript match the requested Talc UI script. The previous issue with spoken English UI labels is resolved.
