# gx10 Silero Talc UI video verification

Date: 2026-07-04

Remote source:
`ashmelev@192.168.86.14:/home/ashmelev/Projects/nornikel_tts_video/demo_video_talc_ui_20260704/nornikel_talc_ui_demo_1080p_ru.mp4`

Local copy:
`presentation/videos/demo_video_talc_ui_20260704/gx10_silero/nornikel_talc_ui_demo_1080p_ru.mp4`

## Render properties

- Duration: `300.0` seconds.
- Video: H.264, `1920x1080`, `24 fps`.
- Audio: AAC, `48000 Hz`, stereo.
- TTS backend: `silero`.
- Speaker: `aidar` Russian male voice.
- Raw narration duration before final fit: `255.6` seconds.
- Final audio fit tempo: `0.854849`.
- Subtitles: burned into the MP4 and exported as `subtitles_ru.srt`.

## STT review

STT command:

```bash
/Users/ashmelev/.local/bin/whisper presentation/videos/demo_video_talc_ui_20260704/gx10_silero/nornikel_talc_ui_demo_1080p_ru.mp4 --model base --language ru --output_dir presentation/videos/demo_video_talc_ui_20260704/gx10_silero/stt --output_format all
```

Transcript:
`presentation/videos/demo_video_talc_ui_20260704/gx10_silero/stt/nornikel_talc_ui_demo_1080p_ru.txt`

Measured against `timeline.json` narration:

- Script words: `510`.
- STT words: `508`.
- Matched script words: `474`.
- Normalized word similarity: `0.9294`.
- Sequence ratio by words: `0.9312`.
- Latin words in script narration: `0`.
- Latin words in burned subtitles: `0`.
- Latin words in Whisper transcript: `0`.

Key concepts are present in the transcript: talc, blue-line draft annotation,
mask editing, sulfides, brush, fill, similar-pixel preview, dark-pixel preview,
save flow, and training output.

## Verdict

Pass for deployment verification. The gx10-local Silero render is complete and
the spoken/subtitle text no longer contains English UI labels. The voice is
functional but less natural than the earlier Edge render because Silero produced
a shorter raw narration that had to be slowed to five minutes; XTTS-v2 or
F5-TTS-Russian should be wired in for the next quality pass.
