# STT Refresh Verification

Date: 2026-07-04

Scope: refreshed Whisper `base` Russian STT text for the final MP4 artifacts
after moving and regenerating the video bundles under `presentation/videos/`.

## Inputs

- `demo_video_talc_ui_20260704/nornikel_talc_ui_demo_1080p_ru.mp4`
- `demo_video_talc_ui_20260704/gx10_silero/nornikel_talc_ui_demo_1080p_ru.mp4`
- `demo_video_v2_ui_only_20260704/nornikel_v2_ui_only_demo_1080p_ru.mp4`
- `demo_video_v2_ui_20260704/nornikel_v2_ui_demo_1080p_ru.mp4`

## STT Outputs

- `demo_video_talc_ui_20260704/stt_refresh_20260704/nornikel_talc_ui_demo_1080p_ru.txt`
- `demo_video_talc_ui_20260704/gx10_silero/stt_refresh_20260704/nornikel_talc_ui_demo_1080p_ru.txt`
- `demo_video_v2_ui_only_20260704/stt_refresh_20260704/nornikel_v2_ui_only_demo_1080p_ru.txt`
- `demo_video_v2_ui_20260704/stt_refresh_20260704/nornikel_v2_ui_demo_1080p_ru.txt`

## Text Match Summary

| Video | Script words | STT words | Word sequence ratio | Latin words in script | Latin words in STT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Talc UI Edge | 510 | 509 | 0.9342 | 0 | 0 |
| Talc UI gx10 Silero | 510 | 508 | 0.9116 | 0 | 0 |
| v2 UI-only | 675 | 686 | 0.8038 | 57 | 8 |
| v2 UI broad | 625 | 591 | 0.7385 | 54 | 20 |

The Talc UI videos remain Russian-only in both script and STT transcript. The
v2 UI videos contain visible English UI/technical labels in the script and
subtitles; Whisper often transcribes those terms phonetically or drops them,
which lowers the raw word-ratio metric. Manual contact-sheet review confirms
the video frames stay on the intended UI screens, with burned-in Russian
subtitles and corner time marks.
