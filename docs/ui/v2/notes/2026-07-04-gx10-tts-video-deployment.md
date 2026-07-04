# gx10 TTS video deployment

Date: 2026-07-04

Scope: dedicated local TTS/video generation on `gx10` for the Talc UI demo. This
does not deploy or restart the ore pipeline web UI on `:8210`.

## Remote layout

- Host: `ashmelev@192.168.86.14`
- Project root: `/home/ashmelev/Projects/nornikel_tts_video`
- Talc demo generator: `/home/ashmelev/Projects/nornikel_tts_video/demo_video_talc_ui_20260704`
- Python runtime: `/home/ashmelev/Projects/benchmark_venv/bin/python3`
- Shared model caches:
  - Silero: `/home/ashmelev/.cache/torch/hub`
  - Hugging Face: `/home/ashmelev/.cache/huggingface/hub`

## Working local TTS backend

The generated Talc UI video now works on gx10 with a local Silero Russian TTS
backend:

- Model loader: `torch.hub.load("snakers4/silero-models", "silero_tts", language="ru", speaker="v4_ru")`
- Speaker: `aidar` (Russian male)
- Device: CPU by default, to avoid competing with active GPU evaluation jobs.
- Audio output: per-sentence WAV files, then a 300-second AAC track in the MP4.
- Video muxing: user-space ffmpeg from `imageio-ffmpeg`, because system ffmpeg is
  not installed and sudo is unavailable.

Validated gx10 output:

- MP4: `/home/ashmelev/Projects/nornikel_tts_video/demo_video_talc_ui_20260704/nornikel_talc_ui_demo_1080p_ru.mp4`
- Duration: `300.0` seconds
- Video: `1920x1080`, H.264, `24 fps`
- Audio: AAC, `48000 Hz`, stereo
- Subtitles: Russian-only, `0` Latin words
- `timeline.json`: `tts_backend=silero`, `silero_speaker=aidar`
- Raw Silero narration before final fit: `255.6` seconds
- Final tempo: `0.854849`
- Local pulled copy and STT report:
  `presentation/videos/demo_video_talc_ui_20260704/gx10_silero/stt_verification_report.md`

## Regeneration command

From the v2 repo on the Mac:

```bash
scripts/generate_talc_ui_video_gx10.sh
```

To also pull the generated MP4 back into the local output directory:

```bash
PULL_RESULT=1 scripts/generate_talc_ui_video_gx10.sh
```

Optional overrides:

```bash
SILERO_SPEAKER=eugene scripts/generate_talc_ui_video_gx10.sh
GX10_HOST=ashmelev@192.168.86.14 scripts/generate_talc_ui_video_gx10.sh
```

## Higher-quality model assets

Silero is the deployed, verified fallback because it is stable on gx10's existing
Python 3.12 / PyTorch CUDA 13 environment and does not need `torchaudio`.

For higher-end TTS experiments, model snapshots are cached under the shared
Hugging Face cache rather than the project directory. Candidate assets:

- `coqui/XTTS-v2`: multilingual voice-cloning TTS with Russian support.
  - Cache size: `2.1G`
  - Snapshot:
    `/home/ashmelev/.cache/huggingface/hub/models--coqui--XTTS-v2/snapshots/6c2b0d75eae4b7047358e3b6bd9325f857d43f77`
- `hotstone228/F5-TTS-Russian`: Russian/English F5-TTS fine-tune; non-commercial
  ShareAlike license.
  - Cache size: `4.7G`
  - Snapshot:
    `/home/ashmelev/.cache/huggingface/hub/models--hotstone228--F5-TTS-Russian/snapshots/4b9fcb51e68b0b7e96dbc8c9df3d80b4a835b914`

These are model assets for the next quality pass. They still need a separate
runtime integration step before replacing the Silero backend in `build_video.py`.

## Notes

- The old `edge_tts` path remains available locally as the default backend, but
  it is not local/offline and should not be treated as the gx10 deployment path.
- `TTS_BACKEND=silero` is the gx10 path.
- The current Silero narration is functional and fully local, but it is shorter
  than the five-minute target before fitting, so the final audio is slowed by
  about 17%. For best naturalness, the next pass should wire XTTS-v2 or F5-TTS
  and use shorter sentence chunks/reference audio as required by those models.
