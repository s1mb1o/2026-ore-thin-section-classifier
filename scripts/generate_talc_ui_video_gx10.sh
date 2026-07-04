#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-${ROOT_DIR}/presentation/videos/demo_video_talc_ui_20260704}"
GX10_HOST="${GX10_HOST:-ashmelev@192.168.86.14}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/ashmelev/Projects/nornikel_tts_video}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-${REMOTE_ROOT}/demo_video_talc_ui_20260704}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/home/ashmelev/Projects/benchmark_venv/bin/python3}"
TTS_BACKEND="${TTS_BACKEND:-silero}"
SILERO_SPEAKER="${SILERO_SPEAKER:-aidar}"
PULL_RESULT="${PULL_RESULT:-0}"

FINAL_MP4="nornikel_talc_ui_demo_1080p_ru.mp4"

die() {
  echo "error: $*" >&2
  exit 1
}

[[ -f "${LOCAL_OUTPUT_DIR}/build_video.py" ]] || die "missing generator: ${LOCAL_OUTPUT_DIR}/build_video.py"
compgen -G "${LOCAL_OUTPUT_DIR}/scene_*.png" >/dev/null || die "missing scene screenshots in ${LOCAL_OUTPUT_DIR}"

ssh "${GX10_HOST}" "mkdir -p '${REMOTE_OUTPUT_DIR}' '${REMOTE_ROOT}/logs'"

rsync -av \
  "${LOCAL_OUTPUT_DIR}/build_video.py" \
  "${LOCAL_OUTPUT_DIR}"/scene_*.png \
  "${GX10_HOST}:${REMOTE_OUTPUT_DIR}/"

ssh "${GX10_HOST}" "
  set -euo pipefail
  '${REMOTE_PYTHON}' -m pip install --quiet imageio-ffmpeg
  cd '${REMOTE_OUTPUT_DIR}'
  TTS_BACKEND='${TTS_BACKEND}' SILERO_SPEAKER='${SILERO_SPEAKER}' '${REMOTE_PYTHON}' build_video.py
  '${REMOTE_PYTHON}' - <<'PY'
import json
import re
import wave
from pathlib import Path

root = Path('.')
meta = json.loads((root / 'build_metadata.json').read_text())
timeline = json.loads((root / 'timeline.json').read_text())
with wave.open(str(root / 'narration_300s.wav'), 'rb') as wav:
    wav_seconds = wav.getnframes() / wav.getframerate()
subtitle_latin = len(re.findall(r'[A-Za-z]+', (root / 'subtitles_ru.srt').read_text()))
final_mp4 = Path(meta['final_mp4'])
print(f\"validated: {final_mp4}\")
print(f\"  exists={final_mp4.exists()} size={final_mp4.stat().st_size if final_mp4.exists() else 0}\")
print(f\"  duration={meta['duration_seconds']} audio_seconds={wav_seconds}\")
print(f\"  tts_backend={timeline.get('tts_backend')} silero_speaker={timeline.get('silero_speaker')}\")
print(f\"  subtitle_latin_words={subtitle_latin}\")
if not final_mp4.exists() or abs(float(meta['duration_seconds']) - 300.0) > 0.05:
    raise SystemExit(1)
if abs(wav_seconds - 300.0) > 0.05 or subtitle_latin:
    raise SystemExit(1)
PY
"

if [[ "${PULL_RESULT}" == "1" ]]; then
  rsync -av \
    "${GX10_HOST}:${REMOTE_OUTPUT_DIR}/${FINAL_MP4}" \
    "${LOCAL_OUTPUT_DIR}/nornikel_talc_ui_demo_1080p_ru_gx10_${TTS_BACKEND}_${SILERO_SPEAKER}.mp4"
fi

echo "gx10 video: ${GX10_HOST}:${REMOTE_OUTPUT_DIR}/${FINAL_MP4}"
