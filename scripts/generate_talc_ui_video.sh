#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-${ROOT_DIR}/outputs/demo_video_talc_ui_20260704}"
if [[ "${OUTPUT_DIR}" != /* ]]; then
  OUTPUT_DIR="${ROOT_DIR}/${OUTPUT_DIR}"
fi
RENDERER="${OUTPUT_DIR}/build_video.py"
FINAL_MP4="${OUTPUT_DIR}/nornikel_talc_ui_demo_1080p_ru.mp4"
CONTACT_SHEET="${OUTPUT_DIR}/contact_sheet.jpg"
CHECK_ONLY="${CHECK_ONLY:-0}"
RUN_STT="${RUN_STT:-1}"
WHISPER_BIN="${WHISPER_BIN:-/Users/ashmelev/.local/bin/whisper}"

if [[ -x /tmp/nornikel_demo_tts_venv/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-/tmp/nornikel_demo_tts_venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

required_scenes=(
  scene_00_intro_original.png
  scene_01_ms_paint_annotation.png
  scene_02_converter_qa_overlay.png
  scene_03_classes_clusters.png
  scene_04_brush_talc_target.png
  scene_05_fill_tool.png
  scene_06_rectangle_tool.png
  scene_07_polygon_tool.png
  scene_08_similar_preview.png
  scene_09_sam2_dark_preview.png
  scene_10_quality_sulfides_clusters.png
  scene_11_save_controls_result.png
)

die() {
  echo "error: $*" >&2
  exit 1
}

need_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

check_inputs() {
  [[ -f "${RENDERER}" ]] || die "renderer not found: ${RENDERER}"
  for scene in "${required_scenes[@]}"; do
    [[ -f "${OUTPUT_DIR}/${scene}" ]] || die "missing scene screenshot: ${OUTPUT_DIR}/${scene}"
  done
}

probe_stream_field() {
  local stream_selector="$1"
  local entries="$2"
  ffprobe -v error -select_streams "${stream_selector}" -show_entries "${entries}" -of default=noprint_wrappers=1:nokey=1 "${FINAL_MP4}"
}

validate_video() {
  [[ -f "${FINAL_MP4}" ]] || die "final MP4 was not produced: ${FINAL_MP4}"

  local width height video_codec audio_codec duration
  video_codec="$(probe_stream_field v:0 stream=codec_name | head -1)"
  width="$(probe_stream_field v:0 stream=width | head -1)"
  height="$(probe_stream_field v:0 stream=height | head -1)"
  audio_codec="$(probe_stream_field a:0 stream=codec_name | head -1)"
  duration="$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${FINAL_MP4}")"

  [[ "${video_codec}" == "h264" ]] || die "unexpected video codec: ${video_codec}"
  [[ "${width}" == "1920" && "${height}" == "1080" ]] || die "unexpected resolution: ${width}x${height}"
  [[ "${audio_codec}" == "aac" ]] || die "unexpected audio codec: ${audio_codec}"
  [[ "${duration}" == "300.000000" ]] || die "unexpected duration: ${duration}"

  echo "validated: ${FINAL_MP4}"
  echo "  duration=${duration}, video=${video_codec} ${width}x${height}, audio=${audio_codec}"
}

make_contact_sheet() {
  ffmpeg -y \
    -i "${FINAL_MP4}" \
    -vf "fps=1/30,scale=384:216,tile=5x2" \
    -frames:v 1 \
    -update 1 \
    "${CONTACT_SHEET}" >/dev/null 2>&1
  echo "contact sheet: ${CONTACT_SHEET}"
}

run_stt() {
  if [[ "${RUN_STT}" != "1" ]]; then
    echo "STT skipped: RUN_STT=${RUN_STT}"
    return
  fi
  if [[ ! -x "${WHISPER_BIN}" ]]; then
    echo "STT skipped: whisper not found at ${WHISPER_BIN}"
    return
  fi

  mkdir -p "${OUTPUT_DIR}/stt"
  "${WHISPER_BIN}" "${FINAL_MP4}" \
    --model base \
    --language ru \
    --output_dir "${OUTPUT_DIR}/stt" \
    --output_format all \
    > "${OUTPUT_DIR}/stt/whisper_base.log" 2>&1
  echo "STT outputs: ${OUTPUT_DIR}/stt"
}

main() {
  cd "${ROOT_DIR}"
  need_command ffmpeg
  need_command ffprobe
  check_inputs

  echo "renderer: ${RENDERER}"
  echo "python: ${PYTHON_BIN}"
  if [[ "${CHECK_ONLY}" == "1" ]]; then
    echo "CHECK_ONLY=1, skipping render"
  else
    "${PYTHON_BIN}" "${RENDERER}"
  fi

  validate_video
  make_contact_sheet
  run_stt
}

main "$@"
