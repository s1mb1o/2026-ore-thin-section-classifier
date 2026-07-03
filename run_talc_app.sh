#!/usr/bin/env bash
# Launch the talc mask review web app (apps/talc_review_web.py).
#
# Defaults to the prepared conversion workspace at
# outputs/talc_blue_line_conversion (contains manifest.json).
#
# Usage:
#   ./run_talc_app.sh                     # prepared-workspace mode, OS-assigned port
#   ./run_talc_app.sh --port 8081         # fixed port
#   ./run_talc_app.sh --reconvert         # regenerate the conversion workspace
#   ./run_talc_app.sh --help              # app help
#
# To start from the raw MS Paint annotation folder instead, pass:
#   ./run_talc_app.sh --annotated-dir "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования"
#
# Any extra arguments are passed straight through to the app.
# The app prints the selected local URL on startup.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Prefer the repo-local venv, fall back to python3 on PATH.
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="python3"
fi

CONVERSION_DIR="${TALC_CONVERSION_DIR:-outputs/talc_blue_line_conversion}"

# If the caller supplies their own source flag, don't force --conversion-dir.
source_override=0
for arg in "$@"; do
  case "$arg" in
    --conversion-dir|--conversion-dir=*|--annotated-dir|--annotated-dir=*) source_override=1 ;;
  esac
done

if [[ "$source_override" -eq 1 ]]; then
  exec "$PY" apps/talc_review_web.py \
    --host "${TALC_HOST:-127.0.0.1}" \
    --port "${TALC_PORT:-0}" \
    "$@"
else
  exec "$PY" apps/talc_review_web.py \
    --conversion-dir "$CONVERSION_DIR" \
    --host "${TALC_HOST:-127.0.0.1}" \
    --port "${TALC_PORT:-0}" \
    "$@"
fi
