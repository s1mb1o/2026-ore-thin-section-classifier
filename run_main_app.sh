#!/usr/bin/env bash
# Launch the main end-to-end ore pipeline web UI (apps/ore_pipeline_web.py).
#
# Usage:
#   ./run_main_app.sh                     # heuristic backend, OS-assigned port
#   ./run_main_app.sh --port 8080         # fixed port
#   ./run_main_app.sh --backend ml        # ML sulfide backend (uses B2 checkpoint by default)
#   ./run_main_app.sh --help              # app help
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

exec "$PY" apps/ore_pipeline_web.py \
  --host "${ORE_HOST:-127.0.0.1}" \
  --port "${ORE_PORT:-0}" \
  "$@"
