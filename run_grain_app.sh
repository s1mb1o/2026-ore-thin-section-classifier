#!/usr/bin/env bash
# Launch the grain review / labeling web app (apps/grain_review_web.py).
#
# Defaults to the grain dataset at outputs/grain_dataset_v0 (grains_manifest.csv
# + crops/, produced by scripts/build_grain_dataset.py). Shows each grain's
# feature report + heuristic reason so the annotator can decide ordinary vs fine.
#
# Usage:
#   ./run_grain_app.sh                                  # default dataset, OS-assigned port
#   ./run_grain_app.sh --port 8082                      # fixed port
#   ./run_grain_app.sh --dataset-dir outputs/grain_dataset_v1
#   ./run_grain_app.sh --help                           # app help
#
# Env overrides: GRAIN_HOST, GRAIN_PORT, GRAIN_DATASET_DIR.
# Any extra arguments are passed straight through. The app prints the local URL.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Prefer the repo-local venv, fall back to python3 on PATH.
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="python3"
fi

DATASET_DIR="${GRAIN_DATASET_DIR:-outputs/grain_dataset_v0}"

# If the caller supplies their own --dataset-dir, don't force the default.
dataset_override=0
for arg in "$@"; do
  case "$arg" in
    --dataset-dir|--dataset-dir=*) dataset_override=1 ;;
  esac
done

if [[ "$dataset_override" -eq 1 ]]; then
  exec "$PY" apps/grain_review_web.py \
    --host "${GRAIN_HOST:-127.0.0.1}" \
    --port "${GRAIN_PORT:-0}" \
    "$@"
else
  exec "$PY" apps/grain_review_web.py \
    --dataset-dir "$DATASET_DIR" \
    --host "${GRAIN_HOST:-127.0.0.1}" \
    --port "${GRAIN_PORT:-0}" \
    "$@"
fi
