#!/usr/bin/env sh
set -eu

if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
  exec "$@"
fi

workspace="${ORE_UI_WORKSPACE:-/data/ore_pipeline_ui}"
mkdir -p "$workspace"

set -- \
  python3 apps/ore_pipeline_web.py \
  --host "${ORE_UI_HOST:-0.0.0.0}" \
  --port "${ORE_UI_PORT:-8080}" \
  --workspace-dir "$workspace" \
  --backend "${ORE_UI_BACKEND:-heuristic}" \
  --processing-max-side "${ORE_UI_PROCESSING_MAX_SIDE:-2600}" \
  --panorama-max-side "${ORE_UI_PANORAMA_MAX_SIDE:-1800}" \
  --preview-max-sides "${ORE_UI_PREVIEW_MAX_SIDES:-1024,2048,4096}" \
  "$@"

if [ -n "${ORE_UI_CHECKPOINT:-}" ]; then
  set -- "$@" --checkpoint "$ORE_UI_CHECKPOINT"
fi

exec "$@"
