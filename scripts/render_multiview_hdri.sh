#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BLENDER_BIN="${BLENDER_BIN:-/data/blender-3.6.21-linux-x64/blender}"
PLY_ROOT="${PLY_ROOT:-/data/h2o_data/h2o_avatar_ply}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/work/multiview_hdri}"
HDRI_ROOT="${HDRI_ROOT:-/data/HandSynthesis_dataset/hdris_4k}"

if [[ ! -x "${BLENDER_BIN}" ]]; then
  echo "Blender executable not found: ${BLENDER_BIN}" >&2
  exit 2
fi
if [[ ! -d "${HDRI_ROOT}" ]]; then
  echo "HDRI directory not found: ${HDRI_ROOT}" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
exec "${BLENDER_BIN}" --background --factory-startup \
  --python generation/render_multiview_hdri.py -- \
  --ply-root "${PLY_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --hdri-root "${HDRI_ROOT}" \
  "$@"
