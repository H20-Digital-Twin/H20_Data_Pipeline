#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_MULTIVIEW_FULL_RUN:-}" != "YES" ]]; then
  echo "Refusing the 515-motion RGB PNG + depth TIFF + mask PNG run." >&2
  echo "Re-run with CONFIRM_MULTIVIEW_FULL_RUN=YES." >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dell/anaconda3/envs/py38/bin/python}"
PLY_ROOT="${PLY_ROOT:-${PROJECT_ROOT}/work/ply}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/work/multiview}"
CANDIDATE_MANIFEST="${CANDIDATE_MANIFEST:-${PROJECT_ROOT}/selection/artifacts/v4/candidate_v4.jsonl}"

if ! mountpoint -q "/data"; then
  echo "Local data volume is not mounted: /data" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/logs"
LOG_PATH="${OUTPUT_ROOT}/logs/multiview_tiff_full_$(date +%Y%m%d_%H%M%S).log"
echo "Writing log to ${LOG_PATH}"
echo "Output formats: RGB=uint8 PNG, depth=float32 TIFF metres, mask=uint8 PNG"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" \
  generation/render_multiview.py \
  --candidate-jsonl "${CANDIDATE_MANIFEST}" \
  --ply-root "${PLY_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --all \
  --num-cameras 4 \
  --width 512 \
  --height 512 \
  --yfov-deg 30 \
  --framing-margin 1.08 \
  --depth-format tiff \
  --resume 2>&1 | tee "${LOG_PATH}"
