#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dell/anaconda3/envs/py38/bin/python}"
PLY_ROOT="${PLY_ROOT:-${PROJECT_ROOT}/work/ply}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/work/multiview}"
CANDIDATE_MANIFEST="${CANDIDATE_MANIFEST:-${PROJECT_ROOT}/selection/artifacts/v4/candidate_v4.jsonl}"

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" \
  generation/render_multiview.py \
  --candidate-jsonl "${CANDIDATE_MANIFEST}" \
  --ply-root "${PLY_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  "$@" \
  --depth-format tiff \
  --validate-only
