#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_SELECTION_REBUILD:-}" != "YES" ]]; then
  echo "Refusing to rebuild V2/V3/V4 selection artifacts." >&2
  echo "Set CONFIRM_SELECTION_REBUILD=YES after reviewing the configs." >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TOOLS_DIR="${PROJECT_ROOT}/selection/tools"
CONFIG_DIR="${PROJECT_ROOT}/selection/configs"
PYTHON_BIN="${PYTHON_BIN:-/home/dell/anaconda3/envs/py38/bin/python}"

"${PYTHON_BIN}" "${TOOLS_DIR}/scan_amass.py" \
  --config "${CONFIG_DIR}/amass_v2.yaml" \
  --overwrite
"${PYTHON_BIN}" "${TOOLS_DIR}/build_v3.py" \
  --config "${CONFIG_DIR}/amass_v3.yaml" \
  --overwrite
"${PYTHON_BIN}" "${TOOLS_DIR}/build_v4.py" \
  --config "${CONFIG_DIR}/amass_v4.yaml" \
  --overwrite

actual_count="$(wc -l < "${PROJECT_ROOT}/selection/artifacts/v4/candidate_v4.jsonl")"
if [[ "${actual_count}" != "515" ]]; then
  echo "Expected 515 V4 candidates, got ${actual_count}." >&2
  exit 1
fi
sha256sum "${PROJECT_ROOT}/selection/artifacts/v4/candidate_v4.jsonl"
