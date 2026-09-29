#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
GENERATION_DIR="${PROJECT_ROOT}/generation"

PYTHON_BIN="${PYTHON_BIN:-/home/dell/anaconda3/envs/py38/bin/python}"
CANONICAL_ROOT="${CANONICAL_ROOT:-${PROJECT_ROOT}/work/canonical}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/work/ply}"
ASSIGNMENT_MANIFEST="${ASSIGNMENT_MANIFEST:-${CANONICAL_ROOT}/generation_manifests/avatar_assignment_v2.jsonl}"

"${PYTHON_BIN}" "${GENERATION_DIR}/avatar_batch_status.py" \
    --output-root "${OUTPUT_ROOT}" \
    --assignment-manifest "${ASSIGNMENT_MANIFEST}"
