#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
GENERATION_DIR="${PROJECT_ROOT}/generation"

if [[ "${CONFIRM_FULL_RUN:-}" != "YES" ]]; then
    echo "Refusing to prepare all 515 motions." >&2
    echo "Set CONFIRM_FULL_RUN=YES after the pilot is approved." >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/home/dell/anaconda3/envs/py38/bin/python}"
CANONICAL_ROOT="${CANONICAL_ROOT:-${PROJECT_ROOT}/work/canonical}"
CANDIDATE_MANIFEST="${CANDIDATE_MANIFEST:-${PROJECT_ROOT}/selection/artifacts/v4/candidate_v4.jsonl}"
AMASS_ROOT="${AMASS_ROOT:-/data/motions/AMASS/smpl-x-n}"

"${PYTHON_BIN}" "${GENERATION_DIR}/build_canonical_motion.py" \
    --candidate-jsonl "${CANDIDATE_MANIFEST}" \
    --amass-root "${AMASS_ROOT}" \
    --output-root "${CANONICAL_ROOT}" \
    --all \
    --resume
