#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
GENERATION_DIR="${PROJECT_ROOT}/generation"

PYTHON_BIN="${PYTHON_BIN:-/home/dell/anaconda3/envs/py38/bin/python}"
CANDIDATE_MANIFEST="${CANDIDATE_MANIFEST:-${PROJECT_ROOT}/selection/artifacts/v4/candidate_v4.jsonl}"
AVATAR_ROOT="${AVATAR_ROOT:-${PROJECT_ROOT}/assets/avatars}"
MANIFEST_DIR="${MANIFEST_DIR:-${PROJECT_ROOT}/work/canonical/generation_manifests}"
ASSIGNMENT_MANIFEST="${ASSIGNMENT_MANIFEST:-${MANIFEST_DIR}/avatar_assignment_v2.jsonl}"
ASSIGNMENT_SUMMARY="${ASSIGNMENT_SUMMARY:-${MANIFEST_DIR}/avatar_assignment_v2_summary.json}"
ASSIGNMENT_VERSION="${ASSIGNMENT_VERSION:-avatar_assignment_v2}"
ASSIGNMENT_SEED="${ASSIGNMENT_SEED:-xavatar-stage2-v2-2026-07-26}"

"${PYTHON_BIN}" "${GENERATION_DIR}/build_avatar_assignment.py" \
    --candidate-manifest "${CANDIDATE_MANIFEST}" \
    --avatar-root "${AVATAR_ROOT}" \
    --output "${ASSIGNMENT_MANIFEST}" \
    --summary "${ASSIGNMENT_SUMMARY}" \
    --assignment-version "${ASSIGNMENT_VERSION}" \
    --seed "${ASSIGNMENT_SEED}"
