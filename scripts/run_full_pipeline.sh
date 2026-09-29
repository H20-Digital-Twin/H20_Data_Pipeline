#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_FULL_PIPELINE:-}" != "YES" ]]; then
  echo "Refusing the full 515-motion pipeline." >&2
  echo "Set CONFIRM_FULL_PIPELINE=YES to continue." >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ "${REBUILD_SELECTION:-NO}" == "YES" ]]; then
  CONFIRM_SELECTION_REBUILD=YES "${SCRIPT_DIR}/build_candidates_all.sh"
fi

CONFIRM_FULL_RUN=YES "${SCRIPT_DIR}/build_canonical_all.sh"
"${SCRIPT_DIR}/prepare_avatar_assignment.sh"
CONFIRM_FULL_RUN=YES "${SCRIPT_DIR}/generate_ply_all.sh"
"${SCRIPT_DIR}/ply_status.sh"
CONFIRM_MULTIVIEW_FULL_RUN=YES "${SCRIPT_DIR}/render_multiview_all.sh"

echo "FULL PIPELINE COMPLETED"
echo "project_root=${PROJECT_ROOT}"
echo "candidate_manifest=${PROJECT_ROOT}/selection/artifacts/v4/candidate_v4.jsonl"
echo "ply_root=${PROJECT_ROOT}/work/ply"
echo "multiview_root=${PROJECT_ROOT}/work/multiview"
