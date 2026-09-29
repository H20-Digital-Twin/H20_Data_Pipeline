#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dell/anaconda3/envs/py38/bin/python}"
CANDIDATE="${PROJECT_ROOT}/selection/artifacts/v4/candidate_v4.jsonl"

test "$(wc -l < "${CANDIDATE}")" = 515
test "$(sha256sum "${CANDIDATE}" | awk '{print $1}')" = \
  55279a1088fbfc71f93f41fd7577e92bc358fed57d302fcef889e2bf3cdede1e

for avatar in 00025_scan 00034_rgbd 00085_scan 00020_scan 00027_scan 00087_rgbd; do
  test -d "${PROJECT_ROOT}/assets/avatars/${avatar}"
  test ! -L "${PROJECT_ROOT}/assets/avatars/${avatar}"
  test -L "${PROJECT_ROOT}/assets/avatars/${avatar}/checkpoints/last.pth"
  test -f "${PROJECT_ROOT}/assets/avatars/${avatar}/checkpoints/last.pth"
  test -f "${PROJECT_ROOT}/assets/avatars/${avatar}/meta_info.npz"
done

test -f "${PROJECT_ROOT}/vendor/xavatar/code/lib/libmise/mise.cpython-38-x86_64-linux-gnu.so"

"${PYTHON_BIN}" -m py_compile \
  "${PROJECT_ROOT}"/selection/tools/*.py \
  "${PROJECT_ROOT}"/generation/*.py

if rg -n '/data/X-Avatar-main|/home/dell/data_link/X-Avatar-main' \
  "${PROJECT_ROOT}" \
  --glob '*.py' --glob '*.sh' --glob '*.yaml' \
  --glob '!preflight.sh'; then
  echo "Found forbidden dependency on the old project root." >&2
  exit 1
fi

echo "preflight=passed"
echo "candidate_count=515"
echo "candidate_sha256=55279a1088fbfc71f93f41fd7577e92bc358fed57d302fcef889e2bf3cdede1e"
