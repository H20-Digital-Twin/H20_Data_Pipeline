#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
GENERATION_DIR="${PROJECT_ROOT}/generation"

if [[ "${CONFIRM_FULL_RUN:-}" != "YES" ]]; then
    echo "Refusing to run the full Avatar batch." >&2
    echo "Set CONFIRM_FULL_RUN=YES after the pilot is approved." >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/home/dell/anaconda3/envs/py38/bin/python}"
CANDIDATE_MANIFEST="${CANDIDATE_MANIFEST:-${PROJECT_ROOT}/selection/artifacts/v4/candidate_v4.jsonl}"
CANONICAL_ROOT="${CANONICAL_ROOT:-${PROJECT_ROOT}/work/canonical}"
AVATAR_ROOT="${AVATAR_ROOT:-${PROJECT_ROOT}/assets/avatars}"
XAVATAR_ROOT="${XAVATAR_ROOT:-${PROJECT_ROOT}/vendor/xavatar}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/work/ply}"
ASSIGNMENT_MANIFEST="${ASSIGNMENT_MANIFEST:-${CANONICAL_ROOT}/generation_manifests/avatar_assignment_v2.jsonl}"
ASSIGNMENT_SUMMARY="${ASSIGNMENT_SUMMARY:-${CANONICAL_ROOT}/generation_manifests/avatar_assignment_v2_summary.json}"
ASSIGNMENT_VERSION="${ASSIGNMENT_VERSION:-avatar_assignment_v2}"
MESH_RESOLUTION="${MESH_RESOLUTION:-256}"
DEVICE="${DEVICE:-0}"
OUTPUT_MOUNT="${OUTPUT_MOUNT:-/data}"
MIN_FREE_BYTES="${MIN_FREE_BYTES:-150000000000}"

# Refuse to use an edited, incomplete, reordered, or checkpoint-stale manifest.
"${PYTHON_BIN}" "${GENERATION_DIR}/build_avatar_assignment.py" \
    --candidate-manifest "${CANDIDATE_MANIFEST}" \
    --avatar-root "${AVATAR_ROOT}" \
    --output "${ASSIGNMENT_MANIFEST}" \
    --summary "${ASSIGNMENT_SUMMARY}" \
    --assignment-version "${ASSIGNMENT_VERSION}" \
    --validate-existing || exit 1

if ! mountpoint -q "${OUTPUT_MOUNT}"; then
    echo "Refusing batch: output volume is not mounted at ${OUTPUT_MOUNT}." >&2
    exit 1
fi
case "${OUTPUT_ROOT}/" in
    "${OUTPUT_MOUNT}/"*) ;;
    *)
        echo "Refusing batch: OUTPUT_ROOT is outside OUTPUT_MOUNT." >&2
        exit 1
        ;;
esac
available_bytes="$(df --output=avail -B1 "${OUTPUT_MOUNT}" | tail -n 1 | tr -d ' ')"
if (( available_bytes < MIN_FREE_BYTES )); then
    echo "Refusing batch: output volume has less than ${MIN_FREE_BYTES} free bytes." >&2
    exit 1
fi

"${PYTHON_BIN}" "${GENERATION_DIR}/preflight_avatar_batch.py" \
    --candidate-manifest "${CANDIDATE_MANIFEST}" \
    --assignment-manifest "${ASSIGNMENT_MANIFEST}" \
    --canonical-root "${CANONICAL_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --expected-count 515 || exit 1

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/generation_manifests"
cp -f "${ASSIGNMENT_MANIFEST}" \
    "${OUTPUT_ROOT}/generation_manifests/avatar_assignment_v2.jsonl"
cp -f "${ASSIGNMENT_SUMMARY}" \
    "${OUTPUT_ROOT}/generation_manifests/avatar_assignment_v2_summary.json"

mapfile -t jobs < <(
    "${PYTHON_BIN}" -c \
        'import json,pathlib,sys
root=pathlib.Path(sys.argv[2])
rows=[]
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    assignment=json.loads(line)
    motion_id=assignment["motion_id"]
    avatar_id=assignment["avatar_id"]
    output_dir=root/motion_id
    status_path=output_dir/"avatar_generation_status.json"
    complete=False
    if status_path.is_file():
        status=json.loads(status_path.read_text())
        planned=int(status.get("planned_frames",-1))
        expected={f"{index:06d}.ply" for index in range(max(planned,0))}
        actual={path.name for path in output_dir.glob("*.ply")}
        complete=(
            status.get("status")=="completed"
            and status.get("motion_id")==motion_id
            and status.get("avatar_id")==avatar_id
            and status.get("checkpoint_sha256")==assignment["checkpoint_sha256"]
            and planned>=0
            and actual==expected
        )
    rows.append(motion_id+"\t"+avatar_id+"\t"+("complete" if complete else "pending"))
print("\n".join(rows))' \
        "${ASSIGNMENT_MANIFEST}" "${OUTPUT_ROOT}"
)

if (( ${#jobs[@]} != 515 )); then
    echo "Refusing batch: expected 515 assignments, got ${#jobs[@]}." >&2
    exit 1
fi

failures=0
job_index=0
for job in "${jobs[@]}"; do
    job_index=$((job_index + 1))
    IFS=$'\t' read -r motion_id avatar_id job_state <<< "${job}"
    source_dir="${CANONICAL_ROOT}/motions/${motion_id}/source"
    output_dir="${OUTPUT_ROOT}/${motion_id}"
    log_path="${OUTPUT_ROOT}/logs/${motion_id}__${avatar_id}.log"
    if [[ "${job_state}" == "complete" ]]; then
        echo "BATCH [${job_index}/${#jobs[@]}] SKIP completed motion=${motion_id}"
        continue
    fi
    echo "BATCH [${job_index}/${#jobs[@]}] avatar=${avatar_id} motion=${motion_id} res=${MESH_RESOLUTION}"
    if ! "${PYTHON_BIN}" "${GENERATION_DIR}/generate_avatar_ply.py" \
        --motion-id "${motion_id}" \
        --canonical-file "${source_dir}/canonical_smplx_30fps.npz" \
        --frame-map "${source_dir}/frame_map_15fps.npz" \
        --avatar-dir "${AVATAR_ROOT}/${avatar_id}" \
        --xavatar-root "${XAVATAR_ROOT}" \
        --output-dir "${output_dir}" \
        --assignment-manifest "${ASSIGNMENT_MANIFEST}" \
        --mesh-resolution "${MESH_RESOLUTION}" \
        --device "${DEVICE}" \
        --resume 2>&1 | tee "${log_path}"; then
        failures=$((failures + 1))
    fi
done

if (( failures > 0 )); then
    echo "Avatar batch finished with ${failures} failed job(s)." >&2
    exit 1
fi
echo "Avatar batch completed successfully."
