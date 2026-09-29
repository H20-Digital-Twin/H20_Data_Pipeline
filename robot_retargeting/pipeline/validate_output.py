#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
GMR_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import atomic_write_json, load_json, load_npz, sha256_file, utc_now
from trajectory_tools import split_qpos


TIME_FIELDS = (
    "qpos_mujoco",
    "root_pos",
    "root_quat",
    "dof_pos",
    "qvel_mujoco",
    "root_lin_vel",
    "root_ang_vel",
    "dof_vel",
    "frame_valid",
    "frame_quality_flags",
)


def _require_fields(data: dict[str, np.ndarray], fields: tuple[str, ...], label: str) -> None:
    missing = sorted(set(fields).difference(data))
    if missing:
        raise KeyError(f"{label} is missing fields: {missing}")


def validate_robot_output(
    robot_dir: Path,
    check_all_mujoco_frames: bool,
    write_report: bool,
) -> dict:
    robot_dir = robot_dir.resolve()
    metadata_path = robot_dir / "metadata.json"
    quality_path = robot_dir / "quality.json"
    trajectory_30_path = robot_dir / "trajectory_30fps.npz"
    trajectory_15_path = robot_dir / "trajectory_15fps.npz"
    diagnostics_path = robot_dir / "diagnostics.npz"
    for path in (
        metadata_path,
        quality_path,
        trajectory_30_path,
        trajectory_15_path,
        diagnostics_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = load_json(metadata_path)
    output_root = robot_dir.parents[3]
    schema_path = output_root / metadata["robot_schema"]
    schema = load_json(schema_path)
    arrays_30 = load_npz(trajectory_30_path)
    arrays_15 = load_npz(trajectory_15_path)
    diagnostics = load_npz(diagnostics_path)

    core_30 = (
        "fps",
        "frame_id",
        "canonical_frame_id",
        "timestamp_sec",
        "source_frame_float",
    ) + TIME_FIELDS
    core_15 = (
        "fps",
        "frame_id",
        "visual_frame_id",
        "canonical_frame_id",
        "robot_frame_id",
        "timestamp_sec",
        "source_frame_float",
    ) + TIME_FIELDS
    _require_fields(arrays_30, core_30, "trajectory_30fps")
    _require_fields(arrays_15, core_15, "trajectory_15fps")

    t30 = len(arrays_30["frame_id"])
    t15 = len(arrays_15["frame_id"])
    if float(arrays_30["fps"]) != 30.0 or float(arrays_15["fps"]) != 15.0:
        raise AssertionError("trajectory FPS metadata is incorrect")
    if not np.array_equal(arrays_30["frame_id"], np.arange(t30)):
        raise AssertionError("30 FPS frame_id is not contiguous")
    if not np.array_equal(arrays_30["canonical_frame_id"], np.arange(t30)):
        raise AssertionError("30 FPS canonical_frame_id is incorrect")
    if not np.array_equal(arrays_15["frame_id"], np.arange(t15)):
        raise AssertionError("15 FPS frame_id is not contiguous")
    if not np.array_equal(arrays_15["visual_frame_id"], np.arange(t15)):
        raise AssertionError("15 FPS visual_frame_id is incorrect")
    canonical_ids = np.asarray(arrays_15["canonical_frame_id"], dtype=np.int64)
    if not np.array_equal(canonical_ids, np.arange(0, t30, 2)):
        raise AssertionError("15 FPS canonical IDs are not 0,2,4,...")
    if not np.array_equal(arrays_15["robot_frame_id"], canonical_ids):
        raise AssertionError("15 FPS robot IDs differ from canonical IDs")
    if not np.all(np.diff(arrays_30["timestamp_sec"]) > 0):
        raise AssertionError("30 FPS timestamps are not strictly increasing")
    if not np.all(np.diff(arrays_15["timestamp_sec"]) > 0):
        raise AssertionError("15 FPS timestamps are not strictly increasing")

    for field in TIME_FIELDS:
        if not np.array_equal(arrays_15[field], arrays_30[field][canonical_ids]):
            raise AssertionError(f"15 FPS {field} is not an exact 30 FPS slice")

    qpos = np.asarray(arrays_30["qpos_mujoco"], dtype=np.float64)
    if qpos.shape != (t30, int(schema["nq"])):
        raise AssertionError("qpos_mujoco shape differs from robot schema")
    root_pos, root_quat, dof_pos = split_qpos(qpos, schema)
    if not np.allclose(root_pos, arrays_30["root_pos"], atol=1e-6):
        raise AssertionError("root_pos cannot be reconstructed from qpos_mujoco")
    if not np.allclose(root_quat, arrays_30["root_quat"], atol=1e-6):
        raise AssertionError("root_quat cannot be reconstructed from qpos_mujoco")
    if not np.allclose(dof_pos, arrays_30["dof_pos"], atol=1e-6):
        raise AssertionError("dof_pos cannot be reconstructed from qpos_mujoco")
    if dof_pos.shape[1] != len(schema["dof_joint_names"]):
        raise AssertionError("dof_pos columns differ from robot schema")
    quat_norm_error = np.max(
        np.abs(np.linalg.norm(arrays_30["root_quat"], axis=1) - 1.0)
    )
    if quat_norm_error > 1e-4:
        raise AssertionError("root quaternion norm error exceeds tolerance")

    xml_path = Path(metadata["mujoco_xml_path"])
    if sha256_file(xml_path) != metadata["mujoco_xml_sha256"]:
        raise AssertionError("MuJoCo XML hash differs from generation metadata")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    if check_all_mujoco_frames:
        replay_ids = np.arange(t30)
    else:
        replay_ids = np.unique(
            np.linspace(0, t30 - 1, min(10, t30), dtype=np.int64)
        )
    for frame in replay_ids:
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.xpos).all() or not np.isfinite(data.xquat).all():
            raise AssertionError(f"MuJoCo replay is nonfinite at frame {frame}")

    if len(diagnostics["ik_converged"]) != t30:
        raise AssertionError("diagnostics frame count differs from trajectory")
    visual_alignment = load_json(quality_path)["visual_ply_alignment"]
    report = {
        "validation_status": "passed",
        "validated_at_utc": utc_now(),
        "motion_id": metadata["motion_id"],
        "robot_id": metadata["robot_id"],
        "num_frames_30fps": t30,
        "num_frames_15fps": t15,
        "num_dof_positions": int(dof_pos.shape[1]),
        "qpos_width": int(qpos.shape[1]),
        "root_quaternion_norm_max_error": float(quat_norm_error),
        "exact_30_to_15_slice_passed": True,
        "qpos_decomposition_passed": True,
        "mujoco_replay_frames_checked": int(len(replay_ids)),
        "visual_ply_alignment": visual_alignment,
    }
    if write_report:
        atomic_write_json(robot_dir / "validation.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-dir", type=Path, required=True)
    parser.add_argument("--all-mujoco-frames", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = validate_robot_output(
        args.robot_dir,
        check_all_mujoco_frames=args.all_mujoco_frames,
        write_report=args.write_report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
