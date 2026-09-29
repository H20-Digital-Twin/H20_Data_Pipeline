from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mujoco
import numpy as np


FLAG_NONFINITE = np.uint32(1 << 0)
FLAG_ROOT_QUAT_INVALID = np.uint32(1 << 1)
FLAG_MUJOCO_FORWARD_FAILED = np.uint32(1 << 2)
FLAG_IK_POSITION_HIGH = np.uint32(1 << 3)
FLAG_IK_ROTATION_HIGH = np.uint32(1 << 4)
FLAG_JOINT_LIMIT = np.uint32(1 << 5)
FLAG_ROOT_SPEED_HIGH = np.uint32(1 << 6)
FLAG_DOF_SPEED_HIGH = np.uint32(1 << 7)

STRUCTURAL_FAILURE_MASK = (
    FLAG_NONFINITE | FLAG_ROOT_QUAT_INVALID | FLAG_MUJOCO_FORWARD_FAILED
)


def free_joint(schema: dict) -> dict:
    roots = [joint for joint in schema["joints"] if joint["type"] == "free"]
    if len(roots) != 1:
        raise NotImplementedError(
            f"V1 requires exactly one free root joint, found {len(roots)}"
        )
    return roots[0]


def actuated_joints(schema: dict) -> list[dict]:
    joints = [joint for joint in schema["joints"] if joint["type"] != "free"]
    for joint in joints:
        if joint["qpos_width"] != 1 or joint["qvel_width"] != 1:
            raise NotImplementedError(
                f"V1 dof_pos requires scalar joints: {joint['name']}"
            )
    return joints


def canonicalize_root_quaternions(qpos: np.ndarray, schema: dict) -> None:
    root = free_joint(schema)
    start = int(root["qpos_address"]) + 3
    quaternions = qpos[:, start : start + 4]
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-12):
        raise ValueError("invalid MuJoCo root quaternion")
    quaternions /= norms[:, None]
    for frame in range(1, len(quaternions)):
        if np.dot(quaternions[frame - 1], quaternions[frame]) < 0:
            quaternions[frame] *= -1.0


def apply_root_offsets(
    qpos: np.ndarray,
    schema: dict,
    normalize_xy: bool,
    ground_z_offset: float,
) -> dict:
    root = free_joint(schema)
    address = int(root["qpos_address"])
    xy_offset = np.zeros(2, dtype=np.float64)
    if normalize_xy:
        xy_offset = qpos[0, address : address + 2].copy()
        qpos[:, address : address + 2] -= xy_offset
    qpos[:, address + 2] += float(ground_z_offset)
    return {
        "root_xy_offset_removed": xy_offset.tolist(),
        "ground_z_offset_added": float(ground_z_offset),
    }


def split_qpos(qpos: np.ndarray, schema: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = free_joint(schema)
    address = int(root["qpos_address"])
    root_pos = qpos[:, address : address + 3]
    root_quat_wxyz = qpos[:, address + 3 : address + 7]
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]
    joints = actuated_joints(schema)
    dof_pos = np.stack(
        [qpos[:, int(joint["qpos_address"])] for joint in joints], axis=1
    )
    return root_pos, root_quat_xyzw, dof_pos


def compute_qvel(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    qvel = np.zeros((len(qpos), model.nv), dtype=np.float64)
    if len(qpos) < 2:
        return qvel
    for frame in range(len(qpos)):
        if frame == 0:
            left, right = 0, 1
        elif frame == len(qpos) - 1:
            left, right = len(qpos) - 2, len(qpos) - 1
        else:
            left, right = frame - 1, frame + 1
        dt = float(timestamps[right] - timestamps[left])
        if dt <= 0:
            raise ValueError("timestamps must be strictly increasing")
        mujoco.mj_differentiatePos(
            model, qvel[frame], dt, qpos[left], qpos[right]
        )
    return qvel


def split_qvel(qvel: np.ndarray, schema: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = free_joint(schema)
    address = int(root["qvel_address"])
    root_lin_vel = qvel[:, address : address + 3]
    root_ang_vel = qvel[:, address + 3 : address + 6]
    joints = actuated_joints(schema)
    dof_vel = np.stack(
        [qvel[:, int(joint["qvel_address"])] for joint in joints], axis=1
    )
    return root_lin_vel, root_ang_vel, dof_vel


def joint_limit_diagnostics(
    dof_pos: np.ndarray, schema: dict, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    joints = actuated_joints(schema)
    margin = np.full_like(dof_pos, np.inf, dtype=np.float64)
    violation = np.zeros_like(dof_pos, dtype=bool)
    for column, joint in enumerate(joints):
        if not joint["limited"] or joint["range"] is None:
            continue
        lower, upper = map(float, joint["range"])
        margin[:, column] = np.minimum(
            dof_pos[:, column] - lower, upper - dof_pos[:, column]
        )
        violation[:, column] = (
            (dof_pos[:, column] < lower - tolerance)
            | (dof_pos[:, column] > upper + tolerance)
        )
    return margin, violation


def replay_and_flags(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    ik_position_error: np.ndarray,
    ik_rotation_error: np.ndarray,
    joint_limit_violation: np.ndarray,
    root_lin_vel: np.ndarray,
    dof_vel: np.ndarray,
    thresholds: dict,
) -> tuple[np.ndarray, np.ndarray]:
    flags = np.zeros(len(qpos), dtype=np.uint32)
    finite = np.isfinite(qpos).all(axis=1)
    flags[~finite] |= FLAG_NONFINITE

    quat_norm = np.linalg.norm(root_quat_xyzw, axis=1)
    quat_ok = np.isfinite(quat_norm) & (
        np.abs(quat_norm - 1.0) <= float(thresholds["quaternion_norm_tolerance"])
    )
    flags[~quat_ok] |= FLAG_ROOT_QUAT_INVALID

    data = mujoco.MjData(model)
    for frame, values in enumerate(qpos):
        if not finite[frame]:
            flags[frame] |= FLAG_MUJOCO_FORWARD_FAILED
            continue
        try:
            data.qpos[:] = values
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.xpos).all() or not np.isfinite(data.xquat).all():
                flags[frame] |= FLAG_MUJOCO_FORWARD_FAILED
        except Exception:
            flags[frame] |= FLAG_MUJOCO_FORWARD_FAILED

    flags[
        ik_position_error > float(thresholds["ik_position_error_threshold_m"])
    ] |= FLAG_IK_POSITION_HIGH
    flags[
        ik_rotation_error > float(thresholds["ik_rotation_error_threshold_rad"])
    ] |= FLAG_IK_ROTATION_HIGH
    flags[np.any(joint_limit_violation, axis=1)] |= FLAG_JOINT_LIMIT
    flags[
        np.linalg.norm(root_lin_vel, axis=1)
        > float(thresholds["root_speed_warning_mps"])
    ] |= FLAG_ROOT_SPEED_HIGH
    flags[
        np.max(np.abs(dof_vel), axis=1)
        > float(thresholds["dof_speed_warning_rps"])
    ] |= FLAG_DOF_SPEED_HIGH

    frame_valid = (flags & STRUCTURAL_FAILURE_MASK) == 0
    return frame_valid, flags


def quaternion_angle_wxyz(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    cosine = float(np.clip(abs(np.dot(left, right)), 0.0, 1.0))
    return float(2.0 * np.arccos(cosine))


@dataclass(frozen=True)
class TaskSpec:
    stage: int
    human_body_name: str
    robot_body_name: str
    robot_body_id: int
    position_weight: float
    rotation_weight: float

    @property
    def task_id(self) -> str:
        return (
            f"stage{self.stage}:{self.robot_body_name}"
            f"<-{self.human_body_name}"
        )


def build_task_specs(retargeter) -> list[TaskSpec]:
    specs: list[TaskSpec] = []
    stages = (
        (1, retargeter.human_body_to_task1, retargeter.ik_match_table1),
        (2, retargeter.human_body_to_task2, retargeter.ik_match_table2),
    )
    for stage, mapping, table in stages:
        for human_name, task in mapping.items():
            entry = table[task.frame_name]
            specs.append(
                TaskSpec(
                    stage=stage,
                    human_body_name=human_name,
                    robot_body_name=task.frame_name,
                    robot_body_id=int(
                        retargeter.configuration.model.body(task.frame_name).id
                    ),
                    position_weight=float(entry[1]),
                    rotation_weight=float(entry[2]),
                )
            )
    return specs


def measure_task_errors(
    retargeter, specs: list[TaskSpec]
) -> tuple[np.ndarray, np.ndarray, float, float]:
    positions = np.zeros(len(specs), dtype=np.float64)
    rotations = np.zeros(len(specs), dtype=np.float64)
    data = retargeter.configuration.data
    for index, spec in enumerate(specs):
        target_pos, target_quat_wxyz = retargeter.scaled_human_data[
            spec.human_body_name
        ]
        positions[index] = np.linalg.norm(
            np.asarray(data.xpos[spec.robot_body_id]) - np.asarray(target_pos)
        )
        rotations[index] = quaternion_angle_wxyz(
            data.xquat[spec.robot_body_id], target_quat_wxyz
        )
    position_active = np.array(
        [spec.position_weight > 0 for spec in specs], dtype=bool
    )
    rotation_active = np.array(
        [spec.rotation_weight > 0 for spec in specs], dtype=bool
    )
    position_summary = (
        float(np.max(positions[position_active])) if np.any(position_active) else 0.0
    )
    rotation_summary = (
        float(np.max(rotations[rotation_active])) if np.any(rotation_active) else 0.0
    )
    return positions, rotations, position_summary, rotation_summary
