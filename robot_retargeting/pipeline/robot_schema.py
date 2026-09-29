from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
GMR_ROOT = HERE.parent
if str(GMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GMR_ROOT))

from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT

from common import atomic_write_json, sha256_file


JOINT_TYPE_NAMES = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}
QPOS_WIDTH = {"free": 7, "ball": 4, "slide": 1, "hinge": 1}
QVEL_WIDTH = {"free": 6, "ball": 3, "slide": 1, "hinge": 1}


def _name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, index: int) -> str:
    value = mujoco.mj_id2name(model, obj_type, index)
    return value if value is not None else f"unnamed_{index}"


def _relative_to_gmr(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(GMR_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def build_robot_schema(
    robot_id: str,
    end_effector_names: list[str] | None = None,
    left_foot_links: list[str] | None = None,
    right_foot_links: list[str] | None = None,
) -> dict:
    if robot_id not in ROBOT_XML_DICT:
        raise KeyError(f"unknown robot_id: {robot_id}")
    if robot_id not in IK_CONFIG_DICT["smplx"]:
        raise KeyError(f"no SMPL-X IK config for robot_id: {robot_id}")

    xml_path = Path(ROBOT_XML_DICT[robot_id]).resolve()
    ik_path = Path(IK_CONFIG_DICT["smplx"][robot_id]).resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    joints: list[dict[str, Any]] = []
    actuated_position_names: list[str] = []
    joint_limits: list[list[float] | None] = []
    root_joint_names: list[str] = []

    for joint_id in range(model.njnt):
        joint_type = JOINT_TYPE_NAMES[int(model.jnt_type[joint_id])]
        joint_name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        body_id = int(model.jnt_bodyid[joint_id])
        limited = bool(model.jnt_limited[joint_id])
        joint_range = (
            [float(x) for x in model.jnt_range[joint_id]]
            if limited and joint_type in {"hinge", "slide"}
            else None
        )
        entry = {
            "joint_id": joint_id,
            "name": joint_name,
            "type": joint_type,
            "body_id": body_id,
            "body_name": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
            "qpos_address": int(model.jnt_qposadr[joint_id]),
            "qpos_width": QPOS_WIDTH[joint_type],
            "qvel_address": int(model.jnt_dofadr[joint_id]),
            "qvel_width": QVEL_WIDTH[joint_type],
            "axis": [float(x) for x in model.jnt_axis[joint_id]],
            "limited": limited,
            "range": joint_range,
        }
        joints.append(entry)
        if joint_type == "free":
            root_joint_names.append(joint_name)
        else:
            if entry["qpos_width"] != 1 or entry["qvel_width"] != 1:
                raise NotImplementedError(
                    f"{robot_id} contains non-root multi-DoF joint {joint_name}; "
                    "define its training representation before using dof_pos"
                )
            actuated_position_names.append(joint_name)
            joint_limits.append(joint_range)

    body_names = [
        _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for body_id in range(1, model.nbody)
    ]
    actuator_names = [
        _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        for actuator_id in range(model.nu)
    ]
    schema = {
        "schema_version": "robot_schema_v1",
        "robot_id": robot_id,
        "model_name": robot_id,
        "mujoco_xml": _relative_to_gmr(xml_path),
        "mujoco_xml_sha256": sha256_file(xml_path),
        "retarget_config": _relative_to_gmr(ik_path),
        "retarget_config_sha256": sha256_file(ik_path),
        "coordinate_system": "right_handed_z_up",
        "up_axis": "z",
        "length_unit": "meter",
        "angle_unit": "radian",
        "dataset_quaternion_order": "xyzw",
        "mujoco_quaternion_order": "wxyz",
        "floating_base": bool(root_joint_names),
        "root_joint_names": root_joint_names,
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "mujoco_njnt": int(model.njnt),
        "mujoco_nbody_including_world": int(model.nbody),
        "num_dof_positions": len(actuated_position_names),
        "dof_joint_names": actuated_position_names,
        "joint_limits": joint_limits,
        "joints": joints,
        "link_names": body_names,
        "actuator_names": actuator_names,
        "end_effector_names": end_effector_names or [],
        "foot_contact_link_groups": {
            "left": left_foot_links or [],
            "right": right_foot_links or [],
        },
    }
    return schema


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    schema = build_robot_schema(args.robot)
    atomic_write_json(args.output, schema)
    print(json.dumps(schema, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
