#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import mujoco
import numpy as np
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
GMR_ROOT = HERE.parent
if str(GMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GMR_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT

from canonical_smplx import load_canonical_human_motion
from common import (
    atomic_write_json,
    atomic_write_npz,
    load_json,
    load_npz,
    load_yaml,
    quantiles,
    remove_exact_directory,
    replace_directory_atomically,
    select_candidate,
    sha256_file,
    utc_now,
)
from robot_schema import build_robot_schema
from trajectory_tools import (
    FLAG_DOF_SPEED_HIGH,
    FLAG_IK_POSITION_HIGH,
    FLAG_IK_ROTATION_HIGH,
    FLAG_JOINT_LIMIT,
    FLAG_ROOT_SPEED_HIGH,
    apply_root_offsets,
    build_task_specs,
    canonicalize_root_quaternions,
    compute_qvel,
    free_joint,
    joint_limit_diagnostics,
    measure_task_errors,
    replay_and_flags,
    split_qpos,
    split_qvel,
)


VISUAL_FROM_CANONICAL = [
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
]


def _code_identity() -> dict:
    generator_files = (
        "generate_one.py",
        "canonical_smplx.py",
        "trajectory_tools.py",
        "robot_schema.py",
        "common.py",
    )
    gmr_files = (
        GMR_ROOT / "general_motion_retargeting" / "motion_retarget.py",
        GMR_ROOT / "general_motion_retargeting" / "utils" / "smpl.py",
        GMR_ROOT / "general_motion_retargeting" / "params.py",
    )
    return {
        "generator_version": "robot_dataset_stage2_v1",
        "generator_files_sha256": {
            name: sha256_file(HERE / name) for name in generator_files
        },
        "gmr_dependency_files_sha256": {
            str(path.relative_to(GMR_ROOT)): sha256_file(path)
            for path in gmr_files
        },
    }


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _validate_frame_map(
    frame_map: dict[str, np.ndarray], canonical_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    required = {
        "visual_frame_id",
        "canonical_frame_id",
        "robot_frame_id",
        "source_frame_float",
        "timestamp_sec",
    }
    missing = sorted(required.difference(frame_map))
    if missing:
        raise KeyError(f"frame map is missing fields: {missing}")
    visual_ids = np.asarray(frame_map["visual_frame_id"], dtype=np.int64)
    canonical_ids = np.asarray(frame_map["canonical_frame_id"], dtype=np.int64)
    robot_ids = np.asarray(frame_map["robot_frame_id"], dtype=np.int64)
    if not np.array_equal(visual_ids, np.arange(len(visual_ids))):
        raise ValueError("visual_frame_id must be contiguous from zero")
    if not np.array_equal(canonical_ids, robot_ids):
        raise ValueError("robot_frame_id must equal canonical_frame_id")
    if not np.array_equal(canonical_ids, np.arange(0, canonical_frames, 2)):
        raise ValueError("frame map is not the canonical even-frame 15 FPS view")
    if not np.allclose(
        frame_map["timestamp_sec"], canonical_ids / 30.0, atol=1e-9
    ):
        raise ValueError("frame-map timestamps do not match canonical IDs")
    return visual_ids, canonical_ids


def _ensure_robot_schema(output_root: Path, robot_id: str, config: dict) -> tuple[dict, Path]:
    contact = config.get("contact", {})
    schema = build_robot_schema(
        robot_id,
        end_effector_names=list(contact.get("end_effector_names", [])),
        left_foot_links=list(contact.get("left_foot_links", [])),
        right_foot_links=list(contact.get("right_foot_links", [])),
    )
    schema_path = output_root / "robot_schemas" / f"{robot_id}.json"
    if schema_path.exists():
        existing = load_json(schema_path)
        if existing != schema:
            raise ValueError(
                f"existing robot schema differs from current MuJoCo model: {schema_path}"
            )
    else:
        atomic_write_json(schema_path, schema)
    return schema, schema_path


def _clear_legacy_history(retargeter) -> None:
    for history in retargeter.comparison_history.values():
        for values in history.values():
            values.clear()


def _seed_first_frame_root(
    retargeter,
    human_frame: dict,
    schema: dict,
) -> None:
    """Initialize the floating base at the first root target before limb IK."""
    retargeter.update_targets(human_frame)
    target_pos, target_quat_wxyz = retargeter.scaled_human_data[
        retargeter.human_root_name
    ]
    root = free_joint(schema)
    address = int(root["qpos_address"])
    quaternion = np.asarray(target_quat_wxyz, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    retargeter.configuration.data.qpos[address : address + 3] = target_pos
    retargeter.configuration.data.qpos[address + 3 : address + 7] = quaternion
    mujoco.mj_forward(
        retargeter.configuration.model, retargeter.configuration.data
    )


def _visual_alignment(ply_root: Path, motion_id: str, expected: int) -> dict:
    directory = ply_root / motion_id
    if not directory.is_dir():
        return {
            "available": False,
            "passed": None,
            "directory": motion_id,
            "num_ply": None,
            "reason": "visual_directory_not_available",
        }
    ids = sorted(
        int(path.stem)
        for path in directory.glob("*.ply")
        if path.stem.isdigit()
    )
    expected_ids = list(range(expected))
    passed = ids == expected_ids
    return {
        "available": True,
        "passed": passed,
        "directory": motion_id,
        "num_ply": len(ids),
        "reason": None if passed else "ply_ids_or_count_mismatch",
    }


def _extract_15fps(
    arrays_30: dict[str, np.ndarray],
    frame_map: dict[str, np.ndarray],
    canonical_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {
        "fps": np.array(15.0, dtype=np.float32),
        "frame_id": np.asarray(frame_map["visual_frame_id"], dtype=np.int64),
        "visual_frame_id": np.asarray(
            frame_map["visual_frame_id"], dtype=np.int64
        ),
        "canonical_frame_id": canonical_ids.astype(np.int64),
        "robot_frame_id": np.asarray(
            frame_map["robot_frame_id"], dtype=np.int64
        ),
        "timestamp_sec": np.asarray(
            frame_map["timestamp_sec"], dtype=np.float64
        ),
        "source_frame_float": np.asarray(
            frame_map["source_frame_float"], dtype=np.float64
        ),
    }
    time_fields = (
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
    for field in time_fields:
        result[field] = np.asarray(arrays_30[field])[canonical_ids]
    return result


def _assert_exact_15fps(
    arrays_30: dict[str, np.ndarray],
    arrays_15: dict[str, np.ndarray],
    canonical_ids: np.ndarray,
) -> None:
    for field in (
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
    ):
        if not np.array_equal(arrays_15[field], arrays_30[field][canonical_ids]):
            raise AssertionError(f"15 FPS field is not an exact 30 FPS slice: {field}")


def _quality_report(
    motion_id: str,
    robot_id: str,
    arrays_30: dict[str, np.ndarray],
    diagnostics: dict[str, np.ndarray],
    joint_limit_violation: np.ndarray,
    visual_alignment: dict,
    config: dict,
) -> dict:
    valid = np.asarray(arrays_30["frame_valid"], dtype=bool)
    flags = np.asarray(arrays_30["frame_quality_flags"], dtype=np.uint32)
    converged = np.asarray(diagnostics["ik_converged"], dtype=bool)
    position = np.asarray(diagnostics["ik_position_error_max"], dtype=np.float64)
    rotation = np.asarray(diagnostics["ik_rotation_error_max"], dtype=np.float64)
    root_speed = np.linalg.norm(arrays_30["root_lin_vel"], axis=1)
    dof_speed = np.max(np.abs(arrays_30["dof_vel"]), axis=1)

    reasons: list[str] = []
    if not np.all(valid):
        reasons.append("structurally_invalid_frames")
    if np.mean(converged) < float(config["quality"]["review_ik_converged_ratio_below"]):
        reasons.append("ik_converged_ratio_low")
    if np.any(flags & FLAG_JOINT_LIMIT):
        reasons.append("joint_limit_violation")
    if np.any(flags & FLAG_ROOT_SPEED_HIGH):
        reasons.append("root_speed_high")
    if np.any(flags & FLAG_DOF_SPEED_HIGH):
        reasons.append("dof_speed_high")
    if visual_alignment["available"] and not visual_alignment["passed"]:
        reasons.append("visual_ply_alignment_failed")

    structural_passed = bool(np.all(valid))
    quality_status = (
        "robot_quality_passed"
        if structural_passed and not reasons
        else "review_robot_quality"
    )
    position_stats = quantiles(position)
    rotation_stats = quantiles(rotation)
    limited_margins = np.asarray(
        diagnostics["joint_limit_margin"], dtype=np.float64
    )
    limited_margins = limited_margins[np.isfinite(limited_margins)]
    return {
        "motion_id": motion_id,
        "robot_id": robot_id,
        "generation_status": (
            "generated" if not reasons else "generated_with_warnings"
        ),
        "quality_status": quality_status,
        "quality_reasons": reasons,
        "decision_mode": "audit",
        "num_frames_30fps": int(len(valid)),
        "num_frames_15fps": int(
            np.count_nonzero(np.arange(len(valid)) % 2 == 0)
        ),
        "valid_frame_count": int(np.count_nonzero(valid)),
        "valid_frame_ratio": float(np.mean(valid)),
        "ik_converged_ratio": float(np.mean(converged)),
        "ik_convergence_definition": (
            "thresholded final task max position/rotation error; "
            "not a native Mink solver status"
        ),
        "ik_position_error_mean_m": position_stats["mean"],
        "ik_position_error_p95_m": position_stats["p95"],
        "ik_position_error_max_m": position_stats["max"],
        "ik_rotation_error_mean_rad": rotation_stats["mean"],
        "ik_rotation_error_p95_rad": rotation_stats["p95"],
        "ik_rotation_error_max_rad": rotation_stats["max"],
        "joint_limit_violation_frame_ratio": float(
            np.mean(np.any(joint_limit_violation, axis=1))
        ),
        "joint_limit_min_margin_rad": (
            float(np.min(limited_margins)) if len(limited_margins) else None
        ),
        "root_speed_max_mps": float(np.max(root_speed)),
        "dof_speed_max_rps": float(np.max(dof_speed)),
        "frame_alignment_passed": True,
        "visual_ply_alignment": visual_alignment,
        "contact_metrics_status": "not_computed_v1",
        "self_collision_status": "not_computed_v1",
    }


def generate(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    paths = config["paths"]
    candidate_path = _require_file(Path(paths["candidate_jsonl"]), "candidate manifest")
    canonical_root = Path(paths["canonical_root"])
    body_model_root = Path(paths["body_model_root"])
    output_root = Path(paths["output_root"])
    ply_root = Path(paths["ply_root"])
    robot_id = args.robot or config["robot"]["robot_id"]

    item = select_candidate(candidate_path, args.motion_id)
    motion_id = item["motion_id"]
    source_dir = canonical_root / motion_id / "source"
    canonical_path = _require_file(
        source_dir / "canonical_smplx_30fps.npz", "canonical motion"
    )
    frame_map_path = _require_file(
        source_dir / "frame_map_15fps.npz", "15 FPS frame map"
    )
    canonical_metadata_path = _require_file(
        source_dir / "canonical_metadata.json", "canonical metadata"
    )

    schema, schema_path = _ensure_robot_schema(output_root, robot_id, config)
    code_identity = _code_identity()
    xml_path = Path(ROBOT_XML_DICT[robot_id]).resolve()
    ik_path = Path(IK_CONFIG_DICT["smplx"][robot_id]).resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if model.nq != schema["nq"] or model.nv != schema["nv"]:
        raise AssertionError("MuJoCo model and robot schema differ")

    target = output_root / "motions" / motion_id / "robots" / robot_id
    if target.exists():
        if args.overwrite:
            remove_exact_directory(target, output_root)
        elif args.resume:
            metadata_path = target / "metadata.json"
            required = (
                target / "trajectory_30fps.npz",
                target / "trajectory_15fps.npz",
                target / "diagnostics.npz",
                target / "quality.json",
            )
            if (
                metadata_path.is_file()
                and all(path.is_file() for path in required)
                and load_json(metadata_path).get("generation_status") == "completed"
            ):
                existing_metadata = load_json(metadata_path)
                expected_identity = {
                    "motion_id": motion_id,
                    "robot_id": robot_id,
                    "candidate_manifest_sha256": sha256_file(candidate_path),
                    "canonical_sha256": sha256_file(canonical_path),
                    "frame_map_sha256": sha256_file(frame_map_path),
                    "robot_schema_sha256": sha256_file(schema_path),
                    "mujoco_xml_sha256": sha256_file(xml_path),
                    "retarget_config_sha256": sha256_file(ik_path),
                    "generation_config_sha256": sha256_file(config_path),
                    "code_identity": code_identity,
                }
                mismatches = [
                    key
                    for key, value in expected_identity.items()
                    if existing_metadata.get(key) != value
                ]
                if mismatches:
                    raise ValueError(
                        "resume identity mismatch for existing output: "
                        + ", ".join(mismatches)
                    )
                from validate_output import validate_robot_output

                validate_robot_output(
                    target,
                    check_all_mujoco_frames=False,
                    write_report=False,
                )
                result = {
                    "motion_id": motion_id,
                    "robot_id": robot_id,
                    "status": "skipped_completed",
                    "output": str(target),
                }
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return result
            raise ValueError(
                f"resume found incomplete or incompatible output: {target}"
            )
        else:
            raise FileExistsError(
                f"output exists; pass --resume or explicit --overwrite: {target}"
            )

    temporary = target.parent / f".{robot_id}.tmp.{os.getpid()}"
    if temporary.exists():
        remove_exact_directory(temporary, output_root)
    temporary.mkdir(parents=True)

    try:
        canonical, human_frames, human_height = load_canonical_human_motion(
            canonical_path, body_model_root
        )
        timestamps = np.asarray(canonical["timestamp_sec"], dtype=np.float64)
        canonical_frames = len(timestamps)
        expected_frames = int(item["expected_num_frames_at_30fps"])
        if canonical_frames != expected_frames:
            raise ValueError(
                f"canonical frames {canonical_frames} != V4 expected {expected_frames}"
            )
        frame_map = load_npz(frame_map_path)
        visual_ids, canonical_ids = _validate_frame_map(
            frame_map, canonical_frames
        )

        retargeter = GMR(
            actual_human_height=human_height,
            src_human="smplx",
            tgt_robot=robot_id,
        )
        task_specs = build_task_specs(retargeter)
        qpos_raw: list[np.ndarray] = []
        task_pos_errors: list[np.ndarray] = []
        task_rot_errors: list[np.ndarray] = []
        pos_error_max: list[float] = []
        rot_error_max: list[float] = []
        minimum_body_origin_z = np.inf

        for frame_index, human_frame in enumerate(tqdm(
            human_frames,
            total=canonical_frames,
            desc=f"GMR {robot_id} {motion_id}",
            unit="frame",
            dynamic_ncols=True,
        )):
            if (
                frame_index == 0
                and bool(config["retarget"]["seed_first_frame_root_from_target"])
            ):
                _seed_first_frame_root(retargeter, human_frame, schema)
            qpos = np.asarray(retargeter.retarget(human_frame), dtype=np.float64)
            if qpos.shape != (model.nq,):
                raise ValueError(
                    f"GMR qpos shape {qpos.shape} != expected {(model.nq,)}"
                )
            positions, rotations, pos_max, rot_max = measure_task_errors(
                retargeter, task_specs
            )
            qpos_raw.append(qpos.copy())
            task_pos_errors.append(positions)
            task_rot_errors.append(rotations)
            pos_error_max.append(pos_max)
            rot_error_max.append(rot_max)
            minimum_body_origin_z = min(
                minimum_body_origin_z,
                float(np.min(retargeter.configuration.data.xpos[1:, 2])),
            )
            _clear_legacy_history(retargeter)

        qpos_retarget_raw = np.stack(qpos_raw, axis=0)
        if len(qpos_retarget_raw) != canonical_frames:
            raise AssertionError("GMR did not return one qpos for every canonical frame")
        qpos_final = qpos_retarget_raw.copy()
        canonicalize_root_quaternions(qpos_final, schema)

        postprocess = config["postprocess"]
        ground_z_offset = (
            float(postprocess["ground_height_m"]) - minimum_body_origin_z
            if postprocess["align_min_body_origin_to_ground"]
            else 0.0
        )
        root_offsets = apply_root_offsets(
            qpos_final,
            schema,
            normalize_xy=bool(postprocess["normalize_root_xy"]),
            ground_z_offset=ground_z_offset,
        )
        root_pos, root_quat, dof_pos = split_qpos(qpos_final, schema)
        qvel = compute_qvel(model, qpos_final, timestamps)
        root_lin_vel, root_ang_vel, dof_vel = split_qvel(qvel, schema)
        joint_margin, joint_violation = joint_limit_diagnostics(
            dof_pos,
            schema,
            tolerance=float(config["quality"]["joint_limit_tolerance_rad"]),
        )

        pos_error_max_array = np.asarray(pos_error_max, dtype=np.float64)
        rot_error_max_array = np.asarray(rot_error_max, dtype=np.float64)
        ik_converged = (
            pos_error_max_array
            <= float(config["quality"]["ik_position_error_threshold_m"])
        ) & (
            rot_error_max_array
            <= float(config["quality"]["ik_rotation_error_threshold_rad"])
        )
        frame_valid, frame_flags = replay_and_flags(
            model=model,
            qpos=qpos_final,
            root_quat_xyzw=root_quat,
            ik_position_error=pos_error_max_array,
            ik_rotation_error=rot_error_max_array,
            joint_limit_violation=joint_violation,
            root_lin_vel=root_lin_vel,
            dof_vel=dof_vel,
            thresholds=config["quality"],
        )

        arrays_30 = {
            "fps": np.array(30.0, dtype=np.float32),
            "frame_id": np.arange(canonical_frames, dtype=np.int64),
            "canonical_frame_id": np.arange(canonical_frames, dtype=np.int64),
            "timestamp_sec": timestamps.astype(np.float64),
            "source_frame_float": np.asarray(
                canonical["source_frame_float"], dtype=np.float64
            ),
            "qpos_mujoco": qpos_final.astype(np.float32),
            "root_pos": root_pos.astype(np.float32),
            "root_quat": root_quat.astype(np.float32),
            "dof_pos": dof_pos.astype(np.float32),
            "qvel_mujoco": qvel.astype(np.float32),
            "root_lin_vel": root_lin_vel.astype(np.float32),
            "root_ang_vel": root_ang_vel.astype(np.float32),
            "dof_vel": dof_vel.astype(np.float32),
            "frame_valid": frame_valid.astype(bool),
            "frame_quality_flags": frame_flags.astype(np.uint32),
        }
        arrays_15 = _extract_15fps(arrays_30, frame_map, canonical_ids)
        _assert_exact_15fps(arrays_30, arrays_15, canonical_ids)

        diagnostics = {
            "frame_id": np.arange(canonical_frames, dtype=np.int64),
            "ik_converged": ik_converged.astype(bool),
            "ik_position_error_max": pos_error_max_array.astype(np.float32),
            "ik_rotation_error_max": rot_error_max_array.astype(np.float32),
            "ik_task_position_error": np.stack(
                task_pos_errors, axis=0
            ).astype(np.float32),
            "ik_task_rotation_error": np.stack(
                task_rot_errors, axis=0
            ).astype(np.float32),
            "ik_task_ids": np.asarray(
                [spec.task_id for spec in task_specs], dtype=np.str_
            ),
            "ik_task_position_weights": np.asarray(
                [spec.position_weight for spec in task_specs], dtype=np.float32
            ),
            "ik_task_rotation_weights": np.asarray(
                [spec.rotation_weight for spec in task_specs], dtype=np.float32
            ),
            "joint_limit_margin": joint_margin.astype(np.float32),
            "joint_limit_violation": joint_violation.astype(bool),
        }
        if bool(config["output"]["save_qpos_retarget_raw_in_diagnostics"]):
            diagnostics["qpos_retarget_raw"] = qpos_retarget_raw.astype(np.float32)

        visual_alignment = _visual_alignment(
            ply_root, motion_id, len(visual_ids)
        )
        quality = _quality_report(
            motion_id,
            robot_id,
            arrays_30,
            diagnostics,
            joint_violation,
            visual_alignment,
            config,
        )
        canonical_metadata = load_json(canonical_metadata_path)
        metadata = {
            "schema_version": "robot_motion_metadata_v1",
            "generation_status": "completed",
            "created_by": "robot_dataset_stage2_v1",
            "created_at_utc": utc_now(),
            "code_identity": code_identity,
            "motion_id": motion_id,
            "robot_id": robot_id,
            "candidate_version": "v4",
            "candidate_manifest": str(candidate_path),
            "candidate_manifest_sha256": sha256_file(candidate_path),
            "source_relative_path": item["relative_path"],
            "source_dataset": item.get("source_dataset"),
            "subject_id": item.get("subject_id"),
            "source_fps": float(item["source_fps"]),
            "source_num_frames": int(item["source_num_frames"]),
            "canonical_fps": 30.0,
            "canonical_num_frames": canonical_frames,
            "visual_fps": 15.0,
            "visual_num_frames": len(visual_ids),
            "timeline_mode": "canonical_30fps_linear_translation_joint_slerp",
            "visual_sampling_mode": "canonical_even_frames_no_interpolation",
            "canonical_path": str(canonical_path),
            "canonical_sha256": sha256_file(canonical_path),
            "frame_map_path": str(frame_map_path),
            "frame_map_sha256": sha256_file(frame_map_path),
            "robot_schema": str(schema_path.relative_to(output_root)),
            "robot_schema_sha256": sha256_file(schema_path),
            "mujoco_xml_path": str(xml_path),
            "mujoco_xml_sha256": sha256_file(xml_path),
            "retarget_config_path": str(ik_path),
            "retarget_config_sha256": sha256_file(ik_path),
            "generation_config_path": str(config_path),
            "generation_config_sha256": sha256_file(config_path),
            "coordinate_system": schema["coordinate_system"],
            "length_unit": schema["length_unit"],
            "angle_unit": schema["angle_unit"],
            "dataset_quaternion_order": "xyzw",
            "mujoco_quaternion_order": "wxyz",
            "root_postprocess": root_offsets,
            "ground_alignment_mode": (
                "sequence_min_mujoco_body_origin_z"
                if postprocess["align_min_body_origin_to_ground"]
                else "none"
            ),
            "ik_iterations_per_enabled_stage": int(retargeter.max_iter),
            "ik_convergence_is_threshold_derived": True,
            "retarget_initialization_mode": (
                "seed_first_frame_free_root_from_human_root_target"
                if config["retarget"]["seed_first_frame_root_from_target"]
                else "gmr_default_configuration"
            ),
            "smplx_hand_pose_used_for_retarget": False,
            "visual_storage_id": config["paths"]["visual_storage_id"],
            "ply_relative_directory": motion_id,
            "ply_filename_pattern": "{visual_frame_id:06d}.ply",
            "visual_from_canonical": VISUAL_FROM_CANONICAL,
            "canonical_metadata": canonical_metadata,
            "generation_time_sec": float(time.perf_counter() - started),
        }

        atomic_write_npz(temporary / "trajectory_30fps.npz", arrays_30)
        atomic_write_npz(temporary / "trajectory_15fps.npz", arrays_15)
        atomic_write_npz(temporary / "diagnostics.npz", diagnostics)
        atomic_write_json(temporary / "metadata.json", metadata)
        atomic_write_json(temporary / "quality.json", quality)
        replace_directory_atomically(temporary, target)

        result = {
            "motion_id": motion_id,
            "robot_id": robot_id,
            "status": quality["generation_status"],
            "quality_status": quality["quality_status"],
            "frames_30fps": canonical_frames,
            "frames_15fps": len(visual_ids),
            "output": str(target),
            "generation_time_sec": metadata["generation_time_sec"],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    except BaseException as error:
        failure_dir = output_root / "failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "motion_id": motion_id,
            "robot_id": robot_id,
            "generation_status": "failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback_path": f"{motion_id}__{robot_id}.traceback.txt",
            "failed_at_utc": utc_now(),
        }
        atomic_write_json(
            failure_dir / f"{motion_id}__{robot_id}.json", failure
        )
        traceback_text = traceback.format_exc().encode("utf-8")
        from common import atomic_write_bytes

        atomic_write_bytes(
            failure_dir / f"{motion_id}__{robot_id}.traceback.txt",
            traceback_text,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retarget one frozen V4 canonical motion to one robot."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--motion-id", required=True)
    parser.add_argument("--robot")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
