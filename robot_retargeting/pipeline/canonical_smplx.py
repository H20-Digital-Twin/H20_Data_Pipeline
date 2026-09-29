from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterator

import numpy as np
from scipy.spatial.transform import Rotation
from smplx.joint_names import JOINT_NAMES

HERE = Path(__file__).resolve().parent
GMR_ROOT = HERE.parent
if str(GMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GMR_ROOT))

from general_motion_retargeting.utils.smpl import load_smplx_file


REQUIRED_CANONICAL_KEYS = {
    "fps",
    "trans",
    "root_orient",
    "pose_body",
    "pose_hand",
    "pose_jaw",
    "betas",
    "gender",
    "num_betas",
    "source_frame_float",
    "timestamp_sec",
}


def load_canonical_human_motion(
    canonical_path: Path,
    body_model_root: Path,
) -> tuple[dict, Iterator[dict[str, tuple[np.ndarray, np.ndarray]]], float]:
    """Load canonical 30 FPS once and yield GMR human dictionaries without resampling."""
    with np.load(canonical_path, allow_pickle=False) as archive:
        missing = sorted(REQUIRED_CANONICAL_KEYS.difference(archive.files))
        if missing:
            raise KeyError(f"canonical motion is missing fields: {missing}")
        canonical = {key: np.asarray(archive[key]) for key in archive.files}

    fps = float(np.asarray(canonical["fps"]).item())
    if not np.isclose(fps, 30.0):
        raise ValueError(f"canonical FPS must be 30, got {fps}")
    timestamps = np.asarray(canonical["timestamp_sec"], dtype=np.float64)
    if not np.allclose(timestamps, np.arange(len(timestamps)) / 30.0, atol=1e-9):
        raise ValueError("canonical timestamp_sec is not the expected 30 FPS grid")

    # Reuse the project's established SMPL-X body-model construction. This does
    # not resample: the input archive is already canonical 30 FPS.
    _, body_model, smplx_output, human_height = load_smplx_file(
        str(canonical_path), str(body_model_root)
    )
    global_orient = (
        smplx_output.global_orient.detach().cpu().numpy().reshape(-1, 3)
    )
    full_pose = (
        smplx_output.full_pose.detach().cpu().numpy().reshape(len(timestamps), -1, 3)
    )
    joints = smplx_output.joints.detach().cpu().numpy()
    parents = np.asarray(body_model.parents, dtype=np.int64)
    joint_names = JOINT_NAMES[: len(parents)]

    if len(global_orient) != len(timestamps) or len(joints) != len(timestamps):
        raise ValueError("SMPL-X body output does not match canonical frame count")

    def frames() -> Iterator[dict[str, tuple[np.ndarray, np.ndarray]]]:
        for frame_index in range(len(timestamps)):
            result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            global_rotations: list[Rotation] = []
            for joint_index, joint_name in enumerate(joint_names):
                if joint_index == 0:
                    rotation = Rotation.from_rotvec(global_orient[frame_index])
                else:
                    parent = int(parents[joint_index])
                    rotation = global_rotations[parent] * Rotation.from_rotvec(
                        full_pose[frame_index, joint_index]
                    )
                global_rotations.append(rotation)
                quat_xyzw = rotation.as_quat()
                quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
                result[joint_name] = (
                    np.asarray(joints[frame_index, joint_index], dtype=np.float64),
                    np.asarray(quat_wxyz, dtype=np.float64),
                )
            yield result

    return canonical, frames(), float(human_height)
