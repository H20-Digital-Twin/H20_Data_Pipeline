"""Read-only SMPL-X/NumPy motion metrics for heuristic auditing."""

import math
from pathlib import Path

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation


METRIC_NAMES = [
    "static_fraction",
    "root_path_length_m",
    "joint_motion_integral_rad",
    "calibration_fraction",
    "sitting_fraction",
    "lying_fraction",
    "crawl_fraction",
    "floor_fraction",
    "root_speed_p99",
    "body_ang_speed_p99",
    "jump_fraction",
    "extreme_frame_fraction",
    "calibration_longest_run_sec",
    "max_consecutive_extreme_frames",
]


def _angle(a, b):
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cosine = np.sum(a * b, axis=-1) / np.maximum(denom, 1e-8)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _sample_indices(num_frames, source_fps, target_fps):
    if source_fps > target_fps:
        frame_skip = max(1, int(source_fps / target_fps))
        new_count = max(1, num_frames // frame_skip)
        return np.rint(np.linspace(0, num_frames - 1, new_count)).astype(int)
    return np.arange(num_frames, dtype=int)


def _angular_displacements(body_pose):
    if len(body_pose) < 2:
        return np.zeros((0, body_pose.shape[1] // 3), dtype=np.float64)
    current = Rotation.from_rotvec(body_pose[:-1].reshape(-1, 3))
    following = Rotation.from_rotvec(body_pose[1:].reshape(-1, 3))
    relative = current.inv() * following
    return relative.magnitude().reshape(len(body_pose) - 1, -1)


def _longest_true_run(mask):
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


class SMPLXMetricExtractor:
    def __init__(self, config):
        self.config = config
        torch.set_num_threads(int(config.get("torch_num_threads", 4)))
        self.device = torch.device(config.get("device", "cpu"))
        self.models = {}

    def _model(self, gender):
        gender = gender if gender in ("male", "female", "neutral") else "neutral"
        if gender not in self.models:
            self.models[gender] = smplx.create(
                self.config["smplx_model_path"],
                model_type="smplx",
                gender=gender,
                use_pca=False,
                num_betas=int(self.config["num_betas"]),
            ).to(self.device).eval()
        return self.models[gender]

    def _forward_joints(self, model, arrays):
        count = len(arrays["body_pose"])
        outputs = []
        batch_size = int(self.config["fk_batch_size"])
        betas = torch.as_tensor(arrays["betas"], dtype=torch.float32)
        for start in range(0, count, batch_size):
            end = min(count, start + batch_size)
            size = end - start
            with torch.no_grad():
                result = model(
                    betas=betas[None].expand(size, -1).to(self.device),
                    global_orient=torch.as_tensor(
                        arrays["root_orient"][start:end], dtype=torch.float32,
                        device=self.device,
                    ),
                    body_pose=torch.as_tensor(
                        arrays["body_pose"][start:end], dtype=torch.float32,
                        device=self.device,
                    ),
                    transl=torch.as_tensor(
                        arrays["trans"][start:end], dtype=torch.float32,
                        device=self.device,
                    ),
                    left_hand_pose=torch.zeros(
                        size, 45, dtype=torch.float32, device=self.device
                    ),
                    right_hand_pose=torch.zeros(
                        size, 45, dtype=torch.float32, device=self.device
                    ),
                    expression=torch.zeros(
                        size, 10, dtype=torch.float32, device=self.device
                    ),
                    jaw_pose=torch.zeros(
                        size, 3, dtype=torch.float32, device=self.device
                    ),
                    leye_pose=torch.zeros(
                        size, 3, dtype=torch.float32, device=self.device
                    ),
                    reye_pose=torch.zeros(
                        size, 3, dtype=torch.float32, device=self.device
                    ),
                    return_verts=False,
                )
            outputs.append(result.joints[:, :22].detach().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def extract_joints(self, path, render_fps=None, max_frames=None):
        """Return read-only Neutral/gender-specific 22-joint FK for review rendering."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            source_fps = float(
                data[
                    "mocap_frame_rate"
                    if "mocap_frame_rate" in data.files
                    else "mocap_framerate"
                ]
            )
            body = np.asarray(data["pose_body"], dtype=np.float64)
            target_fps = float(render_fps or self.config["target_fps"])
            indices = _sample_indices(len(body), source_fps, target_fps)
            if max_frames and len(indices) > int(max_frames):
                indices = indices[
                    np.rint(
                        np.linspace(0, len(indices) - 1, int(max_frames))
                    ).astype(int)
                ]
            arrays = {
                "body_pose": body[indices],
                "root_orient": np.asarray(
                    data["root_orient"], dtype=np.float64
                )[indices],
                "trans": np.asarray(data["trans"], dtype=np.float64)[indices],
                "betas": np.asarray(data["betas"], dtype=np.float64)[
                    : int(self.config["num_betas"])
                ],
            }
            gender = str(np.asarray(data["gender"]).item()).casefold()
            if self.config.get("force_gender"):
                gender = str(self.config["force_gender"]).casefold()
        joints = self._forward_joints(self._model(gender), arrays)
        effective_fps = (
            len(indices) / len(body) * source_fps
            if source_fps > target_fps
            else source_fps
        )
        return joints, float(effective_fps)

    def extract(self, path):
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            source_fps = float(
                data[
                    "mocap_frame_rate"
                    if "mocap_frame_rate" in data.files
                    else "mocap_framerate"
                ]
            )
            body = np.asarray(data["pose_body"], dtype=np.float64)
            indices = _sample_indices(
                len(body), source_fps, float(self.config["target_fps"])
            )
            arrays = {
                "body_pose": body[indices],
                "root_orient": np.asarray(data["root_orient"], dtype=np.float64)[
                    indices
                ],
                "trans": np.asarray(data["trans"], dtype=np.float64)[indices],
                "betas": np.asarray(data["betas"], dtype=np.float64)[
                    : int(self.config["num_betas"])
                ],
            }
            gender = str(np.asarray(data["gender"]).item()).casefold()
            if self.config.get("force_gender"):
                gender = str(self.config["force_gender"]).casefold()

        fps = (
            len(indices) / len(body) * source_fps
            if source_fps > self.config["target_fps"]
            else source_fps
        )
        joints = self._forward_joints(self._model(gender), arrays)
        body_pose = arrays["body_pose"]
        trans = arrays["trans"]
        delta_trans = np.diff(trans, axis=0)
        root_speed = np.linalg.norm(delta_trans, axis=1) * fps
        angular_delta = _angular_displacements(body_pose)
        angular_speed = angular_delta * fps
        frame_body_speed = (
            np.max(angular_speed, axis=1)
            if angular_speed.size else np.zeros(0)
        )
        mean_body_speed = (
            np.mean(angular_speed, axis=1)
            if angular_speed.size else np.zeros(0)
        )

        # Anthropometric scale from stable SMPL-X bone lengths.
        edges = [
            (0, 1), (1, 4), (4, 7), (7, 10),
            (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
        ]
        scale = float(
            np.median(
                sum(np.linalg.norm(joints[:, a] - joints[:, b], axis=1) for a, b in edges)
            )
        )
        scale = max(scale, 1e-6)
        z = joints[:, :, 2]
        vertical_span = (z.max(axis=1) - z.min(axis=1)) / scale
        feet_z = np.min(z[:, [7, 8, 10, 11]], axis=1)
        ground = float(np.percentile(feet_z, 1))
        pelvis_height = (z[:, 0] - ground) / scale
        head_height = (z[:, 15] - ground) / scale
        wrist_height = (z[:, [20, 21]].mean(axis=1) - ground) / scale
        knee_height = (z[:, [4, 5]].mean(axis=1) - ground) / scale

        torso = joints[:, 15] - joints[:, 0]
        torso_vertical = np.abs(torso[:, 2]) / np.maximum(
            np.linalg.norm(torso, axis=1), 1e-8
        )
        left_knee = _angle(joints[:, 1] - joints[:, 4], joints[:, 7] - joints[:, 4])
        right_knee = _angle(
            joints[:, 2] - joints[:, 5], joints[:, 8] - joints[:, 5]
        )
        knee_angle = np.minimum(left_knee, right_knee)

        left_elbow = _angle(
            joints[:, 16] - joints[:, 18], joints[:, 20] - joints[:, 18]
        )
        right_elbow = _angle(
            joints[:, 17] - joints[:, 19], joints[:, 21] - joints[:, 19]
        )
        arms_straight = np.minimum(left_elbow, right_elbow) > math.radians(
            self.config["calibration_elbow_min_deg"]
        )
        shoulder_z = z[:, [16, 17]].mean(axis=1)
        wrist_z = z[:, [20, 21]].mean(axis=1)
        wrist_drop = (shoulder_z - wrist_z) / scale
        wrist_span = (
            np.linalg.norm(joints[:, 20] - joints[:, 21], axis=1) / scale
        )
        t_or_a_pose = (
            arms_straight
            & (wrist_span > self.config["calibration_wrist_span_ratio"])
            & (wrist_drop >= self.config["calibration_wrist_drop_min_ratio"])
            & (wrist_drop <= self.config["calibration_wrist_drop_max_ratio"])
            & (torso_vertical > self.config["upright_torso_ratio"])
        )

        # Speeds live between frames; pad the first frame for fraction metrics.
        root_speed_frame = np.r_[root_speed[:1], root_speed] if root_speed.size else np.zeros(len(joints))
        mean_body_speed_frame = (
            np.r_[mean_body_speed[:1], mean_body_speed]
            if mean_body_speed.size else np.zeros(len(joints))
        )
        max_body_speed_frame = (
            np.r_[frame_body_speed[:1], frame_body_speed]
            if frame_body_speed.size else np.zeros(len(joints))
        )
        static_mask = (
            (root_speed_frame < self.config["static_root_speed_mps"])
            & (mean_body_speed_frame < self.config["static_body_ang_speed_radps"])
        )
        calibration_mask = t_or_a_pose & static_mask
        sitting_mask = (
            (pelvis_height < self.config["sitting_pelvis_height_ratio"])
            & (torso_vertical > self.config["upright_torso_ratio"])
            & (knee_angle < math.radians(self.config["sitting_knee_angle_max_deg"]))
        )
        lying_mask = (
            (vertical_span < self.config["lying_vertical_span_ratio"])
            & (torso_vertical < self.config["lying_torso_vertical_ratio"])
            & (np.minimum(pelvis_height, head_height)
               < self.config["floor_height_ratio"])
        )
        crawl_mask = (
            (pelvis_height < self.config["crawl_pelvis_height_ratio"])
            & (torso_vertical < self.config["crawl_torso_vertical_ratio"])
            & (wrist_height < self.config["crawl_contact_height_ratio"])
            & (knee_height < self.config["crawl_contact_height_ratio"])
        )
        floor_mask = (
            np.minimum(pelvis_height, head_height)
            < self.config["floor_height_ratio"]
        )
        jump_mask = feet_z > ground + self.config["jump_height_m"]
        extreme_mask = (
            (root_speed_frame > self.config["extreme_root_speed_mps"])
            | (max_body_speed_frame > self.config["extreme_body_ang_speed_radps"])
        )

        metrics = {
            "static_fraction": float(np.mean(static_mask)),
            "root_path_length_m": float(np.sum(np.linalg.norm(delta_trans, axis=1))),
            "joint_motion_integral_rad": float(np.sum(angular_delta)),
            "calibration_fraction": float(np.mean(calibration_mask)),
            "sitting_fraction": float(np.mean(sitting_mask)),
            "lying_fraction": float(np.mean(lying_mask)),
            "crawl_fraction": float(np.mean(crawl_mask)),
            "floor_fraction": float(np.mean(floor_mask)),
            "root_speed_p99": float(np.percentile(root_speed, 99)) if root_speed.size else 0.0,
            "body_ang_speed_p99": (
                float(np.percentile(frame_body_speed, 99))
                if frame_body_speed.size else 0.0
            ),
            "jump_fraction": float(np.mean(jump_mask)),
            "extreme_frame_fraction": float(np.mean(extreme_mask)),
            "calibration_longest_run_sec": (
                float(_longest_true_run(calibration_mask) / fps) if fps > 0 else 0.0
            ),
            "max_consecutive_extreme_frames": int(
                _longest_true_run(extreme_mask)
            ),
        }

        descriptor_count = int(self.config["dedup_descriptor_frames"])
        descriptor_idx = np.rint(
            np.linspace(0, len(body_pose) - 1, descriptor_count)
        ).astype(int)
        pose_descriptor = body_pose[descriptor_idx].reshape(-1)
        root_descriptor = trans[descriptor_idx] - trans[descriptor_idx[:1]]
        return {
            "metrics": metrics,
            "sampled_frame_count": len(indices),
            "effective_fps": float(fps),
            "body_scale_m": scale,
            "pose_descriptor": pose_descriptor.astype(np.float32),
            "root_descriptor": root_descriptor.reshape(-1).astype(np.float32),
        }
