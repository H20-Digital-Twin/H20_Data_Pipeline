"""Read AMASS SMPL-X metadata without modifying or resampling source files."""

import hashlib
import re
import traceback
from pathlib import Path

import numpy as np

from schemas import empty_record


def _first_key(available_keys, aliases):
    return next((key for key in aliases if key in available_keys), None)


def _clean_component(value):
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_")
    return cleaned or "unknown"


def _scalar(value):
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("expected a scalar value, got shape={}".format(array.shape))
    item = array.reshape(-1)[0]
    return item.item() if hasattr(item, "item") else item


def _identity(path, amass_root):
    relative = path.relative_to(amass_root).as_posix()
    parts = Path(relative).parts
    if len(parts) >= 3:
        source_dataset, subject_id = parts[0], parts[1]
    elif len(parts) == 2:
        source_dataset, subject_id = amass_root.name, parts[0]
    else:
        source_dataset, subject_id = amass_root.name, ""

    motion_name = path.name
    if motion_name.endswith("_stageii.npz"):
        motion_name = motion_name[: -len("_stageii.npz")]
    else:
        motion_name = path.stem

    base = "_".join(
        _clean_component(value)
        for value in (source_dataset, subject_id, motion_name)
        if value
    )
    short_hash = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:8]
    return {
        "motion_id": "{}_{}".format(base, short_hash),
        "source_dataset": source_dataset,
        "subject_id": subject_id,
        "motion_name": motion_name,
        "relative_path": relative,
    }


def make_base_record(path, amass_root):
    """Build file identity without opening the NPZ archive."""
    path = Path(path)
    record = empty_record()
    record.update(_identity(path, Path(amass_root)))
    record["file_size_mb"] = round(path.stat().st_size / (1024.0 * 1024.0), 6)
    return record


def _numeric_flags(array):
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("expected a numeric array, got dtype={}".format(array.dtype))
    return bool(np.isnan(array).any()), bool(np.isinf(array).any())


def _motion_hashes(poses, trans, dedup_config):
    exact = hashlib.sha1()
    for array in (poses, trans):
        contiguous = np.ascontiguousarray(array)
        exact.update(str(contiguous.shape).encode("ascii"))
        exact.update(str(contiguous.dtype).encode("ascii"))
        exact.update(contiguous.tobytes())

    sample_count = int(dedup_config["sample_frames"])
    indices = np.linspace(0, poses.shape[0] - 1, sample_count).round().astype(int)
    pose_sample = np.asarray(poses[indices], dtype=np.float64)
    trans_sample = np.asarray(trans[indices], dtype=np.float64)
    trans_sample = trans_sample - trans_sample[:1]
    feature = np.concatenate(
        (pose_sample.reshape(sample_count, -1), trans_sample.reshape(sample_count, -1)),
        axis=1,
    )
    tolerance = float(dedup_config["quantization_tolerance"])
    quantized = np.rint(feature / tolerance).astype(np.int64)
    near = hashlib.sha1(quantized.tobytes()).hexdigest()
    return exact.hexdigest(), near


def read_amass_file(path, amass_root, required_fields, scan_config):
    """Return one complete inventory record, including failures."""
    path = Path(path)
    amass_root = Path(amass_root)
    record = make_base_record(path, amass_root)

    try:
        with np.load(str(path), allow_pickle=False) as archive:
            keys = set(archive.files)
            pose_key = _first_key(keys, required_fields["pose_keys"])
            trans_key = _first_key(keys, required_fields["translation_keys"])
            fps_key = _first_key(keys, required_fields["fps_keys"])
            betas_key = _first_key(keys, required_fields.get("betas_keys", ["betas"]))
            gender_key = _first_key(keys, required_fields.get("gender_keys", ["gender"]))

            record.update(
                {
                    "pose_key": pose_key,
                    "trans_key": trans_key,
                    "fps_key": fps_key,
                    "has_pose": pose_key is not None,
                    "has_translation": trans_key is not None,
                    "has_fps": fps_key is not None,
                    "has_betas": betas_key is not None,
                    "has_gender": gender_key is not None,
                }
            )

            if pose_key is not None:
                poses = np.asarray(archive[pose_key])
                record["pose_shape_valid"] = poses.ndim == 2
                record["source_num_frames"] = int(poses.shape[0]) if poses.ndim else 0
                record["pose_dim"] = int(poses.shape[1]) if poses.ndim == 2 else None
                record["frame_count_valid"] = record["source_num_frames"] > 0
                record["pose_has_nan"], record["pose_has_inf"] = _numeric_flags(poses)
                if poses.size:
                    record["pose_abs_max"] = float(np.max(np.abs(poses)))
                    record["pose_motion_mean"] = (
                        float(np.mean(np.abs(np.diff(poses, axis=0))))
                        if poses.shape[0] > 1 else 0.0
                    )

            if trans_key is not None:
                trans = np.asarray(archive[trans_key])
                record["trans_shape_valid"] = trans.ndim == 2
                record["trans_dim"] = int(trans.shape[1]) if trans.ndim == 2 else None
                record["trans_num_frames"] = int(trans.shape[0]) if trans.ndim else 0
                record["trans_has_nan"], record["trans_has_inf"] = _numeric_flags(trans)
                if trans.size:
                    record["trans_abs_max"] = float(np.max(np.abs(trans)))
                if pose_key is not None and poses.ndim > 0 and trans.ndim > 0:
                    record["pose_trans_length_match"] = poses.shape[0] == trans.shape[0]

            if fps_key is not None:
                source_fps = float(_scalar(archive[fps_key]))
                record["source_fps"] = source_fps
                record["fps_valid"] = bool(np.isfinite(source_fps) and source_fps > 0)

            if betas_key is not None:
                betas = np.asarray(archive[betas_key])
                record["num_betas"] = int(betas.size)

            if gender_key is not None:
                record["gender"] = str(_scalar(archive[gender_key]))

            if (
                pose_key is not None
                and trans_key is not None
                and poses.ndim == 2
                and trans.ndim == 2
                and poses.shape[0] > 0
                and poses.shape[0] == trans.shape[0]
                and not any(
                    (
                        record["pose_has_nan"], record["pose_has_inf"],
                        record["trans_has_nan"], record["trans_has_inf"],
                    )
                )
                and scan_config.get("dedup", {}).get("enabled", True)
            ):
                (
                    record["exact_motion_hash"],
                    record["near_motion_hash"],
                ) = _motion_hashes(poses, trans, scan_config["dedup"])

        if record["fps_valid"] and record["frame_count_valid"]:
            duration = record["source_num_frames"] / record["source_fps"]
            target_fps = float(scan_config["target_fps"])
            record["duration_sec"] = round(duration, 6)
            record["expected_num_frames_at_30fps"] = int(round(duration * target_fps))

        record["npz_loaded"] = True
        return record
    except Exception as error:
        record["status"] = "reject_corrupt"
        record["status_reason"] = "{}: {}".format(type(error).__name__, error)
        record["error_type"] = type(error).__name__
        record["error_message"] = str(error)
        if scan_config.get("save_error_trace", True):
            record["error_trace"] = traceback.format_exc()
        return record
