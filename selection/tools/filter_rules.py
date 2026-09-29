"""Independent semantic, basic-validity and NumPy heuristic decisions."""

import math

from babel_reader import all_evidence, useful_frame_labels
from babel_timeline import ActionInterval, compute_timeline_statistics
from babel_timeline import intervals_union_duration
from category_taxonomy import normalize_label


def apply_semantic_filter(record, match, config):
    record["babel_match_status"] = match["babel_match_status"]
    record["babel_match_candidates"] = match.get("babel_match_candidates", [])
    entry = match.get("entry")
    if entry is None:
        record["semantic_status"] = (
            "babel_ambiguous"
            if match["babel_match_status"] == "ambiguous"
            else "babel_unmatched"
        )
        record["semantic_status_reason"] = (
            "no unique BABEL annotation; continue with NumPy filters"
        )
        return record

    record.update(
        {
            "babel_id": entry["babel_id"],
            "babel_split": entry["babel_split"],
            "babel_feat_p": entry["babel_feat_p"],
            "babel_duration_sec": entry["babel_duration_sec"],
            "sequence_tags": sorted(
                {value for label in entry["sequence_labels"] for value in label["evidence"]}
            ),
            "frame_tags": sorted(
                {value for label in entry["frame_labels"] for value in label["evidence"]}
            ),
        }
    )
    evidence = all_evidence(entry)
    hard = {normalize_label(value) for value in config["hard_blacklist"]}
    soft = {normalize_label(value) for value in config["soft_blacklist"]}
    hard_matches = sorted(evidence & hard)
    record["semantic_hard_matches"] = hard_matches
    if hard_matches:
        record["semantic_status"] = "reject_semantic_hard"
        record["semantic_status_reason"] = "hard blacklist label(s): {}".format(
            ", ".join(hard_matches)
        )
        return record

    duration = entry["babel_duration_sec"]
    frames = useful_frame_labels(entry)
    soft_intervals = []
    soft_matches = set()
    if frames:
        for label in frames:
            matched = set(label["evidence"]) & soft
            if matched:
                soft_matches.update(matched)
                soft_intervals.append((label["start"], label["end"]))
        coverage = (
            intervals_union_duration(soft_intervals, duration) / duration
            if duration > 0 else 0.0
        )
    else:
        sequence_evidence = {
            value
            for label in entry["sequence_labels"]
            for value in label["evidence"]
        }
        soft_matches = sequence_evidence & soft
        coverage = 1.0 if soft_matches and duration > 0 else 0.0

    record["semantic_soft_matches"] = sorted(soft_matches)
    record["soft_blacklist_coverage"] = round(coverage, 6)
    if coverage >= float(config["soft_blacklist_min_coverage"]):
        record["semantic_status"] = "reject_semantic_coverage"
        record["semantic_status_reason"] = (
            "soft blacklist union coverage={:.4f}, threshold={:.4f}; labels={}"
        ).format(
            coverage,
            config["soft_blacklist_min_coverage"],
            ", ".join(sorted(soft_matches)),
        )
    else:
        record["semantic_status"] = "semantic_passed"
        record["semantic_status_reason"] = "passed_semantic_filter"
    return record


def attach_timeline(record, entry, taxonomy, timeline_config):
    duration = float(record.get("duration_sec") or record.get("babel_duration_sec") or 0)
    intervals = []
    frames = useful_frame_labels(entry) if entry else []
    if frames and timeline_config["use_frame_ann_when_available"]:
        for label in frames:
            for category in taxonomy.map_labels(label["categories"]):
                intervals.append(
                    ActionInterval(category, label["start"], label["end"])
                )
    elif entry and timeline_config["sequence_ann_fallback"]:
        categories = set()
        for label in entry["sequence_labels"]:
            categories.update(taxonomy.map_labels(label["categories"]))
        for category in categories:
            intervals.append(ActionInterval(category, 0.0, duration))

    stats = compute_timeline_statistics(intervals, duration)
    inclusive = dict(stats["inclusive_duration"])
    allocated = dict(stats["allocated_duration"])
    unlabeled = stats["unlabeled_duration"]
    policy = timeline_config["unlabeled_policy"]
    if policy == "keep_unknown":
        if unlabeled > 0:
            inclusive["unknown"] = unlabeled
            allocated["unknown"] = unlabeled
    elif policy == "fill_with_primary_seq_label":
        seq_categories = set()
        if entry:
            for label in entry["sequence_labels"]:
                seq_categories.update(taxonomy.map_labels(label["categories"]))
        target = sorted(seq_categories)[0] if seq_categories else "unknown"
        inclusive[target] = inclusive.get(target, 0.0) + unlabeled
        allocated[target] = allocated.get(target, 0.0) + unlabeled
    elif policy != "ignore":
        raise ValueError(
            "unsupported unlabeled_policy: {}".format(
                policy
            )
        )

    allocated_total = sum(allocated.values())
    proportions = (
        {key: value / duration for key, value in allocated.items()}
        if duration > 0 else {}
    )
    if allocated:
        primary = sorted(allocated, key=lambda key: (-allocated[key], key))[0]
        dominance = allocated[primary] / duration if duration > 0 else 0.0
    else:
        primary, dominance = "unknown", 0.0
    record.update(
        {
            "category_inclusive_duration": {
                key: round(value, 6) for key, value in sorted(inclusive.items())
            },
            "category_allocated_duration": {
                key: round(value, 6) for key, value in sorted(allocated.items())
            },
            "category_proportions": {
                key: round(value, 8) for key, value in sorted(proportions.items())
            },
            "unlabeled_duration_sec": round(unlabeled, 6),
            "primary_category": primary,
            "dominant_category_ratio": round(dominance, 8),
        }
    )
    # Retain full precision only for the invariant check.
    if policy != "ignore" and duration > 0:
        if not math.isclose(allocated_total, duration, abs_tol=1e-5, rel_tol=1e-6):
            raise AssertionError(
                "allocated timeline {:.6f} != duration {:.6f}".format(
                    allocated_total, duration
                )
            )
    return record


def classify_basic(record, config):
    if record.get("error_type"):
        record["basic_status"] = "reject_corrupt"
        record["basic_status_reason"] = record["status_reason"]
        return record
    missing = []
    if not record["has_pose"]:
        missing.append("pose")
    if config["require_translation"] and not record["has_translation"]:
        missing.append("translation")
    if not record["has_fps"]:
        missing.append("fps")
    if missing:
        record["basic_status"] = "reject_missing_field"
        record["basic_status_reason"] = "missing required field(s): {}".format(
            ", ".join(missing)
        )
        return record

    reasons = []
    if not record["pose_shape_valid"]:
        reasons.append("pose array must be 2D")
    if not record["trans_shape_valid"]:
        reasons.append("translation array must be 2D")
    if not record["frame_count_valid"]:
        reasons.append("pose frame count must be > 0")
    if not record["fps_valid"]:
        reasons.append("FPS must be finite and > 0")
    if config["require_pose_translation_same_length"] and not record[
        "pose_trans_length_match"
    ]:
        reasons.append(
            "pose frames={} do not match translation frames={}".format(
                record["source_num_frames"], record["trans_num_frames"]
            )
        )
    if config["reject_nan"] and (record["pose_has_nan"] or record["trans_has_nan"]):
        reasons.append("pose or translation contains NaN")
    if config["reject_inf"] and (record["pose_has_inf"] or record["trans_has_inf"]):
        reasons.append("pose or translation contains Inf")
    if reasons:
        record["basic_status"] = "reject_invalid_value"
        record["basic_status_reason"] = "; ".join(reasons)
        return record

    duration = float(record["duration_sec"])
    source_fps = float(record["source_fps"])
    target_fps = float(config["target_fps"])
    ratio = source_fps / target_fps
    nearest = round(ratio)
    record["fps_supported_v1"] = bool(
        nearest >= 1
        and math.isclose(
            ratio, nearest, rel_tol=0.0,
            abs_tol=float(config["fps_ratio_tolerance"])
        )
    )
    record["duration_supported_v1"] = bool(
        config["min_duration_sec"] <= duration <= config["max_duration_v1_sec"]
    )
    if duration < config["min_duration_sec"]:
        status = "review_short"
        reason = "duration={:.4f}s, below minimum {:.4f}s".format(
            duration, config["min_duration_sec"]
        )
    elif duration > config["max_duration_defer_sec"]:
        status = "defer_very_long"
        reason = "duration={:.4f}s, above defer limit {:.4f}s".format(
            duration, config["max_duration_defer_sec"]
        )
    elif duration > config["max_duration_v1_sec"]:
        status = "defer_long"
        reason = "duration={:.4f}s, above v1 maximum {:.4f}s".format(
            duration, config["max_duration_v1_sec"]
        )
    elif source_fps < target_fps:
        status = "defer_low_fps"
        reason = "source_fps={:.6g}, below target_fps={:.6g}".format(
            source_fps, target_fps
        )
    elif not record["fps_supported_v1"]:
        status = "defer_fps"
        reason = "source_fps={:.6g}, not an integer multiple of {:.6g}".format(
            source_fps, target_fps
        )
    else:
        status, reason = "basic_passed", "passed_v1_basic_filter"
    record["basic_status"], record["basic_status_reason"] = status, reason
    return record


def classify_heuristic(record, config):
    if record["basic_status"] != "basic_passed":
        record["heuristic_status"] = "not_run"
        record["heuristic_status_reason"] = "basic filter did not pass"
        return record
    reasons = []
    if record["pose_abs_max"] > config["max_abs_pose_value"]:
        reasons.append(
            "pose_abs_max={:.6g} exceeds {:.6g}".format(
                record["pose_abs_max"], config["max_abs_pose_value"]
            )
        )
    if record["trans_abs_max"] > config["max_abs_translation_value"]:
        reasons.append(
            "trans_abs_max={:.6g} exceeds {:.6g}".format(
                record["trans_abs_max"], config["max_abs_translation_value"]
            )
        )
    if record["pose_motion_mean"] < config["min_pose_motion_mean"]:
        reasons.append(
            "pose_motion_mean={:.6g} below {:.6g}".format(
                record["pose_motion_mean"], config["min_pose_motion_mean"]
            )
        )
    record["heuristic_all_reasons"] = reasons
    if reasons:
        record["heuristic_status"] = "reject_heuristic"
        record["heuristic_status_reason"] = "; ".join(reasons)
        record["heuristic_quality_score"] = 0.0
    else:
        record["heuristic_status"] = "heuristic_passed"
        record["heuristic_status_reason"] = "passed_numpy_heuristics"
        motion_scale = max(config["quality_motion_scale"], 1e-12)
        record["heuristic_quality_score"] = round(
            min(1.0, record["pose_motion_mean"] / motion_scale), 8
        )
    return record
