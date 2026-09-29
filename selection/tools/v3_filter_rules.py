"""Three-tier, fully configured V3 robot-suitability heuristic decisions."""

import numpy as np


REJECT_PRIORITY = [
    "reject_floor_high_confidence",
    "reject_crawl_high_confidence",
    "reject_static_high_confidence",
    "reject_sitting_high_confidence",
    "reject_abnormal_high_confidence",
]

REVIEW_PRIORITY = [
    "review_floor_or_crouch",
    "review_sitting_or_squat",
    "review_crawling",
    "review_static",
    "review_body_speed",
    "review_calibration",
]


def resolve_joint_motion_threshold(metric_rows, config):
    static = config["static"]["reject"]
    if static.get("joint_motion_integral_max_rad") is not None:
        return float(static["joint_motion_integral_max_rad"])
    percentile = float(static["joint_motion_integral_max_percentile"])
    return float(
        np.quantile(
            [row["joint_motion_integral_rad"] for row in metric_rows],
            percentile,
        )
    )


def _semantic_sit(record, config):
    labels = set(record.get("semantic_soft_matches") or [])
    sit_labels = {str(value).casefold() for value in config["sitting"]["sit_labels"]}
    coverage = (
        float(record.get("soft_blacklist_coverage") or 0.0)
        if labels & sit_labels
        else 0.0
    )
    primary_supported = record.get("primary_category") in set(
        config["sitting"]["primary_categories"]
    )
    supported = (
        coverage >= config["sitting"]["reject"]["semantic_coverage_min"]
        or primary_supported
    )
    return supported, coverage


def decide_heuristic(record, metrics, config, joint_motion_threshold):
    reject_reasons = []
    review_reasons = []

    floor_reject = config["floor_motion"]["reject"]
    floor_review = config["floor_motion"]["review"]
    if (
        metrics["lying_fraction"] >= floor_reject["lying_fraction_min"]
        or metrics["floor_fraction"] >= floor_reject["floor_fraction_min"]
    ):
        reject_reasons.append("reject_floor_high_confidence")
    elif (
        metrics["lying_fraction"] >= floor_review["lying_fraction_min"]
        or metrics["floor_fraction"] >= floor_review["floor_fraction_min"]
    ):
        review_reasons.append("review_floor_or_crouch")

    crawl_reject = config["crawling"]["reject"]
    crawl_review = config["crawling"]["review"]
    if (
        metrics["crawl_fraction"] >= crawl_reject["crawl_fraction_min"]
        and metrics["floor_fraction"] >= crawl_reject["floor_fraction_min"]
    ):
        reject_reasons.append("reject_crawl_high_confidence")
    elif metrics["crawl_fraction"] >= crawl_review["crawl_fraction_min"]:
        review_reasons.append("review_crawling")

    static_reject = config["static"]["reject"]
    static_review = config["static"]["review"]
    reject_static = (
        metrics["static_fraction"] >= static_reject["static_fraction_min"]
        and metrics["root_path_length_m"]
        <= static_reject["root_path_length_max_m"]
        and (
            not static_reject["require_low_joint_motion"]
            or metrics["joint_motion_integral_rad"] <= joint_motion_threshold
        )
    )
    if reject_static:
        reject_reasons.append("reject_static_high_confidence")
    elif metrics["static_fraction"] >= static_review["static_fraction_min"]:
        review_reasons.append("review_static")

    semantic_sit, sit_coverage = _semantic_sit(record, config)
    sitting_reject = config["sitting"]["reject"]
    sitting_review = config["sitting"]["review"]
    if (
        metrics["sitting_fraction"] >= sitting_reject["sitting_fraction_min"]
        and (
            not sitting_reject["require_semantic_support"] or semantic_sit
        )
    ):
        reject_reasons.append("reject_sitting_high_confidence")
    elif metrics["sitting_fraction"] >= sitting_review["sitting_fraction_min"]:
        review_reasons.append("review_sitting_or_squat")

    abnormal_reject = config["abnormal_motion"]["reject"]
    abnormal_review = config["abnormal_motion"]["review"]
    abnormal_auxiliary = (
        metrics["jump_fraction"] >= abnormal_reject["jump_fraction_min"]
        or metrics["extreme_frame_fraction"]
        >= abnormal_reject["extreme_frame_fraction_min"]
        or metrics["max_consecutive_extreme_frames"]
        >= abnormal_reject["min_consecutive_abnormal_frames"]
    )
    if (
        metrics["body_ang_speed_p99"]
        >= abnormal_reject["body_ang_speed_p99_min_rps"]
        and abnormal_auxiliary
    ):
        reject_reasons.append("reject_abnormal_high_confidence")
    elif (
        metrics["body_ang_speed_p99"]
        >= abnormal_review["body_ang_speed_p99_min_rps"]
    ):
        review_reasons.append("review_body_speed")

    root_reject = config["root_motion"]["reject"]
    root_review = config["root_motion"]["review"]
    if metrics["root_speed_p99"] > root_reject["root_speed_p99_max_mps"]:
        reject_reasons.append("reject_abnormal_high_confidence")
    elif metrics["root_speed_p99"] > root_review["root_speed_p99_max_mps"]:
        review_reasons.append("review_body_speed")

    calibration_reject = config["calibration"]["reject"]
    calibration_review = config["calibration"]["review"]
    if (
        metrics["calibration_fraction"]
        >= calibration_reject["calibration_fraction_min"]
        and metrics["static_fraction"]
        >= calibration_reject["static_fraction_min"]
        and metrics["calibration_longest_run_sec"]
        >= calibration_reject["longest_run_sec_min"]
    ):
        reject_reasons.append("reject_static_high_confidence")
    elif (
        metrics["calibration_fraction"]
        >= calibration_review["calibration_fraction_min"]
    ):
        review_reasons.append("review_calibration")

    reject_reasons = [
        reason for reason in REJECT_PRIORITY if reason in set(reject_reasons)
    ]
    review_reasons = [
        reason for reason in REVIEW_PRIORITY if reason in set(review_reasons)
    ]
    all_reasons = reject_reasons + review_reasons
    if reject_reasons:
        status = "reject_heuristic_high_confidence"
        primary = reject_reasons[0]
    elif len(review_reasons) > 1:
        status = "review_heuristic"
        primary = "review_multiple_heuristics"
    elif review_reasons:
        status = "review_heuristic"
        primary = review_reasons[0]
    else:
        status = "heuristic_passed"
        primary = "heuristic_passed"
    return {
        "heuristic_status": status,
        "heuristic_primary_reason": primary,
        "heuristic_all_reasons": all_reasons,
        "heuristic_review_reasons": review_reasons,
        "heuristic_reject_reasons": reject_reasons,
        "babel_sit_coverage": sit_coverage,
        "semantic_sit_supported": semantic_sit,
        "joint_motion_integral_threshold_rad": joint_motion_threshold,
    }
