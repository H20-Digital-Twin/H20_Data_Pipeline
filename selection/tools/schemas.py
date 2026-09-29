"""Shared output schemas for the auditable AMASS candidate builder."""

INVENTORY_FIELDS = [
    "motion_id", "source_dataset", "subject_id", "motion_name",
    "relative_path", "file_size_mb",
    "babel_id", "babel_split", "babel_feat_p", "babel_match_status",
    "babel_match_candidates", "babel_duration_sec",
    "sequence_tags", "frame_tags",
    "semantic_status", "semantic_status_reason",
    "soft_blacklist_coverage", "semantic_hard_matches",
    "semantic_soft_matches",
    "source_fps", "source_num_frames", "duration_sec",
    "expected_num_frames_at_30fps",
    "pose_key", "trans_key", "fps_key", "pose_dim", "trans_dim",
    "trans_num_frames", "num_betas", "gender",
    "has_pose", "has_translation", "has_fps", "has_betas", "has_gender",
    "pose_has_nan", "pose_has_inf", "trans_has_nan", "trans_has_inf",
    "fps_valid", "frame_count_valid", "pose_shape_valid",
    "trans_shape_valid", "pose_trans_length_match",
    "fps_supported_v1", "duration_supported_v1",
    "basic_status", "basic_status_reason",
    "pose_abs_max", "trans_abs_max", "pose_motion_mean",
    "heuristic_quality_score", "heuristic_status",
    "heuristic_status_reason", "heuristic_all_reasons",
    "exact_motion_hash", "near_motion_hash",
    "dedup_status", "duplicate_of",
    "category_inclusive_duration", "category_allocated_duration",
    "category_proportions", "unlabeled_duration_sec",
    "primary_category", "dominant_category_ratio",
    "selection_score", "quota_status", "quota_status_reason",
    "final_status", "final_status_reason",
    "status", "status_reason",
    "npz_loaded", "error_type", "error_message", "error_trace",
]

CANDIDATE_FIELDS = [
    "motion_id", "source_dataset", "subject_id", "motion_name",
    "relative_path", "source_fps", "source_num_frames", "duration_sec",
    "expected_num_frames_at_30fps", "babel_id", "babel_split",
    "babel_match_status", "sequence_tags", "frame_tags",
    "primary_category", "category_inclusive_duration",
    "category_allocated_duration", "category_proportions",
    "heuristic_quality_score", "selection_score", "final_status",
]

DUAL_FILTER_FIELDS = CANDIDATE_FIELDS + [
    "dedup_status", "duplicate_of", "quota_status", "quota_status_reason"
]

FINAL_STATUSES = [
    "selected_candidate",
    "defer_quota_category",
    "defer_global_budget",
    "reject_semantic_hard",
    "reject_semantic_coverage",
    "review_short",
    "defer_long",
    "defer_very_long",
    "defer_fps",
    "defer_low_fps",
    "reject_corrupt",
    "reject_missing_field",
    "reject_invalid_value",
    "reject_heuristic",
    "defer_duplicate",
]

STRUCTURED_FIELDS = {
    "babel_match_candidates", "sequence_tags", "frame_tags",
    "semantic_hard_matches", "semantic_soft_matches",
    "heuristic_all_reasons", "category_inclusive_duration",
    "category_allocated_duration", "category_proportions",
}


def empty_record():
    record = {field: None for field in INVENTORY_FIELDS}
    for field in (
        "has_pose", "has_translation", "has_fps", "has_betas", "has_gender",
        "pose_has_nan", "pose_has_inf", "trans_has_nan", "trans_has_inf",
        "fps_valid", "frame_count_valid", "pose_shape_valid",
        "trans_shape_valid", "pose_trans_length_match",
        "fps_supported_v1", "duration_supported_v1", "npz_loaded",
    ):
        record[field] = False
    for field in (
        "babel_match_candidates", "sequence_tags", "frame_tags",
        "semantic_hard_matches", "semantic_soft_matches",
        "heuristic_all_reasons",
    ):
        record[field] = []
    for field in (
        "category_inclusive_duration", "category_allocated_duration",
        "category_proportions",
    ):
        record[field] = {}
    return record
