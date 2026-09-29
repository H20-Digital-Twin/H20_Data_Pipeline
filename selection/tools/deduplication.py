"""Deterministic exact and quantized near-duplicate suppression."""


def apply_deduplication(records, config):
    eligible = [
        record for record in records
        if record["semantic_status"] in ("semantic_passed", "babel_unmatched", "babel_ambiguous")
        and record["heuristic_status"] == "heuristic_passed"
    ]
    if not config["enabled"]:
        for record in eligible:
            record["dedup_status"] = "unique"
        return records

    eligible.sort(
        key=lambda item: (
            -float(item["heuristic_quality_score"]),
            item["motion_id"],
        )
    )
    seen_exact = {}
    seen_near = {}
    for record in eligible:
        scope = record[config["scope_field"]]
        exact_key = (scope, record["exact_motion_hash"])
        near_key = (scope, record["near_motion_hash"])
        duplicate_of = None
        reason = None
        if record["exact_motion_hash"] and exact_key in seen_exact:
            duplicate_of = seen_exact[exact_key]
            reason = "duplicate_exact"
        elif record["near_motion_hash"] and near_key in seen_near:
            duplicate_of = seen_near[near_key]
            reason = "duplicate_near_quantized"
        if duplicate_of:
            record["dedup_status"] = reason
            record["duplicate_of"] = duplicate_of
        else:
            record["dedup_status"] = "unique"
            if record["exact_motion_hash"]:
                seen_exact[exact_key] = record["motion_id"]
            if record["near_motion_hash"]:
                seen_near[near_key] = record["motion_id"]
    return records
