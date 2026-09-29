"""Multi-stage category and rejection statistics for auditing and papers."""

from collections import Counter, defaultdict
from pathlib import Path

from schemas import FINAL_STATUSES


def category_scope(records):
    total_count = len(records)
    total_duration = sum(
        float(item.get("duration_sec") or item.get("babel_duration_sec") or 0.0)
        for item in records
    )
    matched = sum(
        str(item.get("babel_match_status", "")).startswith("match_")
        for item in records
    )
    category_data = defaultdict(
        lambda: {
            "sequence_count_any": 0,
            "primary_sequence_count": 0,
            "inclusive_duration_sec": 0.0,
            "allocated_duration_sec": 0.0,
        }
    )
    for item in records:
        inclusive = item.get("category_inclusive_duration") or {}
        allocated = item.get("category_allocated_duration") or {}
        for category in set(inclusive) | set(allocated):
            category_data[category]["sequence_count_any"] += 1
            category_data[category]["inclusive_duration_sec"] += inclusive.get(
                category, 0.0
            )
            category_data[category]["allocated_duration_sec"] += allocated.get(
                category, 0.0
            )
        category_data[item.get("primary_category") or "unknown"][
            "primary_sequence_count"
        ] += 1

    categories = {}
    allocated_total = sum(
        values["allocated_duration_sec"] for values in category_data.values()
    )
    for category, values in sorted(category_data.items()):
        values = dict(values)
        values["inclusive_duration_sec"] = round(
            values["inclusive_duration_sec"], 6
        )
        values["allocated_duration_sec"] = round(
            values["allocated_duration_sec"], 6
        )
        values["any_sequence_count_ratio"] = round(
            values["sequence_count_any"] / total_count if total_count else 0.0, 8
        )
        values["primary_sequence_count_ratio"] = round(
            values["primary_sequence_count"] / total_count if total_count else 0.0,
            8,
        )
        values["allocated_duration_ratio"] = round(
            values["allocated_duration_sec"] / allocated_total
            if allocated_total else 0.0,
            8,
        )
        categories[category] = values
    return {
        "total_sequence_count": total_count,
        "total_duration_sec": round(total_duration, 6),
        "babel_matched_count": matched,
        "babel_unmatched_or_ambiguous_count": total_count - matched,
        "allocated_duration_total_sec": round(allocated_total, 6),
        "categories": categories,
    }


def build_pipeline_summary(
    records, amass_root, files_found, target_fps, path_summary
):
    final_counts = Counter(item["final_status"] for item in records)
    semantic_counts = Counter(item["semantic_status"] for item in records)
    basic_counts = Counter(item["basic_status"] for item in records)
    heuristic_counts = Counter(item["heuristic_status"] for item in records)
    dedup_counts = Counter(item["dedup_status"] for item in records)
    quota_counts = Counter(item["quota_status"] for item in records)
    dual = [
        item for item in records
        if item["heuristic_status"] == "heuristic_passed"
        and item["semantic_status"] in (
            "semantic_passed", "babel_unmatched", "babel_ambiguous"
        )
    ]
    selected = [
        item for item in records if item["final_status"] == "selected_candidate"
    ]
    summary = {
        "amass_root": str(Path(amass_root).resolve()),
        "target_fps": float(target_fps),
        "num_files_found": files_found,
        "num_inventory_records": len(records),
        "num_npz_loaded": sum(bool(item["npz_loaded"]) for item in records),
        "num_semantic_rejected_without_npz_load": sum(
            item["semantic_status"].startswith("reject_semantic")
            and not item["npz_loaded"]
            for item in records
        ),
        "path_mapping": path_summary,
        "stage_status_counts": {
            "semantic": dict(sorted(semantic_counts.items())),
            "basic": dict(sorted(basic_counts.items())),
            "heuristic": dict(sorted(heuristic_counts.items())),
            "dedup": dict(sorted(dedup_counts.items())),
            "quota": dict(sorted(quota_counts.items())),
        },
        "final_status_distribution": {
            status: final_counts.get(status, 0) for status in FINAL_STATUSES
        },
        "category_statistics": {
            "before_semantic_filter": category_scope(records),
            "after_dual_filter": category_scope(dual),
            "final_selected": category_scope(selected),
        },
        "selected_expected_frames_30fps": sum(
            item["expected_num_frames_at_30fps"] for item in selected
        ),
        "selected_duration_hours": round(
            sum(item["duration_sec"] for item in selected) / 3600.0, 6
        ),
    }
    return summary
