"""V3 deterministic reports, stage distributions and V2/V3 comparison."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from summary_report import category_scope


V3_OUTPUTS = [
    "amass_inventory.csv",
    "amass_inventory.jsonl",
    "dual_filter_passed.jsonl",
    "review_candidates.jsonl",
    "candidate_v3.jsonl",
    "scan_summary.json",
    "dedup_clusters.jsonl",
    "quota_report.json",
    "v2_v3_comparison.json",
]


def file_hashes(paths):
    return {
        str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in paths
    }


def group_distribution(records, key_function):
    groups = defaultdict(lambda: {"sequence_count": 0, "duration_sec": 0.0})
    total_duration = sum(
        float(item.get("duration_sec") or item.get("babel_duration_sec") or 0.0)
        for item in records
    )
    for item in records:
        key = key_function(item)
        groups[key]["sequence_count"] += 1
        groups[key]["duration_sec"] += float(
            item.get("duration_sec") or item.get("babel_duration_sec") or 0.0
        )
    for values in groups.values():
        values["duration_sec"] = round(values["duration_sec"], 6)
        values["sequence_ratio"] = round(
            values["sequence_count"] / len(records) if records else 0.0, 8
        )
        values["duration_ratio"] = round(
            values["duration_sec"] / total_duration if total_duration else 0.0, 8
        )
    return dict(sorted(groups.items()))


def metric_quantiles(metric_records, metric_names):
    quantiles = {
        "min": 0.0, "p01": 0.01, "p05": 0.05, "p25": 0.25,
        "p50": 0.5, "p75": 0.75, "p90": 0.9, "p95": 0.95,
        "p99": 0.99, "max": 1.0,
    }
    return {
        metric: {
            label: float(
                np.quantile([item[metric] for item in metric_records], value)
            )
            for label, value in quantiles.items()
        }
        for metric in metric_names
    }


def build_summary(
    records,
    metric_records,
    clusters,
    quota_report,
    neutral_validation,
    metric_names,
):
    final_counts = Counter(item["final_status"] for item in records)
    heuristic_counts = Counter(
        item["heuristic_status_v3"] for item in metric_records
    )
    passed = [
        item for item in metric_records
        if item["heuristic_status_v3"] == "heuristic_passed"
    ]
    after_dedup = [
        item for item in passed
        if item["dedup_status_v3"] in ("unique", "unique_cluster_representative")
    ]
    selected = [
        item for item in records if item["final_status"] == "selected_candidate"
    ]
    reason_reject = Counter(
        reason for item in metric_records for reason in item["heuristic_reject_reasons"]
    )
    reason_review = Counter(
        reason for item in metric_records for reason in item["heuristic_review_reasons"]
    )
    reason_overlap = Counter(
        reason
        for item in metric_records
        if len(item["heuristic_all_reasons_v3"]) > 1
        for reason in item["heuristic_all_reasons_v3"]
    )
    stages = {
        "before_filter": records,
        "after_heuristic": passed,
        "after_dedup": after_dedup,
        "final_selected": selected,
    }
    source_distributions = {
        stage: group_distribution(values, lambda item: item["source_dataset"])
        for stage, values in stages.items()
    }
    subject_final = group_distribution(
        selected,
        lambda item: "{}/{}".format(item["source_dataset"], item["subject_id"]),
    )
    top_duration = sorted(
        subject_final.items(),
        key=lambda value: (-value[1]["duration_sec"], value[0]),
    )[:50]
    top_count = sorted(
        subject_final.items(),
        key=lambda value: (-value[1]["sequence_count"], value[0]),
    )[:50]
    return {
        "num_files_found": len(records),
        "final_status_distribution": dict(sorted(final_counts.items())),
        "heuristic_evaluated_count": len(metric_records),
        "heuristic_status_distribution": dict(sorted(heuristic_counts.items())),
        "heuristic_rule_statistics": {
            reason: {
                "reject_count": reason_reject.get(reason, 0),
                "review_count": reason_review.get(reason, 0),
                "overlap_count": reason_overlap.get(reason, 0),
            }
            for reason in sorted(
                set(reason_reject) | set(reason_review) | set(reason_overlap)
            )
        },
        "heuristic_metric_quantiles": metric_quantiles(
            metric_records, metric_names
        ),
        "exact_duplicate_count": final_counts.get("defer_exact_duplicate", 0),
        "near_duplicate_cluster_count": len(clusters),
        "defer_redundant_take_count": final_counts.get(
            "defer_redundant_take", 0
        ),
        "quota_defer_counts": {
            key: final_counts.get(key, 0)
            for key in (
                "defer_category_quota", "defer_source_quota",
                "defer_subject_quota", "defer_global_budget",
            )
        },
        "selected_sequence_count": len(selected),
        "selected_duration_sec": round(
            sum(item["duration_sec"] for item in selected), 6
        ),
        "selected_expected_frames_30fps": sum(
            item["expected_num_frames_at_30fps"] for item in selected
        ),
        "source_distributions": source_distributions,
        "subject_distribution_final": {
            "top_50_by_duration": dict(top_duration),
            "top_50_by_sequence_count": dict(top_count),
        },
        "category_distributions": {
            stage: category_scope(values) for stage, values in stages.items()
        },
        "neutral_gender_validation": neutral_validation,
        "quota_report": quota_report,
    }


def build_comparison(v2_candidates, v3_candidates, records, example_count):
    v2 = {item["motion_id"]: item for item in v2_candidates}
    v3 = {item["motion_id"]: item for item in v3_candidates}
    intersection = sorted(set(v2) & set(v3))
    only_v2 = sorted(set(v2) - set(v3))
    only_v3 = sorted(set(v3) - set(v2))
    changes = defaultdict(list)
    for item in records:
        if item["motion_id"] in only_v2:
            changes[item["final_status"]].append(item["motion_id"])
    return {
        "v2_selected_count": len(v2),
        "v2_selected_duration_sec": round(
            sum(item["duration_sec"] for item in v2.values()), 6
        ),
        "v3_selected_count": len(v3),
        "v3_selected_duration_sec": round(
            sum(item["duration_sec"] for item in v3.values()), 6
        ),
        "v3_high_confidence_reject_count": sum(
            item["heuristic_status_v3"] == "reject_heuristic_high_confidence"
            for item in records
        ),
        "v3_review_count": sum(
            item["heuristic_status_v3"] == "review_heuristic"
            for item in records
        ),
        "v3_exact_duplicate_count": sum(
            item["final_status"] == "defer_exact_duplicate" for item in records
        ),
        "v3_redundant_take_count": sum(
            item["final_status"] == "defer_redundant_take" for item in records
        ),
        "v3_source_quota_count": sum(
            item["final_status"] == "defer_source_quota" for item in records
        ),
        "v3_subject_quota_count": sum(
            item["final_status"] == "defer_subject_quota" for item in records
        ),
        "candidate_intersection_count": len(intersection),
        "only_v2_count": len(only_v2),
        "only_v3_count": len(only_v3),
        "intersection_motion_ids": intersection,
        "only_v2_motion_ids": only_v2,
        "only_v3_motion_ids": only_v3,
        "change_examples": {
            status: sorted(values)[:example_count]
            for status, values in sorted(changes.items())
        },
    }


def _jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def write_outputs(
    output,
    records,
    dual,
    reviews,
    selected,
    clusters,
    summary,
    quota_report,
    comparison,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _jsonl(output / "amass_inventory.jsonl", records)
    _jsonl(output / "dual_filter_passed.jsonl", dual)
    _jsonl(output / "review_candidates.jsonl", reviews)
    _jsonl(output / "candidate_v3.jsonl", selected)
    _jsonl(output / "dedup_clusters.jsonl", clusters)
    all_fields = []
    identity = [
        "motion_id", "source_dataset", "subject_id", "motion_name",
        "relative_path", "final_status", "final_status_reason",
    ]
    field_set = set()
    for field in identity:
        all_fields.append(field)
        field_set.add(field)
    for item in records:
        for field in sorted(item):
            if field not in field_set:
                all_fields.append(field)
                field_set.add(field)
    with (output / "amass_inventory.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields)
        writer.writeheader()
        for item in records:
            writer.writerow(
                {
                    field: (
                        json.dumps(item.get(field), ensure_ascii=False, sort_keys=True)
                        if isinstance(item.get(field), (list, dict))
                        else item.get(field)
                    )
                    for field in all_fields
                }
            )
    for name, value in (
        ("scan_summary.json", summary),
        ("quota_report.json", quota_report),
        ("v2_v3_comparison.json", comparison),
    ):
        with (output / name).open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
