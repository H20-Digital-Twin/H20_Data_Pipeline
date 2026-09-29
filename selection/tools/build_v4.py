#!/usr/bin/env python3
"""Build the V4 user-confirmed BABEL semantic-core AMASS candidate set."""

import argparse
import copy
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from babel_path_matcher import build_path_mapping
from babel_reader import load_babel_entries
from summary_report import category_scope
from v3_deduplication import apply_v3_deduplication
from v3_quota import apply_v3_quota
from v3_report import group_distribution
from v4_semantic_policy import (
    apply_v4_semantic_policy,
    semantic_statistics,
    validate_policy,
)


OUTPUT_NAMES = (
    "amass_inventory.csv",
    "amass_inventory.jsonl",
    "semantic_core_eligible.jsonl",
    "review_candidates.jsonl",
    "candidate_v4.jsonl",
    "dedup_clusters.jsonl",
    "quota_report.json",
    "scan_summary.json",
    "v3_v4_comparison.json",
)


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _file_hashes(paths):
    return {str(Path(path).resolve()): _sha256(path) for path in paths}


def _sample_indices(num_frames, source_fps, target_fps):
    if source_fps > target_fps:
        frame_skip = max(1, int(source_fps / target_fps))
        new_count = max(1, num_frames // frame_skip)
        return np.rint(np.linspace(0, num_frames - 1, new_count)).astype(int)
    return np.arange(num_frames, dtype=int)


def _descriptors(path, target_fps, descriptor_frames):
    """Read pose/root arrays only; V3 Neutral metrics remain authoritative."""
    with np.load(path, allow_pickle=False) as data:
        fps_key = (
            "mocap_frame_rate"
            if "mocap_frame_rate" in data.files
            else "mocap_framerate"
        )
        source_fps = float(data[fps_key])
        pose_key = "pose_body" if "pose_body" in data.files else "poses"
        body = np.asarray(data[pose_key], dtype=np.float64)
        if body.shape[1] > 63:
            body = body[:, 3:66]
        trans_key = "trans" if "trans" in data.files else "transl"
        trans = np.asarray(data[trans_key], dtype=np.float64)
    indices = _sample_indices(len(body), source_fps, target_fps)
    body = body[indices]
    trans = trans[indices]
    descriptor_idx = np.rint(
        np.linspace(0, len(body) - 1, descriptor_frames)
    ).astype(int)
    pose_descriptor = body[descriptor_idx].reshape(-1).astype(np.float32)
    root_descriptor = (
        trans[descriptor_idx] - trans[descriptor_idx[:1]]
    ).reshape(-1).astype(np.float32)
    return pose_descriptor, root_descriptor


def _prepare_records(
    source_records, mapping, semantic_config, v3_candidate_ids
):
    records = copy.deepcopy(source_records)
    for item in records:
        item["final_status_v3"] = item["final_status"]
        item["final_status_reason_v3"] = item["final_status_reason"]
        item["dedup_status_v3_preserved"] = item.get("dedup_status_v3")
        item["quota_status_v3_preserved"] = item.get("quota_status_v3")
        item["selection_score_v3_preserved"] = item.get("selection_score_v3")
        item["in_v3_candidate"] = item["motion_id"] in v3_candidate_ids
        apply_v4_semantic_policy(
            item, mapping[item["relative_path"]], semantic_config
        )
        item["dedup_status_v4"] = "not_eligible"
        item["quota_status_v4"] = "not_eligible"
        item["selection_score_v4"] = None
        item["final_status_v4"] = None
        item["final_status_reason_v4"] = None
    return records


def _reset_dedup_fields(item):
    for field in (
        "dedup_status", "duplicate_of", "duplicate_similarity",
        "duplicate_pose_distance", "duplicate_root_distance",
        "duplicate_cluster_id", "cluster_size", "cluster_rank",
        "cluster_representative",
    ):
        item[field] = None


def _finalize(records):
    for item in records:
        semantic = item["semantic_status_v4"]
        if not item["in_v3_candidate"]:
            status = "defer_not_in_v3_candidate"
            reason = "not selected by frozen V3 candidate set"
        elif semantic != "semantic_core_eligible":
            status = semantic
            reason = item["semantic_status_reason_v4"]
        elif item["basic_status"] != "basic_passed":
            status = item["basic_status"]
            reason = item["basic_status_reason"]
        elif item["heuristic_status_v3"] == "reject_heuristic_high_confidence":
            status = item["heuristic_primary_reason"]
            reason = "; ".join(item["heuristic_all_reasons_v3"])
        elif item["heuristic_status_v3"] == "review_heuristic":
            status = item["heuristic_primary_reason"]
            reason = "; ".join(item["heuristic_all_reasons_v3"])
        elif item["heuristic_status_v3"] == "heuristic_passed":
            if item["dedup_status_v4"] in (
                "defer_exact_duplicate", "defer_redundant_take"
            ):
                status = item["dedup_status_v4"]
                reason = "{} of {}".format(
                    status, item.get("duplicate_of")
                )
            else:
                status = item["quota_status_v4"]
                reason = item["quota_status_reason"]
        else:
            raise AssertionError(
                "unresolved V4 status for {}".format(item["motion_id"])
            )
        item["final_status_v4"] = status
        item["final_status_reason_v4"] = reason
        item["final_status"] = status
        item["final_status_reason"] = reason
        item["status"] = status
        item["status_reason"] = reason


def _quota_with_order_check(pool, quota_config):
    ordered = copy.deepcopy(sorted(pool, key=lambda item: item["motion_id"]))
    reversed_input = copy.deepcopy(list(reversed(ordered)))
    first = apply_v3_quota(ordered, quota_config)
    second = apply_v3_quota(reversed_input, quota_config)
    first_status = {
        item["motion_id"]: (item["quota_status"], item["quota_status_reason"])
        for item in ordered
    }
    second_status = {
        item["motion_id"]: (item["quota_status"], item["quota_status_reason"])
        for item in reversed_input
    }
    if first_status != second_status or first != second:
        raise AssertionError("V4 quota result depends on input order")
    original = {item["motion_id"]: item for item in pool}
    for decided in ordered:
        target = original[decided["motion_id"]]
        target["quota_status"] = decided["quota_status"]
        target["quota_status_reason"] = decided["quota_status_reason"]
        target["selection_score_v3"] = decided["selection_score_v3"]
    return first


def _top_act_cat(records, limit):
    counts = Counter(
        category
        for item in records
        for category in item.get("babel_act_cat_all", [])
    )
    return [
        {"act_cat": key, "sequence_count_any": value}
        for key, value in sorted(
            counts.items(), key=lambda pair: (-pair[1], pair[0])
        )[:limit]
    ]


def _comparison(v3_candidates, v4_candidates, records, example_count):
    v3 = {item["motion_id"]: item for item in v3_candidates}
    v4 = {item["motion_id"]: item for item in v4_candidates}
    intersection = sorted(set(v3) & set(v4))
    only_v3 = sorted(set(v3) - set(v4))
    only_v4 = sorted(set(v4) - set(v3))
    by_status = defaultdict(list)
    indexed = {item["motion_id"]: item for item in records}
    for motion_id in only_v3:
        by_status[indexed[motion_id]["final_status_v4"]].append(motion_id)
    return {
        "v3_selected_count": len(v3),
        "v3_selected_duration_sec": round(
            sum(float(item["duration_sec"]) for item in v3.values()), 6
        ),
        "v4_selected_count": len(v4),
        "v4_selected_duration_sec": round(
            sum(float(item["duration_sec"]) for item in v4.values()), 6
        ),
        "intersection_count": len(intersection),
        "only_v3_count": len(only_v3),
        "only_v4_count": len(only_v4),
        "change_examples": {
            key: sorted(values)[:example_count]
            for key, values in sorted(by_status.items())
        },
    }


def _summary(
    records, selected, semantic_validation, clusters, quota_report
):
    statuses = Counter(item["final_status_v4"] for item in records)
    semantic = semantic_statistics(records)
    semantic_eligible = [
        item for item in records
        if item["in_v3_candidate"]
        and item["semantic_status_v4"] == "semantic_core_eligible"
    ]
    movement_passed = [
        item for item in semantic_eligible
        if item["basic_status"] == "basic_passed"
        and item["heuristic_status_v3"] == "heuristic_passed"
    ]
    after_dedup = [
        item for item in movement_passed
        if item["dedup_status_v4"] == "inherited_v3_candidate"
    ]
    stages = {
        "before_filter": records,
        "semantic_core_eligible": semantic_eligible,
        "after_neutral_heuristic": movement_passed,
        "after_dedup": after_dedup,
        "final_selected": selected,
    }
    tier_presence = {
        tier: sum(
            bool(item["semantic_tier_hits"][tier]) for item in selected
        )
        for tier in ("R", "C", "A", "M")
    }
    return {
        "version": "v4",
        "selection_base": "strict subset of frozen candidate_v3",
        "num_files_found": len(records),
        "v3_candidate_count": sum(item["in_v3_candidate"] for item in records),
        "semantic_policy_validation": semantic_validation,
        "semantic_filter": semantic,
        "final_status_distribution": dict(sorted(statuses.items())),
        "semantic_core_eligible_count": len(semantic_eligible),
        "neutral_heuristic_passed_count": len(movement_passed),
        "near_duplicate_cluster_count": len(clusters),
        "after_dedup_count": len(after_dedup),
        "selected_sequence_count": len(selected),
        "selected_duration_sec": round(
            sum(float(item["duration_sec"]) for item in selected), 6
        ),
        "selected_expected_frames_30fps": sum(
            int(item["expected_num_frames_at_30fps"]) for item in selected
        ),
        "selected_semantic_tier_presence": tier_presence,
        "selected_top_act_cat": _top_act_cat(selected, 50),
        "source_distributions": {
            name: group_distribution(values, lambda item: item["source_dataset"])
            for name, values in stages.items()
        },
        "subject_distribution_selected": group_distribution(
            selected,
            lambda item: "{}/{}".format(
                item["source_dataset"], item["subject_id"]
            ),
        ),
        "category_distributions": {
            name: category_scope(values) for name, values in stages.items()
        },
        "quota_report": quota_report,
    }


def _jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(
                json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            )


def _write_outputs(
    output, records, semantic_eligible, reviews, selected, clusters,
    quota_report, summary, comparison,
):
    output.mkdir(parents=True, exist_ok=True)
    _jsonl(output / "amass_inventory.jsonl", records)
    _jsonl(output / "semantic_core_eligible.jsonl", semantic_eligible)
    _jsonl(output / "review_candidates.jsonl", reviews)
    _jsonl(output / "candidate_v4.jsonl", selected)
    _jsonl(output / "dedup_clusters.jsonl", clusters)

    identity = [
        "motion_id", "source_dataset", "subject_id", "motion_name",
        "relative_path", "final_status_v4", "final_status_reason_v4",
    ]
    fields = list(identity)
    seen = set(fields)
    for item in records:
        for field in sorted(item):
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with (output / "amass_inventory.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in records:
            writer.writerow(
                {
                    field: (
                        json.dumps(
                            item.get(field), ensure_ascii=False, sort_keys=True
                        )
                        if isinstance(item.get(field), (list, dict))
                        else item.get(field)
                    )
                    for field in fields
                }
            )
    for name, value in (
        ("quota_report.json", quota_report),
        ("scan_summary.json", summary),
        ("v3_v4_comparison.json", comparison),
    ):
        with (output / name).open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")


def _validate(records, selected, quota_report, config):
    if len(records) != 8345:
        raise AssertionError("V4 inventory must contain all 8345 files")
    if len({item["motion_id"] for item in records}) != len(records):
        raise AssertionError("motion_id values are not unique")
    if any(not item["final_status_v4"] for item in records):
        raise AssertionError("every V4 record needs exactly one final status")
    selected_ids = {item["motion_id"] for item in selected}
    inventory_ids = {item["motion_id"] for item in records}
    if not selected_ids < inventory_ids:
        raise AssertionError("candidate_v4 must be a strict inventory subset")
    for item in selected:
        if not item["in_v3_candidate"]:
            raise AssertionError("V3-external action entered strict V4")
        if item["semantic_status_v4"] != "semantic_core_eligible":
            raise AssertionError("unverified/rejected semantics entered V4")
        if item["basic_status"] != "basic_passed":
            raise AssertionError("basic-invalid action entered V4")
        if item["heuristic_status_v3"] != "heuristic_passed":
            raise AssertionError("non-passed Neutral heuristic entered V4")
        if item["dedup_status_v4"] != "inherited_v3_candidate":
            raise AssertionError("V4 did not inherit frozen V3 dedup result")
    total = len(selected)
    duration = sum(float(item["duration_sec"]) for item in selected)
    source_cfg = config["quota"]["source_dataset"]
    for source, count in quota_report["source_count"].items():
        override = source_cfg.get("overrides", {}).get(source, {})
        ratio = float(
            override.get(
                "max_sequence_ratio",
                source_cfg["max_sequence_ratio_default"],
            )
        )
        if total and count / total > ratio + 1e-9:
            raise AssertionError("source sequence ratio exceeded: " + source)
    for source, value in quota_report["source_duration"].items():
        override = source_cfg.get("overrides", {}).get(source, {})
        ratio = float(
            override.get(
                "max_duration_ratio",
                source_cfg["max_duration_ratio_default"],
            )
        )
        if duration and value / duration > ratio + 1e-9:
            raise AssertionError("source duration ratio exceeded: " + source)
    subject_cfg = config["quota"]["subject"]
    if max(quota_report["subject_count"].values(), default=0) > int(
        subject_cfg["max_sequence_count"]
    ):
        raise AssertionError("subject sequence quota exceeded")
    if max(quota_report["subject_duration"].values(), default=0.0) > float(
        subject_cfg["max_duration_sec"]
    ) + 1e-9:
        raise AssertionError("subject duration quota exceeded")


def run(config):
    output = Path(config["output"]["root"]).expanduser().resolve()
    existing = [output / name for name in OUTPUT_NAMES if (output / name).exists()]
    if existing and not config["output"]["overwrite"]:
        raise FileExistsError("V4 output exists; use --overwrite")

    v3_path = Path(config["input"]["v3_inventory_jsonl"]).resolve()
    v3_dir = v3_path.parent
    v3_files = sorted(path for path in v3_dir.iterdir() if path.is_file())
    v3_hashes_before = _file_hashes(v3_files)
    source_records = load_jsonl(v3_path)
    v3_candidates = load_jsonl(config["input"]["v3_candidate_jsonl"])
    v3_candidate_ids = {item["motion_id"] for item in v3_candidates}
    if config["input"].get("selection_base") != "v3_candidate_strict_subset":
        raise ValueError("V4 selection_base must be v3_candidate_strict_subset")

    root = Path(config["input"]["amass_root"]).resolve()
    local_files = [root / item["relative_path"] for item in source_records]
    babel_entries = load_babel_entries(config["input"]["babel_json_files"])
    semantic_validation = validate_policy(
        config["semantic_policy"], babel_entries
    )
    mapping, _, mapping_summary = build_path_mapping(
        local_files, root, babel_entries, config["path_matching"]
    )
    records = _prepare_records(
        source_records,
        mapping,
        config["semantic_policy"],
        v3_candidate_ids,
    )

    movement_passed = [
        item for item in records
        if item["in_v3_candidate"]
        and item["semantic_status_v4"] == "semantic_core_eligible"
        and item["basic_status"] == "basic_passed"
        and item["heuristic_status_v3"] == "heuristic_passed"
    ]
    # V3 candidates have already passed the Neutral heuristic and V3
    # deduplication. V4 is deliberately a one-way refinement and never
    # reopens or replenishes that frozen pool.
    clusters = []
    for item in movement_passed:
        item["dedup_status_v4"] = "inherited_v3_candidate"

    quota_pool = list(movement_passed)
    quota_report = _quota_with_order_check(quota_pool, config["quota"])
    for item in quota_pool:
        item["quota_status_v4"] = item["quota_status"]
        item["selection_score_v4"] = item["selection_score_v3"]

    _finalize(records)
    records.sort(key=lambda item: item["relative_path"])
    semantic_eligible = sorted(
        [
            item for item in records
            if item["in_v3_candidate"]
            and item["semantic_status_v4"] == "semantic_core_eligible"
        ],
        key=lambda item: item["motion_id"],
    )
    reviews = sorted(
        [
            item for item in records
            if item["in_v3_candidate"]
            and item["semantic_status_v4"] == "semantic_core_eligible"
            and item["basic_status"] == "basic_passed"
            and item["heuristic_status_v3"] == "review_heuristic"
        ],
        key=lambda item: item["motion_id"],
    )
    selected = sorted(
        [
            item for item in records
            if item["final_status_v4"] == "selected_candidate"
        ],
        key=lambda item: item["motion_id"],
    )
    _validate(records, selected, quota_report, config)

    summary = _summary(
        records, selected, semantic_validation, clusters, quota_report
    )
    summary["babel_path_mapping"] = mapping_summary
    comparison = _comparison(
        v3_candidates,
        selected,
        records,
        int(config["report"]["comparison_examples_per_status"]),
    )

    _write_outputs(
        output, records, semantic_eligible, reviews, selected, clusters,
        quota_report, summary, comparison,
    )
    v3_hashes_after = _file_hashes(v3_files)
    if v3_hashes_before != v3_hashes_after:
        raise AssertionError("V3 artifacts changed during V4 build")
    return summary, comparison


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.overwrite:
        config["output"]["overwrite"] = True
    summary, comparison = run(config)
    print(
        json.dumps(
            {"summary": summary, "comparison": comparison},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
