#!/usr/bin/env python3
"""Build the independent V3 three-tier AMASS candidate dataset."""

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from heuristic_metrics import METRIC_NAMES, SMPLXMetricExtractor
from v3_deduplication import apply_v3_deduplication
from v3_filter_rules import decide_heuristic, resolve_joint_motion_threshold
from v3_quota import apply_v3_quota
from v3_report import (
    V3_OUTPUTS,
    build_comparison,
    build_summary,
    file_hashes,
    write_outputs,
)


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _prepare_records(v2_records):
    records = copy.deepcopy(v2_records)
    for item in records:
        item["final_status_v2"] = item.get("final_status")
        item["final_status_reason_v2"] = item.get("final_status_reason")
        item["heuristic_status_v2"] = item.get("heuristic_status")
        item["heuristic_all_reasons_v2"] = item.get("heuristic_all_reasons", [])
        item["dedup_status_v2"] = item.get("dedup_status")
        item["quota_status_v2"] = item.get("quota_status")
        item["babel_path_matched"] = str(
            item.get("babel_match_status", "")
        ).startswith("match_")
        item["babel_label_available"] = bool(
            item.get("sequence_tags") or item.get("frame_tags")
        )
        item["heuristic_status"] = "not_evaluated"
        item["heuristic_status_v3"] = "not_evaluated"
        item["heuristic_primary_reason"] = "not_evaluated"
        item["heuristic_all_reasons"] = []
        item["heuristic_all_reasons_v3"] = []
        item["heuristic_review_reasons"] = []
        item["heuristic_reject_reasons"] = []
        item["dedup_status"] = None
        item["dedup_status_v3"] = "not_eligible"
        item["quota_status"] = "not_eligible"
        item["quota_status_v3"] = "not_eligible"
        item["final_status"] = None
        item["final_status_reason"] = None
    return records


def _eligible(item):
    return (
        item["heuristic_status_v2"] == "heuristic_passed"
        and item["semantic_status"] in (
            "semantic_passed", "babel_unmatched", "babel_ambiguous"
        )
        and item["basic_status"] == "basic_passed"
    )


def _status_digest(records):
    return hashlib.sha256(
        json.dumps(
            [(item["motion_id"], item["final_status_v2"]) for item in records],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _neutral_gender_validation(
    eligible,
    neutral_results,
    config,
    heuristic_config,
    joint_threshold,
    amass_root,
):
    validation = config["neutral_gender_validation"]
    if not validation["enabled"]:
        return {"enabled": False}
    sample_count = min(int(validation["sample_count"]), len(eligible))
    ordered = sorted(eligible, key=lambda item: item["motion_id"])
    rng = random.Random(int(validation["deterministic_seed"]))
    sampled = sorted(rng.sample(ordered, sample_count), key=lambda item: item["motion_id"])
    gender_config = dict(config["metrics"])
    gender_config.pop("force_gender", None)
    extractor = SMPLXMetricExtractor(gender_config)
    changes = []
    matches = 0
    compared_metrics = (
        "sitting_fraction", "lying_fraction", "crawl_fraction", "floor_fraction"
    )
    metric_differences = {metric: [] for metric in compared_metrics}
    for index, item in enumerate(sampled, start=1):
        gender_result = extractor.extract(amass_root / item["relative_path"])
        gender_decision = decide_heuristic(
            item, gender_result["metrics"], heuristic_config, joint_threshold
        )
        neutral = neutral_results[item["motion_id"]]
        neutral_decision = decide_heuristic(
            item, neutral["metrics"], heuristic_config, joint_threshold
        )
        if (
            gender_decision["heuristic_status"]
            == neutral_decision["heuristic_status"]
        ):
            matches += 1
        else:
            changes.append(
                {
                    "motion_id": item["motion_id"],
                    "relative_path": item["relative_path"],
                    "neutral_status": neutral_decision["heuristic_status"],
                    "gender_specific_status": gender_decision["heuristic_status"],
                    "neutral_primary_reason": neutral_decision[
                        "heuristic_primary_reason"
                    ],
                    "gender_specific_primary_reason": gender_decision[
                        "heuristic_primary_reason"
                    ],
                    "metric_differences": {
                        metric: {
                            "neutral": neutral["metrics"][metric],
                            "gender_specific": gender_result["metrics"][metric],
                            "difference": (
                                neutral["metrics"][metric]
                                - gender_result["metrics"][metric]
                            ),
                        }
                        for metric in compared_metrics
                    },
                }
            )
        for metric in compared_metrics:
            metric_differences[metric].append(
                neutral["metrics"][metric] - gender_result["metrics"][metric]
            )
        if index % 50 == 0 or index == sample_count:
            print(
                "Neutral/gender validation {}/{}".format(index, sample_count),
                file=sys.stderr,
            )
    agreement = matches / sample_count if sample_count else 1.0
    difference_summary = {
        metric: {
            "mean_signed_difference": sum(values) / len(values) if values else 0.0,
            "mean_absolute_difference": (
                sum(abs(value) for value in values) / len(values) if values else 0.0
            ),
            "max_absolute_difference": max(
                (abs(value) for value in values), default=0.0
            ),
        }
        for metric, values in metric_differences.items()
    }
    return {
        "enabled": True,
        "sample_count": sample_count,
        "sample_original_gender_distribution": dict(
            sorted(Counter(item.get("gender") for item in sampled).items())
        ),
        "status_match_count": matches,
        "status_change_count": len(changes),
        "status_agreement": agreement,
        "minimum_required_agreement": validation["minimum_status_agreement"],
        "meets_minimum": agreement >= validation["minimum_status_agreement"],
        "metric_difference_summary": difference_summary,
        "changed_actions": changes,
    }


def _finalize(records):
    for item in records:
        if str(item["semantic_status"]).startswith("reject_semantic"):
            status = item["semantic_status"]
            reason = item["semantic_status_reason"]
        elif item["basic_status"] != "basic_passed":
            status = item["basic_status"]
            reason = item["basic_status_reason"]
        elif item["heuristic_status"] == "reject_heuristic_high_confidence":
            status = item["heuristic_primary_reason"]
            reason = "; ".join(item["heuristic_all_reasons"])
        elif item["heuristic_status"] == "review_heuristic":
            status = item["heuristic_primary_reason"]
            reason = "; ".join(item["heuristic_all_reasons"])
        elif item["dedup_status"] in (
            "defer_exact_duplicate", "defer_redundant_take"
        ):
            status = item["dedup_status"]
            reason = "{} of {}".format(
                item["dedup_status"], item.get("duplicate_of")
            )
        elif item["heuristic_status"] == "heuristic_passed":
            status = item["quota_status"]
            reason = item["quota_status_reason"]
        else:
            raise AssertionError(
                "unresolved V3 state for {}".format(item["motion_id"])
            )
        item["final_status"] = status
        item["final_status_reason"] = reason
        item["status"] = status
        item["status_reason"] = reason


def _validate(records, eligible, reviews, selected):
    if len(records) != 8345:
        raise AssertionError("V3 inventory must contain all 8345 local files")
    if len({item["motion_id"] for item in records}) != len(records):
        raise AssertionError("motion IDs are not unique")
    tiers = Counter(item["heuristic_status"] for item in eligible)
    if sum(tiers.values()) != 5069:
        raise AssertionError("all 5069 formal heuristic candidates were not decided")
    if len(reviews) != tiers["review_heuristic"]:
        raise AssertionError("review list does not match heuristic review tier")
    selected_ids = {item["motion_id"] for item in selected}
    if not selected_ids < {item["motion_id"] for item in eligible}:
        raise AssertionError("candidate_v3 must be a strict eligible subset")
    ambiguous = {
        Path(item["relative_path"]).name: item
        for item in records
        if Path(item["relative_path"]).name
        in {
            "A10-_Lie_to_crouch_stageii.npz",
            "A7-_Crouch_stageii.npz",
            "A8-_Crouch_to_Lie_stageii.npz",
        }
    }
    if ambiguous["A10-_Lie_to_crouch_stageii.npz"]["final_status"] != (
        "reject_floor_high_confidence"
    ):
        raise AssertionError("A10 floor regression failed")
    if ambiguous["A8-_Crouch_to_Lie_stageii.npz"]["final_status"] != (
        "reject_floor_high_confidence"
    ):
        raise AssertionError("A8 floor regression failed")
    if ambiguous["A7-_Crouch_stageii.npz"]["final_status"] != (
        "review_floor_or_crouch"
    ):
        raise AssertionError("A7 crouch review regression failed")
    for item in eligible:
        if (
            item["heuristic_primary_reason"]
            == "reject_sitting_high_confidence"
            and not item["semantic_sit_supported"]
        ):
            raise AssertionError("sitting rejected without semantic support")
        if (
            item["heuristic_primary_reason"]
            == "reject_abnormal_high_confidence"
            and item["body_ang_speed_p99"] >= 40
            and not (
                item["jump_fraction"] >= 0.05
                or item["extreme_frame_fraction"] >= 0.05
                or item["max_consecutive_extreme_frames"] >= 2
            )
            and item["root_speed_p99"] <= 6
        ):
            raise AssertionError("fast action rejected on speed alone")
    selected_by_source = Counter(item["source_dataset"] for item in selected)
    selected_by_subject = Counter(
        "{}/{}".format(item["source_dataset"], item["subject_id"])
        for item in selected
    )
    if max(selected_by_subject.values(), default=0) > 15:
        raise AssertionError("subject sequence quota exceeded")


def run(config):
    output = Path(config["output"]["root"]).expanduser().resolve()
    existing = [output / name for name in V3_OUTPUTS if (output / name).exists()]
    if existing and not config["output"]["overwrite"]:
        raise FileExistsError("V3 output exists; use --overwrite")
    v2_directory = Path(config["input"]["v2_inventory_jsonl"]).resolve().parent
    v2_paths = sorted(path for path in v2_directory.iterdir() if path.is_file())
    v2_hashes_before = file_hashes(v2_paths)
    v2_records = load_jsonl(config["input"]["v2_inventory_jsonl"])
    v2_candidates = load_jsonl(config["input"]["v2_candidate_jsonl"])
    records = _prepare_records(v2_records)
    v2_status_digest = _status_digest(records)
    eligible = [item for item in records if _eligible(item)]
    if len(eligible) != 5069:
        raise AssertionError("expected 5069 V2 heuristic candidates")

    extractor = SMPLXMetricExtractor(config["metrics"])
    root = Path(config["input"]["amass_root"])
    results = {}
    descriptors, root_descriptors = {}, {}
    for index, item in enumerate(eligible, start=1):
        result = extractor.extract(root / item["relative_path"])
        results[item["motion_id"]] = result
        descriptors[item["motion_id"]] = result["pose_descriptor"]
        root_descriptors[item["motion_id"]] = result["root_descriptor"]
        item.update(result["metrics"])
        item["metric_sampled_frame_count"] = result["sampled_frame_count"]
        item["metric_effective_fps"] = result["effective_fps"]
        item["metric_body_scale_m"] = result["body_scale_m"]
        if index % 100 == 0 or index == len(eligible):
            print("V3 Neutral metrics {}/{}".format(index, len(eligible)), file=sys.stderr)

    joint_threshold = resolve_joint_motion_threshold(
        eligible, config["heuristic_filter"]
    )
    for item in eligible:
        decision = decide_heuristic(
            item,
            {metric: item[metric] for metric in METRIC_NAMES},
            config["heuristic_filter"],
            joint_threshold,
        )
        item.update(decision)
        item["heuristic_status_v3"] = item["heuristic_status"]
        item["heuristic_all_reasons_v3"] = item["heuristic_all_reasons"]

    neutral_validation = _neutral_gender_validation(
        eligible,
        results,
        config,
        config["heuristic_filter"],
        joint_threshold,
        root,
    )

    passed = [
        item for item in eligible if item["heuristic_status"] == "heuristic_passed"
    ]
    clusters = apply_v3_deduplication(
        passed, descriptors, root_descriptors, config["deduplication"]
    )
    for item in passed:
        item["dedup_status_v3"] = item["dedup_status"]
    quota_pool = [
        item for item in passed
        if item["dedup_status"] in ("unique", "unique_cluster_representative")
    ]
    quota_report = apply_v3_quota(quota_pool, config["quota"])
    for item in quota_pool:
        item["quota_status_v3"] = item["quota_status"]
    _finalize(records)
    records.sort(key=lambda item: item["relative_path"])
    dual = sorted(eligible, key=lambda item: item["motion_id"])
    reviews = sorted(
        [item for item in eligible if item["heuristic_status"] == "review_heuristic"],
        key=lambda item: item["motion_id"],
    )
    selected = sorted(
        [item for item in records if item["final_status"] == "selected_candidate"],
        key=lambda item: item["motion_id"],
    )
    _validate(records, eligible, reviews, selected)
    if max(quota_report["subject_count"].values(), default=0) > int(
        config["quota"]["subject"]["max_sequence_count"]
    ):
        raise AssertionError("subject sequence quota exceeded")
    if max(quota_report["subject_duration"].values(), default=0.0) > float(
        config["quota"]["subject"]["max_duration_sec"]
    ) + 1e-6:
        raise AssertionError("subject duration quota exceeded")
    selected_count = len(selected)
    selected_duration = sum(item["duration_sec"] for item in selected)
    for source, duration in quota_report["source_duration"].items():
        source_cfg = dict(
            max_duration_ratio=config["quota"]["source_dataset"][
                "max_duration_ratio_default"
            ],
            max_sequence_ratio=config["quota"]["source_dataset"][
                "max_sequence_ratio_default"
            ],
        )
        source_cfg.update(
            config["quota"]["source_dataset"].get("overrides", {}).get(source, {})
        )
        if (
            duration
            > float(source_cfg["max_duration_ratio"]) * selected_duration
            + 1e-6
        ):
            raise AssertionError("{} source duration quota exceeded".format(source))
        count = quota_report["source_count"].get(source, 0)
        if count > int(
            float(source_cfg["max_sequence_ratio"]) * selected_count
        ):
            raise AssertionError("{} source sequence ratio exceeded".format(source))

    summary = build_summary(
        records,
        eligible,
        clusters,
        quota_report,
        neutral_validation,
        METRIC_NAMES,
    )
    summary["joint_motion_integral_threshold_rad"] = joint_threshold
    summary["v2_input_hashes"] = v2_hashes_before
    summary["v2_status_digest"] = v2_status_digest
    summary["generation_budget"] = config["generation_budget"]
    comparison = build_comparison(
        v2_candidates,
        selected,
        records,
        int(config["report"]["comparison_examples_per_status"]),
    )
    write_outputs(
        output,
        records,
        dual,
        reviews,
        selected,
        clusters,
        summary,
        quota_report,
        comparison,
    )
    v2_hashes_after = file_hashes(v2_paths)
    if v2_hashes_before != v2_hashes_after:
        raise AssertionError("V3 modified a V2 artifact")
    print("V3 reports written to {}".format(output), file=sys.stderr)
    return summary, comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.overwrite:
        config["output"]["overwrite"] = True
    summary, comparison = run(config)
    print(
        json.dumps(
            {
                "heuristic_status_distribution": summary[
                    "heuristic_status_distribution"
                ],
                "final_status_distribution": summary[
                    "final_status_distribution"
                ],
                "selected_sequence_count": summary["selected_sequence_count"],
                "selected_duration_sec": summary["selected_duration_sec"],
                "neutral_gender_validation": summary[
                    "neutral_gender_validation"
                ],
                "v2_v3": {
                    key: comparison[key]
                    for key in (
                        "v2_selected_count", "v3_selected_count",
                        "candidate_intersection_count", "only_v2_count",
                        "only_v3_count",
                    )
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
