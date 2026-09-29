"""Order-independent two-pass category quota selection."""

import math
from collections import defaultdict


def _entropy(proportions):
    return -sum(value * math.log(value + 1e-12) for value in proportions.values())


def compute_selection_scores(candidates, config):
    pool_duration = defaultdict(float)
    for item in candidates:
        for category, value in item["category_allocated_duration"].items():
            pool_duration[category] += value
    weights = config["selection"]
    for item in candidates:
        proportions = item["category_proportions"]
        rarity = sum(
            proportion / math.sqrt(pool_duration.get(category, 0.0) + 1.0)
            for category, proportion in proportions.items()
            if category != "unknown"
        )
        score = (
            weights["quality_weight"] * item["heuristic_quality_score"]
            + weights["rarity_weight"] * rarity
            + weights["multi_action_weight"] * _entropy(proportions)
        )
        item["selection_score"] = round(score, 10)
    return pool_duration


def _category_limit(category, quota):
    category_cfg = quota["categories"].get(category, {})
    limits = []
    if category_cfg.get("max_duration_sec") is not None:
        limits.append(float(category_cfg["max_duration_sec"]))
    if category_cfg.get("max_ratio") is not None:
        limits.append(
            float(category_cfg["max_ratio"])
            * float(quota["global"]["max_total_duration_sec"])
        )
    return min(limits) if limits else None


def _can_accept(item, state, config):
    tolerance = float(config["overshoot_tolerance_sec"])
    global_cfg = config["global"]
    if state["total_count"] + 1 > int(global_cfg["max_sequence_count"]):
        return False, "defer_global_budget", "global sequence count limit reached"
    if (
        state["total_duration"] + item["duration_sec"]
        > float(global_cfg["max_total_duration_sec"]) + tolerance
    ):
        return False, "defer_global_budget", "global duration limit reached"

    for category, contribution in item["category_allocated_duration"].items():
        limit = _category_limit(category, config)
        category_cfg = config["categories"].get(category, {})
        if (
            category_cfg.get("max_sequence_count") is not None
            and item["primary_category"] == category
            and state["category_count"][category] + 1
            > int(category_cfg["max_sequence_count"])
        ):
            return (
                False, "defer_quota_category",
                "{} primary sequence count limit reached".format(category),
            )
        if limit is None:
            continue
        exceeds = state["category_duration"][category] + contribution > limit + tolerance
        mixed_exception = (
            config["allow_mixed_over_category_max"]
            and (
                item["primary_category"] != category
                or item["dominant_category_ratio"]
                < config["pure_action_dominance_threshold"]
            )
        )
        if exceeds and not mixed_exception:
            return (
                False, "defer_quota_category",
                "{} allocation would exceed {:.3f}s quota".format(category, limit),
            )
    return True, "selected_candidate", "accepted_by_quota"


def _accept(item, state, reason):
    item["quota_status"] = "selected_candidate"
    item["quota_status_reason"] = reason
    state["total_count"] += 1
    state["total_duration"] += item["duration_sec"]
    state["category_count"][item["primary_category"]] += 1
    for category, value in item["category_allocated_duration"].items():
        state["category_duration"][category] += value


def apply_quota(records, config):
    candidates = [
        item for item in records
        if item["heuristic_status"] == "heuristic_passed"
        and item["dedup_status"] == "unique"
        and item["semantic_status"] in (
            "semantic_passed", "babel_unmatched", "babel_ambiguous"
        )
    ]
    compute_selection_scores(candidates, config)
    ranked = sorted(candidates, key=lambda x: (-x["selection_score"], x["motion_id"]))
    if not config["enabled"]:
        for item in ranked:
            item["quota_status"] = "selected_candidate"
            item["quota_status_reason"] = "quota_disabled"
        return records

    state = {
        "total_count": 0,
        "total_duration": 0.0,
        "category_count": defaultdict(int),
        "category_duration": defaultdict(float),
    }
    decided = set()

    minimum_categories = sorted(
        (
            (category, float(values.get("min_duration_sec", 0.0)))
            for category, values in config["categories"].items()
            if float(values.get("min_duration_sec", 0.0)) > 0
        ),
        key=lambda item: (item[1], item[0]),
    )
    for category, minimum in minimum_categories:
        for item in sorted(
            ranked,
            key=lambda x: (
                -x["category_allocated_duration"].get(category, 0.0),
                -x["selection_score"],
                x["motion_id"],
            ),
        ):
            if state["category_duration"][category] >= minimum:
                break
            if item["motion_id"] in decided:
                continue
            if item["category_allocated_duration"].get(category, 0.0) <= 0:
                continue
            accepted, status, reason = _can_accept(item, state, config)
            if accepted:
                _accept(item, state, "minimum target for {}".format(category))
                decided.add(item["motion_id"])

    for item in ranked:
        if item["motion_id"] in decided:
            continue
        accepted, status, reason = _can_accept(item, state, config)
        if accepted:
            _accept(item, state, reason)
        else:
            item["quota_status"] = status
            item["quota_status_reason"] = reason
        decided.add(item["motion_id"])
    return records
