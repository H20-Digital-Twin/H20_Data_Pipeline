"""Deterministic V3 category -> source -> subject -> global quota selection."""

import math
from collections import defaultdict


def _entropy(proportions):
    return -sum(value * math.log(value + 1e-12) for value in proportions.values())


def _limits(config, group, global_duration, global_count):
    duration = group.get("max_duration_sec")
    ratio = group.get("max_duration_ratio", group.get("max_ratio"))
    if ratio is not None:
        ratio_value = float(ratio) * global_duration
        duration = min(float(duration), ratio_value) if duration is not None else ratio_value
    count = group.get("max_sequence_count")
    count_ratio = group.get("max_sequence_ratio")
    if count_ratio is not None:
        ratio_value = int(math.floor(float(count_ratio) * global_count))
        count = min(int(count), ratio_value) if count is not None else ratio_value
    return (
        float(duration) if duration is not None else None,
        int(count) if count is not None else None,
    )


def _source_config(source, config):
    result = {
        "max_duration_ratio": config["max_duration_ratio_default"],
        "max_sequence_ratio": config["max_sequence_ratio_default"],
    }
    result.update(config.get("overrides", {}).get(source, {}))
    return result


def compute_scores(candidates, config):
    category_pool = defaultdict(float)
    source_pool = defaultdict(float)
    for item in candidates:
        source_pool[item["source_dataset"]] += item["duration_sec"]
        for category, value in item["category_allocated_duration"].items():
            category_pool[category] += value
    weights = config["selection"]
    for item in candidates:
        category_rarity = sum(
            proportion / math.sqrt(category_pool[category] + 1.0)
            for category, proportion in item["category_proportions"].items()
        )
        source_rarity = 1.0 / math.sqrt(
            source_pool[item["source_dataset"]] + 1.0
        )
        score = (
            weights["quality_weight"]
            * float(item.get("heuristic_quality_score") or 1.0)
            + weights["category_rarity_weight"] * category_rarity
            + weights["source_rarity_weight"] * source_rarity
            + weights["multi_action_weight"]
            * _entropy(item["category_proportions"])
        )
        item["selection_score_v3"] = round(score, 10)


def _can_accept(
    item,
    state,
    config,
    source_duration_basis,
    source_count_basis,
):
    global_cfg = config["global"]
    global_duration = float(global_cfg["max_total_duration_sec"])
    global_count = int(global_cfg["max_sequence_count"])
    tolerance = float(config["overshoot_tolerance_sec"])

    # 1. Action-category quota.
    for category, contribution in item["category_allocated_duration"].items():
        category_cfg = config["categories"].get(category)
        if not category_cfg:
            continue
        duration_limit, count_limit = _limits(
            config, category_cfg, global_duration, global_count
        )
        mixed_exception = (
            config["allow_mixed_over_category_max"]
            and (
                item["primary_category"] != category
                or item["dominant_category_ratio"]
                < config["pure_action_dominance_threshold"]
            )
        )
        if (
            duration_limit is not None
            and state["category_duration"][category] + contribution
            > duration_limit + tolerance
            and not mixed_exception
        ):
            return (
                False,
                "defer_category_quota",
                "{} duration quota {:.3f}s reached".format(
                    category, duration_limit
                ),
            )
        if (
            count_limit is not None
            and item["primary_category"] == category
            and state["category_count"][category] + 1 > count_limit
        ):
            return (
                False,
                "defer_category_quota",
                "{} primary sequence quota {} reached".format(
                    category, count_limit
                ),
            )

    # 2. Source-dataset quota.
    if config["source_dataset"]["enabled"]:
        source = item["source_dataset"]
        source_cfg = _source_config(source, config["source_dataset"])
        duration_limit, count_limit = _limits(
            config, source_cfg, source_duration_basis, source_count_basis
        )
        if (
            duration_limit is not None
            and state["source_duration"][source] + item["duration_sec"]
            > duration_limit
        ):
            return (
                False,
                "defer_source_quota",
                "{} source duration quota {:.3f}s reached".format(
                    source, duration_limit
                ),
            )
        if (
            count_limit is not None
            and state["source_count"][source] + 1 > count_limit
        ):
            return (
                False,
                "defer_source_quota",
                "{} source sequence quota {} reached".format(
                    source, count_limit
                ),
            )

    # 3. Subject quota.
    if config["subject"]["enabled"]:
        subject = "{}/{}".format(item["source_dataset"], item["subject_id"])
        if (
            state["subject_duration"][subject] + item["duration_sec"]
            > float(config["subject"]["max_duration_sec"])
        ):
            return (
                False,
                "defer_subject_quota",
                "{} subject duration quota reached".format(subject),
            )
        if (
            state["subject_count"][subject] + 1
            > int(config["subject"]["max_sequence_count"])
        ):
            return (
                False,
                "defer_subject_quota",
                "{} subject sequence quota reached".format(subject),
            )

    # 4. Global budget.
    if state["total_count"] + 1 > global_count:
        return False, "defer_global_budget", "global sequence quota reached"
    if (
        state["total_duration"] + item["duration_sec"]
        > global_duration + tolerance
    ):
        return False, "defer_global_budget", "global duration quota reached"
    return True, "selected_candidate", "accepted_by_v3_quota"


def _accept(item, state, reason):
    item["quota_status"] = "selected_candidate"
    item["quota_status_reason"] = reason
    state["total_count"] += 1
    state["total_duration"] += item["duration_sec"]
    source = item["source_dataset"]
    subject = "{}/{}".format(source, item["subject_id"])
    state["source_count"][source] += 1
    state["source_duration"][source] += item["duration_sec"]
    state["subject_count"][subject] += 1
    state["subject_duration"][subject] += item["duration_sec"]
    state["category_count"][item["primary_category"]] += 1
    for category, value in item["category_allocated_duration"].items():
        state["category_duration"][category] += value


def _select_once(
    candidates,
    config,
    source_duration_basis,
    source_count_basis,
):
    ranked = sorted(
        candidates, key=lambda item: (-item["selection_score_v3"], item["motion_id"])
    )
    for item in candidates:
        item["quota_status"] = "not_decided"
        item["quota_status_reason"] = "not_decided"
    state = {
        "total_count": 0,
        "total_duration": 0.0,
        "category_count": defaultdict(int),
        "category_duration": defaultdict(float),
        "source_count": defaultdict(int),
        "source_duration": defaultdict(float),
        "subject_count": defaultdict(int),
        "subject_duration": defaultdict(float),
    }
    decided = set()

    # First pass: category minimum targets, rarest target first.
    minimums = sorted(
        (
            (category, float(values.get("min_duration_sec", 0.0)))
            for category, values in config["categories"].items()
            if float(values.get("min_duration_sec", 0.0)) > 0
        ),
        key=lambda value: (value[1], value[0]),
    )
    for category, target in minimums:
        category_ranked = sorted(
            ranked,
            key=lambda item: (
                -item["category_allocated_duration"].get(category, 0.0),
                -item["selection_score_v3"],
                item["motion_id"],
            ),
        )
        for item in category_ranked:
            if state["category_duration"][category] >= target:
                break
            if item["motion_id"] in decided:
                continue
            if item["category_allocated_duration"].get(category, 0.0) <= 0:
                continue
            accepted, _, reason = _can_accept(
                item,
                state,
                config,
                source_duration_basis,
                source_count_basis,
            )
            if accepted:
                _accept(item, state, "minimum target for {}".format(category))
                decided.add(item["motion_id"])

    for item in ranked:
        if item["motion_id"] in decided:
            continue
        accepted, status, reason = _can_accept(
            item,
            state,
            config,
            source_duration_basis,
            source_count_basis,
        )
        if accepted:
            _accept(item, state, reason)
        else:
            item["quota_status"] = status
            item["quota_status_reason"] = reason
        decided.add(item["motion_id"])
    return state


def apply_v3_quota(candidates, config):
    compute_scores(candidates, config)
    source_duration_basis = float(config["global"]["max_total_duration_sec"])
    source_count_basis = int(config["global"]["max_sequence_count"])
    state = None

    # A ratio derived only from the configured global budgets can exceed the
    # configured ratio when either final budget is not completely filled.
    # Monotonically tighten both bases until the actual selected totals are
    # the ratio denominators.
    for _ in range(64):
        state = _select_once(
            candidates,
            config,
            source_duration_basis,
            source_count_basis,
        )
        next_duration_basis = min(
            source_duration_basis, state["total_duration"]
        )
        next_count_basis = min(source_count_basis, state["total_count"])
        if (
            abs(next_duration_basis - source_duration_basis) < 1e-9
            and next_count_basis == source_count_basis
        ):
            break
        source_duration_basis = next_duration_basis
        source_count_basis = next_count_basis
    else:
        raise AssertionError("source ratio selection did not converge")

    report = {
        "selected_sequence_count": state["total_count"],
        "selected_duration_sec": round(state["total_duration"], 6),
        "source_duration_ratio_basis_sec": round(source_duration_basis, 6),
        "source_sequence_ratio_basis_count": source_count_basis,
        "category_count": dict(sorted(state["category_count"].items())),
        "category_duration": {
            key: round(value, 6)
            for key, value in sorted(state["category_duration"].items())
        },
        "source_count": dict(sorted(state["source_count"].items())),
        "source_duration": {
            key: round(value, 6)
            for key, value in sorted(state["source_duration"].items())
        },
        "subject_count": dict(sorted(state["subject_count"].items())),
        "subject_duration": {
            key: round(value, 6)
            for key, value in sorted(state["subject_duration"].items())
        },
    }
    return report
