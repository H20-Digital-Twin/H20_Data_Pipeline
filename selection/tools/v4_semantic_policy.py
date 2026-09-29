"""Exhaustive, user-confirmed BABEL act_cat policy for the V4 semantic core."""

from collections import Counter

from category_taxonomy import normalize_label


TIER_KEYS = ("reject_any", "conditional", "allowed", "meta")


def _sets(config):
    return {
        key: {normalize_label(value) for value in config[key]}
        for key in TIER_KEYS
    }


def validate_policy(config, babel_entries):
    """Require a disjoint and exhaustive partition of dense train/val act_cat."""
    tiers = _sets(config)
    owner = {}
    duplicates = {}
    for tier, labels in tiers.items():
        for label in labels:
            if label in owner:
                duplicates.setdefault(label, [owner[label]]).append(tier)
            owner[label] = tier
    if duplicates:
        raise ValueError("V4 semantic tiers overlap: {}".format(duplicates))

    semantic_splits = {"train", "val"}
    observed = {
        category
        for entry in babel_entries
        if entry["babel_split"] in semantic_splits
        for label in entry["sequence_labels"] + entry["frame_labels"]
        for category in label["act_cat"]
    }
    configured = set(owner)
    missing = sorted(observed - configured)
    extra = sorted(configured - observed)
    if missing or extra:
        raise ValueError(
            "V4 policy must exactly cover dense train/val act_cat; "
            "missing={}, extra={}".format(missing, extra)
        )
    return {
        "observed_category_count": len(observed),
        "tier_counts": {
            tier: len(tiers[tier]) for tier in TIER_KEYS
        },
        "duplicate_category_count": 0,
        "missing_category_count": 0,
        "extra_category_count": 0,
    }


def _entry_categories(entry, frame_min_duration):
    sequence = {
        category
        for label in entry["sequence_labels"]
        for category in label["act_cat"]
    }
    frame = set()
    short_frame = set()
    frame_durations = {}
    for label in entry["frame_labels"]:
        start, end = label.get("start"), label.get("end")
        if start is None or end is None:
            continue
        duration = max(0.0, float(end) - float(start))
        target = frame if duration >= frame_min_duration else short_frame
        for category in label["act_cat"]:
            target.add(category)
            if duration >= frame_min_duration:
                frame_durations[category] = (
                    frame_durations.get(category, 0.0) + duration
                )
    return sequence, frame, short_frame, frame_durations


def apply_v4_semantic_policy(record, match, config):
    """Attach exact act_cat evidence and decide reject/eligible/unverified."""
    tiers = _sets(config)
    entry = match.get("entry")
    record["semantic_policy_version"] = config["version"]
    record["babel_act_cat_sequence"] = []
    record["babel_act_cat_frame"] = []
    record["babel_act_cat_short_frame_ignored"] = []
    record["babel_act_cat_all"] = []
    record["semantic_tier_hits"] = {
        "R": [], "C": [], "A": [], "M": [],
    }
    record["semantic_reject_matches"] = []
    record["semantic_reject_frame_duration_sec"] = {}

    if entry is None or entry["babel_split"] not in ("train", "val"):
        record["semantic_status_v4"] = "defer_semantic_unverified"
        record["semantic_status_reason_v4"] = (
            "no usable dense train/val BABEL act_cat"
        )
        return record

    sequence, frame, short_frame, frame_durations = _entry_categories(
        entry, float(config["frame_event_min_duration_sec"])
    )
    all_categories = sequence | frame
    record["babel_act_cat_sequence"] = sorted(sequence)
    record["babel_act_cat_frame"] = sorted(frame)
    record["babel_act_cat_short_frame_ignored"] = sorted(short_frame - frame)
    record["babel_act_cat_all"] = sorted(all_categories)
    record["semantic_tier_hits"] = {
        "R": sorted(all_categories & tiers["reject_any"]),
        "C": sorted(all_categories & tiers["conditional"]),
        "A": sorted(all_categories & tiers["allowed"]),
        "M": sorted(all_categories & tiers["meta"]),
    }

    reject_matches = all_categories & tiers["reject_any"]
    calibration_evidence = {
        normalize_label(value) for value in config["calibration_evidence"]
    }
    legacy_calibration = (
        set(record.get("semantic_hard_matches") or []) & calibration_evidence
    )
    if reject_matches or legacy_calibration:
        combined = sorted(reject_matches | legacy_calibration)
        record["semantic_reject_matches"] = combined
        record["semantic_reject_frame_duration_sec"] = {
            category: round(frame_durations.get(category, 0.0), 6)
            for category in sorted(reject_matches & frame)
        }
        record["semantic_status_v4"] = "reject_semantic_any_occurrence"
        record["semantic_status_reason_v4"] = (
            "confirmed R label/evidence occurs: {}".format(", ".join(combined))
        )
    elif (
        record["semantic_tier_hits"]["A"]
        or record["semantic_tier_hits"]["C"]
    ):
        record["semantic_status_v4"] = "semantic_core_eligible"
        record["semantic_status_reason_v4"] = (
            "usable A/C BABEL act_cat and no R occurrence"
        )
    else:
        record["semantic_status_v4"] = "defer_semantic_unverified"
        record["semantic_status_reason_v4"] = (
            "only meta labels or no usable physical act_cat"
        )
    return record


def semantic_statistics(records):
    statuses = Counter(item["semantic_status_v4"] for item in records)
    reject_labels = Counter(
        label
        for item in records
        for label in item.get("semantic_reject_matches", [])
    )
    return {
        "status_distribution": dict(sorted(statuses.items())),
        "reject_label_sequence_counts": dict(
            sorted(reject_labels.items(), key=lambda value: (-value[1], value[0]))
        ),
    }
