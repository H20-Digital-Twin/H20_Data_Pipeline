"""Interval-safe BABEL timeline statistics."""

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class ActionInterval:
    category: str
    start: float
    end: float


def clip_interval(start, end, duration):
    start = max(0.0, min(float(start), float(duration)))
    end = max(0.0, min(float(end), float(duration)))
    return None if end <= start else (start, end)


def merge_intervals(intervals):
    if not intervals:
        return []
    merged = [item for item in sorted(intervals)]
    output = [merged[0]]
    for start, end in merged[1:]:
        last_start, last_end = output[-1]
        if start <= last_end:
            output[-1] = (last_start, max(last_end, end))
        else:
            output.append((start, end))
    return output


def union_duration(intervals):
    return sum(end - start for start, end in merge_intervals(intervals))


def compute_timeline_statistics(intervals, duration, excluded_from_allocation=None):
    """Compute per-class inclusive/allocated durations without double counting."""
    excluded = set(excluded_from_allocation or [])
    duration = max(0.0, float(duration))
    boundaries = {0.0, duration}
    valid = []
    for item in intervals:
        clipped = clip_interval(item.start, item.end, duration)
        if clipped is None:
            continue
        start, end = clipped
        valid.append(ActionInterval(item.category, start, end))
        boundaries.update((start, end))

    inclusive_intervals = defaultdict(list)
    for item in valid:
        inclusive_intervals[item.category].append((item.start, item.end))

    allocated = defaultdict(float)
    unlabeled = 0.0
    boundaries = sorted(boundaries)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right <= left:
            continue
        middle = (left + right) * 0.5
        active = {
            item.category
            for item in valid
            if item.start <= middle < item.end
        } - excluded
        delta = right - left
        if not active:
            unlabeled += delta
            continue
        share = delta / len(active)
        for category in active:
            allocated[category] += share

    return {
        "inclusive_duration": {
            category: union_duration(values)
            for category, values in sorted(inclusive_intervals.items())
        },
        "allocated_duration": dict(sorted(allocated.items())),
        "unlabeled_duration": unlabeled,
    }


def intervals_union_duration(intervals, duration):
    clipped = [clip_interval(start, end, duration) for start, end in intervals]
    return union_duration([item for item in clipped if item is not None])
