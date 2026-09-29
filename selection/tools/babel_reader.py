"""Load dense BABEL annotations and preserve sequence/frame semantics separately."""

import json
from pathlib import Path

from category_taxonomy import normalize_label


def _label_categories(label):
    categories = label.get("act_cat")
    if categories:
        return sorted({normalize_label(value) for value in categories if value})
    fallback = label.get("proc_label") or label.get("raw_label")
    return [normalize_label(fallback)] if fallback else []


def _label_evidence(label):
    values = set(_label_categories(label))
    for key in ("proc_label", "raw_label"):
        if label.get(key):
            values.add(normalize_label(label[key]))
    return sorted(value for value in values if value)


def _parse_labels(annotation, with_time):
    if not annotation:
        return []
    parsed = []
    for label in annotation.get("labels") or []:
        act_cat = sorted(
            {
                normalize_label(value)
                for value in (label.get("act_cat") or [])
                if value
            }
        )
        item = {
            "categories": _label_categories(label),
            # Preserve standardized BABEL categories separately.  Existing
            # callers retain their fallback behavior through ``categories``;
            # semantic-core policies can avoid treating free text as act_cat.
            "act_cat": act_cat,
            "evidence": _label_evidence(label),
            "raw_label": label.get("raw_label"),
            "proc_label": label.get("proc_label"),
        }
        if with_time:
            item["start"] = label.get("start_t")
            item["end"] = label.get("end_t")
        parsed.append(item)
    return parsed


def load_babel_entries(paths):
    entries = []
    for path in paths:
        path = Path(path).expanduser().resolve()
        split = path.stem
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        for babel_id, source in data.items():
            entries.append(
                {
                    "babel_id": str(babel_id),
                    "babel_split": split,
                    "babel_feat_p": source.get("feat_p", ""),
                    "babel_duration_sec": float(source.get("dur") or 0.0),
                    "sequence_labels": _parse_labels(
                        source.get("seq_ann"), with_time=False
                    ),
                    "frame_labels": _parse_labels(
                        source.get("frame_ann"), with_time=True
                    ),
                }
            )
    entries.sort(
        key=lambda item: (
            item["babel_split"],
            item["babel_feat_p"].casefold(),
            item["babel_id"],
        )
    )
    return entries


def all_evidence(entry):
    evidence = set()
    for label in entry["sequence_labels"] + entry["frame_labels"]:
        evidence.update(label["evidence"])
    return evidence


def useful_frame_labels(entry):
    return [
        label
        for label in entry["frame_labels"]
        if label["categories"]
        and label.get("start") is not None
        and label.get("end") is not None
    ]
