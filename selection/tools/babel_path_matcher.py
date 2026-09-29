"""Conservative multi-level mapping from BABEL paths to local AMASS files."""

import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def normalize_path_text(value):
    value = unicodedata.normalize("NFKC", str(value)).replace("\\", "/")
    value = re.sub(r"/+", "/", value)
    return value.strip("/").casefold()


def normalize_token(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^0-9a-z]+", "_", value).strip("_")


def normalize_motion_stem(filename):
    name = Path(filename).stem
    name = re.sub(r"_(poses|stageii|stagei)$", "", name, flags=re.IGNORECASE)
    return normalize_token(name)


def _dataset(value, aliases):
    normalized = normalize_token(value)
    return normalize_token(aliases.get(normalized, normalized))


def _local_identity(path, root, aliases):
    relative = path.relative_to(root)
    parts = relative.parts
    return {
        "relative_path": relative.as_posix(),
        "dataset": _dataset(parts[0], aliases),
        "parent": normalize_token(parts[-2]) if len(parts) > 1 else "",
        "stem": normalize_motion_stem(parts[-1]),
        "normalized_path": "/".join(normalize_token(part) for part in parts),
    }


def _babel_identity(entry, aliases):
    parts = normalize_path_text(entry["babel_feat_p"]).split("/")
    return {
        "dataset": _dataset(parts[0] if parts else "", aliases),
        "parent": normalize_token(parts[-2]) if len(parts) > 1 else "",
        "stem": normalize_motion_stem(parts[-1]) if parts else "",
        "normalized_path": "/".join(normalize_token(part) for part in parts),
    }


def build_path_mapping(local_files, amass_root, babel_entries, config):
    aliases = {
        normalize_token(key): value
        for key, value in config.get("dataset_aliases", {}).items()
    }
    local_identities = [
        _local_identity(path, amass_root, aliases) for path in local_files
    ]
    indices = {
        "exact_path": defaultdict(list),
        "anchor_exact": defaultdict(list),
        "parent_stem_unique": defaultdict(list),
        "stem_unique": defaultdict(list),
    }
    for item in local_identities:
        indices["exact_path"][item["normalized_path"]].append(item["relative_path"])
        indices["anchor_exact"][
            (item["dataset"], item["parent"], item["stem"])
        ].append(item["relative_path"])
        indices["parent_stem_unique"][(item["parent"], item["stem"])].append(
            item["relative_path"]
        )
        indices["stem_unique"][item["stem"]].append(item["relative_path"])

    rows = []
    local_matches = defaultdict(list)
    levels = [
        ("match_exact_path", "exact_path"),
        ("match_anchor_exact", "anchor_exact"),
        ("match_parent_stem_unique", "parent_stem_unique"),
        ("match_stem_unique", "stem_unique"),
    ]
    allowed = {
        "exact_path": True,
        "anchor_exact": config.get("allow_anchor_match", True),
        "parent_stem_unique": config.get("allow_parent_stem_match", True),
        "stem_unique": config.get("allow_unique_stem_match", True),
    }
    for entry in babel_entries:
        identity = _babel_identity(entry, aliases)
        keys = {
            "exact_path": identity["normalized_path"],
            "anchor_exact": (
                identity["dataset"],
                identity["parent"],
                identity["stem"],
            ),
            "parent_stem_unique": (identity["parent"], identity["stem"]),
            "stem_unique": identity["stem"],
        }
        status = "unmatched"
        candidates = []
        for status_name, index_name in levels:
            if not allowed[index_name]:
                continue
            found = indices[index_name].get(keys[index_name], [])
            if len(found) == 1:
                status, candidates = status_name, list(found)
                break
            if len(found) > 1:
                status, candidates = "ambiguous", list(found)
                break
        row = {
            "babel_id": entry["babel_id"],
            "babel_split": entry["babel_split"],
            "babel_feat_p": entry["babel_feat_p"],
            "babel_match_status": status,
            "local_relative_path": candidates[0] if len(candidates) == 1 else None,
            "babel_match_candidates": sorted(candidates),
        }
        rows.append(row)
        if row["local_relative_path"]:
            local_matches[row["local_relative_path"]].append((entry, row))

    # Never guess if multiple dense annotations resolve to one local sequence.
    mapping = {}
    for relative in (item["relative_path"] for item in local_identities):
        matches = local_matches.get(relative, [])
        if len(matches) == 1:
            entry, row = matches[0]
            mapping[relative] = {"entry": entry, **row}
        elif len(matches) > 1:
            conflict_ids = [
                "{}:{}".format(item[0]["babel_split"], item[0]["babel_id"])
                for item in matches
            ]
            for _, conflict_row in matches:
                conflict_row["babel_match_status"] = "ambiguous_local_conflict"
                conflict_row["babel_match_candidates"] = conflict_ids
            mapping[relative] = {
                "entry": None,
                "babel_id": None,
                "babel_split": None,
                "babel_feat_p": None,
                "babel_match_status": "ambiguous",
                "local_relative_path": relative,
                "babel_match_candidates": conflict_ids,
            }
        else:
            mapping[relative] = {
                "entry": None,
                "babel_id": None,
                "babel_split": None,
                "babel_feat_p": None,
                "babel_match_status": "unmatched",
                "local_relative_path": relative,
                "babel_match_candidates": [],
            }

    status_counts = Counter(row["babel_match_status"] for row in rows)
    summary = {
        "local_motion_count": len(local_files),
        "babel_entry_count": len(babel_entries),
        "babel_mapping_status": dict(sorted(status_counts.items())),
        "matched_local_count": sum(
            value["entry"] is not None for value in mapping.values()
        ),
        "ambiguous_local_count": sum(
            value["babel_match_status"] == "ambiguous" for value in mapping.values()
        ),
        "unmatched_local_count": sum(
            value["babel_match_status"] == "unmatched" for value in mapping.values()
        ),
    }
    return mapping, rows, summary
