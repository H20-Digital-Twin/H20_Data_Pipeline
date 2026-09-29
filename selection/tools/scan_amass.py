#!/usr/bin/env python3
"""Build an auditable BABEL-aware AMASS candidate dataset."""

import argparse
import fnmatch
import json
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from amass_reader import make_base_record, read_amass_file
from babel_path_matcher import build_path_mapping
from babel_reader import load_babel_entries
from category_taxonomy import CategoryTaxonomy
from deduplication import apply_deduplication
from filter_rules import (
    apply_semantic_filter,
    attach_timeline,
    classify_basic,
    classify_heuristic,
)
from quota_selector import apply_quota
from report import ensure_writable, write_reports
from summary_report import build_pipeline_summary


SEMANTIC_FIELDS = [
    "babel_id", "babel_split", "babel_feat_p", "babel_match_status",
    "babel_match_candidates", "babel_duration_sec", "sequence_tags",
    "frame_tags", "semantic_status", "semantic_status_reason",
    "soft_blacklist_coverage", "semantic_hard_matches",
    "semantic_soft_matches",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--amass-root")
    parser.add_argument("--output-root")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = (
        "amass", "output", "required_fields", "filter", "heuristic",
        "babel", "path_matching", "category_taxonomy", "dedup",
        "quota", "report", "scan",
    )
    for section in required:
        if section not in config:
            raise ValueError("missing configuration section: {}".format(section))
    return config


def discover_files(root, pattern, exclude_patterns, recursive, sort_files):
    iterator = root.glob(pattern) if recursive else root.glob(Path(pattern).name)
    files = [
        path for path in iterator
        if path.is_file()
        and not any(
            fnmatch.fnmatch(path.relative_to(root).as_posix(), excluded)
            for excluded in exclude_patterns
        )
    ]
    if sort_files:
        files.sort(key=lambda path: path.relative_to(root).as_posix())
    return files


def _empty_mapping(files, root):
    return {
        path.relative_to(root).as_posix(): {
            "entry": None,
            "babel_id": None,
            "babel_split": None,
            "babel_feat_p": None,
            "babel_match_status": "unmatched",
            "local_relative_path": path.relative_to(root).as_posix(),
            "babel_match_candidates": [],
        }
        for path in files
    }


def _finalize(records):
    for record in records:
        if str(record["semantic_status"]).startswith("reject_semantic"):
            status = record["semantic_status"]
            reason = record["semantic_status_reason"]
        elif record["basic_status"] != "basic_passed":
            status = record["basic_status"]
            reason = record["basic_status_reason"]
        elif record["heuristic_status"] != "heuristic_passed":
            status = record["heuristic_status"]
            reason = record["heuristic_status_reason"]
        elif record["dedup_status"] != "unique":
            status = "defer_duplicate"
            reason = "{} of {}".format(
                record["dedup_status"], record["duplicate_of"]
            )
        else:
            status = record["quota_status"]
            reason = record["quota_status_reason"]
        record["final_status"] = status
        record["final_status_reason"] = reason
        # Compatibility aliases for consumers of the original first-stage reports.
        record["status"] = status
        record["status_reason"] = reason


def validate_results(records, files_found):
    if len(records) != files_found:
        raise AssertionError("inventory count differs from discovered file count")
    ids = [item["motion_id"] for item in records]
    if len(ids) != len(set(ids)):
        raise AssertionError("motion_id values are not unique")
    if any(not item["final_status"] for item in records):
        raise AssertionError("every item must have exactly one final status")
    selected = [r for r in records if r["final_status"] == "selected_candidate"]
    if not {r["motion_id"] for r in selected}.issubset(set(ids)):
        raise AssertionError("selected candidates are not an inventory subset")
    semantic_rejected = [
        r for r in records if str(r["semantic_status"]).startswith("reject_semantic")
    ]
    if any(r["npz_loaded"] for r in semantic_rejected):
        raise AssertionError("semantic rejection must happen before np.load")


def run_pipeline(config, limit=None, dry_run=False):
    root = Path(config["amass"]["root"]).expanduser().resolve()
    output_root = Path(config["output"]["root"]).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError("AMASS root does not exist: {}".format(root))
    if limit is not None and limit < 0:
        raise ValueError("--limit must be >= 0")
    ensure_writable(output_root, config["output"]["overwrite"], dry_run)

    all_files = discover_files(
        root, config["amass"]["pattern"],
        config["amass"].get("exclude_patterns", []),
        config["scan"]["recursive"], config["scan"]["sort_files"],
    )
    files = all_files[:limit] if limit is not None else all_files

    if config["babel"]["enabled"]:
        dense_paths = config["babel"]["dense_json_files"]
        if config["babel"].get("use_extra_annotations", False):
            raise NotImplementedError(
                "extra annotations require a consensus policy and are disabled in v1"
            )
        babel_entries = load_babel_entries(dense_paths)
        mapping, mapping_rows, mapping_summary = build_path_mapping(
            all_files, root, babel_entries, config["path_matching"]
        )
    else:
        babel_entries = []
        mapping = _empty_mapping(all_files, root)
        mapping_rows = []
        mapping_summary = {
            "local_motion_count": len(all_files), "babel_entry_count": 0,
            "babel_mapping_status": {}, "matched_local_count": 0,
            "ambiguous_local_count": 0, "unmatched_local_count": len(all_files),
        }

    taxonomy = CategoryTaxonomy(config["category_taxonomy"])
    reader_config = dict(config["filter"])
    reader_config["save_error_trace"] = config["scan"]["save_error_trace"]
    reader_config["dedup"] = config["dedup"]
    records = []
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(root).as_posix()
        match = mapping[relative]
        base = make_base_record(path, root)
        apply_semantic_filter(base, match, config["babel"])
        entry = match.get("entry")
        if str(base["semantic_status"]).startswith("reject_semantic"):
            base["basic_status"] = "not_loaded_semantic_reject"
            base["basic_status_reason"] = "semantic rejection precedes NPZ loading"
            base["heuristic_status"] = "not_run"
            base["heuristic_status_reason"] = "semantic filter rejected sequence"
            base["dedup_status"] = "not_eligible"
            base["quota_status"] = "not_eligible"
            attach_timeline(
                base, entry, taxonomy, config["babel"]["timeline"]
            )
            records.append(base)
        else:
            record = read_amass_file(
                path, root, config["required_fields"], reader_config
            )
            for field in SEMANTIC_FIELDS:
                record[field] = base[field]
            if (
                record.get("error_type")
                and not config["scan"]["continue_on_error"]
            ):
                raise RuntimeError(
                    "failed to read {}: {}".format(path, record["error_message"])
                )
            classify_basic(record, config["filter"])
            classify_heuristic(record, config["heuristic"])
            attach_timeline(
                record, entry, taxonomy, config["babel"]["timeline"]
            )
            if record["heuristic_status"] != "heuristic_passed":
                record["dedup_status"] = "not_eligible"
                record["quota_status"] = "not_eligible"
            records.append(record)
        if index % 250 == 0 or index == len(files):
            print("Processed {}/{} files".format(index, len(files)), file=sys.stderr)

    apply_deduplication(records, config["dedup"])
    for record in records:
        if record["dedup_status"] not in (None, "unique"):
            record["quota_status"] = "not_eligible"
        elif record["dedup_status"] is None:
            record["dedup_status"] = "not_eligible"
    apply_quota(records, config["quota"])
    _finalize(records)
    records.sort(key=lambda item: item["relative_path"])
    validate_results(records, len(files))
    summary = build_pipeline_summary(
        records, root, len(files), config["filter"]["target_fps"],
        mapping_summary,
    )
    if not dry_run:
        write_reports(
            records, summary, mapping_rows, mapping_summary, output_root
        )
        print("Reports written to {}".format(output_root), file=sys.stderr)
    return records, summary


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.amass_root:
        config["amass"]["root"] = args.amass_root
    if args.output_root:
        config["output"]["root"] = args.output_root
    if args.overwrite:
        config["output"]["overwrite"] = True
    _, summary = run_pipeline(config, limit=args.limit, dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
