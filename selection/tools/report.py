"""Deterministic CSV/JSONL output for the upgraded AMASS pipeline."""

import csv
import json
from pathlib import Path

from schemas import (
    CANDIDATE_FIELDS,
    DUAL_FILTER_FIELDS,
    INVENTORY_FIELDS,
    STRUCTURED_FIELDS,
)


OUTPUT_NAMES = (
    "amass_inventory.csv",
    "amass_inventory.jsonl",
    "dual_filter_passed.jsonl",
    "candidate_v1.jsonl",
    "scan_summary.json",
    "babel_path_mapping.csv",
    "babel_path_summary.json",
)

PATH_FIELDS = [
    "babel_id", "babel_split", "babel_feat_p", "babel_match_status",
    "local_relative_path", "babel_match_candidates",
]


def _write_jsonl(path, records, fields):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            item = {field: record.get(field) for field in fields}
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def _csv_value(field, value):
    if field in STRUCTURED_FIELDS or isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def ensure_writable(output_root, overwrite, dry_run):
    if dry_run:
        return
    existing = [
        str(Path(output_root) / name)
        for name in OUTPUT_NAMES
        if (Path(output_root) / name).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "output files already exist; use --overwrite or another directory:\n{}"
            .format("\n".join(existing))
        )


def write_reports(records, summary, mapping_rows, mapping_summary, output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "amass_inventory.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    field: _csv_value(field, record.get(field))
                    for field in INVENTORY_FIELDS
                }
            )
    _write_jsonl(output_root / "amass_inventory.jsonl", records, INVENTORY_FIELDS)
    dual = [
        item for item in records
        if item["heuristic_status"] == "heuristic_passed"
        and item["semantic_status"] in (
            "semantic_passed", "babel_unmatched", "babel_ambiguous"
        )
    ]
    selected = [
        item for item in records if item["final_status"] == "selected_candidate"
    ]
    _write_jsonl(
        output_root / "dual_filter_passed.jsonl", dual, DUAL_FILTER_FIELDS
    )
    _write_jsonl(
        output_root / "candidate_v1.jsonl", selected, CANDIDATE_FIELDS
    )
    with (output_root / "scan_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    with (output_root / "babel_path_mapping.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=PATH_FIELDS)
        writer.writeheader()
        for row in mapping_rows:
            writer.writerow(
                {field: _csv_value(field, row.get(field)) for field in PATH_FIELDS}
            )
    with (output_root / "babel_path_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(mapping_summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
