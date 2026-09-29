#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (
    atomic_write_bytes,
    atomic_write_json,
    load_json,
    load_yaml,
    read_candidate_manifest,
    utc_now,
)
from validate_output import validate_robot_output


def select_items(
    items: list[dict],
    motion_ids: list[str],
    limit: int | None,
    select_all: bool,
) -> list[dict]:
    if select_all and motion_ids:
        raise ValueError("--all and --motion-id cannot be combined")
    if not select_all and not motion_ids and limit is None:
        raise ValueError("pass --motion-id, --limit, or explicit --all")
    if motion_ids:
        by_id = {item["motion_id"]: item for item in items}
        missing = [motion_id for motion_id in motion_ids if motion_id not in by_id]
        if missing:
            raise KeyError(f"motion IDs are not in frozen V4: {missing}")
        selected = [by_id[motion_id] for motion_id in motion_ids]
    else:
        selected = list(items)
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit cannot be negative")
        selected = selected[:limit]
    return selected


def inventory_record(item: dict, output_root: Path, robot_id: str) -> dict:
    motion_id = item["motion_id"]
    robot_dir = output_root / "motions" / motion_id / "robots" / robot_id
    failure_path = output_root / "failures" / f"{motion_id}__{robot_id}.json"
    base = {
        "motion_id": motion_id,
        "robot_id": robot_id,
        "source_dataset": item.get("source_dataset"),
        "subject_id": item.get("subject_id"),
        "source_relative_path": item["relative_path"],
        "source_fps": item.get("source_fps"),
        "source_num_frames": item.get("source_num_frames"),
        "expected_num_frames_at_30fps": item.get("expected_num_frames_at_30fps"),
        "trajectory_30fps_path": None,
        "trajectory_15fps_path": None,
        "diagnostics_path": None,
        "metadata_path": None,
        "quality_path": None,
        "generation_status": "pending",
        "quality_status": None,
        "quality_reasons": [],
        "error_type": None,
        "error_message": None,
    }
    metadata_path = robot_dir / "metadata.json"
    quality_path = robot_dir / "quality.json"
    if metadata_path.is_file() and quality_path.is_file():
        metadata = load_json(metadata_path)
        quality = load_json(quality_path)
        relative = robot_dir.relative_to(output_root)
        base.update(
            {
                "generation_status": quality["generation_status"],
                "quality_status": quality["quality_status"],
                "quality_reasons": quality["quality_reasons"],
                "robot_num_frames_30fps": quality["num_frames_30fps"],
                "robot_num_frames_15fps": quality["num_frames_15fps"],
                "valid_frame_ratio": quality["valid_frame_ratio"],
                "ik_converged_ratio": quality["ik_converged_ratio"],
                "ik_position_error_mean_m": quality["ik_position_error_mean_m"],
                "ik_position_error_p95_m": quality["ik_position_error_p95_m"],
                "ik_rotation_error_mean_rad": quality[
                    "ik_rotation_error_mean_rad"
                ],
                "ik_rotation_error_p95_rad": quality[
                    "ik_rotation_error_p95_rad"
                ],
                "joint_limit_violation_frame_ratio": quality[
                    "joint_limit_violation_frame_ratio"
                ],
                "generation_time_sec": metadata["generation_time_sec"],
                "trajectory_30fps_path": str(relative / "trajectory_30fps.npz"),
                "trajectory_15fps_path": str(relative / "trajectory_15fps.npz"),
                "diagnostics_path": str(relative / "diagnostics.npz"),
                "metadata_path": str(relative / "metadata.json"),
                "quality_path": str(relative / "quality.json"),
            }
        )
    elif failure_path.is_file():
        failure = load_json(failure_path)
        base.update(
            {
                "generation_status": "failed",
                "error_type": failure.get("error_type"),
                "error_message": failure.get("error_message"),
            }
        )
    return base


def rebuild_inventory(
    items: list[dict], output_root: Path, robot_id: str
) -> tuple[Path, Path]:
    records = [inventory_record(item, output_root, robot_id) for item in items]
    manifest_dir = output_root / "manifests"
    inventory_path = manifest_dir / f"robot_generation_inventory_{robot_id}.jsonl"
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    ).encode("utf-8")
    atomic_write_bytes(inventory_path, content)

    status_counts: dict[str, int] = {}
    quality_counts: dict[str, int] = {}
    generated = []
    for record in records:
        status_counts[record["generation_status"]] = (
            status_counts.get(record["generation_status"], 0) + 1
        )
        if record["quality_status"]:
            quality_counts[record["quality_status"]] = (
                quality_counts.get(record["quality_status"], 0) + 1
            )
        if record["generation_status"] in {
            "generated",
            "generated_with_warnings",
        }:
            generated.append(record)
    summary = {
        "schema_version": "robot_generation_summary_v1",
        "created_at_utc": utc_now(),
        "robot_id": robot_id,
        "input_motion_count": len(records),
        "generation_status_counts": status_counts,
        "quality_status_counts": quality_counts,
        "total_robot_frames_30fps": int(
            sum(record.get("robot_num_frames_30fps", 0) for record in generated)
        ),
        "total_robot_frames_15fps": int(
            sum(record.get("robot_num_frames_15fps", 0) for record in generated)
        ),
        "total_generation_time_sec": float(
            sum(record.get("generation_time_sec", 0.0) for record in generated)
        ),
        "inventory_path": str(inventory_path.relative_to(output_root)),
    }
    summary_path = manifest_dir / f"robot_generation_summary_{robot_id}.json"
    atomic_write_json(summary_path, summary)
    return inventory_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--robot")
    parser.add_argument("--motion-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--all-mujoco-frames", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    robot_id = args.robot or config["robot"]["robot_id"]
    candidate_path = Path(config["paths"]["candidate_jsonl"])
    output_root = Path(config["paths"]["output_root"])
    all_items = read_candidate_manifest(candidate_path)
    selected = select_items(
        all_items, args.motion_id, args.limit, args.all
    )
    failures = 0
    started = time.perf_counter()
    try:
        for item in tqdm(
            selected,
            desc=f"BATCH {robot_id}",
            unit="motion",
            dynamic_ncols=True,
        ):
            motion_id = item["motion_id"]
            try:
                if args.validate_only:
                    robot_dir = (
                        output_root
                        / "motions"
                        / motion_id
                        / "robots"
                        / robot_id
                    )
                    validate_robot_output(
                        robot_dir,
                        check_all_mujoco_frames=args.all_mujoco_frames,
                        write_report=True,
                    )
                else:
                    command = [
                        sys.executable,
                        str(HERE / "generate_one.py"),
                        "--config",
                        str(args.config),
                        "--motion-id",
                        motion_id,
                        "--robot",
                        robot_id,
                        "--overwrite" if args.overwrite else "--resume",
                    ]
                    completed = subprocess.run(command, check=False)
                    if completed.returncode:
                        failures += 1
            except Exception as error:
                failures += 1
                print(f"[{motion_id}] validation failed: {type(error).__name__}: {error}")
    finally:
        inventory_path, summary_path = rebuild_inventory(
            all_items, output_root, robot_id
        )

    report = {
        "robot_id": robot_id,
        "selected_motion_count": len(selected),
        "failed_this_run": failures,
        "elapsed_sec": time.perf_counter() - started,
        "inventory": str(inventory_path),
        "summary": str(summary_path),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
