#!/usr/bin/env python3
"""Filter an NPZ path list by actual source duration and/or raw frame count."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="One absolute NPZ path per line")
    parser.add_argument("--output", required=True, help="Filtered path list")
    parser.add_argument("--metadata-output", help="Optional per-sequence JSONL audit")
    parser.add_argument("--summary-output", help="Optional JSON summary")
    parser.add_argument("--min-duration-sec", type=float)
    parser.add_argument("--min-frames", type=int)
    args = parser.parse_args()
    if args.min_duration_sec is None and args.min_frames is None:
        parser.error("at least one of --min-duration-sec or --min-frames is required")
    if args.min_duration_sec is not None and args.min_duration_sec < 0:
        parser.error("--min-duration-sec must be non-negative")
    if args.min_frames is not None and args.min_frames < 0:
        parser.error("--min-frames must be non-negative")
    return args


def load_paths(path: Path) -> list[Path]:
    paths = [Path(line.strip()) for line in path.open(encoding="utf-8") if line.strip()]
    if len(paths) != len(set(paths)):
        raise ValueError(f"input contains duplicate paths: {path}")
    return paths


def inspect_npz(path: Path) -> dict:
    if not path.is_absolute() or not path.is_file() or path.suffix.casefold() != ".npz":
        raise FileNotFoundError(f"invalid absolute NPZ path: {path}")
    with np.load(path, allow_pickle=False) as data:
        poses = data["poses"]
        trans = data["trans"]
        fps = float(np.asarray(data["mocap_frame_rate"]).reshape(-1)[0])
        if poses.ndim != 2 or trans.shape != (poses.shape[0], 3):
            raise ValueError(
                f"invalid poses/trans shapes in {path}: {poses.shape}, {trans.shape}"
            )
        if not np.isfinite(poses).all() or not np.isfinite(trans).all():
            raise ValueError(f"non-finite poses/trans in {path}")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"invalid mocap_frame_rate in {path}: {fps}")
        frame_count = int(poses.shape[0])
    return {
        "path": str(path),
        "frame_count": frame_count,
        "source_fps": fps,
        # Match the AMASS/V4 convention used by this repository: N / FPS.
        "duration_sec": frame_count / fps,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    metadata_path = Path(args.metadata_output).resolve() if args.metadata_output else None
    summary_path = Path(args.summary_output).resolve() if args.summary_output else None

    records = [inspect_npz(path) for path in load_paths(input_path)]
    for record in records:
        duration_pass = (
            args.min_duration_sec is None
            or record["duration_sec"] >= args.min_duration_sec
        )
        frame_pass = args.min_frames is None or record["frame_count"] >= args.min_frames
        record["selected"] = bool(duration_pass and frame_pass)
    selected = [record for record in records if record["selected"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(record["path"] + "\n")

    if metadata_path:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "min_duration_sec": args.min_duration_sec,
        "min_frames": args.min_frames,
        "input_count": len(records),
        "selected_count": len(selected),
        "rejected_count": len(records) - len(selected),
        "selected_duration_sec": round(
            sum(record["duration_sec"] for record in selected), 6
        ),
        "selected_fps_count": {
            str(fps): count
            for fps, count in sorted(
                Counter(record["source_fps"] for record in selected).items()
            )
        },
        "minimum_selected_duration_sec": (
            round(min(record["duration_sec"] for record in selected), 6)
            if selected
            else None
        ),
        "maximum_rejected_duration_sec": (
            round(
                max(
                    record["duration_sec"]
                    for record in records
                    if not record["selected"]
                ),
                6,
            )
            if len(selected) != len(records)
            else None
        ),
    }
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
