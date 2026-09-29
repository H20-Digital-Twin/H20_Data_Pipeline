#!/usr/bin/env python3
"""Visualize 100 HDRIs using render_multiview_hdri.py's selection/orientation.

Run with:
  /home/dell/anaconda3/envs/myenv/bin/python visualize_hdri_100.py --overwrite

This file is standalone and may be deleted after the previews are generated.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "selection" / "artifacts" / "v4" / "candidate_v4.jsonl"
)
DEFAULT_HDRI_ROOT = Path("/data/HandSynthesis_dataset/hdris_4k")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "work" / "hdri_preview_100"
DEFAULT_SEED = 20260929


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-jsonl", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--hdri-root", type=Path, default=DEFAULT_HDRI_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--view-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--yfov-deg", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.candidate_jsonl = args.candidate_jsonl.resolve()
    args.hdri_root = args.hdri_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.count < 1 or args.view_size < 64 or args.workers < 1:
        parser.error("count/workers must be positive and view-size must be >= 64")
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")
    return args


def read_motion_ids(path: Path) -> list[str]:
    motion_ids = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "motion_id" not in record:
                raise KeyError(f"missing motion_id at {path}:{line_number}")
            motion_ids.append(record["motion_id"])
    return motion_ids


def select_unique_hdris(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = sorted(
        path
        for path in args.hdri_root.iterdir()
        if path.suffix.lower() in {".hdr", ".exr"}
    )
    if len(candidates) < args.count:
        raise ValueError(
            f"requested {args.count} unique HDRIs, but only {len(candidates)} exist"
        )

    selected = []
    used_paths: set[Path] = set()
    for motion_id in read_motion_ids(args.candidate_jsonl):
        digest = hashlib.sha256(f"{args.seed}:{motion_id}".encode()).digest()
        index = int.from_bytes(digest[:8], "big") % len(candidates)
        hdri_path = candidates[index]
        if hdri_path in used_paths:
            continue
        used_paths.add(hdri_path)
        selected.append(
            {
                "selection_index": len(selected),
                "motion_id": motion_id,
                "hdri_index_in_sorted_catalog": index,
                "hdri_name": hdri_path.name,
                "hdri_path": str(hdri_path),
                "hdri_rotation_radians": 0.0,
            }
        )
        if len(selected) == args.count:
            return selected
    raise ValueError(
        f"candidate manifest produced only {len(selected)} unique HDRIs; "
        f"cannot reach requested count {args.count}"
    )


def look_at_pose(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    backward = eye - target
    backward /= np.linalg.norm(backward)
    world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(world_up, backward)
    right /= np.linalg.norm(right)
    camera_up = np.cross(backward, right)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.column_stack((right, camera_up, backward))
    pose[:3, 3] = eye
    return pose


def camera_poses() -> list[np.ndarray]:
    # Same azimuth order as render_multiview_hdri.py with four cameras:
    # eye positions +X, +Z, -X, -Z, all looking at the origin, Y-up.
    target = np.zeros(3, dtype=np.float64)
    return [
        look_at_pose(np.asarray([math.cos(angle), 0.0, math.sin(angle)]), target)
        for angle in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
    ]


def camera_rays(pose: np.ndarray, view_size: int, yfov_deg: float) -> np.ndarray:
    focal = 0.5 * view_size / math.tan(0.5 * math.radians(yfov_deg))
    pixel = np.arange(view_size, dtype=np.float32) + 0.5
    x = (pixel - 0.5 * view_size) / focal
    y = (pixel - 0.5 * view_size) / focal
    xx, yy = np.meshgrid(x, y)
    camera_directions = np.stack(
        (xx, yy, -np.ones_like(xx)), axis=-1
    )
    camera_directions /= np.linalg.norm(camera_directions, axis=-1, keepdims=True)
    world_directions = camera_directions @ pose[:3, :3].T
    # Same mapping as Rotation=(pi/2, 0, 0): convert Y-up world rays to
    # Blender's Z-up environment coordinates.  HDR heading rotation is zero.
    return np.stack(
        (
            world_directions[..., 0],
            -world_directions[..., 2],
            world_directions[..., 1],
        ),
        axis=-1,
    )


def sample_environment(source: np.ndarray, directions: np.ndarray) -> np.ndarray:
    height, width, _ = source.shape
    x, y, z = np.moveaxis(directions, -1, 0)
    u = (0.5 + np.arctan2(y, x) / (2.0 * math.pi)) % 1.0
    v = np.clip(0.5 + np.arcsin(np.clip(z, -1.0, 1.0)) / math.pi, 0.0, 1.0)
    px = u * width - 0.5
    py = v * height - 0.5
    x0 = np.floor(px).astype(np.int64) % width
    y0 = np.clip(np.floor(py).astype(np.int64), 0, height - 1)
    x1 = (x0 + 1) % width
    y1 = np.clip(y0 + 1, 0, height - 1)
    wx = (px - np.floor(px))[..., None]
    wy = (py - np.floor(py))[..., None]
    top = source[y0, x0] * (1.0 - wx) + source[y0, x1] * wx
    bottom = source[y1, x0] * (1.0 - wx) + source[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def save_contact_sheet(
    hdri_path: Path,
    poses: list[np.ndarray],
    destination: Path,
    args: argparse.Namespace,
) -> None:
    source_bgr = cv2.imread(str(hdri_path), cv2.IMREAD_UNCHANGED)
    if source_bgr is None or source_bgr.ndim != 3 or source_bgr.shape[2] < 3:
        raise ValueError(f"failed to decode HDRI: {hdri_path}")
    source = source_bgr[..., :3].astype(np.float32, copy=False)
    views = [
        sample_environment(
            source, camera_rays(pose, args.view_size, args.yfov_deg)
        )
        for pose in poses
    ]
    view_size = args.view_size
    canvas = np.empty((view_size * 2, view_size * 2, 3), dtype=np.float32)
    canvas[view_size:, :view_size] = views[0]
    canvas[view_size:, view_size:] = views[1]
    canvas[:view_size, :view_size] = views[2]
    canvas[:view_size, view_size:] = views[3]
    # Automatic exposure followed by an ACES-style display transform.  This
    # affects preview brightness only; the actual renderer still consumes the
    # original floating-point HDR values.
    luminance = (
        0.0722 * canvas[..., 0]
        + 0.7152 * canvas[..., 1]
        + 0.2126 * canvas[..., 2]
    )
    log_average = float(np.exp(np.mean(np.log(np.maximum(luminance, 1e-6)))))
    exposed = canvas * (0.18 / max(log_average, 1e-6))
    mapped = np.clip(
        exposed * (2.51 * exposed + 0.03)
        / (exposed * (2.43 * exposed + 0.59) + 0.14),
        0.0,
        1.0,
    )
    srgb = np.where(
        mapped <= 0.0031308,
        12.92 * mapped,
        1.055 * np.power(mapped, 1.0 / 2.4) - 0.055,
    )
    output_bgr = np.rint(srgb * 255.0).astype(np.uint8)
    if not cv2.imwrite(str(destination), output_bgr):
        raise OSError(f"failed to save preview: {destination}")


def save_overview(
    selected: list[dict[str, Any]], output_root: Path, tile_size: int = 160
) -> Path:
    columns = 10
    rows = math.ceil(len(selected) / columns)
    overview = np.full(
        (rows * tile_size, columns * tile_size, 3), 24, dtype=np.uint8
    )
    for item in selected:
        index = int(item["selection_index"])
        preview = cv2.imread(str(output_root / item["preview"]), cv2.IMREAD_COLOR)
        if preview is None:
            raise FileNotFoundError(output_root / item["preview"])
        tile = cv2.resize(
            preview, (tile_size, tile_size), interpolation=cv2.INTER_AREA
        )
        cv2.putText(
            tile,
            f"{index:03d}",
            (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            f"{index:03d}",
            (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        row, column = divmod(index, columns)
        overview[
            row * tile_size : (row + 1) * tile_size,
            column * tile_size : (column + 1) * tile_size,
        ] = tile
    destination = output_root / "overview.png"
    if not cv2.imwrite(str(destination), overview):
        raise OSError(f"failed to save overview: {destination}")
    return destination


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        if args.overwrite:
            shutil.rmtree(args.output_root)
        elif not args.resume:
            raise FileExistsError(
                f"output exists; pass --resume or --overwrite: {args.output_root}"
            )
    previews_dir = args.output_root / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    selected = select_unique_hdris(args)
    poses = camera_poses()

    futures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for item in selected:
            selection_index = int(item["selection_index"])
            hdri_path = Path(item["hdri_path"])
            preview_name = f"{selection_index:03d}_{hdri_path.stem}.png"
            preview_path = previews_dir / preview_name
            item["preview"] = str(preview_path.relative_to(args.output_root))
            if args.resume and preview_path.is_file():
                print(
                    f"HDRI PREVIEW {selection_index + 1}/{len(selected)}: "
                    f"SKIP {hdri_path.name}",
                    flush=True,
                )
                continue
            future = executor.submit(
                save_contact_sheet, hdri_path, poses, preview_path, args
            )
            futures[future] = (selection_index, hdri_path)
        for future in as_completed(futures):
            selection_index, hdri_path = futures[future]
            future.result()
            print(
                f"HDRI PREVIEW {selection_index + 1}/{len(selected)}: "
                f"{hdri_path.name}",
                flush=True,
            )

    overview_path = save_overview(selected, args.output_root)
    manifest = {
        "selection_logic": "sha256(f'{seed}:{motion_id}') first 8 bytes modulo sorted HDR catalog",
        "seed": args.seed,
        "count": len(selected),
        "view_order": ["eye_+X", "eye_+Z", "eye_-X", "eye_-Z"],
        "contact_sheet_layout_top_to_bottom": [
            ["eye_-X", "eye_-Z"],
            ["eye_+X", "eye_+Z"],
        ],
        "coordinate_system": "Y-up; HDR mapping X rotation pi/2; HDR heading rotation 0",
        "yfov_deg": args.yfov_deg,
        "view_size": args.view_size,
        "workers": args.workers,
        "contact_sheet_size": [args.view_size * 2, args.view_size * 2],
        "items": selected,
        "overview": str(overview_path.relative_to(args.output_root)),
    }
    with (args.output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Saved {len(selected)} previews to {previews_dir}", flush=True)


if __name__ == "__main__":
    main()
