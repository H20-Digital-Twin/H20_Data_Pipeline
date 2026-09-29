from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, Mapping
import zipfile

import numpy as np
import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, content)


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Create compressed NPZ bytes with stable member ordering and timestamps."""
    output = BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for key in sorted(arrays):
            npy = BytesIO()
            np.lib.format.write_array(
                npy, np.asanyarray(arrays[key]), allow_pickle=False
            )
            info = zipfile.ZipInfo(
                filename=f"{key}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                npy.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
            )
    return output.getvalue()


def atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    atomic_write_bytes(path, deterministic_npz_bytes(arrays))


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def read_candidate_manifest(path: Path) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            motion_id = item.get("motion_id")
            relative_path = item.get("relative_path")
            if not motion_id or not relative_path:
                raise ValueError(
                    f"{path}:{line_number} lacks motion_id/relative_path"
                )
            if motion_id in seen:
                raise ValueError(f"duplicate motion_id in candidate manifest: {motion_id}")
            seen.add(motion_id)
            records.append(item)
    return records


def select_candidate(path: Path, motion_id: str) -> dict:
    matches = [
        item
        for item in read_candidate_manifest(path)
        if item["motion_id"] == motion_id
    ]
    if len(matches) != 1:
        raise KeyError(
            f"expected exactly one V4 record for {motion_id}, got {len(matches)}"
        )
    return matches[0]


def replace_directory_atomically(staging: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(staging), str(target))


def remove_exact_directory(path: Path, allowed_parent: Path) -> None:
    resolved = path.resolve()
    parent = allowed_parent.resolve()
    if resolved == parent or parent not in resolved.parents:
        raise ValueError(f"refusing to remove path outside output root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def quantiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }
