#!/usr/bin/env python3
"""Build a conservative AMASS subset for the 29-DoF Unitree G1 without hands."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import yaml


OUTPUT_FILENAMES = (
    "candidate_g1_no_hands_v1.jsonl",
    "review_fixed_hand_gestures.jsonl",
    "rejected_hand_dependent.jsonl",
    "npz_paths_absolute.txt",
    "npz_paths_relative.txt",
    "npz_mapping.tsv",
    "summary.json",
    "README.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="selection/configs/g1_no_hands_v1.yaml",
        help="YAML policy and path configuration",
    )
    parser.add_argument(
        "--output-root",
        help="Override output.root from the YAML (useful for validation)",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_set(values: list[str], name: str) -> set[str]:
    normalized = {str(value).casefold().strip() for value in values}
    if "" in normalized:
        raise ValueError(f"{name} contains an empty label")
    if len(normalized) != len(values):
        raise ValueError(f"{name} contains duplicate labels")
    return normalized


def normalized_phrase_set(values: list[str], name: str) -> set[str]:
    normalized = {normalize_free_text(value) for value in values}
    if "" in normalized:
        raise ValueError(f"{name} contains an empty phrase")
    if len(normalized) != len(values):
        raise ValueError(f"{name} contains duplicate phrases")
    return normalized


def normalize_free_text(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).casefold())
    text = re.sub(r"(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def free_text_hits(record: dict, phrases: set[str]) -> list[str]:
    values = [record.get("motion_name", "")]
    values.extend(record.get("sequence_tags") or [])
    values.extend(record.get("frame_tags") or [])
    normalized_values = [f" {normalize_free_text(value)} " for value in values]
    return sorted(
        phrase
        for phrase in phrases
        if any(f" {phrase} " in value for value in normalized_values)
    )


def decorate(
    record: dict,
    source: Path,
    policy_version: str,
    status: str,
    reasons: list[str],
) -> dict:
    result = dict(record)
    result.update(
        {
            "g1_no_hands_policy_version": policy_version,
            "g1_no_hands_status": status,
            "g1_no_hands_reasons": reasons,
            "source_npz_absolute": str(source),
        }
    )
    return result


def counter_dict(records: list[dict], field: str) -> dict[str, int]:
    return dict(sorted(Counter(record[field] for record in records).items()))


def duration(records: list[dict]) -> float:
    return round(sum(float(record["duration_sec"]) for record in records), 6)


def render_readme(summary: dict) -> str:
    return f"""# Unitree G1 无手训练 AMASS 子集

该目录从冻结的 V4 候选集筛出适合 **Unitree G1 29-DoF、无灵巧手** 的动作。
默认核心集不会包含依赖物体操控、抓握、手指动作或手部接触语义的序列。

- 核心集：{summary['selected_count']} 条，{summary['selected_duration_sec']:.6f} 秒
- 固定手手势复核集：{summary['review_count']} 条，{summary['review_duration_sec']:.6f} 秒
- 排除集：{summary['rejected_count']} 条，{summary['rejected_duration_sec']:.6f} 秒

`candidate_g1_no_hands_v1.jsonl` 是推荐训练清单；
`npz_paths_absolute.txt` 可直接交给支持逐行路径的训练项目；
`npz/` 是保留 AMASS 相对目录结构的软链接树，不复制原始 NPZ；
`review_fixed_hand_gestures.jsonl` 只适合在确认固定橡胶手可接受后追加。

筛选主要使用 V4 中经过 BABEL 校验的 `babel_act_cat_all` 精确类别，并补充少量
人工审计确认的自由文本短语（例如 `walk with box`、`throwing hard` 和
`check watch`）；不使用宽泛的模糊关键词。输入 V4 清单 SHA-256 为
`{summary['candidate_manifest_sha256']}`。

注意：源文件仍是 SMPL-X Stage-II NPZ（pose_dim=165），其中仍保存手部姿态参数。
“无手”是动作语义筛选，不是改写 NPZ。训练项目若只接受 SMPL/SMPL-H 或机器人
qpos，需要另外做格式转换或使用本仓库的 G1 重定向流程。动作在真实 G1 上是否
可跟踪，还必须以 29-DoF 重定向后的关节限位、速度和接触质量检查为准。

复现命令：

```bash
cd /data/h2o_data_engine
/home/dell/anaconda3/envs/py38/bin/python selection/tools/filter_g1_no_hands.py
```
"""


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    candidate_path = Path(config["input"]["candidate_jsonl"]).resolve()
    amass_root = Path(config["input"]["amass_root"]).resolve()
    output_root = Path(args.output_root or config["output"]["root"]).resolve()
    policy = config["policy"]
    manipulation = normalized_set(
        policy["manipulation_required"], "manipulation_required"
    )
    hand_contact = normalized_set(
        policy["hand_contact_dependent"], "hand_contact_dependent"
    )
    manipulation_phrases = normalized_phrase_set(
        policy.get("free_text_manipulation_phrases", []),
        "free_text_manipulation_phrases",
    )
    hand_review_phrases = normalized_phrase_set(
        policy.get("free_text_hand_review_phrases", []),
        "free_text_hand_review_phrases",
    )
    overlap = manipulation & hand_contact
    if overlap:
        raise ValueError(f"policy lists overlap: {sorted(overlap)}")

    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    if not amass_root.is_dir():
        raise NotADirectoryError(amass_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(candidate_path)
    ids = [record.get("motion_id") for record in records]
    if any(not motion_id for motion_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("candidate manifest has missing or duplicate motion_id values")

    selected: list[dict] = []
    reviews: list[dict] = []
    rejected: list[dict] = []
    missing_sources: list[str] = []
    source_sizes = 0
    for record in records:
        relative = Path(record["relative_path"])
        source = (amass_root / relative).resolve()
        try:
            source.relative_to(amass_root)
        except ValueError as error:
            raise ValueError(f"source escapes AMASS root: {relative}") from error
        if source.suffix.casefold() != ".npz" or not source.is_file():
            missing_sources.append(str(source))
            continue
        source_sizes += source.stat().st_size

        labels = {
            str(value).casefold().strip()
            for value in record.get("babel_act_cat_all", [])
        }
        manipulation_hits = sorted(labels & manipulation)
        hand_hits = sorted(labels & hand_contact)
        manipulation_text_hits = [
            f"free_text:{value}"
            for value in free_text_hits(record, manipulation_phrases)
        ]
        hand_text_hits = [
            f"free_text:{value}"
            for value in free_text_hits(record, hand_review_phrases)
        ]
        if manipulation_hits or manipulation_text_hits:
            rejected.append(
                decorate(
                    record,
                    source,
                    policy["version"],
                    "rejected_manipulation_required",
                    manipulation_hits + manipulation_text_hits,
                )
            )
        elif hand_hits or hand_text_hits:
            reviews.append(
                decorate(
                    record,
                    source,
                    policy["version"],
                    "review_fixed_hand_gesture",
                    hand_hits + hand_text_hits,
                )
            )
        else:
            selected.append(
                decorate(
                    record,
                    source,
                    policy["version"],
                    "selected_core",
                    [],
                )
            )

    if missing_sources:
        raise FileNotFoundError(
            "missing or invalid source NPZ files: " + ", ".join(missing_sources[:10])
        )
    if len(selected) + len(reviews) + len(rejected) != len(records):
        raise AssertionError("classification is not exhaustive")

    selected.sort(key=lambda item: item["motion_id"])
    reviews.sort(key=lambda item: item["motion_id"])
    rejected.sort(key=lambda item: item["motion_id"])
    write_jsonl(output_root / OUTPUT_FILENAMES[0], selected)
    write_jsonl(output_root / OUTPUT_FILENAMES[1], reviews)
    write_jsonl(output_root / OUTPUT_FILENAMES[2], rejected)

    with (output_root / "npz_paths_absolute.txt").open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(record["source_npz_absolute"] + "\n")
    with (output_root / "npz_paths_relative.txt").open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(record["relative_path"] + "\n")
    with (output_root / "npz_mapping.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ("motion_id", "relative_path", "absolute_path", "primary_category")
        )
        for record in selected:
            writer.writerow(
                (
                    record["motion_id"],
                    record["relative_path"],
                    record["source_npz_absolute"],
                    record["primary_category"],
                )
            )

    if config["output"].get("create_symlink_tree", False):
        link_root = output_root / "npz"
        for record in selected:
            link = link_root / record["relative_path"]
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(record["source_npz_absolute"])

    reason_counts = Counter(
        reason for record in reviews + rejected for reason in record["g1_no_hands_reasons"]
    )
    summary = {
        "policy_version": policy["version"],
        "candidate_manifest": str(candidate_path),
        "candidate_manifest_sha256": sha256(candidate_path),
        "amass_root": str(amass_root),
        "input_count": len(records),
        "input_duration_sec": duration(records),
        "selected_count": len(selected),
        "selected_duration_sec": duration(selected),
        "review_count": len(reviews),
        "review_duration_sec": duration(reviews),
        "rejected_count": len(rejected),
        "rejected_duration_sec": duration(rejected),
        "selected_source_count": counter_dict(selected, "source_dataset"),
        "selected_primary_category_count": counter_dict(selected, "primary_category"),
        "blocked_label_occurrence_count": dict(sorted(reason_counts.items())),
        "all_source_npz_size_bytes": source_sizes,
        "symlink_tree_created": bool(
            config["output"].get("create_symlink_tree", False)
        ),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_root / "README.md").open("w", encoding="utf-8") as handle:
        handle.write(render_readme(summary))

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
