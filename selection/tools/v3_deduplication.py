"""Deterministic V3 exact-duplicate and cross-subject take clustering."""

import hashlib
import math
from collections import defaultdict

import numpy as np


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first, second):
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _candidate_pairs(features, config):
    rng = np.random.RandomState(int(config["deterministic_seed"]))
    projection = rng.normal(
        0.0,
        1.0 / math.sqrt(features.shape[1]),
        size=(features.shape[1], int(config["projection_dim"])),
    ).astype(np.float32)
    projected = features @ projection
    projected /= np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-8)
    pairs = set()
    neighbors = int(config["neighbors_per_item"])
    for start in range(0, len(features), 256):
        similarities = projected[start : start + 256] @ projected.T
        for offset, values in enumerate(similarities):
            index = start + offset
            values[index] = -np.inf
            count = min(neighbors, len(values) - 1)
            for other in np.argpartition(values, -count)[-count:]:
                pairs.add(tuple(sorted((index, int(other)))))
    return sorted(pairs)


def _pair_metrics(first, second, features, roots, records):
    cosine = float(np.dot(features[first], features[second]))
    pose_distance = float(
        np.sqrt(np.mean((features[first] - features[second]) ** 2))
    )
    root_distance = float(
        np.mean(
            np.linalg.norm(
                roots[first].reshape(-1, 3) - roots[second].reshape(-1, 3),
                axis=1,
            )
        )
    )
    duration_ratio = min(
        records[first]["duration_sec"], records[second]["duration_sec"]
    ) / max(records[first]["duration_sec"], records[second]["duration_sec"])
    return cosine, pose_distance, root_distance, duration_ratio


def apply_v3_deduplication(records, descriptors, root_descriptors, config):
    """Mutate passed records with exact/cluster audit fields; return clusters."""
    if not records:
        return []
    # Fix floating-point reduction order and all index-based graph operations.
    records.sort(key=lambda item: item["motion_id"])
    raw = np.stack([descriptors[item["motion_id"]] for item in records]).astype(
        np.float32
    )
    sequence = raw.reshape(len(raw), -1, 63)
    centered = (sequence - sequence.mean(axis=1, keepdims=True)).reshape(
        len(raw), -1
    )
    centered /= np.maximum(
        np.linalg.norm(centered, axis=1, keepdims=True), 1e-8
    )
    roots = np.stack([root_descriptors[item["motion_id"]] for item in records])
    pairs = _candidate_pairs(centered, config["cluster"])

    exact_uf = UnionFind(len(records))
    near_edges = []
    pair_details = {}
    exact_cfg = config["exact"]
    cluster_cfg = config["cluster"]
    for first, second in pairs:
        cosine, pose_distance, root_distance, duration_ratio = _pair_metrics(
            first, second, centered, roots, records
        )
        pair_details[(first, second)] = (
            cosine, pose_distance, root_distance, duration_ratio
        )
        same_subject = (
            records[first]["source_dataset"] == records[second]["source_dataset"]
            and records[first]["subject_id"] == records[second]["subject_id"]
        )
        exact_hash = (
            records[first].get("exact_motion_hash")
            and records[first].get("exact_motion_hash")
            == records[second].get("exact_motion_hash")
        )
        strict_near = (
            same_subject
            and cosine >= exact_cfg["cosine_similarity_min"]
            and duration_ratio >= exact_cfg["duration_ratio_min"]
            and root_distance <= exact_cfg["root_distance_max"]
            and pose_distance <= exact_cfg["pose_distance_max"]
        )
        if exact_hash or strict_near:
            exact_uf.union(first, second)
        if (
            cosine >= cluster_cfg["cosine_similarity_min"]
            and duration_ratio >= cluster_cfg["duration_ratio_min"]
            and root_distance <= cluster_cfg["root_distance_max"]
        ):
            near_edges.append((first, second))

    # Raw exact hashes must be caught even if approximate neighbor recall misses them.
    by_hash = defaultdict(list)
    for index, item in enumerate(records):
        if item.get("exact_motion_hash"):
            by_hash[item["exact_motion_hash"]].append(index)
    for values in by_hash.values():
        for other in values[1:]:
            exact_uf.union(values[0], other)

    exact_groups = defaultdict(list)
    for index in range(len(records)):
        exact_groups[exact_uf.find(index)].append(index)
    deferred_exact = set()
    for members in exact_groups.values():
        if len(members) < 2:
            continue
        members.sort(
            key=lambda idx: (
                -float(records[idx].get("heuristic_quality_score") or 0.0),
                records[idx]["motion_id"],
            )
        )
        representative = members[0]
        for index in members[1:]:
            deferred_exact.add(index)
            cosine, pose_distance, root_distance, _ = _pair_metrics(
                representative, index, centered, roots, records
            )
            records[index]["dedup_status"] = "defer_exact_duplicate"
            records[index]["duplicate_of"] = records[representative]["motion_id"]
            records[index]["duplicate_similarity"] = cosine
            records[index]["duplicate_pose_distance"] = pose_distance
            records[index]["duplicate_root_distance"] = root_distance

    near_uf = UnionFind(len(records))
    for first, second in near_edges:
        if first not in deferred_exact and second not in deferred_exact:
            near_uf.union(first, second)
    groups = defaultdict(list)
    for index in range(len(records)):
        if index not in deferred_exact:
            groups[near_uf.find(index)].append(index)

    clusters = []
    max_keep = int(cluster_cfg["max_sequences_per_cluster"])
    for members in sorted(
        (values for values in groups.values() if len(values) >= 2),
        key=lambda values: min(records[index]["motion_id"] for index in values),
    ):
        member_features = centered[members]
        centroid = member_features.mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-8)
        centrality = {
            index: float(np.dot(centered[index], centroid)) for index in members
        }
        ranked = sorted(
            members,
            key=lambda idx: (
                -centrality[idx],
                -float(records[idx].get("heuristic_quality_score") or 0.0),
                records[idx]["motion_id"],
            ),
        )
        kept = []
        used_subjects = set()
        for index in ranked:
            subject = "{}/{}".format(
                records[index]["source_dataset"], records[index]["subject_id"]
            )
            if subject not in used_subjects and len(kept) < max_keep:
                kept.append(index)
                used_subjects.add(subject)
        for index in ranked:
            if len(kept) >= max_keep:
                break
            if index not in kept:
                kept.append(index)

        cluster_ids = sorted(records[index]["motion_id"] for index in members)
        cluster_id = "cluster_" + hashlib.sha1(
            "\n".join(cluster_ids).encode("utf-8")
        ).hexdigest()[:12]
        representative = kept[0]
        cluster_row = {
            "duplicate_cluster_id": cluster_id,
            "cluster_size": len(members),
            "kept_count": len(kept),
            "cluster_representative": records[representative]["motion_id"],
            "members": [],
        }
        for rank, index in enumerate(ranked, start=1):
            item = records[index]
            item["duplicate_cluster_id"] = cluster_id
            item["cluster_size"] = len(members)
            item["cluster_rank"] = rank
            item["cluster_representative"] = records[representative]["motion_id"]
            item["duplicate_similarity"] = centrality[index]
            if index not in kept:
                cosine, pose_distance, root_distance, _ = _pair_metrics(
                    representative, index, centered, roots, records
                )
                item["dedup_status"] = "defer_redundant_take"
                item["duplicate_of"] = records[representative]["motion_id"]
                item["duplicate_similarity"] = cosine
                item["duplicate_pose_distance"] = pose_distance
                item["duplicate_root_distance"] = root_distance
            elif item.get("dedup_status") is None:
                item["dedup_status"] = "unique_cluster_representative"
            cluster_row["members"].append(
                {
                    "motion_id": item["motion_id"],
                    "subject": "{}/{}".format(
                        item["source_dataset"], item["subject_id"]
                    ),
                    "rank": rank,
                    "kept": index in kept,
                    "centroid_similarity": centrality[index],
                }
            )
        clusters.append(cluster_row)

    for index, item in enumerate(records):
        if item.get("dedup_status") is None:
            item["dedup_status"] = "unique"
    return clusters
