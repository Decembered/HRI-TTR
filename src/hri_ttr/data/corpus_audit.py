"""Verify a prepared corpus as one immutable training boundary."""

# pyright: reportAny=false

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import TYPE_CHECKING, cast

import numpy as np

from hri_ttr.data.same_motion_quality import TARGET_FPS
from hri_ttr.representations.g1.normalizer import G1FeatureNormalizer
from hri_ttr.representations.human.normalizer import HumanFeatureNormalizer

if TYPE_CHECKING:
    from pathlib import Path

Json = dict[str, object]


def audit_corpus(root: Path) -> Json:
    """Check hashes, schemas, finite arrays, boundaries, and split isolation."""
    _verify_checksums(root)
    schema = _json(root / "schema.json")
    if schema.get("target_fps") != TARGET_FPS:
        detail = "corpus target FPS is not 20"
        raise ValueError(detail)
    _ = HumanFeatureNormalizer.load(root / "normalizers" / "human.json")
    _ = G1FeatureNormalizer.load(root / "normalizers" / "g1.json")
    identifiers: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    sequences = 0
    frames = 0
    for split in ("train", "val", "test"):
        for shard in sorted((root / "shards" / split).glob("shard-*")):
            human = np.load(shard / "human.npy", mmap_mode="r")
            g1 = np.load(shard / "g1.npy", mmap_mode="r")
            offsets = np.load(shard / "offsets.npy", mmap_mode="r")
            rows = _jsonl(shard / "sequences.jsonl")
            if (
                human.dtype != np.float32
                or g1.dtype != np.float32
                or human.shape != (len(g1), 262)
                or g1.shape[1:] != (75,)
                or offsets.shape != (len(rows), 2)
                or not np.isfinite(human).all()
                or not np.isfinite(g1).all()
            ):
                detail = f"invalid corpus shard: {shard}"
                raise ValueError(detail)
            expected = 0
            for row, bounds in zip(rows, offsets, strict=True):
                start, stop = (int(value) for value in bounds)
                if start != expected or stop <= start or row.get("split") != split:
                    detail = f"invalid sequence boundary: {shard}"
                    raise ValueError(detail)
                expected = stop
                sample_id = str(row["sample_id"])
                if sample_id in identifiers:
                    detail = f"duplicate sample id: {sample_id}"
                    raise ValueError(detail)
                identifiers.add(sample_id)
                group_splits[str(row["source_group_id"])].add(split)
            if expected != len(human):
                detail = f"offset tail mismatch: {shard}"
                raise ValueError(detail)
            _verify_optional_space(human, shard / "human_space.npy")
            _verify_optional_space(g1, shard / "g1_space.npy")
            sequences += len(rows)
            frames += len(human)
    leaking = [group for group, splits in group_splits.items() if len(splits) > 1]
    if leaking:
        detail = f"source groups cross splits: {leaking[:5]}"
        raise ValueError(detail)
    return {
        "sequences": sequences,
        "frames": frames,
        "fps": 20,
        "duplicate_sample_ids": 0,
        "cross_split_source_groups": 0,
        "checksums": "verified",
    }


def _verify_checksums(root: Path) -> None:
    inventory = root / "checksums" / "SHA256SUMS"
    expected: dict[str, str] = {}
    with inventory.open(encoding="utf-8") as stream:
        for line in stream:
            digest, relative = line.rstrip("\n").split("  ", maxsplit=1)
            expected[relative] = digest
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path != inventory
    }
    if actual != expected.keys():
        detail = "checksum inventory file set mismatch"
        raise ValueError(detail)
    for relative, expected_digest in expected.items():
        digest = hashlib.sha256()
        with (root / relative).open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != expected_digest:
            detail = f"checksum mismatch: {relative}"
            raise ValueError(detail)


def _json(path: Path) -> Json:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        detail = f"expected JSON object: {path}"
        raise TypeError(detail)
    return cast("Json", value)


def _verify_optional_space(features: np.ndarray, path: Path) -> None:
    if not path.exists():
        return
    values = np.load(path, mmap_mode="r")
    if (
        values.dtype != np.float32
        or values.shape != (len(features), 3)
        or not np.isfinite(values).all()
    ):
        detail = f"invalid optional space array: {path}"
        raise ValueError(detail)


def _jsonl(path: Path) -> list[Json]:
    with path.open(encoding="utf-8") as stream:
        return [_mapping(json.loads(line), path) for line in stream]


def _mapping(value: object, path: Path) -> Json:
    if not isinstance(value, dict):
        detail = f"expected JSON object: {path}"
        raise TypeError(detail)
    return cast("Json", value)
