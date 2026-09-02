"""Streaming shard writer and train-only normalization statistics."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import numpy.typing as npt

from hri_ttr.data.corpus_artifacts import (
    schema,
    write_checksums,
    write_json,
    write_jsonl,
)
from hri_ttr.data.same_motion_preprocess import TARGET_FPS, PreparedPair
from hri_ttr.data.same_motion_quality import QualityError
from hri_ttr.representations.g1.normalizer import G1FeatureNormalizer
from hri_ttr.representations.g1.schema import G1_FEATURE_SLICES, G1_SCHEMA_VERSION
from hri_ttr.representations.human.normalizer import (
    SCHEMA_ID,
    HumanFeatureNormalizer,
)

if TYPE_CHECKING:
    from pathlib import Path

Json = dict[str, object]
DEFAULT_SHARD_FRAMES: Final = 250_000


@dataclass(slots=True)
class _Statistics:
    width: int
    count: int = 0
    total: npt.NDArray[np.float64] = field(init=False)
    squared: npt.NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        self.total = np.zeros(self.width, dtype=np.float64)
        self.squared = np.zeros(self.width, dtype=np.float64)

    def update(self, values: npt.NDArray[np.float32]) -> None:
        self.count += len(values)
        self.total += np.sum(values, axis=0, dtype=np.float64)
        self.squared += np.sum(values.astype(np.float64) ** 2, axis=0)

    def parameters(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if self.count == 0:
            detail = "normalizer requires accepted train frames"
            raise ValueError(detail)
        mean = self.total / self.count
        variance = np.maximum(self.squared / self.count - mean * mean, 0.0)
        return mean, np.maximum(np.sqrt(variance), 1e-4)


@dataclass(slots=True)
class _ShardBuffer:
    human: list[npt.NDArray[np.float32]] = field(default_factory=list)
    g1: list[npt.NDArray[np.float32]] = field(default_factory=list)
    metadata: list[Json] = field(default_factory=list)
    frames: int = 0
    index: int = 0


class CorpusWriter:
    """Write aligned shards, provenance, quality reports, and normalizers."""

    def __init__(
        self, root: Path, *, max_shard_frames: int = DEFAULT_SHARD_FRAMES
    ) -> None:
        """Create one new corpus root and reject accidental overwrite."""
        if root.exists() and any(root.iterdir()):
            detail = f"output directory is not empty: {root}"
            raise ValueError(detail)
        root.mkdir(parents=True, exist_ok=True)
        self.root: Path = root
        self.max_shard_frames: int = max_shard_frames
        self.buffers: dict[str, _ShardBuffer] = {
            name: _ShardBuffer() for name in ("train", "val", "test")
        }
        self.human_statistics: _Statistics = _Statistics(262)
        self.g1_statistics: _Statistics = _Statistics(75)
        self.accepted: list[Json] = []
        self.quarantined: list[Json] = []
        self.datasets: Counter[str] = Counter()
        self.splits: Counter[str] = Counter()
        self.failures: Counter[str] = Counter()
        self.source_failures: dict[str, Counter[str]] = defaultdict(Counter)
        self.quality_values: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def add(self, record: Json, pair: PreparedPair) -> None:
        """Append one aligned pair and flush only at sequence boundaries."""
        split = str(record["split"])
        buffer = self.buffers[split]
        if (
            buffer.frames
            and buffer.frames + len(pair.human_features) > self.max_shard_frames
        ):
            self._flush(split)
            buffer = self.buffers[split]
        start = buffer.frames
        stop = start + len(pair.human_features)
        metadata = {
            "sample_id": record["sample_id"],
            "source_dataset": record["source_dataset"],
            "source_sequence_id": record["source_sequence_id"],
            "source_group_id": record["source_group_id"],
            "split": split,
            "frame_start": start,
            "frame_end": stop,
            "frames": len(pair.human_features),
            "fps": TARGET_FPS,
            "anchor_origin": pair.anchor_origin.tolist(),
            "anchor_basis": pair.anchor_basis.tolist(),
            "quality": pair.quality,
            "texts": record.get("texts", []),
            "has_text": bool(record.get("has_text")),
            "human_source": record["human"],
            "g1_source": record["g1"],
        }
        buffer.human.append(pair.human_features)
        buffer.g1.append(pair.g1_features)
        buffer.metadata.append(metadata)
        buffer.frames = stop
        accepted = dict(record)
        accepted["prepared_frames"] = len(pair.human_features)
        accepted["prepared_fps"] = TARGET_FPS
        accepted["quality"] = pair.quality
        self.accepted.append(accepted)
        dataset = str(record["source_dataset"])
        self.datasets[dataset] += 1
        self.splits[split] += 1
        for name, value in pair.quality.items():
            self.quality_values[dataset][name].append(value)
        if split == "train":
            self.human_statistics.update(pair.human_features)
            self.g1_statistics.update(pair.g1_features)

    def quarantine(self, record: Json, error: Exception) -> None:
        """Record a recoverable rejection without copying or changing source data."""
        reason = (
            error.reason if isinstance(error, QualityError) else type(error).__name__
        )
        self.failures[str(reason)] += 1
        dataset = str(record.get("source_dataset", "unknown"))
        self.source_failures[dataset][str(reason)] += 1
        self.quarantined.append(
            {
                "sample_id": record.get("sample_id"),
                "source_dataset": record.get("source_dataset"),
                "human_source": record.get("human"),
                "g1_source": record.get("g1"),
                "reason": str(reason),
                "detail": str(error),
            }
        )

    def finish(self) -> Json:
        """Flush shards and write deterministic corpus-level artifacts."""
        for split in self.buffers:
            self._flush(split)
        write_jsonl(self.root / "manifest" / "accepted.jsonl", self.accepted)
        write_jsonl(self.root / "manifest" / "quarantined.jsonl", self.quarantined)
        write_jsonl(
            self.root / "manifest" / "split_assignment.jsonl",
            (
                {"sample_id": row["sample_id"], "split": row["split"]}
                for row in self.accepted
            ),
        )
        write_jsonl(
            self.root / "texts" / "annotations.jsonl",
            (
                {"sample_id": row["sample_id"], "texts": row.get("texts", [])}
                for row in self.accepted
            ),
        )
        self._write_normalizers()
        summary = self._summary()
        write_json(self.root / "quality" / "summary.json", summary)
        write_json(
            self.root / "quality" / "failures_by_reason.json", dict(self.failures)
        )
        write_json(
            self.root / "quality" / "source_statistics.json", self._quality_summary()
        )
        write_json(self.root / "schema.json", schema())
        write_checksums(self.root)
        return summary

    def _flush(self, split: str) -> None:
        buffer = self.buffers[split]
        if not buffer.metadata:
            return
        destination = self.root / "shards" / split / f"shard-{buffer.index:05d}"
        destination.mkdir(parents=True)
        human = np.concatenate(buffer.human, axis=0).astype(np.float32, copy=False)
        g1 = np.concatenate(buffer.g1, axis=0).astype(np.float32, copy=False)
        offsets = np.asarray(
            [[row["frame_start"], row["frame_end"]] for row in buffer.metadata],
            dtype=np.int64,
        )
        np.save(destination / "human.npy", human)
        np.save(destination / "g1.npy", g1)
        np.save(destination / "offsets.npy", offsets)
        write_jsonl(destination / "sequences.jsonl", buffer.metadata)
        self.buffers[split] = _ShardBuffer(index=buffer.index + 1)

    def _write_normalizers(self) -> None:
        human_mean, human_std = self.human_statistics.parameters()
        human = HumanFeatureNormalizer.create(
            human_mean.astype(np.float32),
            human_std.astype(np.float32),
            source_dataset="hri_ttr_same_motion_20hz_v1:train",
            fps=TARGET_FPS,
        )
        human.save(self.root / "normalizers" / "human.json")
        g1_mean, g1_std = self.g1_statistics.parameters()
        contact = G1_FEATURE_SLICES["foot_contact_lr"]
        g1_mean[contact] = 0.0
        g1_std[contact] = 1.0
        G1FeatureNormalizer(
            g1_mean, g1_std, "hri_ttr_same_motion_20hz_v1:train", TARGET_FPS
        ).save(self.root / "normalizers" / "g1.json")

    def _summary(self) -> Json:
        captioned = sum(bool(row.get("has_text")) for row in self.accepted)
        return {
            "accepted_sequences": len(self.accepted),
            "quarantined_sequences": len(self.quarantined),
            "accepted_duration_hours": sum(
                cast("int", row["prepared_frames"]) for row in self.accepted
            )
            / TARGET_FPS
            / 3600,
            "by_dataset": dict(sorted(self.datasets.items())),
            "by_split": dict(sorted(self.splits.items())),
            "captioned_sequences": captioned,
            "text_coverage": captioned / len(self.accepted) if self.accepted else 0.0,
            "target_fps": TARGET_FPS,
            "human_schema": SCHEMA_ID,
            "g1_schema": G1_SCHEMA_VERSION,
        }

    def _quality_summary(self) -> Json:
        result: Json = {}
        for dataset, metrics in sorted(self.quality_values.items()):
            summaries: Json = {}
            for name, values in metrics.items():
                array = np.asarray(values, dtype=np.float64)
                summaries[name] = {
                    "min": float(np.min(array)),
                    "median": float(np.median(array)),
                    "p95": float(np.percentile(array, 95)),
                    "max": float(np.max(array)),
                }
            result[dataset] = {
                "accepted": self.datasets[dataset],
                "accepted_duration_hours": sum(
                    cast("int", row["prepared_frames"])
                    for row in self.accepted
                    if row["source_dataset"] == dataset
                )
                / TARGET_FPS
                / 3600,
                "quarantined": sum(self.source_failures[dataset].values()),
                "failures_by_reason": dict(
                    sorted(self.source_failures[dataset].items())
                ),
                "quality_metrics": summaries,
            }
        return result
