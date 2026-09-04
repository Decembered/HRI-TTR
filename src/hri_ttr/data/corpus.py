"""Memory-mapped aligned corpus access for Human and G1 training."""

# pyright: reportAny=false

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset
from typing_extensions import override

from hri_ttr.representations.g1.normalizer import G1FeatureNormalizer  # noqa: TC001
from hri_ttr.representations.human.normalizer import HumanFeatureNormalizer
from hri_ttr.training.data import FRAMES_PER_TOKEN, TrainingWindow, WindowConfig

CorpusSplit = Literal["train", "val", "test"]
CorpusDomain = Literal["human", "g1"]
FEATURE_NDIM = 2

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class _Sequence:
    sample_id: str
    shard: int
    start: int
    stop: int
    windows: int


class CorpusWindowDataset(Dataset[TrainingWindow]):
    """Lazy fixed-window dataset backed by read-only NumPy mmap shards."""

    _arrays: tuple[npt.NDArray[np.float32], ...]
    _sequences: tuple[_Sequence, ...]
    _cumulative: tuple[int, ...]
    window: WindowConfig
    normalizer: HumanFeatureNormalizer | G1FeatureNormalizer | None

    def __init__(
        self,
        root: Path,
        split: CorpusSplit,
        domain: CorpusDomain,
        window: WindowConfig,
        normalizer: HumanFeatureNormalizer | G1FeatureNormalizer | None = None,
    ) -> None:
        """Index sequence boundaries without loading frame arrays into memory."""
        shard_roots = sorted((root / "shards" / split).glob("shard-*"))
        arrays: list[npt.NDArray[np.float32]] = []
        sequences: list[_Sequence] = []
        cumulative: list[int] = []
        total = 0
        for shard_index, shard in enumerate(shard_roots):
            loaded_array: object = np.load(shard / f"{domain}.npy", mmap_mode="r")
            array = np.asarray(loaded_array)
            if array.ndim != FEATURE_NDIM or array.dtype != np.dtype(np.float32):
                detail = f"invalid {domain} shard: {shard}"
                raise ValueError(detail)
            arrays.append(cast("npt.NDArray[np.float32]", array))
            loaded_offsets: object = np.load(shard / "offsets.npy", mmap_mode="r")
            offsets = np.asarray(loaded_offsets, dtype=np.int64)
            metadata = _jsonl(shard / "sequences.jsonl")
            if offsets.shape != (len(metadata), 2):
                detail = f"invalid offsets: {shard}"
                raise ValueError(detail)
            for row, bounds in zip(metadata, offsets, strict=True):
                start, stop = (int(value) for value in bounds)
                length = stop - start
                count = max(
                    1,
                    (max(0, length - window.frames) + window.stride - 1)
                    // window.stride
                    + 1,
                )
                total += count
                sequences.append(
                    _Sequence(str(row["sample_id"]), shard_index, start, stop, count)
                )
                cumulative.append(total)
        if not sequences:
            detail = f"empty corpus split: {split}"
            raise ValueError(detail)
        self._arrays = tuple(arrays)
        self._sequences = tuple(sequences)
        self._cumulative = tuple(cumulative)
        self.window = window
        self.normalizer = normalizer

    def __len__(self) -> int:
        """Return the number of sequence-bounded windows."""
        return self._cumulative[-1]

    @override
    def __getitem__(self, index: int) -> TrainingWindow:
        """Read and repeat-pad one window without crossing its sequence boundary."""
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        sequence_index = bisect_right(self._cumulative, index)
        previous = 0 if sequence_index == 0 else self._cumulative[sequence_index - 1]
        sequence = self._sequences[sequence_index]
        start = sequence.start + (index - previous) * self.window.stride
        source_stop = min(start + self.window.frames, sequence.stop)
        source = np.asarray(
            self._arrays[sequence.shard][start:source_stop], dtype=np.float32
        )
        if self.normalizer is not None:
            if isinstance(self.normalizer, HumanFeatureNormalizer):
                normalized = self.normalizer.normalize(source)
            else:
                normalized = self.normalizer.normalize(source.astype(np.float64))
            source = np.asarray(normalized, dtype=np.float32)
        values = np.empty((self.window.frames, source.shape[1]), dtype=np.float32)
        values[: len(source)] = source
        values[len(source) :] = source[-1]
        frame_mask = np.zeros(self.window.frames, dtype=np.bool_)
        frame_mask[: len(source)] = True
        token_mask = frame_mask.reshape(-1, FRAMES_PER_TOKEN).all(axis=1)
        return TrainingWindow(
            sequence.sample_id,
            start - sequence.start,
            values,
            frame_mask,
            token_mask,
        )


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value: object = json.loads(line)
            if not isinstance(value, dict):
                detail = f"invalid JSONL object: {path}"
                raise TypeError(detail)
            rows.append(cast("dict[str, object]", value))
    return rows
