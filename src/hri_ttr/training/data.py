"""Aligned fixed-window training data with explicit validity masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from hri_ttr.training.errors import TrainingError, TrainingReason

if TYPE_CHECKING:
    from numpy.typing import NDArray

FRAMES_PER_TOKEN: Final = 4
FEATURE_NDIM: Final = 2


@dataclass(frozen=True, slots=True)
class FeatureSequence:
    """One canonical feature sequence before training windows are formed."""

    sequence_id: str
    features: NDArray[np.float32]
    frame_mask: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        """Validate the untrusted feature-array boundary."""
        if (
            not self.sequence_id
            or self.features.ndim != FEATURE_NDIM
            or self.features.shape[0] == 0
        ):
            raise TrainingError(TrainingReason.FEATURE_SHAPE)
        if self.features.dtype != np.dtype(np.float32):
            raise TrainingError(TrainingReason.FEATURE_DTYPE)
        if not bool(np.isfinite(self.features).all()):
            raise TrainingError(TrainingReason.FEATURE_FINITE)
        if self.frame_mask is not None and (
            self.frame_mask.dtype != np.dtype(np.bool_)
            or self.frame_mask.shape != (len(self.features),)
            or not bool(self.frame_mask.any())
        ):
            raise TrainingError(TrainingReason.FEATURE_MASK)


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """Aligned fixed window dimensions."""

    frames: int
    stride: int

    def __post_init__(self) -> None:
        """Reject dimensions that break token alignment."""
        if min(self.frames, self.stride) <= 0:
            raise TrainingError(TrainingReason.WINDOW_DIMENSION)
        if self.frames % FRAMES_PER_TOKEN or self.stride % FRAMES_PER_TOKEN:
            raise TrainingError(TrainingReason.WINDOW_ALIGNMENT)


@dataclass(frozen=True, slots=True)
class TrainingWindow:
    """One repeat-padded window and both validity masks."""

    sequence_id: str
    start_frame: int
    features: NDArray[np.float32]
    frame_mask: NDArray[np.bool_]
    token_mask: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """Batch-major tensors accepted by the causal tokenizer."""

    features: torch.Tensor
    frame_mask: torch.Tensor
    token_mask: torch.Tensor


def build_windows(
    sequences: tuple[FeatureSequence, ...], config: WindowConfig
) -> tuple[TrainingWindow, ...]:
    """Build deterministic aligned windows and repeat-pad only the final tail."""
    windows: list[TrainingWindow] = []
    for sequence in sequences:
        frame_count = len(sequence.features)
        authoritative_mask = (
            np.ones(frame_count, dtype=np.bool_)
            if sequence.frame_mask is None
            else sequence.frame_mask
        )
        feature_dim = sequence.features.size // frame_count
        for start in range(0, frame_count, config.stride):
            source_count = min(config.frames, frame_count - start)
            source = sequence.features[start : start + source_count]
            values = np.empty((config.frames, feature_dim), dtype=np.float32)
            values[:source_count] = source
            frame_mask = np.zeros(config.frames, dtype=np.bool_)
            frame_mask[:source_count] = authoritative_mask[start : start + source_count]
            if source_count < config.frames:
                values[source_count:] = source[-1]
            grouped = frame_mask.reshape(-1, FRAMES_PER_TOKEN)
            windows.append(
                TrainingWindow(
                    sequence.sequence_id,
                    start,
                    values.astype(np.float32, copy=False),
                    frame_mask.astype(np.bool_),
                    grouped.all(axis=1),
                )
            )
            if start + config.frames >= frame_count:
                break
    return tuple(windows)


class AlignedWindowDataset(Dataset[TrainingWindow]):
    """Torch dataset over a frozen window collection."""

    _windows: tuple[TrainingWindow, ...]

    def __init__(self, windows: tuple[TrainingWindow, ...]) -> None:
        """Store one non-empty immutable window sequence."""
        if not windows:
            raise TrainingError(TrainingReason.DATASET_EMPTY)
        self._windows = windows

    def __len__(self) -> int:
        """Return the fixed window count."""
        return len(self._windows)

    @override
    def __getitem__(self, index: int) -> TrainingWindow:
        """Return one frozen window by index."""
        return self._windows[index]


class WindowDataset(Protocol):
    """Sequence-bounded window access shared by memory and mmap datasets."""

    def __len__(self) -> int:
        """Return the available window count."""
        ...

    def __getitem__(self, index: int) -> TrainingWindow:
        """Return one deterministic training window."""
        ...


def collate_windows(windows: list[TrainingWindow]) -> TrainingBatch:
    """Stack windows without inferring validity from padded values."""
    return TrainingBatch(
        features=torch.stack(
            [torch.as_tensor(item.features, dtype=torch.float32) for item in windows]
        ),
        frame_mask=torch.stack(
            [torch.as_tensor(item.frame_mask, dtype=torch.bool) for item in windows]
        ),
        token_mask=torch.stack(
            [torch.as_tensor(item.token_mask, dtype=torch.bool) for item in windows]
        ),
    )
