"""Tail padding that preserves every real motion frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from typing_extensions import override

if TYPE_CHECKING:
    from numpy.typing import NDArray

    Float32Array: TypeAlias = NDArray[np.float32]
    BoolArray: TypeAlias = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class PaddingError(ValueError):
    """Reports invalid frame arrays or padding multiples."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class PaddedFrames:
    """Contains tail-padded frames and a mask marking original samples."""

    frames: Float32Array
    frame_mask: BoolArray
    valid_frame_count: int


def pad_frames_to_multiple(
    frames: Float32Array,
    multiple: int,
) -> PaddedFrames:
    """Repeat the final frame until the time axis is divisible by ``multiple``."""
    values = frames
    if values.ndim < 1 or values.shape[0] < 1 or multiple < 1:
        detail = "frames must be non-empty and multiple must be positive"
        raise PaddingError(detail)
    if values.dtype != np.dtype(np.float32) or not np.isfinite(values).all():
        detail = "frames must be a finite float32 array"
        raise PaddingError(detail)
    valid_count = len(values)
    padding_count = (-valid_count) % multiple
    padded: Float32Array = np.empty(
        (valid_count + padding_count, *values.shape[1:]),
        dtype=np.float32,
    )
    padded[:valid_count] = values
    padded[valid_count:] = values[-1]
    mask: BoolArray = np.zeros(len(padded), dtype=np.bool_)
    mask[:valid_count] = True
    return PaddedFrames(padded, mask, valid_count)
