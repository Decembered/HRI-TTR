"""Tail padding for the four-frame motion-token cadence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
import numpy.typing as npt
from typing_extensions import override

FRAME_MATRIX_NDIM: Final = 2


class FramePaddingReason(StrEnum):
    """Stable reasons for rejecting a padding request."""

    SHAPE = "frames must be a non-empty [T,F] float32 array"
    DTYPE = "frames must use float32"
    CADENCE = "frames_per_token must be positive"


@dataclass(frozen=True, slots=True)
class PaddedFrames:
    """Frames padded by repetition plus the real-frame mask."""

    features: npt.NDArray[np.float32]
    valid_mask: npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class FramePaddingError(ValueError):
    """Reports malformed input to the tail-padding boundary."""

    reason: FramePaddingReason

    @override
    def __str__(self) -> str:
        """Render the stable boundary failure."""
        return self.reason


def pad_frames_to_token_multiple(
    frames: npt.NDArray[np.float32],
    *,
    frames_per_token: int,
) -> PaddedFrames:
    """Repeat the last real frame until the token block is complete."""
    if frames.ndim != FRAME_MATRIX_NDIM or len(frames) == 0:
        raise FramePaddingError(FramePaddingReason.SHAPE)
    if frames.dtype != np.dtype(np.float32):
        raise FramePaddingError(FramePaddingReason.DTYPE)
    if frames_per_token <= 0:
        raise FramePaddingError(FramePaddingReason.CADENCE)
    missing = (-len(frames)) % frames_per_token
    valid_mask = np.ones(len(frames) + missing, dtype=np.bool_)
    valid_mask[len(frames) :] = False
    if missing == 0:
        return PaddedFrames(frames.copy(), valid_mask)
    padded = np.empty((len(frames) + missing, frames.shape[1]), dtype=np.float32)
    padded[: len(frames)] = frames
    padded[len(frames) :] = frames[-1]
    return PaddedFrames(padded, valid_mask)
