"""Derive binary G1 foot contacts from the verified 20-joint layout."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from enum import StrEnum
from math import floor, isfinite
from typing import Final

import numpy as np
import numpy.typing as npt
from typing_extensions import override

G1_FOOT_INDICES: Final = (4, 8)
FLOOR_PERCENTILE: Final = 2.0
JOINT_MATRIX_NDIM: Final = 3


class G1ContactReason(StrEnum):
    """Stable reasons for rejecting contact derivation input."""

    SHAPE = "joints20_interaction must be a non-empty [T,20,3] array"
    FINITE = "joints20_interaction values must be finite"
    FPS = "fps must be finite and positive"
    THRESHOLD = "contact thresholds must be finite and positive"


@dataclass(frozen=True, slots=True)
class G1ContactError(ValueError):
    """Reports an invalid contact derivation request."""

    reason: G1ContactReason

    @override
    def __str__(self) -> str:
        """Render the stable boundary failure."""
        return self.reason


@dataclass(frozen=True, slots=True)
class G1FootContactThresholds:
    """Physical thresholds for floor proximity and foot speed."""

    height_m: float = 0.06
    speed_m_s: float = 0.25

    def __post_init__(self) -> None:
        """Reject thresholds that cannot describe a physical contact boundary."""
        if (
            not isfinite(self.height_m)
            or not isfinite(self.speed_m_s)
            or self.height_m <= 0.0
            or self.speed_m_s <= 0.0
        ):
            raise G1ContactError(G1ContactReason.THRESHOLD)


DEFAULT_G1_FOOT_CONTACT_THRESHOLDS: Final = G1FootContactThresholds()


def compute_g1_foot_contacts(
    joints20_interaction: npt.NDArray[np.float64],
    *,
    fps: float,
    thresholds: G1FootContactThresholds = DEFAULT_G1_FOOT_CONTACT_THRESHOLDS,
) -> npt.NDArray[np.float64]:
    """Return left/right contacts using floor height and causal backward speed."""
    if (
        joints20_interaction.ndim != JOINT_MATRIX_NDIM
        or joints20_interaction.shape[1:] != (20, 3)
        or len(joints20_interaction) == 0
    ):
        raise G1ContactError(G1ContactReason.SHAPE)
    if not np.isfinite(joints20_interaction).all():
        raise G1ContactError(G1ContactReason.FINITE)
    if not isfinite(fps) or fps <= 0.0:
        raise G1ContactError(G1ContactReason.FPS)
    feet = joints20_interaction[:, G1_FOOT_INDICES]
    floor_height = _linear_percentile(feet[:, :, 1], FLOOR_PERCENTILE)
    velocity = np.zeros_like(feet)
    velocity[1:] = (feet[1:] - feet[:-1]) * fps
    speed = np.sqrt(
        velocity[:, :, 0] ** 2 + velocity[:, :, 1] ** 2 + velocity[:, :, 2] ** 2
    )
    height_contact = feet[:, :, 1] - floor_height <= thresholds.height_m
    speed_contact = speed <= thresholds.speed_m_s
    return np.asarray(height_contact & speed_contact, dtype=np.float64)


def _linear_percentile(values: npt.NDArray[np.float64], percentile: float) -> float:
    flattened: array[float] = array("d")
    flattened.frombytes(np.ascontiguousarray(values).tobytes())
    ordered = sorted(flattened)
    index = percentile / 100.0 * (len(ordered) - 1)
    lower = floor(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
