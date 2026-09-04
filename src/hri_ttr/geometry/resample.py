"""Timestamp-aware interpolation for vectors and scalar-last quaternions."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from itertools import pairwise
from math import floor, isfinite

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from hri_ttr.geometry.quaternion import normalize_xyzw, slerp_xyzw

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TimelineError(ValueError):
    """Reports malformed or incompatible sample timelines."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


def _validate_timeline(timestamps: FloatArray, *, at_least_two: bool = False) -> None:
    required = 2 if at_least_two else 1
    values = tuple(float(value) for value in timestamps.flat)
    if (
        timestamps.ndim != 1
        or len(values) < required
        or any(not isfinite(value) for value in values)
        or any(current <= previous for previous, current in pairwise(values))
    ):
        detail = "timestamps must be finite, one-dimensional, and strictly increasing"
        raise TimelineError(detail)


def target_timestamps(source_timestamps: FloatArray, target_fps: float) -> FloatArray:
    """Build an inclusive regular timeline inside the source time extent."""
    source = np.asarray(source_timestamps, dtype=np.float64)
    _validate_timeline(source, at_least_two=True)
    if not isfinite(target_fps) or target_fps <= 0.0:
        detail = "target FPS must be finite and positive"
        raise TimelineError(detail)
    source_values = tuple(float(value) for value in source.flat)
    count = floor((source_values[-1] - source_values[0]) * target_fps + 1e-9) + 1
    return np.asarray(
        [source_values[0] + index / target_fps for index in range(count)],
        dtype=np.float64,
    )


def resample_linear(
    values: FloatArray,
    source_time: FloatArray,
    target_time: FloatArray,
) -> FloatArray:
    """Linearly resample any array whose first dimension is time."""
    source = np.asarray(source_time, dtype=np.float64)
    target = np.asarray(target_time, dtype=np.float64)
    samples = np.asarray(values, dtype=np.float64)
    _validate_timeline(source, at_least_two=True)
    _validate_timeline(target)
    if len(samples) != len(source) or not np.isfinite(samples).all():
        detail = "values must be finite and match the source timeline"
        raise TimelineError(detail)
    source_values = tuple(float(value) for value in source.flat)
    target_values = tuple(float(value) for value in target.flat)
    if target_values[0] < source_values[0] or target_values[-1] > source_values[-1]:
        detail = "target timeline must remain inside the source extent"
        raise TimelineError(detail)
    flat = samples.reshape(len(source), -1)
    column_count = flat.size // len(source)
    result: FloatArray = np.empty((len(target), column_count), dtype=np.float64)
    for column in range(column_count):
        result[:, column] = np.interp(target, source, flat[:, column])
    return result.reshape((len(target), *samples.shape[1:]))


def resample_quaternion_xyzw(
    quaternions: FloatArray,
    source_time: FloatArray,
    target_time: FloatArray,
) -> FloatArray:
    """Spherically resample scalar-last quaternions without antipodal flips."""
    source = np.asarray(source_time, dtype=np.float64)
    target = np.asarray(target_time, dtype=np.float64)
    _validate_timeline(source, at_least_two=True)
    _validate_timeline(target)
    rotations = normalize_xyzw(np.asarray(quaternions, dtype=np.float64))
    if rotations.shape != (len(source), 4):
        detail = "quaternions must have shape [len(source_time),4]"
        raise TimelineError(detail)
    source_values = tuple(float(value) for value in source.flat)
    target_values = tuple(float(value) for value in target.flat)
    if target_values[0] < source_values[0] or target_values[-1] > source_values[-1]:
        detail = "target timeline must remain inside the source extent"
        raise TimelineError(detail)
    right: NDArray[np.int64] = np.empty(len(target), dtype=np.int64)
    for index, timestamp in enumerate(target_values):
        right[index] = min(
            max(bisect_right(source_values, timestamp), 1), len(source) - 1
        )
    left = right - 1
    fraction = (target - source[left]) / (source[right] - source[left])
    return slerp_xyzw(rotations[left], rotations[right], fraction)
