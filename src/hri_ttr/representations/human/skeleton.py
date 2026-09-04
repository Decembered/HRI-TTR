"""Minimal HumanML3D inverse kinematics used by TTR's 262D converter."""

from __future__ import annotations

from itertools import pairwise
from math import exp
from typing import Final

import numpy as np
from numpy.typing import NDArray

from hri_ttr.representations.human.quaternion_wxyz import (
    between_wxyz,
    inverse_wxyz,
    multiply_wxyz,
)

Float32Array = NDArray[np.float32]
RAW_OFFSETS: Final = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
    ],
    dtype=np.float32,
)
KINEMATIC_CHAINS: Final = (
    (0, 2, 5, 8, 11),
    (0, 1, 4, 7, 10),
    (0, 3, 6, 9, 12, 15),
    (9, 14, 17, 19, 21),
    (9, 13, 16, 18, 20),
)
FACE_JOINTS: Final = (2, 1, 17, 16)


def _gaussian_filter_nearest(values: Float32Array, sigma: float) -> Float32Array:
    radius = int(4.0 * sigma + 0.5)
    unnormalized = [
        exp(-0.5 * ((offset / sigma) ** 2)) for offset in range(-radius, radius + 1)
    ]
    weight_sum = sum(unnormalized)
    weights = tuple(value / weight_sum for value in unnormalized)
    weight_array: Float32Array = np.asarray(weights, dtype=np.float32)
    padded: Float32Array = np.empty((len(values) + 2 * radius, 3), dtype=np.float32)
    padded[:radius] = values[0]
    padded[radius : radius + len(values)] = values
    padded[radius + len(values) :] = values[-1]
    result: Float32Array = np.empty_like(values)
    for column in range(3):
        result[:, column] = np.convolve(
            padded[:, column], weight_array, mode="valid"
        ).astype(np.float32)
    return result


def inverse_kinematics_wxyz(joints: Float32Array) -> Float32Array:
    """Port TTR's pinned HumanML3D IK while preserving its historical behavior."""
    # Ported from HumanML3D via TTR commit
    # 9b7e395f740a68cbd30c027b4952dedb0ebf8b6d.
    positions = np.asarray(joints, dtype=np.float32)
    right_hip, left_hip, right_shoulder, left_shoulder = FACE_JOINTS
    across = (
        positions[:, left_hip]
        - positions[:, right_hip]
        + positions[:, right_shoulder]
        - positions[:, left_shoulder]
    )
    across_norm: Float32Array = np.sqrt((across * across).sum(axis=-1, keepdims=True))
    across /= across_norm
    forward: Float32Array = np.cross(
        np.array([[0.0, 1.0, 0.0]], dtype=np.float32), across
    )
    forward = _gaussian_filter_nearest(forward, 20.0)
    forward_norm: Float32Array = np.sqrt(
        (forward * forward).sum(axis=-1, keepdims=True)
    )
    forward /= forward_norm
    target: Float32Array = np.empty((len(forward), 3), dtype=np.float32)
    target[:] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    root = between_wxyz(forward, target)
    root[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    result = np.zeros((*positions.shape[:-1], 4), dtype=np.float32)
    result[:, 0] = root
    for chain in KINEMATIC_CHAINS:
        rotation = root
        for parent, child in pairwise(chain):
            source: Float32Array = np.empty((len(positions), 3), dtype=np.float32)
            source[:] = RAW_OFFSETS[child]
            destination = positions[:, child] - positions[:, parent]
            destination_norm: Float32Array = np.sqrt(
                (destination * destination).sum(axis=-1, keepdims=True)
            )
            destination /= destination_norm
            global_rotation = between_wxyz(source, destination)
            local_rotation = multiply_wxyz(inverse_wxyz(rotation), global_rotation)
            result[:, child] = local_rotation
            rotation = multiply_wxyz(rotation, local_rotation)
    return result
