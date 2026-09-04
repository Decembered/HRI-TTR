"""Quaternion operations whose public convention is always scalar-last ``xyzw``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

FloatArray = NDArray[np.float64]
SLERP_EPSILON: Final = 1e-8


@dataclass(frozen=True, slots=True)
class QuaternionError(ValueError):
    """Reports a malformed quaternion array at a public boundary."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


def normalize_xyzw(quaternions: FloatArray) -> FloatArray:
    """Return unit ``xyzw`` quaternions and reject zero or non-finite inputs."""
    values = np.asarray(quaternions, dtype=np.float64)
    if values.shape[-1:] != (4,) or not np.isfinite(values).all():
        detail = "quaternions must be finite with final dimension four"
        raise QuaternionError(detail)
    norms: FloatArray = np.sqrt((values * values).sum(axis=-1, keepdims=True))
    if any(float(value) <= np.finfo(np.float64).eps for value in norms.flat):
        detail = "quaternion norm must be positive"
        raise QuaternionError(detail)
    return values / norms


def xyzw_to_matrix(quaternions: FloatArray) -> FloatArray:
    """Convert unit-normalized ``xyzw`` quaternions to rotation matrices."""
    q = normalize_xyzw(quaternions)
    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]
    w = q[..., 3]
    matrix: FloatArray = np.empty((*q.shape[:-1], 3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - z * w)
    matrix[..., 0, 2] = 2 * (x * z + y * w)
    matrix[..., 1, 0] = 2 * (x * y + z * w)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - x * w)
    matrix[..., 2, 0] = 2 * (x * z - y * w)
    matrix[..., 2, 1] = 2 * (y * z + x * w)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def matrix_to_xyzw(matrices: FloatArray) -> FloatArray:
    """Convert proper rotation matrices to deterministically signed ``xyzw``."""
    values = np.asarray(matrices, dtype=np.float64)
    if values.shape[-2:] != (3, 3) or not np.isfinite(values).all():
        detail = "rotation matrices must be finite [...,3,3] arrays"
        raise QuaternionError(detail)
    flat = values.reshape(-1, 3, 3)
    output = np.empty((flat.shape[0], 4), dtype=np.float64)
    for index, matrix in enumerate(flat):
        eigenvalues, eigenvectors = np.linalg.eigh(
            np.array(
                [
                    [
                        matrix[0, 0] - matrix[1, 1] - matrix[2, 2],
                        matrix[1, 0] + matrix[0, 1],
                        matrix[2, 0] + matrix[0, 2],
                        matrix[1, 2] - matrix[2, 1],
                    ],
                    [
                        matrix[1, 0] + matrix[0, 1],
                        matrix[1, 1] - matrix[0, 0] - matrix[2, 2],
                        matrix[2, 1] + matrix[1, 2],
                        matrix[2, 0] - matrix[0, 2],
                    ],
                    [
                        matrix[2, 0] + matrix[0, 2],
                        matrix[2, 1] + matrix[1, 2],
                        matrix[2, 2] - matrix[0, 0] - matrix[1, 1],
                        matrix[0, 1] - matrix[1, 0],
                    ],
                    [
                        matrix[1, 2] - matrix[2, 1],
                        matrix[2, 0] - matrix[0, 2],
                        matrix[0, 1] - matrix[1, 0],
                        matrix.trace(),
                    ],
                ],
                dtype=np.float64,
            )
            / 3.0,
        )
        largest = int(eigenvalues.argmax())
        quaternion = eigenvectors[:, largest]
        quaternion[:3] *= -1.0
        output[index] = quaternion if quaternion[3] >= 0 else -quaternion
    return normalize_xyzw(output.reshape((*values.shape[:-2], 4)))


def slerp_xyzw(q0: FloatArray, q1: FloatArray, fraction: FloatArray) -> FloatArray:
    """Interpolate matching ``xyzw`` arrays along their shortest arcs."""
    start = normalize_xyzw(q0)
    end = normalize_xyzw(q1)
    dots: FloatArray = (start * end).sum(axis=-1, keepdims=True)
    flip = dots[..., 0] < 0.0
    end[flip] *= -1.0
    dots = np.abs(dots)
    dots[dots > 1.0] = 1.0
    angle = np.arccos(dots)
    sine = np.sin(angle)
    weight = np.asarray(fraction, dtype=np.float64)[..., None]
    near = sine < SLERP_EPSILON
    safe_sine = sine.copy()
    safe_sine[near] = 1.0
    left = np.sin((1.0 - weight) * angle) / safe_sine
    right = np.sin(weight * angle) / safe_sine
    left[near] = (1.0 - weight)[near]
    right[near] = weight[near]
    return normalize_xyzw(left * start + right * end)
