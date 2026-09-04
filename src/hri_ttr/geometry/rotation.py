"""Continuous 6D, matrix, and rotation-vector conversions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from hri_ttr.geometry.quaternion import matrix_to_xyzw

FloatArray = NDArray[np.float64]
ROTATION_EPSILON: Final = 1e-12


@dataclass(frozen=True, slots=True)
class RotationError(ValueError):
    """Reports malformed or degenerate rotation representations."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


def matrix_to_rotation_6d(matrices: FloatArray) -> FloatArray:
    """Flatten the first two rotation-matrix columns in TTR order."""
    values = np.asarray(matrices, dtype=np.float64)
    if values.shape[-2:] != (3, 3) or not np.isfinite(values).all():
        detail = "rotation matrices must be finite [...,3,3] arrays"
        raise RotationError(detail)
    result: FloatArray = np.empty((*values.shape[:-2], 6), dtype=np.float64)
    result[..., :3] = values[..., 0]
    result[..., 3:] = values[..., 1]
    return result


def rotation_6d_to_matrix(rotations: FloatArray) -> FloatArray:
    """Orthonormalize two 3D axes into proper rotation matrices."""
    values = np.asarray(rotations, dtype=np.float64)
    if values.shape[-1:] != (6,) or not np.isfinite(values).all():
        detail = "6D rotations must be finite with final dimension six"
        raise RotationError(detail)
    first = values[..., :3]
    second = values[..., 3:]
    first_norm: FloatArray = np.sqrt((first * first).sum(axis=-1, keepdims=True))
    if any(float(value) <= np.finfo(np.float64).eps for value in first_norm.flat):
        detail = "first 6D rotation axis must be non-zero"
        raise RotationError(detail)
    x_axis = first / first_norm
    z_axis = np.cross(x_axis, second, axis=-1)
    z_norm: FloatArray = np.sqrt((z_axis * z_axis).sum(axis=-1, keepdims=True))
    if any(float(value) <= np.finfo(np.float64).eps for value in z_norm.flat):
        detail = "6D rotation axes must not be parallel"
        raise RotationError(detail)
    z_axis /= z_norm
    y_axis = np.cross(z_axis, x_axis, axis=-1)
    result: FloatArray = np.empty((*values.shape[:-1], 3, 3), dtype=np.float64)
    result[..., 0] = x_axis
    result[..., 1] = y_axis
    result[..., 2] = z_axis
    return result


def matrix_to_rotvec(matrices: FloatArray) -> FloatArray:
    """Convert proper matrices to axis-angle rotation vectors."""
    quaternions = matrix_to_xyzw(np.asarray(matrices, dtype=np.float64))
    vector = quaternions[..., :3]
    vector_norm: FloatArray = np.sqrt((vector * vector).sum(axis=-1, keepdims=True))
    angle = 2.0 * np.arctan2(vector_norm, quaternions[..., 3:])
    scale = np.divide(
        angle,
        vector_norm,
        out=np.full_like(angle, 2.0),
        where=vector_norm > ROTATION_EPSILON,
    )
    return vector * scale


def rotvec_to_matrix(rotations: FloatArray) -> FloatArray:
    """Convert axis-angle rotation vectors to proper matrices."""
    values = np.asarray(rotations, dtype=np.float64)
    if values.shape[-1:] != (3,) or not np.isfinite(values).all():
        detail = "rotation vectors must be finite with final dimension three"
        raise RotationError(detail)
    angle: FloatArray = np.sqrt((values * values).sum(axis=-1, keepdims=True))
    axis = np.divide(
        values,
        angle,
        out=np.zeros_like(values),
        where=angle > ROTATION_EPSILON,
    )
    x = axis[..., 0]
    y = axis[..., 1]
    z = axis[..., 2]
    skew: FloatArray = np.zeros((*values.shape[:-1], 3, 3), dtype=np.float64)
    skew[..., 0, 1] = -z
    skew[..., 0, 2] = y
    skew[..., 1, 0] = z
    skew[..., 1, 2] = -x
    skew[..., 2, 0] = -y
    skew[..., 2, 1] = x
    identity: FloatArray = np.zeros_like(skew)
    identity[..., 0, 0] = 1.0
    identity[..., 1, 1] = 1.0
    identity[..., 2, 2] = 1.0
    sine = np.sin(angle)[..., None]
    cosine = np.cos(angle)[..., None]
    squared: FloatArray = skew @ skew
    return identity + sine * skew + (1.0 - cosine) * squared
