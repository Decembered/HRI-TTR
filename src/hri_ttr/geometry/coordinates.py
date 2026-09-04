"""Explicit G1 Z-up and InteractionWorld Y-up basis transforms."""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from hri_ttr.geometry.quaternion import matrix_to_xyzw, xyzw_to_matrix

FloatArray = NDArray[np.float64]
G1_TO_INTERACTION_BASIS: Final = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)


def g1_z_up_to_interaction_y_up(points: FloatArray) -> FloatArray:
    """Map G1 ``(x,y,z-up)`` vectors into InteractionWorld Y-up vectors."""
    values = np.asarray(points, dtype=np.float64)
    result: FloatArray = np.empty_like(values)
    result[..., 0] = values[..., 0]
    result[..., 1] = values[..., 2]
    result[..., 2] = -values[..., 1]
    return result


def interaction_y_up_to_g1_z_up(points: FloatArray) -> FloatArray:
    """Map InteractionWorld vectors back into the G1 Z-up basis."""
    values = np.asarray(points, dtype=np.float64)
    result: FloatArray = np.empty_like(values)
    result[..., 0] = values[..., 0]
    result[..., 1] = -values[..., 2]
    result[..., 2] = values[..., 1]
    return result


def quaternion_xyzw_g1_to_interaction(quaternions: FloatArray) -> FloatArray:
    """Change the basis of scalar-last G1 root rotations."""
    rotations = xyzw_to_matrix(quaternions)
    transformed: FloatArray = (
        G1_TO_INTERACTION_BASIS @ rotations @ G1_TO_INTERACTION_BASIS.T
    )
    return matrix_to_xyzw(transformed)


def quaternion_xyzw_interaction_to_g1(quaternions: FloatArray) -> FloatArray:
    """Change the basis of scalar-last InteractionWorld root rotations."""
    rotations = xyzw_to_matrix(quaternions)
    transformed: FloatArray = (
        G1_TO_INTERACTION_BASIS.T @ rotations @ G1_TO_INTERACTION_BASIS
    )
    return matrix_to_xyzw(transformed)
