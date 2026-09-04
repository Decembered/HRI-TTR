"""Isolated HumanML3D quaternion math using its internal ``wxyz`` order."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]


def normalize_wxyz(quaternions: Float32Array) -> Float32Array:
    """Normalize internal scalar-first quaternions with float32 arithmetic."""
    values = np.asarray(quaternions, dtype=np.float32)
    norms: Float32Array = np.sqrt((values * values).sum(axis=-1, keepdims=True))
    return values / norms


def inverse_wxyz(quaternions: Float32Array) -> Float32Array:
    """Invert unit scalar-first quaternions."""
    result = np.asarray(quaternions, dtype=np.float32).copy()
    result[..., 1:] *= -1.0
    return result


def multiply_wxyz(left: Float32Array, right: Float32Array) -> Float32Array:
    """Multiply equally-shaped scalar-first quaternion arrays."""
    q = np.asarray(left, dtype=np.float32)
    r = np.asarray(right, dtype=np.float32)
    result: Float32Array = np.empty_like(q)
    result[..., 0] = (
        r[..., 0] * q[..., 0]
        - r[..., 1] * q[..., 1]
        - r[..., 2] * q[..., 2]
        - r[..., 3] * q[..., 3]
    )
    result[..., 1] = (
        r[..., 0] * q[..., 1]
        + r[..., 1] * q[..., 0]
        - r[..., 2] * q[..., 3]
        + r[..., 3] * q[..., 2]
    )
    result[..., 2] = (
        r[..., 0] * q[..., 2]
        + r[..., 1] * q[..., 3]
        + r[..., 2] * q[..., 0]
        - r[..., 3] * q[..., 1]
    )
    result[..., 3] = (
        r[..., 0] * q[..., 3]
        - r[..., 1] * q[..., 2]
        + r[..., 2] * q[..., 1]
        + r[..., 3] * q[..., 0]
    )
    return result


def rotate_wxyz(quaternions: Float32Array, vectors: Float32Array) -> Float32Array:
    """Rotate vectors with matching scalar-first quaternion arrays."""
    q = np.asarray(quaternions, dtype=np.float32)
    v = np.asarray(vectors, dtype=np.float32)
    qvec = q[..., 1:]
    uv = np.cross(qvec, v, axis=-1)
    uuv = np.cross(qvec, uv, axis=-1)
    return (v + np.float32(2.0) * (q[..., :1] * uv + uuv)).astype(np.float32)


def between_wxyz(v0: Float32Array, v1: Float32Array) -> Float32Array:
    """Return scalar-first quaternions rotating normalized ``v0`` onto ``v1``."""
    start = np.asarray(v0, dtype=np.float32)
    end = np.asarray(v1, dtype=np.float32)
    vector = np.cross(start, end, axis=-1)
    scalar = np.sqrt(
        np.sum(start * start, axis=-1, keepdims=True)
        * np.sum(end * end, axis=-1, keepdims=True),
    ) + np.sum(start * end, axis=-1, keepdims=True)
    result: Float32Array = np.empty((*start.shape[:-1], 4), dtype=np.float32)
    result[..., :1] = scalar
    result[..., 1:] = vector
    return normalize_wxyz(result)


def wxyz_to_cont6d(quaternions: Float32Array) -> Float32Array:
    """Return the first two matrix columns in HumanML3D's 6D layout."""
    q = np.asarray(quaternions, dtype=np.float32)
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]
    two_s = np.float32(2.0) / np.sum(q * q, axis=-1)
    matrix: Float32Array = np.empty((*q.shape[:-1], 3, 3), dtype=np.float32)
    matrix[..., 0, 0] = 1 - two_s * (y * y + z * z)
    matrix[..., 0, 1] = two_s * (x * y - z * w)
    matrix[..., 0, 2] = two_s * (x * z + y * w)
    matrix[..., 1, 0] = two_s * (x * y + z * w)
    matrix[..., 1, 1] = 1 - two_s * (x * x + z * z)
    matrix[..., 1, 2] = two_s * (y * z - x * w)
    matrix[..., 2, 0] = two_s * (x * z - y * w)
    matrix[..., 2, 1] = two_s * (y * z + x * w)
    matrix[..., 2, 2] = 1 - two_s * (x * x + y * y)
    result: Float32Array = np.empty((*q.shape[:-1], 6), dtype=np.float32)
    result[..., :3] = matrix[..., 0]
    result[..., 3:] = matrix[..., 1]
    return result
