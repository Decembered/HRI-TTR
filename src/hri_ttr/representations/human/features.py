"""Official TTR-compatible normalization and 262D feature construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2
from typing import Final

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from hri_ttr.representations.human.quaternion_wxyz import (
    between_wxyz,
    rotate_wxyz,
    wxyz_to_cont6d,
)
from hri_ttr.representations.human.skeleton import inverse_kinematics_wxyz

FloatArray = NDArray[np.float32] | NDArray[np.float64]
Float32Array = NDArray[np.float32]
JOINT_ARRAY_NDIM: Final = 3
HUMAN_FEATURE_DIM: Final = 262
MINIMUM_FEATURE_NDIM: Final = 2
FACE_JOINTS: Final = (2, 1, 17, 16)
LEFT_FEET: Final = (7, 10)
RIGHT_FEET: Final = (8, 11)


@dataclass(frozen=True, slots=True)
class HumanRepresentationError(ValueError):
    """Reports malformed 22-joint motion at the representation boundary."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class Human262Layout:
    """Named slices for TTR's historical 262D InterGen representation."""

    position: slice = field(default_factory=lambda: slice(0, 66))
    velocity: slice = field(default_factory=lambda: slice(66, 132))
    rotation: slice = field(default_factory=lambda: slice(132, 258))
    contact: slice = field(default_factory=lambda: slice(258, 262))


HUMAN_262_LAYOUT: Final = Human262Layout()


@dataclass(frozen=True, slots=True)
class HumanSpace:
    """Initial planar pose needed to reverse official single-person normalization."""

    x_m: float
    z_m: float
    yaw_rad: float


@dataclass(frozen=True, slots=True)
class NormalizedHuman:
    """Floor-normalized joints plus their initial planar pose."""

    joints: Float32Array
    space: HumanSpace


def _require_joints(joints: FloatArray, *, minimum_frames: int = 1) -> Float32Array:
    values = np.asarray(joints)
    if values.ndim != JOINT_ARRAY_NDIM or values.shape[1:] != (22, 3):
        detail = "joints must have shape [T,22,3]"
        raise HumanRepresentationError(detail)
    if values.shape[0] < minimum_frames or not np.isfinite(values).all():
        detail = f"joints require {minimum_frames} finite frames"
        raise HumanRepresentationError(detail)
    return values.astype(np.float32)


def _forward(joints: Float32Array) -> Float32Array:
    right_hip, left_hip, _right_shoulder, _left_shoulder = FACE_JOINTS
    across = joints[:, right_hip] - joints[:, left_hip]
    norms: Float32Array = np.sqrt((across * across).sum(axis=-1, keepdims=True))
    if any(float(value) <= np.finfo(np.float32).eps for value in norms.flat):
        detail = "hip direction must be non-degenerate"
        raise HumanRepresentationError(detail)
    across /= norms
    forward: Float32Array = np.cross(
        np.array([[0.0, 1.0, 0.0]], dtype=np.float32), across
    )
    forward_norms: Float32Array = np.sqrt(
        (forward * forward).sum(axis=-1, keepdims=True)
    )
    return forward / forward_norms


def human_space_states(joints: FloatArray) -> Float32Array:
    """Return per-frame world ``x/z/yaw`` without modifying the joint sequence."""
    values = _require_joints(joints)
    forward = _forward(values)
    yaw = np.arctan2(forward[:, 2], forward[:, 0])
    states: Float32Array = np.empty((len(values), 3), dtype=np.float32)
    states[:, 0] = values[:, 0, 0]
    states[:, 1] = values[:, 0, 2]
    states[:, 2] = yaw
    return states


def normalize_single_joints22(joints: FloatArray) -> NormalizedHuman:
    """Apply the official floor, root-XZ, and initial-facing transform."""
    values = _require_joints(joints).copy()
    values[:, :, 1] -= np.min(values[:, :, 1])
    root_xz: Float32Array = values[:1, :1, :].reshape(3).copy()
    root_xz[1] = 0.0
    forward = _forward(values[:1])
    x_m, _height_m, z_m = (float(value) for value in root_xz.flat)
    forward_x, _forward_y, forward_z = (
        float(value) for value in forward.reshape(3).flat
    )
    space = HumanSpace(x_m, z_m, atan2(forward_z, forward_x))
    values -= root_xz
    target = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    rotation = between_wxyz(forward, target)
    rotations: Float32Array = np.empty((*values.shape[:-1], 4), dtype=np.float32)
    rotations[:] = rotation[:, None]
    return NormalizedHuman(rotate_wxyz(rotations, values), space)


def denormalize_single_joints22(joints: FloatArray, space: HumanSpace) -> Float32Array:
    """Reverse yaw and XZ normalization; floor height is intentionally absent."""
    values = _require_joints(joints).copy()
    source = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    target = np.array(
        [[np.cos(space.yaw_rad), 0.0, np.sin(space.yaw_rad)]], dtype=np.float32
    )
    rotation = between_wxyz(source, target)
    rotations: Float32Array = np.empty((*values.shape[:-1], 4), dtype=np.float32)
    rotations[:] = rotation[:, None]
    restored = rotate_wxyz(rotations, values)
    restored += np.array([space.x_m, 0.0, space.z_m], dtype=np.float32)
    return restored


def _foot_contacts(joints: Float32Array, threshold: float) -> Float32Array:
    differences = joints[1:] - joints[:-1]
    squared_speed = np.sum(differences * differences, axis=-1)
    left = (squared_speed[:, LEFT_FEET] < threshold).astype(np.float32)
    right = (squared_speed[:, RIGHT_FEET] < threshold).astype(np.float32)
    contacts: Float32Array = np.empty((len(joints), 4), dtype=np.float32)
    contacts[:-1, :2] = left
    contacts[:-1, 2:] = right
    contacts[-1] = contacts[-2]
    return contacts


def joints22_to_human262(
    joints: FloatArray, foot_threshold: float = 0.002
) -> Float32Array:
    """Convert normalized joints to TTR layout: pos66, delta66, rot126, foot4."""
    values = _require_joints(joints, minimum_frames=2)
    if not np.isfinite(foot_threshold) or foot_threshold <= 0.0:
        detail = "foot threshold must be finite and positive"
        raise HumanRepresentationError(detail)
    rotations = wxyz_to_cont6d(inverse_kinematics_wxyz(values))[:, 1:]
    displacement = values[1:] - values[:-1]
    repeated_displacement: Float32Array = np.empty_like(values)
    repeated_displacement[:-1] = displacement
    repeated_displacement[-1] = displacement[-1]
    features: Float32Array = np.empty(
        (len(values), HUMAN_FEATURE_DIM), dtype=np.float32
    )
    features[:, HUMAN_262_LAYOUT.position] = values.reshape(len(values), 66)
    features[:, HUMAN_262_LAYOUT.velocity] = repeated_displacement.reshape(
        len(values), 66
    )
    features[:, HUMAN_262_LAYOUT.rotation] = rotations.reshape(len(values), 126)
    features[:, HUMAN_262_LAYOUT.contact] = _foot_contacts(values, foot_threshold)
    return features


def human262_to_joints22(features: FloatArray) -> Float32Array:
    """Recover positions, the only lossless joint component of historical 262D."""
    values = np.asarray(features)
    if (
        values.ndim < MINIMUM_FEATURE_NDIM
        or values.shape[-1] != HUMAN_FEATURE_DIM
        or not np.isfinite(values).all()
    ):
        detail = "features must be finite with final dimension 262"
        raise HumanRepresentationError(detail)
    return (
        values[..., HUMAN_262_LAYOUT.position]
        .reshape((*values.shape[:-1], 22, 3))
        .astype(np.float32)
    )
