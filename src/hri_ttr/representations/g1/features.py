"""Deterministic G1 motion conversion to and from the canonical 75D schema."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from enum import StrEnum
from math import fsum, sqrt
from typing import Final

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from hri_ttr.geometry.quaternion import matrix_to_xyzw, xyzw_to_matrix
from hri_ttr.geometry.rotation import (
    matrix_to_rotation_6d,
    matrix_to_rotvec,
    rotation_6d_to_matrix,
)
from hri_ttr.representations.g1.constants import G1_DOF_COUNT
from hri_ttr.representations.g1.episode import EpisodeFrame
from hri_ttr.representations.g1.schema import G1_FEATURE_DIM, G1_FEATURE_SLICES

QUATERNION_NORM_TOLERANCE: Final = 1e-8
FEATURE_MATRIX_NDIM: Final = 2


class G1RepresentationReason(StrEnum):
    """Stable reasons for rejecting G1 motion or canonical features."""

    SHAPES = "expected root [T,3], root rotation [T,4], dof [T,29], contact [T,2]"
    FINITE = "G1 motion values must be finite"
    QUATERNION = "root rotations must be unit xyzw quaternions"
    CONVENTION = "quaternion convention must be xyzw"
    CONTACT = "foot contact values must be binary zero or one"
    FPS = "fps must be finite and positive"
    FEATURES = "features must be finite with shape [T,75]"


@dataclass(frozen=True, slots=True)
class G1RepresentationError(ValueError):
    """Reports a violation of the G1 75D representation boundary."""

    reason: G1RepresentationReason

    @override
    def __str__(self) -> str:
        """Render the stable boundary failure."""
        return self.reason


@dataclass(frozen=True, slots=True)
class G1MotionInput:
    """Unbatched G1 motion expressed in InteractionWorld Y-up coordinates."""

    root_pos_interaction_m: npt.NDArray[np.float64]
    root_rot_interaction_xyzw: npt.NDArray[np.float64]
    dof_pos_rad: npt.NDArray[np.float64]
    foot_contact_lr: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class EncodedG1Motion:
    """Canonical 75D frames and the EpisodeFrame needed to decode them."""

    features: npt.NDArray[np.float64]
    anchor: EpisodeFrame


@dataclass(frozen=True, slots=True)
class DecodedG1Motion:
    """Decoded InteractionWorld root pose, actuator positions, and contacts."""

    root_pos_interaction_m: npt.NDArray[np.float64]
    root_rot_interaction_xyzw: npt.NDArray[np.float64]
    dof_pos_rad: npt.NDArray[np.float64]
    foot_contact_lr: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class VelocityIntegrationDiagnostic:
    """Difference between explicit root positions and velocity integration."""

    reconstructed_episode_m: npt.NDArray[np.float64]
    max_position_error_m: float


def encode_g1_features(
    motion: G1MotionInput,
    *,
    fps: float,
    quaternion_convention: str,
) -> EncodedG1Motion:
    """Encode authoritative root pose and causal auxiliary derivatives."""
    _require_motion(motion, fps, quaternion_convention)
    root_rot_interaction = xyzw_to_matrix(motion.root_rot_interaction_xyzw)
    initial_position = motion.root_pos_interaction_m[:1].reshape(3)
    initial_rotation = root_rot_interaction[:1].reshape(3, 3)
    anchor = EpisodeFrame.from_initial_root(initial_position, initial_rotation)
    positions = anchor.positions_to_episode(motion.root_pos_interaction_m)
    rotations = anchor.rotations_to_episode(root_rot_interaction)
    frames = len(positions)
    features = np.zeros((frames, G1_FEATURE_DIM), dtype=np.float64)
    features[:, G1_FEATURE_SLICES["root_pos_episode_m"]] = positions
    features[:, G1_FEATURE_SLICES["root_rot6d_episode"]] = matrix_to_rotation_6d(
        rotations
    )
    features[:, G1_FEATURE_SLICES["dof_pos_rad"]] = motion.dof_pos_rad
    if frames > 1:
        displacement = (positions[1:] - positions[:-1]) * fps
        features[1:, G1_FEATURE_SLICES["root_linear_vel_local_m_s"]] = np.einsum(
            "tji,tj->ti", rotations[:-1], displacement
        )
        relative_rotation = np.asarray(
            rotations[:-1].transpose(0, 2, 1) @ rotations[1:],
            dtype=np.float64,
        )
        features[1:, G1_FEATURE_SLICES["root_angular_vel_local_rad_s"]] = (
            matrix_to_rotvec(relative_rotation) * fps
        )
        features[1:, G1_FEATURE_SLICES["dof_vel_rad_s"]] = (
            motion.dof_pos_rad[1:] - motion.dof_pos_rad[:-1]
        ) * fps
    features[:, G1_FEATURE_SLICES["foot_contact_lr"]] = motion.foot_contact_lr
    return EncodedG1Motion(features, anchor)


def decode_g1_features(
    features: npt.NDArray[np.float64], anchor: EpisodeFrame
) -> DecodedG1Motion:
    """Decode explicit root and DoF fields without integrating velocities."""
    _require_features(features)
    positions_episode = features[:, G1_FEATURE_SLICES["root_pos_episode_m"]]
    rotations_episode = rotation_6d_to_matrix(
        features[:, G1_FEATURE_SLICES["root_rot6d_episode"]]
    )
    return DecodedG1Motion(
        anchor.positions_to_interaction(positions_episode),
        matrix_to_xyzw(anchor.rotations_to_interaction(rotations_episode)),
        features[:, G1_FEATURE_SLICES["dof_pos_rad"]].copy(),
        features[:, G1_FEATURE_SLICES["foot_contact_lr"]].copy(),
    )


def velocity_integration_diagnostic(
    features: npt.NDArray[np.float64], *, fps: float
) -> VelocityIntegrationDiagnostic:
    """Integrate auxiliary local velocity without altering decoded root pose."""
    _require_features(features)
    if not np.isfinite(fps) or fps <= 0.0:
        raise G1RepresentationError(G1RepresentationReason.FPS)
    explicit = features[:, G1_FEATURE_SLICES["root_pos_episode_m"]]
    rotations = rotation_6d_to_matrix(
        features[:, G1_FEATURE_SLICES["root_rot6d_episode"]]
    )
    velocity = features[:, G1_FEATURE_SLICES["root_linear_vel_local_m_s"]]
    integrated = np.empty_like(explicit)
    integrated[0] = explicit[0]
    for frame in range(1, len(explicit)):
        integrated[frame] = (
            integrated[frame - 1] + rotations[frame - 1] @ velocity[frame] / fps
        )
    errors = np.sqrt(np.sum((integrated - explicit) ** 2, axis=1))
    return VelocityIntegrationDiagnostic(integrated, max(_flat_values(errors)))


def _require_motion(motion: G1MotionInput, fps: float, convention: str) -> None:
    frames = len(motion.root_pos_interaction_m)
    shapes = (
        motion.root_pos_interaction_m.shape,
        motion.root_rot_interaction_xyzw.shape,
        motion.dof_pos_rad.shape,
        motion.foot_contact_lr.shape,
    )
    if shapes != ((frames, 3), (frames, 4), (frames, G1_DOF_COUNT), (frames, 2)):
        raise G1RepresentationError(G1RepresentationReason.SHAPES)
    arrays = (
        motion.root_pos_interaction_m,
        motion.root_rot_interaction_xyzw,
        motion.dof_pos_rad,
        motion.foot_contact_lr,
    )
    if frames == 0 or not all(np.isfinite(values).all() for values in arrays):
        raise G1RepresentationError(G1RepresentationReason.FINITE)
    quaternion_values = _flat_values(motion.root_rot_interaction_xyzw)
    norms = (
        sqrt(fsum(value * value for value in quaternion_values[start : start + 4]))
        for start in range(0, len(quaternion_values), 4)
    )
    if any(abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE for norm in norms):
        raise G1RepresentationError(G1RepresentationReason.QUATERNION)
    if convention != "xyzw":
        raise G1RepresentationError(G1RepresentationReason.CONVENTION)
    contacts = _flat_values(motion.foot_contact_lr)
    if any(contact not in {0.0, 1.0} for contact in contacts):
        raise G1RepresentationError(G1RepresentationReason.CONTACT)
    if not np.isfinite(fps) or fps <= 0.0:
        raise G1RepresentationError(G1RepresentationReason.FPS)


def _require_features(features: npt.NDArray[np.float64]) -> None:
    if (
        features.ndim != FEATURE_MATRIX_NDIM
        or features.shape[1] != G1_FEATURE_DIM
        or len(features) == 0
        or not np.isfinite(features).all()
    ):
        raise G1RepresentationError(G1RepresentationReason.FEATURES)


def _flat_values(values: npt.NDArray[np.float64]) -> array[float]:
    flattened: array[float] = array("d")
    flattened.frombytes(values.tobytes())
    return flattened
