"""Canonical same-motion conversion and strict quality boundaries."""

# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import numpy.typing as npt

from hri_ttr.data.same_motion_quality import (
    QualityError,
    QualityReason,
    aligned_quality_metrics,
    validate_g1_dof,
    validate_raw_motion,
)
from hri_ttr.geometry.coordinates import (
    g1_z_up_to_interaction_y_up,
    quaternion_xyzw_g1_to_interaction,
)
from hri_ttr.geometry.resample import (
    resample_linear,
    resample_quaternion_xyzw,
    target_timestamps,
)
from hri_ttr.representations.g1 import (
    G1MotionInput,
    compute_g1_foot_contacts,
    encode_g1_features,
    g1_space_states,
)
from hri_ttr.representations.human.features import (
    human_space_states,
    joints22_to_human262,
    normalize_single_joints22,
)

TARGET_FPS: Final = 20.0
MINIMUM_FRAMES: Final = 2
TIMELINE_TOLERANCE_SECONDS: Final = 0.025

if TYPE_CHECKING:
    from hri_ttr.data.g1_kinematics import G1Kinematics


@dataclass(frozen=True, slots=True)
class RawPair:
    """Source-decoded Human joints and G1 configuration on native timelines."""

    human_joints_z_up: npt.NDArray[np.float64]
    human_fps: float
    g1_root_z_up: npt.NDArray[np.float64]
    g1_root_xyzw_z_up: npt.NDArray[np.float64]
    g1_dof_rad: npt.NDArray[np.float64]
    g1_fps: float
    g1_body_positions_z_up: npt.NDArray[np.float64] | None = None
    g1_body_names: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class PreparedPair:
    """Aligned unnormalized Human262 and G1-75 training features."""

    human_features: npt.NDArray[np.float32]
    g1_features: npt.NDArray[np.float32]
    quality: dict[str, float]
    anchor_origin: npt.NDArray[np.float64]
    anchor_basis: npt.NDArray[np.float64]
    human_space: npt.NDArray[np.float64]
    g1_space: npt.NDArray[np.float64]


def wxyz_to_xyzw(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert scalar-first quaternions to the project's scalar-last convention."""
    source = np.asarray(values, dtype=np.float64)
    if source.shape[-1:] != (4,):
        raise QualityError(QualityReason.G1_QUATERNION_SHAPE, str(source.shape))
    return source[..., [1, 2, 3, 0]]


def common_timeline(
    human_frames: int,
    human_fps: float,
    g1_frames: int,
    g1_fps: float,
    target_fps: float = TARGET_FPS,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Return native timestamps and one target grid inside both time extents."""
    if (
        min(human_frames, g1_frames) < MINIMUM_FRAMES
        or min(human_fps, g1_fps, target_fps) <= 0
    ):
        raise QualityError(
            QualityReason.TIMELINE_SHAPE, "requires two frames and positive FPS"
        )
    human_time = np.arange(human_frames, dtype=np.float64) / human_fps
    g1_time = np.arange(g1_frames, dtype=np.float64) / g1_fps
    human_duration = human_frames / human_fps
    g1_duration = g1_frames / g1_fps
    difference = abs(human_duration - g1_duration)
    quantization = 1.0 / human_fps + 1.0 / g1_fps
    tolerance = max(TIMELINE_TOLERANCE_SECONDS, quantization)
    if difference > tolerance:
        raise QualityError(
            QualityReason.TIMELINE_MISMATCH,
            f"duration_error={difference:.6f}s tolerance={tolerance:.6f}s",
        )
    common_end = min(_scalar(human_time[-1]), _scalar(g1_time[-1]))
    target = target_timestamps(
        np.asarray([0.0, common_end], dtype=np.float64), target_fps
    )
    if len(target) < MINIMUM_FRAMES:
        raise QualityError(QualityReason.TIMELINE_TOO_SHORT, f"{common_end:.6f}s")
    return human_time, g1_time, target


def prepare_pair(raw: RawPair, kinematics: G1Kinematics) -> PreparedPair:
    """Validate, align, transform, and encode one same-motion pair."""
    validate_raw_motion(
        raw.human_joints_z_up,
        raw.human_fps,
        raw.g1_root_z_up,
        raw.g1_root_xyzw_z_up,
        raw.g1_dof_rad,
    )
    human_time, g1_time, target_time = common_timeline(
        len(raw.human_joints_z_up),
        raw.human_fps,
        len(raw.g1_root_z_up),
        raw.g1_fps,
    )
    human_world = g1_z_up_to_interaction_y_up(
        resample_linear(raw.human_joints_z_up, human_time, target_time)
    )
    root_native = resample_linear(raw.g1_root_z_up, g1_time, target_time)
    rotation_native = resample_quaternion_xyzw(
        raw.g1_root_xyzw_z_up, g1_time, target_time
    )
    dof = resample_linear(raw.g1_dof_rad, g1_time, target_time)
    validate_g1_dof(dof)
    root = g1_z_up_to_interaction_y_up(root_native)
    rotation = quaternion_xyzw_g1_to_interaction(rotation_native)
    feet_native = _foot_positions(raw, kinematics, g1_time, target_time)
    feet = g1_z_up_to_interaction_y_up(feet_native)
    contact_input = np.repeat(feet[:, :1], 20, axis=1)
    contact_input[:, 4] = feet[:, 0]
    contact_input[:, 8] = feet[:, 1]
    contacts = compute_g1_foot_contacts(contact_input, fps=TARGET_FPS)
    normalized_human = normalize_single_joints22(human_world.astype(np.float32))
    human_features = joints22_to_human262(normalized_human.joints)
    encoded = encode_g1_features(
        G1MotionInput(root, rotation, dof, contacts),
        fps=TARGET_FPS,
        quaternion_convention="xyzw",
    )
    quality = aligned_quality_metrics(human_world, root, rotation, dof, feet)
    human_space = human_space_states(human_world.astype(np.float32)).astype(
        np.float64
    )
    g1_space = g1_space_states(root, rotation)
    return PreparedPair(
        human_features.astype(np.float32),
        encoded.features.astype(np.float32),
        quality,
        encoded.anchor.origin_interaction_m,
        encoded.anchor.episode_to_interaction,
        human_space,
        g1_space,
    )


def _foot_positions(
    raw: RawPair,
    kinematics: G1Kinematics,
    source_time: npt.NDArray[np.float64],
    target_time: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    if raw.g1_body_positions_z_up is not None and raw.g1_body_names is not None:
        names = raw.g1_body_names
        try:
            indices = (
                names.index("left_ankle_roll_link"),
                names.index("right_ankle_roll_link"),
            )
        except ValueError as error:
            raise QualityError(
                QualityReason.G1_BODY_ORDER, "ankle bodies are absent"
            ) from error
        bodies = raw.g1_body_positions_z_up[:, indices]
        return resample_linear(bodies, source_time, target_time)
    root = resample_linear(raw.g1_root_z_up, source_time, target_time)
    rotation = resample_quaternion_xyzw(raw.g1_root_xyzw_z_up, source_time, target_time)
    dof = resample_linear(raw.g1_dof_rad, source_time, target_time)
    positions = kinematics.body_positions(root, rotation, dof)
    return np.stack(
        (positions["left_ankle_roll_link"], positions["right_ankle_roll_link"]),
        axis=1,
    )


def _scalar(value: object) -> float:
    return cast("float", np.asarray(value).item())
