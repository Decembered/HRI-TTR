"""Strict, machine-readable quality gates for aligned motion pairs."""

# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from hri_ttr.representations.g1.constants import G1_DOF_LIMITS_RAD

TARGET_FPS: Final = 20.0
DOF_LIMIT_TOLERANCE_RAD: Final = 1e-3
QUATERNION_NORM_TOLERANCE: Final = 1e-3
MAX_ROOT_SPEED_M_S: Final = 5.0
MAX_ROOT_ANGULAR_SPEED_RAD_S: Final = 20.0
MAX_DOF_SPEED_RAD_S: Final = 30.0
MAX_HUMAN_JOINT_SPEED_M_S: Final = 15.0
MAX_BONE_LENGTH_CV: Final = 0.02
MIN_HUMAN_HEIGHT_M: Final = 0.5
MAX_HUMAN_HEIGHT_M: Final = 2.5
NONZERO_MOTION_STD: Final = 1e-5
MINIMUM_ENERGY_CORRELATION: Final = -0.1
NEAR_FLOOR_HEIGHT_M: Final = 0.06
MAX_G1_FOOT_PENETRATION_M: Final = 0.05
MATRIX_NDIM: Final = 2
JOINT_ARRAY_NDIM: Final = 3

# The 22-joint order is the HumanML3D/SMPL order used by the feature encoder.
# Keep the chains explicit so the height and bone-length checks cannot silently
# drift from the kinematic tree used by the representation code.
HUMAN_CHAINS: Final = (
    (0, 2, 5, 8, 11),
    (0, 1, 4, 7, 10),
    (0, 3, 6, 9, 12, 15),
    (9, 14, 17, 19, 21),
    (9, 13, 16, 18, 20),
)


class QualityReason(StrEnum):
    """Stable machine-readable quarantine reasons."""

    G1_QUATERNION_SHAPE = "g1_quaternion_shape"
    TIMELINE_SHAPE = "timeline_shape"
    TIMELINE_MISMATCH = "timeline_mismatch"
    TIMELINE_TOO_SHORT = "timeline_too_short"
    G1_DOF_SHAPE_OR_FINITE = "g1_dof_shape_or_finite"
    G1_DOF_LIMIT = "g1_dof_limit"
    SOURCE_SHAPE = "source_shape"
    SOURCE_NONFINITE = "source_nonfinite"
    G1_QUATERNION_NORM = "g1_quaternion_norm"
    HUMAN_HEIGHT = "human_height"
    HUMAN_BONE_LENGTH = "human_bone_length"
    HUMAN_JOINT_SPEED = "human_joint_speed"
    G1_BODY_ORDER = "g1_body_order"
    G1_ROOT_SPEED = "g1_root_speed"
    G1_ROOT_ANGULAR_SPEED = "g1_root_angular_speed"
    G1_DOF_SPEED = "g1_dof_speed"
    G1_FOOT_PENETRATION = "g1_foot_penetration"
    PAIR_ENERGY_ALIGNMENT = "pair_energy_alignment"


@dataclass(frozen=True, slots=True)
class QualityError(ValueError):
    """Stable quarantine reason and measurement for one rejected sequence."""

    reason: QualityReason | str
    detail: str

    @override
    def __str__(self) -> str:
        return f"{self.reason}: {self.detail}"


def validate_g1_dof(values: npt.NDArray[np.float64]) -> None:
    """Reject malformed or mechanically impossible 29-DoF configurations."""
    dof = np.asarray(values, dtype=np.float64)
    if dof.ndim != MATRIX_NDIM or dof.shape[1:] != (29,) or not np.isfinite(dof).all():
        raise QualityError(QualityReason.G1_DOF_SHAPE_OR_FINITE, str(dof.shape))
    limits = np.asarray(G1_DOF_LIMITS_RAD, dtype=np.float64)
    violation = _scalar(
        max(np.max(limits[:, 0] - dof), np.max(dof - limits[:, 1]), 0.0)
    )
    if violation > DOF_LIMIT_TOLERANCE_RAD:
        raise QualityError(
            QualityReason.G1_DOF_LIMIT, f"max_violation={violation:.6f}rad"
        )


def validate_raw_motion(
    human: npt.NDArray[np.float64],
    human_fps: float,
    root: npt.NDArray[np.float64],
    rotation: npt.NDArray[np.float64],
    dof: npt.NDArray[np.float64],
) -> None:
    """Validate native Human and G1 arrays before interpolation."""
    arrays = (human, root, rotation, dof)
    shapes = tuple(value.shape for value in arrays)
    if (
        human.ndim != JOINT_ARRAY_NDIM
        or human.shape[1:] != (22, 3)
        or root.shape != (len(dof), 3)
        or rotation.shape != (len(dof), 4)
    ):
        raise QualityError(QualityReason.SOURCE_SHAPE, str(shapes))
    if not all(np.isfinite(value).all() for value in arrays):
        raise QualityError(
            QualityReason.SOURCE_NONFINITE, "Human or G1 contains NaN/Inf"
        )
    norm_error = _scalar(np.max(np.abs(np.linalg.norm(rotation, axis=1) - 1.0)))
    if norm_error > QUATERNION_NORM_TOLERANCE:
        raise QualityError(
            QualityReason.G1_QUATERNION_NORM, f"max_error={norm_error:.6f}"
        )
    validate_g1_dof(dof)
    _validate_human(human, human_fps)


def aligned_quality_metrics(
    human: npt.NDArray[np.float64],
    root: npt.NDArray[np.float64],
    rotation: npt.NDArray[np.float64],
    dof: npt.NDArray[np.float64],
    feet: npt.NDArray[np.float64],
) -> dict[str, float]:
    """Measure aligned motion and reject physically implausible discontinuities."""
    root_speed = np.linalg.norm(np.diff(root, axis=0), axis=1) * TARGET_FPS
    quaternion_dot = np.abs(np.sum(rotation[:-1] * rotation[1:], axis=1))
    angular_speed = 2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0)) * TARGET_FPS
    dof_speed = np.abs(np.diff(dof, axis=0)) * TARGET_FPS
    human_speed = np.linalg.norm(np.diff(human, axis=0), axis=2) * TARGET_FPS
    maximum_root = _scalar(np.max(root_speed))
    maximum_angular = _scalar(np.max(angular_speed))
    maximum_dof = _scalar(np.max(dof_speed))
    if maximum_root > MAX_ROOT_SPEED_M_S:
        raise QualityError(QualityReason.G1_ROOT_SPEED, f"max={maximum_root:.6f}m/s")
    if maximum_angular > MAX_ROOT_ANGULAR_SPEED_RAD_S:
        raise QualityError(
            QualityReason.G1_ROOT_ANGULAR_SPEED,
            f"max={maximum_angular:.6f}rad/s",
        )
    if maximum_dof > MAX_DOF_SPEED_RAD_S:
        raise QualityError(QualityReason.G1_DOF_SPEED, f"max={maximum_dof:.6f}rad/s")
    human_energy = np.mean(human_speed, axis=1)
    g1_energy = np.mean(dof_speed, axis=1)
    correlation = 1.0
    if (
        np.std(human_energy) > NONZERO_MOTION_STD
        and np.std(g1_energy) > NONZERO_MOTION_STD
    ):
        correlation = _scalar(np.corrcoef(human_energy, g1_energy)[0, 1])
        if correlation < MINIMUM_ENERGY_CORRELATION:
            raise QualityError(
                QualityReason.PAIR_ENERGY_ALIGNMENT,
                f"correlation={correlation:.6f}",
            )
    floor = _scalar(np.percentile(feet[:, :, 1], 2.0))
    penetration = max(0.0, -_scalar(np.min(feet[:, :, 1])))
    if penetration > MAX_G1_FOOT_PENETRATION_M:
        raise QualityError(
            QualityReason.G1_FOOT_PENETRATION,
            f"max={penetration:.6f}m",
        )
    horizontal_speed = (
        np.linalg.norm(np.diff(feet[:, :, (0, 2)], axis=0), axis=2) * TARGET_FPS
    )
    near_floor = feet[1:, :, 1] - floor <= NEAR_FLOOR_HEIGHT_M
    slide = (
        _scalar(np.percentile(horizontal_speed[near_floor], 95.0))
        if np.any(near_floor)
        else 0.0
    )
    return {
        "max_human_joint_speed_m_s": _scalar(np.max(human_speed)),
        "max_g1_root_speed_m_s": maximum_root,
        "max_g1_root_angular_speed_rad_s": maximum_angular,
        "max_g1_dof_speed_rad_s": maximum_dof,
        "energy_correlation": correlation,
        "g1_foot_floor_y_m": floor,
        "max_g1_foot_penetration_m": penetration,
        "g1_near_floor_slide_p95_m_s": slide,
        "root_quaternion_norm_error": _scalar(
            np.max(np.abs(np.linalg.norm(rotation, axis=1) - 1.0))
        ),
    }


def _validate_human(joints: npt.NDArray[np.float64], fps: float) -> None:
    chain_lengths = tuple(
        np.linalg.norm(
            joints[:, np.asarray(chain[1:])] - joints[:, np.asarray(chain[:-1])],
            axis=2,
        )
        for chain in HUMAN_CHAINS
    )
    edge_lengths = np.concatenate(chain_lengths, axis=1)
    left_leg, right_leg, trunk = chain_lengths[:3]
    height = _scalar(
        np.median(
            (left_leg.sum(axis=1) + right_leg.sum(axis=1)) * 0.5 + trunk.sum(axis=1)
        )
    )
    if not MIN_HUMAN_HEIGHT_M <= height <= MAX_HUMAN_HEIGHT_M:
        raise QualityError(QualityReason.HUMAN_HEIGHT, f"stature={height:.6f}m")
    means = np.mean(edge_lengths, axis=0)
    variable = means > NONZERO_MOTION_STD
    cv = _scalar(np.max(np.std(edge_lengths[:, variable], axis=0) / means[variable]))
    if cv > MAX_BONE_LENGTH_CV:
        raise QualityError(QualityReason.HUMAN_BONE_LENGTH, f"max_cv={cv:.6f}")
    if len(joints) > 1:
        maximum = _scalar(np.max(np.linalg.norm(np.diff(joints, axis=0), axis=2) * fps))
        if maximum > MAX_HUMAN_JOINT_SPEED_M_S:
            raise QualityError(QualityReason.HUMAN_JOINT_SPEED, f"max={maximum:.6f}m/s")


def _scalar(value: object) -> float:
    return cast("float", np.asarray(value).item())
