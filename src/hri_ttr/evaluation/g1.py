"""Physical-unit metrics for the G1 75D representation."""

# pyright: reportAny=false

from __future__ import annotations

from array import array
from dataclasses import dataclass
from math import acos, atan2, cos, fsum, pi, sin, sqrt

import numpy as np
import numpy.typing as npt

from hri_ttr.evaluation.common import (
    MaskedFeatureMetrics,
    masked_feature_metrics,
    validate_features,
)
from hri_ttr.evaluation.contact import ContactMetrics, contact_metrics
from hri_ttr.geometry.rotation import rotation_6d_to_matrix
from hri_ttr.representations.g1 import G1_FEATURE_SLICES


@dataclass(frozen=True, slots=True)
class G1ReconstructionMetrics:
    """Masked 75D reconstruction metrics in physical units."""

    features: MaskedFeatureMetrics
    per_joint_mae_rad: npt.NDArray[np.float64]
    per_joint_mae_degree: npt.NDArray[np.float64]
    worst_joint_index: int
    worst_joint_error_rad: float
    root_position_ade_m: float
    root_position_fde_m: float
    root_rotation_geodesic_rad: float
    local_linear_velocity_mae_m_s: float
    world_linear_velocity_mae_m_s: float
    angular_velocity_mae_rad_s: float
    dof_velocity_mae_rad_s: float
    yaw_mae_rad: float
    height_mae_m: float
    contact: ContactMetrics


def _mean_absolute(values: npt.NDArray[np.float64]) -> float:
    flattened: array[float] = array("d")
    flattened.frombytes(values.tobytes())
    absolute = tuple(abs(value) for value in flattened)
    return fsum(absolute) / len(absolute)


def _row_norms(values: npt.NDArray[np.float64]) -> tuple[float, ...]:
    return tuple(
        sqrt(
            fsum(float(values[row, column]) ** 2 for column in range(len(values[row])))
        )
        for row in range(len(values))
    )


def _rotation_geodesic(
    target: npt.NDArray[np.float64], prediction: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    angles = np.empty(len(target), dtype=np.float64)
    for frame in range(len(target)):
        trace = fsum(
            float(target[frame, row, column] * prediction[frame, row, column])
            for row in range(3)
            for column in range(3)
        )
        angles[frame] = acos(min(1.0, max(-1.0, (trace - 1.0) / 2.0)))
    return angles


def _world_velocity(
    rotations: npt.NDArray[np.float64], local: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    world = np.empty_like(local)
    for frame in range(len(local)):
        for row in range(3):
            world[frame, row] = fsum(
                float(rotations[frame, row, column] * local[frame, column])
                for column in range(3)
            )
    return world


def _yaw_difference(
    target: npt.NDArray[np.float64], prediction: npt.NDArray[np.float64]
) -> tuple[float, ...]:
    differences: list[float] = []
    for frame in range(len(target)):
        target_yaw = atan2(float(target[frame, 0, 2]), float(target[frame, 2, 2]))
        predicted_yaw = atan2(
            float(prediction[frame, 0, 2]), float(prediction[frame, 2, 2])
        )
        delta = predicted_yaw - target_yaw
        differences.append(atan2(sin(delta), cos(delta)))
    return tuple(differences)


def evaluate_g1_reconstruction(
    target: npt.NDArray[np.float64],
    prediction: npt.NDArray[np.float64],
    frame_mask: npt.NDArray[np.bool_],
) -> G1ReconstructionMetrics:
    """Evaluate G1 root, joints, velocities, and contacts on valid frames."""
    expected, reconstructed = validate_features(target, prediction, frame_mask, 75)
    dof_slice = G1_FEATURE_SLICES["dof_pos_rad"]
    joint_difference = reconstructed[:, dof_slice] - expected[:, dof_slice]
    joint_errors = np.asarray(
        tuple(
            _mean_absolute(joint_difference[:, joint : joint + 1])
            for joint in range(29)
        ),
        dtype=np.float64,
    )
    root_slice = G1_FEATURE_SLICES["root_pos_episode_m"]
    root_errors = _row_norms(reconstructed[:, root_slice] - expected[:, root_slice])
    rotation_slice = G1_FEATURE_SLICES["root_rot6d_episode"]
    target_rotation = rotation_6d_to_matrix(expected[:, rotation_slice])
    predicted_rotation = rotation_6d_to_matrix(reconstructed[:, rotation_slice])
    local_slice = G1_FEATURE_SLICES["root_linear_vel_local_m_s"]
    target_local = expected[:, local_slice]
    predicted_local = reconstructed[:, local_slice]
    target_world = _world_velocity(target_rotation, target_local)
    predicted_world = _world_velocity(predicted_rotation, predicted_local)
    yaw_errors = _yaw_difference(target_rotation, predicted_rotation)
    worst = max(range(len(joint_errors)), key=lambda joint: float(joint_errors[joint]))
    angular_slice = G1_FEATURE_SLICES["root_angular_vel_local_rad_s"]
    dof_velocity_slice = G1_FEATURE_SLICES["dof_vel_rad_s"]
    contact_slice = G1_FEATURE_SLICES["foot_contact_lr"]
    return G1ReconstructionMetrics(
        masked_feature_metrics(target, prediction, frame_mask),
        joint_errors,
        joint_errors * (180.0 / pi),
        worst,
        float(joint_errors[worst]),
        fsum(root_errors) / len(root_errors),
        root_errors[-1],
        _mean_absolute(_rotation_geodesic(target_rotation, predicted_rotation)),
        _mean_absolute(predicted_local - target_local),
        _mean_absolute(predicted_world - target_world),
        _mean_absolute(reconstructed[:, angular_slice] - expected[:, angular_slice]),
        _mean_absolute(
            reconstructed[:, dof_velocity_slice] - expected[:, dof_velocity_slice]
        ),
        fsum(abs(value) for value in yaw_errors) / len(yaw_errors),
        _mean_absolute(reconstructed[:, 1:2] - expected[:, 1:2]),
        contact_metrics(expected[:, contact_slice], reconstructed[:, contact_slice]),
    )
