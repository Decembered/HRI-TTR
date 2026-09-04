"""Recoverable metrics for the historical Human 262D representation."""

# pyright: reportAny=false

from __future__ import annotations

from dataclasses import dataclass
from math import acos, fsum, sqrt
from typing import TYPE_CHECKING

from hri_ttr.evaluation.common import (
    MaskedFeatureMetrics,
    masked_feature_metrics,
    validate_features,
)
from hri_ttr.evaluation.contact import ContactMetrics, contact_metrics
from hri_ttr.geometry.rotation import rotation_6d_to_matrix
from hri_ttr.representations.human.features import HUMAN_262_LAYOUT

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class HumanReconstructionMetrics:
    """Masked Human 262D position, local rotation, and contact metrics."""

    features: MaskedFeatureMetrics
    mpjpe_m: float
    root_relative_mpjpe_m: float
    mean_joint_rotation_geodesic_rad: float
    contact: ContactMetrics


def _mean_joint_distance(values: npt.NDArray[np.float64]) -> float:
    distances = tuple(
        sqrt(
            fsum(
                float(values[frame, joint, coordinate]) ** 2 for coordinate in range(3)
            )
        )
        for frame in range(len(values))
        for joint in range(22)
    )
    return fsum(distances) / len(distances)


def _mean_geodesic(
    target: npt.NDArray[np.float64], prediction: npt.NDArray[np.float64]
) -> float:
    angles: list[float] = []
    for frame in range(len(target)):
        for joint in range(21):
            trace = fsum(
                float(
                    target[frame, joint, row, column]
                    * prediction[frame, joint, row, column]
                )
                for row in range(3)
                for column in range(3)
            )
            angles.append(acos(min(1.0, max(-1.0, (trace - 1.0) / 2.0))))
    return fsum(angles) / len(angles)


def evaluate_human_reconstruction(
    target: npt.NDArray[np.float64],
    prediction: npt.NDArray[np.float64],
    frame_mask: npt.NDArray[np.bool_],
) -> HumanReconstructionMetrics:
    """Evaluate fields recoverable directly from the historical 262D layout."""
    expected, reconstructed = validate_features(target, prediction, frame_mask, 262)
    target_joints = expected[:, HUMAN_262_LAYOUT.position].reshape(-1, 22, 3)
    predicted_joints = reconstructed[:, HUMAN_262_LAYOUT.position].reshape(-1, 22, 3)
    target_relative = target_joints - target_joints[:, :1]
    predicted_relative = predicted_joints - predicted_joints[:, :1]
    target_rotation = rotation_6d_to_matrix(
        expected[:, HUMAN_262_LAYOUT.rotation].reshape(-1, 21, 6)
    )
    predicted_rotation = rotation_6d_to_matrix(
        reconstructed[:, HUMAN_262_LAYOUT.rotation].reshape(-1, 21, 6)
    )
    return HumanReconstructionMetrics(
        masked_feature_metrics(target, prediction, frame_mask),
        _mean_joint_distance(predicted_joints - target_joints),
        _mean_joint_distance(predicted_relative - target_relative),
        _mean_geodesic(target_rotation, predicted_rotation),
        contact_metrics(
            expected[:, HUMAN_262_LAYOUT.contact],
            reconstructed[:, HUMAN_262_LAYOUT.contact],
        ),
    )
