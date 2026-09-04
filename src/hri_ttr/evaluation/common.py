"""Shared mask validation and feature reconstruction metrics."""

# pyright: reportAny=false

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, sqrt

import numpy as np
import numpy.typing as npt

from hri_ttr.evaluation.errors import reject_evaluation

FEATURE_MATRIX_NDIM = 2


@dataclass(frozen=True, slots=True)
class MaskedFeatureMetrics:
    """Scalar reconstruction errors over valid frames and all feature channels."""

    mae: float
    rmse: float
    maximum_absolute_error: float


def validate_features(
    target: npt.NDArray[np.float64],
    prediction: npt.NDArray[np.float64],
    frame_mask: npt.NDArray[np.bool_],
    feature_dim: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return valid rows after enforcing one unbatched feature contract."""
    if (
        target.ndim != FEATURE_MATRIX_NDIM
        or target.shape != prediction.shape
        or target.shape[1] != feature_dim
        or frame_mask.shape != target.shape[:1]
        or frame_mask.dtype != np.bool_
        or target.dtype != np.float64
        or prediction.dtype != np.float64
        or not frame_mask.any()
        or not np.isfinite(target).all()
        or not np.isfinite(prediction).all()
    ):
        reject_evaluation("features and mask do not satisfy evaluation contract")
    return target[frame_mask], prediction[frame_mask]


def masked_feature_metrics(
    target: npt.NDArray[np.float64],
    prediction: npt.NDArray[np.float64],
    frame_mask: npt.NDArray[np.bool_],
) -> MaskedFeatureMetrics:
    """Compute basic errors for equally shaped finite feature matrices."""
    if target.ndim != FEATURE_MATRIX_NDIM:
        reject_evaluation("features must be two-dimensional")
    valid_target, valid_prediction = validate_features(
        target, prediction, frame_mask, len(target[0])
    )
    difference = valid_prediction - valid_target
    absolute_values = tuple(abs(float(value)) for value in difference.flat)
    squared_values = tuple(float(value) ** 2 for value in difference.flat)
    return MaskedFeatureMetrics(
        fsum(absolute_values) / len(absolute_values),
        sqrt(fsum(squared_values) / len(squared_values)),
        max(absolute_values),
    )
