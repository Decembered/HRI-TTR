"""Binary contact reconstruction metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

CONTACT_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class ContactMetrics:
    """Binary contact precision, recall, and F1 score."""

    precision: float
    recall: float
    f1: float


def contact_metrics(
    target: npt.NDArray[np.float64], prediction: npt.NDArray[np.float64]
) -> ContactMetrics:
    """Threshold reconstructed contact logits or probabilities at one half."""
    pairs = tuple(zip(target.flat, prediction.flat, strict=False))
    true_positive = sum(
        float(expected) >= CONTACT_THRESHOLD and float(predicted) >= CONTACT_THRESHOLD
        for expected, predicted in pairs
    )
    false_positive = sum(
        float(expected) < CONTACT_THRESHOLD and float(predicted) >= CONTACT_THRESHOLD
        for expected, predicted in pairs
    )
    false_negative = sum(
        float(expected) >= CONTACT_THRESHOLD and float(predicted) < CONTACT_THRESHOLD
        for expected, predicted in pairs
    )
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ContactMetrics(precision, recall, f1)
