from __future__ import annotations

from math import sqrt

import numpy as np
import pytest

from hri_ttr.geometry.coordinates import (
    g1_z_up_to_interaction_y_up,
    interaction_y_up_to_g1_z_up,
    quaternion_xyzw_g1_to_interaction,
    quaternion_xyzw_interaction_to_g1,
)
from hri_ttr.geometry.quaternion import matrix_to_xyzw, xyzw_to_matrix
from hri_ttr.geometry.resample import (
    TimelineError,
    resample_linear,
    resample_quaternion_xyzw,
    target_timestamps,
)


def test_points_roundtrip_when_converting_coordinate_basis() -> None:
    # Given
    points = np.array([[1.0, 2.0, 3.0], [-2.0, 1.0, 0.5]])

    # When
    restored = interaction_y_up_to_g1_z_up(
        g1_z_up_to_interaction_y_up(points),
    )

    # Then
    np.testing.assert_allclose(restored, points, atol=1e-12)


def test_quaternion_roundtrip_when_antipodal_input_is_used() -> None:
    # Given
    half = sqrt(0.5)
    rotations = np.array([[0.0, 0.0, half, half], [0.0, 0.0, -half, -half]])

    # When
    restored = quaternion_xyzw_interaction_to_g1(
        quaternion_xyzw_g1_to_interaction(rotations),
    )

    # Then
    dots = np.abs(np.sum(restored * rotations, axis=-1))
    np.testing.assert_allclose(dots, 1.0, atol=1e-12)


def test_matrix_quaternion_roundtrip_preserves_positive_yaw() -> None:
    # Given
    quaternion = np.array([[0.0, np.sin(0.2), 0.0, np.cos(0.2)]])

    # When
    restored = matrix_to_xyzw(xyzw_to_matrix(quaternion))

    # Then
    np.testing.assert_allclose(restored, quaternion, atol=1e-12)


def test_resampling_is_timestamp_aware_when_source_is_irregular() -> None:
    # Given
    source_time = np.array([0.0, 0.1, 0.4])
    values = np.array([[0.0], [1.0], [4.0]])
    target_time = np.array([0.0, 0.2, 0.4])

    # When
    result = resample_linear(values, source_time, target_time)

    # Then
    np.testing.assert_allclose(result[:, 0], [0.0, 2.0, 4.0])


def test_slerp_uses_short_arc_when_quaternions_are_antipodes() -> None:
    # Given
    source_time = np.array([0.0, 1.0])
    quaternions = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]])

    # When
    result = resample_quaternion_xyzw(quaternions, source_time, np.array([0.5]))

    # Then
    np.testing.assert_allclose(result, [[0.0, 0.0, 0.0, 1.0]], atol=1e-12)


@pytest.mark.parametrize("fps", [0.0, -20.0, np.nan])
def test_timeline_rejects_invalid_fps(fps: float) -> None:
    # Given
    source_time = np.array([0.0, 1.0])

    # When / Then
    with pytest.raises(TimelineError):
        _ = target_timestamps(source_time, fps)


def test_resampling_rejects_non_increasing_timestamps() -> None:
    # Given
    source_time = np.array([0.0, 0.2, 0.2])

    # When / Then
    with pytest.raises(TimelineError):
        _ = resample_linear(
            np.zeros((3, 1), dtype=np.float64),
            source_time,
            np.array([0.1]),
        )
