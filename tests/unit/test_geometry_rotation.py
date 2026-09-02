from __future__ import annotations

import numpy as np

from hri_ttr.geometry.rotation import (
    matrix_to_rotation_6d,
    matrix_to_rotvec,
    rotation_6d_to_matrix,
    rotvec_to_matrix,
)


def test_rotation_6d_roundtrip_when_matrix_is_proper() -> None:
    # Given
    matrix = rotvec_to_matrix(np.array([[0.2, -0.4, 0.1]]))

    # When
    restored = rotation_6d_to_matrix(matrix_to_rotation_6d(matrix))

    # Then
    np.testing.assert_allclose(restored, matrix, atol=1e-12)


def test_rotation_vector_roundtrip_when_angle_is_pi() -> None:
    # Given
    rotation = np.array([[np.pi, 0.0, 0.0]])

    # When
    restored_matrix = rotvec_to_matrix(matrix_to_rotvec(rotvec_to_matrix(rotation)))

    # Then
    np.testing.assert_allclose(restored_matrix, rotvec_to_matrix(rotation), atol=1e-12)
