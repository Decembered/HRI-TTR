from __future__ import annotations

import numpy as np
import pytest

from hri_ttr.representations.g1.contacts import (
    G1ContactError,
    G1FootContactThresholds,
    compute_g1_foot_contacts,
)


def _joints(frames: int) -> np.ndarray:
    joints = np.zeros((frames, 20, 3), dtype=np.float64)
    joints[:, :, 1] = 0.5
    joints[:, (4, 8), 1] = 0.0
    return joints


def test_stationary_feet_are_both_in_contact() -> None:
    # Given: both feet are stationary on the estimated floor.
    joints = _joints(5)

    # When: contact is derived with the default verified thresholds.
    contact = compute_g1_foot_contacts(joints, fps=20.0)

    # Then: every left/right frame is a binary positive contact.
    np.testing.assert_array_equal(contact, np.ones((5, 2), dtype=np.float64))


def test_moving_left_foot_is_not_in_contact_after_first_frame() -> None:
    # Given: left foot travels 0.1 metres per frame while right foot is stationary.
    joints = _joints(5)
    joints[:, 4, 0] = np.arange(5, dtype=np.float64) * 0.1

    # When: causal backward speed is used for contact.
    contact = compute_g1_foot_contacts(joints, fps=20.0)

    # Then: frame zero has no past velocity and later moving-left contacts are false.
    np.testing.assert_array_equal(contact[:, 0], (1.0, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_array_equal(contact[:, 1], np.ones(5))


def test_height_threshold_controls_elevated_stationary_foot() -> None:
    # Given: left foot is 0.1 metres above a floor fixed by the right foot.
    joints = _joints(100)
    joints[:, 4, 1] = 0.1

    # When: contacts are computed using strict and permissive height thresholds.
    strict = compute_g1_foot_contacts(joints, fps=20.0)
    permissive = compute_g1_foot_contacts(
        joints,
        fps=20.0,
        thresholds=G1FootContactThresholds(height_m=0.11, speed_m_s=0.25),
    )

    # Then: only the explicitly raised threshold labels the left foot as contact.
    assert not strict[:, 0].any()
    assert permissive[:, 0].all()


@pytest.mark.parametrize(
    ("joints", "fps"),
    [
        (np.zeros((2, 19, 3), dtype=np.float64), 20.0),
        (np.full((2, 20, 3), np.nan, dtype=np.float64), 20.0),
        (np.zeros((0, 20, 3), dtype=np.float64), 20.0),
        (np.zeros((2, 20, 3), dtype=np.float64), 0.0),
    ],
)
def test_contact_derivation_rejects_malformed_input(
    joints: np.ndarray,
    fps: float,
) -> None:
    # Given: malformed shape, non-finite values, empty input, or non-positive FPS.
    # When/Then: the public derivation boundary rejects it instead of returning zeros.
    with pytest.raises(G1ContactError):
        _ = compute_g1_foot_contacts(joints, fps=fps)


def test_contact_thresholds_reject_nonpositive_values() -> None:
    # Given/When/Then: an invalid physical threshold cannot be constructed.
    with pytest.raises(G1ContactError):
        _ = G1FootContactThresholds(height_m=0.0, speed_m_s=0.25)
