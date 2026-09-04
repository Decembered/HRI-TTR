from __future__ import annotations

import hashlib
from itertools import pairwise

import numpy as np
import pytest

from hri_ttr.representations.human.features import (
    HUMAN_262_LAYOUT,
    denormalize_single_joints22,
    human_space_states,
    joints22_to_human262,
    normalize_single_joints22,
)
from hri_ttr.representations.human.normalizer import (
    HumanFeatureNormalizer,
    NormalizerError,
)

RAW_OFFSETS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
    ],
    dtype=np.float64,
)
CHAINS = (
    (0, 2, 5, 8, 11),
    (0, 1, 4, 7, 10),
    (0, 3, 6, 9, 12, 15),
    (9, 14, 17, 19, 21),
    (9, 13, 16, 18, 20),
)


def synthetic_motion(frames: int = 5) -> np.ndarray:
    base = np.zeros((22, 3), dtype=np.float64)
    for chain in CHAINS:
        for parent, child in pairwise(chain):
            base[child] = base[parent] + RAW_OFFSETS[child]
    base[[14, 17, 19, 21], 0] -= 0.25
    time = np.arange(frames, dtype=np.float64)
    motion = np.empty((frames, 22, 3), dtype=np.float64)
    motion[:] = base
    motion += np.array([3.0, 1.5, -2.0])[None, None, :]
    trajectory = np.zeros((frames, 3), dtype=np.float64)
    trajectory[:, 0] = 0.1 * time
    trajectory[:, 2] = 0.05 * time
    motion += trajectory[:, None]
    return motion


def test_human262_matches_pinned_ttr_golden() -> None:
    # Given: digest from TTR commit
    # 9b7e395f740a68cbd30c027b4952dedb0ebf8b6d.
    normalized = normalize_single_joints22(synthetic_motion()).joints

    # When
    features = joints22_to_human262(normalized)

    # Then
    rounded = np.round(features.astype(np.float64), 6)
    assert hashlib.sha256(rounded.tobytes()).hexdigest() == (
        "c6d26842f5566fca0625dc7f8d9502c8467dece84522f8d7df95e7febbc92b80"
    )


def test_human262_has_historical_layout_and_tail_repetition() -> None:
    # Given
    normalized = normalize_single_joints22(synthetic_motion()).joints

    # When
    features = joints22_to_human262(normalized)

    # Then
    assert features.shape == (5, 262)
    np.testing.assert_allclose(
        features[-1, HUMAN_262_LAYOUT.velocity], features[-2, HUMAN_262_LAYOUT.velocity]
    )
    np.testing.assert_array_equal(features[:, HUMAN_262_LAYOUT.contact], 0.0)


def test_human262_marks_stationary_feet_as_contact() -> None:
    # Given
    motion = synthetic_motion()
    motion[:] = motion[:1]
    normalized = normalize_single_joints22(motion).joints

    # When
    features = joints22_to_human262(normalized)

    # Then
    np.testing.assert_array_equal(features[:, HUMAN_262_LAYOUT.contact], 1.0)


def test_normalization_roundtrip_and_space_state() -> None:
    # Given
    motion = synthetic_motion()

    # When
    normalized = normalize_single_joints22(motion)
    restored = denormalize_single_joints22(normalized.joints, normalized.space)
    space = human_space_states(motion)

    # Then
    floor_height = -0.5
    np.testing.assert_allclose(
        restored, motion - np.array([0.0, floor_height, 0.0]), atol=1e-6
    )
    np.testing.assert_allclose(space[:1], [[3.0, -2.0, np.pi / 2.0]], atol=1e-7)


def test_feature_normalizer_roundtrip_and_contact_passthrough() -> None:
    # Given
    features = joints22_to_human262(
        normalize_single_joints22(synthetic_motion()).joints
    )
    mean = np.zeros(262, dtype=np.float32)
    std = np.full(262, 2.0, dtype=np.float32)
    normalizer = HumanFeatureNormalizer.create(
        mean, std, source_dataset="synthetic", fps=20.0
    )

    # When
    normalized = normalizer.normalize(features)
    restored = normalizer.denormalize(normalized)

    # Then
    np.testing.assert_allclose(restored, features, atol=1e-6)
    np.testing.assert_array_equal(
        normalized[:, HUMAN_262_LAYOUT.contact], features[:, HUMAN_262_LAYOUT.contact]
    )


@pytest.mark.parametrize("fps", [np.nan, np.inf, -np.inf])
def test_feature_normalizer_rejects_non_finite_fps(fps: float) -> None:
    # Given
    mean = np.zeros(262, dtype=np.float32)
    std = np.ones(262, dtype=np.float32)

    # When / Then
    with pytest.raises(NormalizerError):
        _ = HumanFeatureNormalizer.create(
            mean, std, source_dataset="synthetic", fps=fps
        )
