from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from hri_ttr.representations.g1.features import (
    G1MotionInput,
    G1RepresentationError,
    decode_g1_features,
    encode_g1_features,
    velocity_integration_diagnostic,
)
from hri_ttr.representations.g1.schema import G1_FEATURE_SLICES


def _xyzw_from_yaw(yaw: np.ndarray) -> np.ndarray:
    result = np.zeros((len(yaw), 4), dtype=np.float64)
    result[:, 1] = np.sin(yaw / 2.0)
    result[:, 3] = np.cos(yaw / 2.0)
    return result


def _motion(frames: int = 9) -> G1MotionInput:
    time = np.arange(frames, dtype=np.float64) / 20.0
    root = np.empty((frames, 3), dtype=np.float64)
    root[:, 0] = 0.3 + time
    root[:, 1] = 0.78 + 0.02 * time
    root[:, 2] = -0.4 + 0.5 * time
    rotation = _xyzw_from_yaw(0.4 + 1.7 * time)
    dof = np.sin(time[:, None] * np.arange(1, 30, dtype=np.float64)[None, :]) * 0.1
    contact = np.empty((frames, 2), dtype=np.float64)
    contact[:, 0] = (np.arange(frames) % 2) == 0
    contact[:, 1] = (np.arange(frames) % 2) == 1
    return G1MotionInput(root, rotation, dof, contact)


def test_g1_75d_roundtrip_preserves_authoritative_root_and_dofs() -> None:
    # Given: translating, rotating, articulated G1 motion in InteractionWorld.
    motion = _motion()

    # When: it is encoded to 75D and decoded from explicit pose channels.
    encoded = encode_g1_features(motion, fps=20.0, quaternion_convention="xyzw")
    decoded = decode_g1_features(encoded.features, encoded.anchor)

    # Then: root pose and all 29 DoFs roundtrip at numerical precision.
    np.testing.assert_allclose(
        decoded.root_pos_interaction_m, motion.root_pos_interaction_m, atol=1e-10
    )
    np.testing.assert_allclose(decoded.dof_pos_rad, motion.dof_pos_rad, atol=1e-12)
    dots = np.abs(
        np.sum(
            decoded.root_rot_interaction_xyzw * motion.root_rot_interaction_xyzw, axis=1
        )
    )
    np.testing.assert_allclose(dots, 1.0, atol=1e-10)
    np.testing.assert_array_equal(decoded.foot_contact_lr, motion.foot_contact_lr)
    diagnostic = velocity_integration_diagnostic(encoded.features, fps=20.0)
    assert diagnostic.max_position_error_m <= 1e-12


def test_g1_derivatives_are_causal_and_frame_zero_is_zero() -> None:
    # Given: a sequence whose final frame is changed drastically.
    original = _motion()
    changed_root = original.root_pos_interaction_m.copy()
    changed_root[-1] += 100.0
    changed_dof = original.dof_pos_rad.copy()
    changed_dof[-1] += 2.0
    changed = G1MotionInput(
        changed_root,
        original.root_rot_interaction_xyzw,
        changed_dof,
        original.foot_contact_lr,
    )

    # When: both sequences are encoded using backward-only differences.
    before = encode_g1_features(original, fps=20.0, quaternion_convention="xyzw")
    after = encode_g1_features(changed, fps=20.0, quaternion_convention="xyzw")

    # Then: prior features are unchanged and every derivative starts at zero.
    np.testing.assert_array_equal(before.features[:-1], after.features[:-1])
    derivative = slice(38, 73)
    np.testing.assert_array_equal(before.features[0, derivative], np.zeros(35))


def test_decoder_does_not_overwrite_explicit_root_with_velocity() -> None:
    # Given: valid encoded motion with deliberately corrupted velocity targets.
    encoded = encode_g1_features(_motion(), fps=20.0, quaternion_convention="xyzw")
    features = encoded.features.copy()
    features[:, G1_FEATURE_SLICES["root_linear_vel_local_m_s"]] = 1_000.0

    # When: the representation is decoded.
    decoded = decode_g1_features(features, encoded.anchor)

    # Then: explicit root position remains authoritative.
    np.testing.assert_allclose(
        decoded.root_pos_interaction_m, _motion().root_pos_interaction_m, atol=1e-10
    )
    diagnostic = velocity_integration_diagnostic(features, fps=20.0)
    assert diagnostic.max_position_error_m > 100.0


def test_high_angular_motion_uses_causal_shortest_arc_velocity() -> None:
    # Given: constant 2.8-radian yaw steps, close to the principal-angle boundary.
    frames = 4
    yaw = np.arange(frames, dtype=np.float64) * 2.8
    base = _motion(frames)
    motion = G1MotionInput(
        base.root_pos_interaction_m,
        _xyzw_from_yaw(yaw),
        base.dof_pos_rad,
        base.foot_contact_lr,
    )

    # When: causal angular velocity is encoded at 20 FPS.
    encoded = encode_g1_features(motion, fps=20.0, quaternion_convention="xyzw")
    angular = encoded.features[:, G1_FEATURE_SLICES["root_angular_vel_local_rad_s"]]

    # Then: frame zero is zero and each later Y-axis velocity is 2.8 * 20.
    np.testing.assert_allclose(angular[:1], np.zeros((1, 3)), atol=1e-12)
    np.testing.assert_allclose(angular[1:, 1], 56.0, atol=1e-10)


@pytest.mark.parametrize(
    ("motion", "convention"),
    [
        (
            G1MotionInput(
                np.zeros((2, 2)), np.zeros((2, 4)), np.zeros((2, 29)), np.zeros((2, 2))
            ),
            "xyzw",
        ),
        (
            G1MotionInput(
                np.zeros((2, 3)),
                np.full((2, 4), np.nan),
                np.zeros((2, 29)),
                np.zeros((2, 2)),
            ),
            "xyzw",
        ),
        (
            G1MotionInput(
                np.zeros((2, 3)), np.ones((2, 4)), np.zeros((2, 29)), np.zeros((2, 2))
            ),
            "xyzw",
        ),
        (_motion(2), "wxyz"),
    ],
)
def test_g1_encoder_rejects_malformed_motion(
    motion: G1MotionInput,
    convention: Literal["xyzw", "wxyz"],
) -> None:
    # Given: malformed shapes, values, quaternion norms, or convention.
    # When/Then: the representation boundary rejects the input.
    with pytest.raises(G1RepresentationError):
        _ = encode_g1_features(motion, fps=20.0, quaternion_convention=convention)
