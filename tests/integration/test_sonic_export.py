from __future__ import annotations

from typing import TYPE_CHECKING
from zipfile import ZipFile

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path
from hri_ttr.representations.g1.features import G1MotionInput, encode_g1_features
from hri_ttr.sonic.export import build_sonic_motion, save_sonic_motion


def test_sonic_export_emits_native_z_up_xyzw_payload(tmp_path: Path) -> None:
    # Given: simple Y-up InteractionWorld motion and a 29-DoF pose.
    frames = 5
    root = np.empty((frames, 3), dtype=np.float64)
    root[:, 0] = np.linspace(0.0, 0.4, frames)
    root[:, 1] = 0.8
    root[:, 2] = np.linspace(0.0, 0.2, frames)
    rotation = np.zeros((frames, 4), dtype=np.float64)
    rotation[:, 3] = 1.0
    dof = np.linspace(-0.1, 0.1, frames * 29).reshape(frames, 29)
    contacts = np.zeros((frames, 2), dtype=np.float64)
    encoded = encode_g1_features(
        G1MotionInput(root, rotation, dof, contacts),
        fps=20.0,
        quaternion_convention="xyzw",
    )

    # When: canonical motion is converted and persisted for SONIC.
    sonic = build_sonic_motion(encoded.features, encoded.anchor, fps=20.0)
    output = tmp_path / "sonic_motion.npz"
    save_sonic_motion(sonic, output)

    # Then: native fields, shapes, convention, and axis-angle values are exact.
    assert sonic.root_trans_offset.shape == (frames, 3)
    assert sonic.root_rot_xyzw.shape == (frames, 4)
    assert sonic.dof.shape == (frames, 29)
    assert sonic.pose_aa.shape == (frames, 30, 3)
    assert sonic.fps == 20.0
    np.testing.assert_allclose(
        np.sqrt(np.sum(sonic.root_rot_xyzw**2, axis=1)), 1.0, atol=1e-12
    )
    np.testing.assert_allclose(sonic.pose_aa[:, 0], 0.0, atol=1e-12)
    expected_native_position = np.empty_like(root)
    expected_native_position[:, 0] = root[:, 0]
    expected_native_position[:, 1] = -root[:, 2]
    expected_native_position[:, 2] = root[:, 1]
    np.testing.assert_allclose(
        sonic.root_trans_offset, expected_native_position, atol=1e-12
    )
    np.testing.assert_allclose(
        sonic.pose_aa[:, 1:], dof[:, :, None] * sonic.dof_axes[None], atol=1e-12
    )
    with ZipFile(output) as archive:
        expected_names = {
            "root_trans_offset.npy",
            "root_rot.npy",
            "dof.npy",
            "pose_aa.npy",
            "fps.npy",
        }
        assert set(archive.namelist()) == expected_names
        assert all(archive.getinfo(name).file_size > 0 for name in expected_names)
