"""Convert canonical G1 motion into SONIC's native Z-up payload."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import numpy.typing as npt

from hri_ttr.geometry.coordinates import (
    interaction_y_up_to_g1_z_up,
    quaternion_xyzw_interaction_to_g1,
)
from hri_ttr.geometry.quaternion import xyzw_to_matrix
from hri_ttr.geometry.rotation import matrix_to_rotvec
from hri_ttr.representations.g1.constants import G1_DOF_AXES, G1_DOF_COUNT
from hri_ttr.representations.g1.features import decode_g1_features

if TYPE_CHECKING:
    from pathlib import Path

    from hri_ttr.representations.g1.episode import EpisodeFrame


@dataclass(frozen=True, slots=True)
class SonicMotion:
    """Deployment-ready G1 arrays in native Z-up and scalar-last convention."""

    root_trans_offset: npt.NDArray[np.float64]
    root_rot_xyzw: npt.NDArray[np.float64]
    dof: npt.NDArray[np.float64]
    pose_aa: npt.NDArray[np.float64]
    fps: float

    @property
    def dof_axes(self) -> npt.NDArray[np.float64]:
        """Return the verified actuator axes used to construct ``pose_aa``."""
        return np.asarray(G1_DOF_AXES, dtype=np.float64)


def build_sonic_motion(
    features: npt.NDArray[np.float64],
    anchor: EpisodeFrame,
    *,
    fps: float,
) -> SonicMotion:
    """Decode authoritative root motion and change basis for SONIC."""
    decoded = decode_g1_features(features, anchor)
    root_position = interaction_y_up_to_g1_z_up(decoded.root_pos_interaction_m)
    root_rotation = quaternion_xyzw_interaction_to_g1(decoded.root_rot_interaction_xyzw)
    pose = np.zeros((len(features), G1_DOF_COUNT + 1, 3), dtype=np.float64)
    pose[:, 0] = matrix_to_rotvec(xyzw_to_matrix(root_rotation))
    pose[:, 1:] = (
        decoded.dof_pos_rad[:, :, None]
        * np.asarray(G1_DOF_AXES, dtype=np.float64)[None, :, :]
    )
    return SonicMotion(root_position, root_rotation, decoded.dof_pos_rad, pose, fps)


def save_sonic_motion(motion: SonicMotion, path: Path) -> None:
    """Persist the five SONIC fields as a non-pickle NumPy archive."""
    arrays = (
        ("root_trans_offset", motion.root_trans_offset),
        ("root_rot", motion.root_rot_xyzw),
        ("dof", motion.dof),
        ("pose_aa", motion.pose_aa),
        ("fps", np.asarray(motion.fps, dtype=np.float64)),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, mode="w", compression=ZIP_DEFLATED) as archive:
        for name, values in arrays:
            buffer = BytesIO()
            np.lib.format.write_array(buffer, values, allow_pickle=False)
            archive.writestr(f"{name}.npy", buffer.getvalue())
