"""G1 EpisodeFrame anchored to the initial root ground pose."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import StrEnum
from math import sqrt
from typing import Final

import numpy as np
import numpy.typing as npt
from typing_extensions import override

UP_Y: Final = np.array([0.0, 1.0, 0.0], dtype=np.float64)
FORWARD_EPSILON: Final = 1e-8


class EpisodeFrameReason(StrEnum):
    """Stable reasons for rejecting an EpisodeFrame input."""

    POSITION = "initial root position must be finite with shape [3]"
    ROTATION = "initial root rotation must be finite with shape [3,3]"
    FORWARD = "initial root forward direction cannot be vertical"


@dataclass(frozen=True, slots=True)
class EpisodeFrameError(ValueError):
    """Reports an invalid EpisodeFrame anchor."""

    reason: EpisodeFrameReason

    @override
    def __str__(self) -> str:
        """Render the stable boundary failure."""
        return self.reason


@dataclass(frozen=True, slots=True)
class EpisodeFrame:
    """Rigid transform from a sequence-local Y-up frame to InteractionWorld."""

    origin_interaction_m: npt.NDArray[np.float64]
    episode_to_interaction: npt.NDArray[np.float64]

    @classmethod
    def from_initial_root(
        cls,
        root_position_interaction_m: npt.NDArray[np.float64],
        root_rotation_interaction: npt.NDArray[np.float64],
    ) -> EpisodeFrame:
        """Set ground origin and point EpisodeFrame +Z along initial G1 forward."""
        if (
            root_position_interaction_m.shape != (3,)
            or not np.isfinite(root_position_interaction_m).all()
        ):
            raise EpisodeFrameError(EpisodeFrameReason.POSITION)
        if (
            root_rotation_interaction.shape != (3, 3)
            or not np.isfinite(root_rotation_interaction).all()
        ):
            raise EpisodeFrameError(EpisodeFrameReason.ROTATION)
        forward = root_rotation_interaction[:, 0].copy()
        forward[1] = 0.0
        forward_x, _, forward_z = struct.unpack("3d", forward.tobytes())
        length = sqrt(forward_x**2 + forward_z**2)
        if length <= FORWARD_EPSILON:
            raise EpisodeFrameError(EpisodeFrameReason.FORWARD)
        episode_z = forward / length
        episode_x = np.cross(UP_Y, episode_z)
        basis = np.empty((3, 3), dtype=np.float64)
        basis[:, 0] = episode_x
        basis[:, 1] = UP_Y
        basis[:, 2] = episode_z
        origin = root_position_interaction_m.copy()
        origin[1] = 0.0
        return cls(origin, basis)

    def positions_to_episode(
        self, positions_interaction_m: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Transform InteractionWorld positions into this EpisodeFrame."""
        return np.asarray(
            (positions_interaction_m - self.origin_interaction_m)
            @ self.episode_to_interaction,
            dtype=np.float64,
        )

    def positions_to_interaction(
        self, positions_episode_m: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Transform EpisodeFrame positions back into InteractionWorld."""
        return np.asarray(
            positions_episode_m @ self.episode_to_interaction.T
            + self.origin_interaction_m,
            dtype=np.float64,
        )

    def rotations_to_episode(
        self, rotations_interaction: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Express active root rotations in this EpisodeFrame."""
        return np.asarray(
            self.episode_to_interaction.T[None, :, :] @ rotations_interaction,
            dtype=np.float64,
        )

    def rotations_to_interaction(
        self, rotations_episode: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Express EpisodeFrame active rotations in InteractionWorld."""
        return np.asarray(
            self.episode_to_interaction[None, :, :] @ rotations_episode,
            dtype=np.float64,
        )
