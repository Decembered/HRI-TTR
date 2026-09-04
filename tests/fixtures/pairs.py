"""Deterministic synthetic retargeted-pair writer for CLI tests."""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, Literal

import joblib
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

OFFSETS = np.array(
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
    dtype=np.float32,
)
CHAINS = (
    (0, 2, 5, 8, 11),
    (0, 1, 4, 7, 10),
    (0, 3, 6, 9, 12, 15),
    (9, 14, 17, 19, 21),
    (9, 13, 16, 18, 20),
)


def write_synthetic_pair(
    root: Path,
    frames: int = 8,
    fps: float = 20.0,
    source_format: Literal["npz", "pkl"] = "npz",
    reactor_fps: float | None = None,
) -> str:
    """Write one valid 20 FPS pair using only generated numeric motion."""
    sequence_id = "G001T001A001R001"
    base = np.zeros((24, 3), dtype=np.float32)
    for chain in CHAINS:
        for parent, child in pairwise(chain):
            base[child] = base[parent] + OFFSETS[child]
    base[[14, 17, 19, 21], 0] -= 0.25
    joints = np.empty((frames, 24, 3), dtype=np.float32)
    joints[:] = base
    joints += np.array([0.1, 2.0, -0.2], dtype=np.float32)
    joints[:, :, 0] += np.arange(frames, dtype=np.float32)[:, None] * 0.01
    actor_root = joints[:, 0].copy()
    identity = np.zeros((frames, 4), dtype=np.float32)
    identity[:, 3] = 1.0
    reactor_root = np.zeros((frames, 3), dtype=np.float32)
    reactor_root[:, 0] = np.arange(frames, dtype=np.float32) * 0.01
    reactor_root[:, 2] = 0.8
    reactor_joints = np.zeros((frames, 20, 3), dtype=np.float32)
    reactor_joints[:, :, 2] = 0.4
    reactor_joints[:, (4, 8), 2] = 0.0
    root.mkdir(parents=True, exist_ok=True)
    actor = {
        "fps": fps,
        "root_pos": actor_root,
        "root_rot": identity,
        "pose_body": np.zeros((frames, 69), dtype=np.float32),
        "joints_pos": joints,
        "betas": np.zeros(10, dtype=np.float32),
        "gender": "neutral",
    }
    reactor = {
        "fps": fps if reactor_fps is None else reactor_fps,
        "root_pos": reactor_root,
        "root_rot": identity,
        "dof_pos": np.zeros((frames, 29), dtype=np.float32),
        "joints_pos": reactor_joints,
    }
    match source_format:
        case "npz":
            np.savez_compressed(
                root / f"{sequence_id}_actor.npz",
                fps=np.asarray(actor["fps"]),
                root_pos=actor_root,
                root_rot=identity,
                pose_body=actor["pose_body"],
                joints_pos=joints,
                betas=actor["betas"],
                gender=np.asarray("neutral"),
            )
            np.savez_compressed(
                root / f"{sequence_id}_reactor.npz",
                fps=np.asarray(reactor["fps"]),
                root_pos=reactor_root,
                root_rot=identity,
                dof_pos=reactor["dof_pos"],
                joints_pos=reactor_joints,
            )
        case "pkl":
            _ = joblib.dump(actor, root / f"{sequence_id}_actor.pkl")
            _ = joblib.dump(reactor, root / f"{sequence_id}_reactor.pkl")
    return sequence_id
