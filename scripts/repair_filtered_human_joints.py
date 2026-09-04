"""Undo the historical HumanML3D joint-resampling reshape bug.

The old HRI-Datasets resampler interpolated local joints in ``[xyz][joint]``
order and then reshaped as ``[joint][xyz]``.  This tool reverses that exact
per-frame permutation using only the frozen filtered pickle's ``poses``,
``trans`` and ``joints`` fields.  It never reads Bone-Seed or another motion
source.

The command is intentionally opt-in.  Use ``--apply`` only after choosing a
backup directory; the original pickle files are copied there before they are
replaced atomically.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Final

import numpy as np

JOINT_COUNT = 22
AXIS_DIM = 3
SMALL_ANGLE: Final = 1e-10


def _rotation_matrices(rotvec: np.ndarray) -> np.ndarray:
    """Return batched Rodrigues matrices for axis-angle vectors."""
    angles = np.linalg.norm(rotvec, axis=1)
    axes = rotvec / np.maximum(angles[:, None], 1e-12)
    x, y, z = axes.T
    skew = np.zeros((len(rotvec), AXIS_DIM, AXIS_DIM), dtype=np.float64)
    skew[:, 0, 1] = -z
    skew[:, 0, 2] = y
    skew[:, 1, 0] = z
    skew[:, 1, 2] = -x
    skew[:, 2, 0] = -y
    skew[:, 2, 1] = x
    identity = np.eye(AXIS_DIM, dtype=np.float64)[None]
    cosine = np.cos(angles)[:, None, None]
    sine = np.sin(angles)[:, None, None]
    result = (
        cosine * identity
        + (1.0 - cosine) * axes[:, :, None] * axes[:, None, :]
        + sine * skew
    )
    result[angles < SMALL_ANGLE] = identity
    return result


def undo_resample_reshape_bug(
    joints: np.ndarray, poses: np.ndarray, trans: np.ndarray
) -> np.ndarray:
    """Recover the correctly ordered joints from the historical output."""
    values = np.asarray(joints, dtype=np.float64)
    pose_values = np.asarray(poses, dtype=np.float64)
    translations = np.asarray(trans, dtype=np.float64)
    if (
        values.shape != (len(values), JOINT_COUNT, AXIS_DIM)
        or pose_values.shape != (len(values), 66)
        or translations.shape != (len(values), AXIS_DIM)
    ):
        detail = "expected joints [T,22,3], poses [T,66], trans [T,3]"
        raise ValueError(detail)
    rotations = _rotation_matrices(pose_values[:, :AXIS_DIM])
    local_bug = np.einsum("tji,tkj->tki", rotations, values - translations[:, None, :])
    flattened = local_bug.reshape(len(values), JOINT_COUNT * AXIS_DIM)
    local = np.stack(
        [
            flattened[:, offset * JOINT_COUNT : (offset + 1) * JOINT_COUNT]
            for offset in range(AXIS_DIM)
        ],
        axis=-1,
    )
    return np.einsum("tij,tkj->tki", rotations, local) + translations[:, None, :]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered-root", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    return parser


def main() -> None:
    """Validate all files, optionally back them up, and repair them."""
    args = _parser().parse_args()
    source_dir = args.filtered_root / "humanl3d" / "human"
    files = sorted(source_dir.glob("*.pkl"))
    if not files:
        detail = f"no HumanML3D pickle files under {source_dir}"
        raise SystemExit(detail)
    if args.apply and args.backup_root.exists() and any(args.backup_root.iterdir()):
        detail = f"backup directory must be absent or empty: {args.backup_root}"
        raise SystemExit(detail)

    repaired: list[tuple[Path, dict[str, object]]] = []
    for path in files:
        with path.open("rb") as stream:
            payload = pickle.load(stream)  # noqa: S301 - local trusted pickle
        motion = payload["motion"]
        fixed = undo_resample_reshape_bug(
            np.asarray(motion["joints"]),
            np.asarray(motion["poses"]),
            np.asarray(motion["trans"]),
        ).astype(np.asarray(motion["joints"]).dtype, copy=False)
        repaired.append(
            (
                path,
                {
                    "path": str(path),
                    "frames": len(fixed),
                    "max_abs_change_m": float(
                        np.max(np.abs(fixed.astype(np.float64) - motion["joints"]))
                    ),
                },
            )
        )
        if args.apply:
            backup = args.backup_root / path.name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            motion["joints"] = fixed
            temporary = path.with_suffix(".repair.tmp.pkl")
            with temporary.open("wb") as stream:
                pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(path)

    report = {
        "dataset": "HumanML3D",
        "files": len(repaired),
        "apply": args.apply,
        "backup_root": str(args.backup_root) if args.apply else None,
        "max_abs_change_m": max(
            float(item["max_abs_change_m"]) for _, item in repaired
        ),
    }
    if args.apply:
        args.backup_root.mkdir(parents=True, exist_ok=True)
        (args.backup_root / "repair_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    _ = sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
