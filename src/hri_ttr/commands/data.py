"""Dataset audit and deterministic pair preparation commands."""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 - Typer resolves runtime annotations.
from typing import Annotated, Final

import numpy as np
import typer

from hri_ttr.commands.common import fail
from hri_ttr.data.pairs import (
    PairPaths,
    PickleConsentRequiredError,
    PickleTrustPolicy,
    RetargetedPair,
    discover_pairs,
    load_retargeted_pair,
)
from hri_ttr.geometry.coordinates import (
    g1_z_up_to_interaction_y_up,
    quaternion_xyzw_g1_to_interaction,
)
from hri_ttr.geometry.quaternion import xyzw_to_matrix
from hri_ttr.geometry.resample import (
    resample_linear,
    resample_quaternion_xyzw,
    target_timestamps,
)
from hri_ttr.representations.g1 import (
    G1MotionInput,
    compute_g1_foot_contacts,
    encode_g1_features,
)
from hri_ttr.representations.g1.tail import pad_frames_to_token_multiple
from hri_ttr.representations.human.features import (
    human_space_states,
    joints22_to_human262,
    normalize_single_joints22,
)

app = typer.Typer(no_args_is_help=True)
INPUT_FPS: Final = 20.0
MINIMUM_TARGET_FRAMES: Final = 2
PICKLE_WARNING: Final = (
    "Pickle can execute code. Permit only the trusted local Stage0 corpus."
)


def _pickle_policy(allow_trusted_pickle: bool) -> PickleTrustPolicy:
    return (
        PickleTrustPolicy.TRUSTED_LOCAL
        if allow_trusted_pickle
        else PickleTrustPolicy.DENY
    )


def _selected_pair(root: Path, sequence_id: str | None) -> PairPaths:
    discovery = discover_pairs(root)
    if discovery.actor_only or discovery.reactor_only:
        fail("dataset contains unmatched actor/reactor files")
    matches = (
        discovery.pairs
        if sequence_id is None
        else tuple(item for item in discovery.pairs if item.sequence_id == sequence_id)
    )
    if len(matches) != 1:
        fail("select exactly one paired sequence with --sequence-id")
    return matches[0]


def _load_pair(paths: PairPaths, *, allow_trusted_pickle: bool) -> RetargetedPair:
    try:
        return load_retargeted_pair(
            paths,
            pickle_policy=_pickle_policy(allow_trusted_pickle),
        )
    except PickleConsentRequiredError:
        fail(f"{PICKLE_WARNING} Pass --allow-trusted-pickle to consent.")
    except ValueError as error:
        fail(f"motion pair is invalid: {error}", cause=error)


@app.command("audit")
def audit(
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input-dir",
            exists=True,
            file_okay=False,
            readable=True,
            help=(
                "Directory containing paired <ID>_actor/reactor.npz files; legacy "
                ".pkl requires --allow-trusted-pickle."
            ),
        ),
    ],
    allow_trusted_pickle: Annotated[
        bool,
        typer.Option("--allow-trusted-pickle", help=PICKLE_WARNING),
    ] = False,
) -> None:
    """Validate names, pairing, schemas, FPS, and timeline alignment."""
    discovery = discover_pairs(input_dir)
    validated = 0
    source_formats: set[str] = set()
    for paths in discovery.pairs:
        pair = _load_pair(paths, allow_trusted_pickle=allow_trusted_pickle)
        source_formats.add(pair.source_format.value)
        validated += 1
    typer.echo(
        json.dumps(
            {
                "paired": len(discovery.pairs),
                "validated": validated,
                "actor_only": list(discovery.actor_only),
                "reactor_only": list(discovery.reactor_only),
                "source_provenance": sorted(source_formats),
            },
            sort_keys=True,
        )
    )
    if discovery.actor_only or discovery.reactor_only or validated == 0:
        fail("audit requires at least one complete pair and no unmatched files")


@app.command("prepare")
def prepare(
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input-dir",
            exists=True,
            file_okay=False,
            readable=True,
            help=(
                "Directory containing safe paired NPZ files; legacy .pkl requires "
                "--allow-trusted-pickle because pickle can execute code."
            ),
        ),
    ],
    output: Annotated[Path, typer.Option("--output", help="Output prepared NPZ.")],
    sequence_id: Annotated[
        str | None,
        typer.Option(help="Pair ID; optional only when the directory has one pair."),
    ] = None,
    allow_trusted_pickle: Annotated[
        bool,
        typer.Option("--allow-trusted-pickle", help=PICKLE_WARNING),
    ] = False,
) -> None:
    """Create aligned Human 262D, G1 75D, space, mask, and anchor arrays."""
    pair = _load_pair(
        _selected_pair(input_dir, sequence_id),
        allow_trusted_pickle=allow_trusted_pickle,
    )
    source_fps = pair.actor.fps
    source_time = np.arange(len(pair.actor.root_pos), dtype=np.float64) / source_fps
    target_time = target_timestamps(source_time, INPUT_FPS)
    if len(target_time) < MINIMUM_TARGET_FRAMES:
        fail("source timeline is too short to produce two frames at 20 FPS")
    joints22 = resample_linear(
        pair.actor.joints_pos[:, :22].astype(np.float64), source_time, target_time
    ).astype(np.float32)
    normalized = normalize_single_joints22(joints22)
    human_features = joints22_to_human262(normalized.joints)
    root_native = resample_linear(
        pair.reactor.root_pos.astype(np.float64), source_time, target_time
    )
    rotation_native = resample_quaternion_xyzw(
        pair.reactor.root_rot.astype(np.float64), source_time, target_time
    )
    dof = resample_linear(
        pair.reactor.dof_pos.astype(np.float64), source_time, target_time
    )
    joints_native = resample_linear(
        pair.reactor.joints_pos.astype(np.float64), source_time, target_time
    )
    root = g1_z_up_to_interaction_y_up(root_native)
    rotation = quaternion_xyzw_g1_to_interaction(rotation_native)
    joints20 = g1_z_up_to_interaction_y_up(joints_native)
    encoded = encode_g1_features(
        G1MotionInput(
            root,
            rotation,
            dof,
            compute_g1_foot_contacts(joints20, fps=INPUT_FPS),
        ),
        fps=INPUT_FPS,
        quaternion_convention="xyzw",
    )
    human_episode = encoded.anchor.positions_to_episode(
        joints22.astype(np.float64).reshape(-1, 3)
    ).reshape(joints22.shape)
    human_space = human_space_states(human_episode.astype(np.float32))
    g1_positions = encoded.features[:, :3]
    g1_rotations = encoded.anchor.rotations_to_episode(xyzw_to_matrix(rotation))
    g1_space = np.empty((len(root), 3), dtype=np.float64)
    g1_space[:, :2] = g1_positions[:, (0, 2)]
    g1_space[:, 2] = np.arctan2(g1_rotations[:, 2, 0], g1_rotations[:, 0, 0])
    padded_human = pad_frames_to_token_multiple(human_features, frames_per_token=4)
    padded_g1 = pad_frames_to_token_multiple(
        encoded.features.astype(np.float32), frames_per_token=4
    )
    padded_human_space = pad_frames_to_token_multiple(human_space, frames_per_token=4)
    padded_g1_space = pad_frames_to_token_multiple(
        g1_space.astype(np.float32), frames_per_token=4
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        sequence_id=np.asarray(pair.sequence_id),
        human_features=padded_human.features,
        g1_features=padded_g1.features.astype(np.float64),
        human_space=padded_human_space.features,
        g1_space=padded_g1_space.features.astype(np.float64),
        human_mask=padded_human.valid_mask,
        g1_mask=padded_g1.valid_mask,
        anchor_origin=encoded.anchor.origin_interaction_m,
        anchor_basis=encoded.anchor.episode_to_interaction,
        fps=np.asarray(INPUT_FPS, dtype=np.float64),
        source_fps=np.asarray(source_fps, dtype=np.float64),
        source_format=np.asarray(pair.source_format.value),
        target_fps=np.asarray(INPUT_FPS, dtype=np.float64),
        valid_frames=np.asarray(len(root), dtype=np.int64),
    )
    typer.echo(json.dumps({"output": str(output), "frames": len(root)}, sort_keys=True))
