"""Canonical G1 to SONIC export command."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 - Typer resolves runtime annotations.
from typing import Annotated

import typer

from hri_ttr.commands.common import load_prepared
from hri_ttr.representations.g1 import EpisodeFrame
from hri_ttr.sonic import build_sonic_motion, save_sonic_motion

app = typer.Typer(no_args_is_help=True)


@app.command("sonic")
def export_sonic(
    prepared: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Prepared pair NPZ.")
    ],
    output: Annotated[Path, typer.Option(help="Output SONIC NPZ payload.")],
) -> None:
    """Emit root_trans_offset, root_rot, dof, pose_aa, and FPS arrays."""
    motion = load_prepared(prepared)
    valid = motion.metadata.valid_frames
    anchor = EpisodeFrame(motion.anchor_origin, motion.anchor_basis)
    sonic = build_sonic_motion(
        motion.g1_features[:valid], anchor, fps=motion.metadata.fps
    )
    save_sonic_motion(sonic, output)
    typer.echo(json.dumps({"output": str(output), "frames": valid}, sort_keys=True))
