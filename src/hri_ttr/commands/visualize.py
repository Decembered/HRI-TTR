"""Ground-truth versus reconstruction visualization command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import typer

from hri_ttr.checkpoints.kinds import ModelKind
from hri_ttr.commands.common import load_model, load_prepared
from hri_ttr.visualization import (
    render_feature_comparison,
    render_trajectory_comparison,
)

app = typer.Typer(no_args_is_help=True)
ExistingFile = Annotated[Path, typer.Option(exists=True, dir_okay=False)]


@app.command("reconstruction")
def visualize_reconstruction(
    prepared: ExistingFile,
    config: ExistingFile,
    checkpoint: ExistingFile,
    output: Annotated[Path, typer.Option(help="Output PNG path.")],
) -> None:
    """Render observed features against one checkpoint reconstruction."""
    motion = load_prepared(prepared)
    model, train_config = load_model(config, checkpoint)
    features = (
        motion.human_features
        if train_config.model_kind is ModelKind.HUMAN
        else motion.g1_features.astype(np.float32)
    )
    mask = (
        motion.human_mask
        if train_config.model_kind is ModelKind.HUMAN
        else motion.g1_mask
    )
    with torch.no_grad():
        prediction = (
            model.forward(
                torch.as_tensor(features, dtype=torch.float32)[None],
                torch.as_tensor(mask, dtype=torch.bool)[None],
            )
            .reconstruction[0]
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    valid = motion.metadata.valid_frames
    if train_config.model_kind is ModelKind.G1:
        _ = render_trajectory_comparison(
            features[:valid, :3].astype(np.float64), prediction[:valid, :3], output
        )
    else:
        _ = render_feature_comparison(
            features[:valid, :3].astype(np.float64), prediction[:valid, :3], output
        )
    typer.echo(json.dumps({"output": str(output), "bytes": output.stat().st_size}))
