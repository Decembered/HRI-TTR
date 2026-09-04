"""Tokenizer reconstruction and causality evaluation command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import typer

from hri_ttr.checkpoints.kinds import ModelKind
from hri_ttr.commands.common import load_model, load_prepared
from hri_ttr.evaluation import (
    codebook_statistics,
    evaluate_g1_reconstruction,
    evaluate_g1_tokenizer_causality,
    evaluate_human_reconstruction,
    evaluate_human_tokenizer_causality,
)
from hri_ttr.tokenizers import G1Tokenizer, HumanTokenizer

app = typer.Typer(no_args_is_help=True)
ExistingFile = Annotated[Path, typer.Option(exists=True, dir_okay=False)]


@app.command("tokenizer")
def evaluate_tokenizer(
    prepared: ExistingFile,
    config: ExistingFile,
    checkpoint: ExistingFile,
    output: Annotated[Path, typer.Option(help="Output metrics JSON.")],
) -> None:
    """Measure one checkpoint without claiming trained reconstruction quality."""
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
    tensor = torch.as_tensor(features, dtype=torch.float32)[None]
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool)[None]
    with torch.no_grad():
        result = model.forward(tensor, mask_tensor)
    target = features.astype(np.float64)
    prediction = result.reconstruction[0].cpu().numpy().astype(np.float64)
    token_ids = result.encoding.token_ids.cpu().numpy().astype(np.int64)
    token_mask = result.encoding.token_mask.cpu().numpy().astype(np.bool_)
    codebook = codebook_statistics(token_ids, token_mask)
    match train_config.model_kind:
        case ModelKind.HUMAN:
            reconstruction = evaluate_human_reconstruction(target, prediction, mask)
            causality = evaluate_human_tokenizer_causality(
                model if isinstance(model, HumanTokenizer) else HumanTokenizer(),
                tensor,
                mask_tensor,
            )
            metric_values = {
                "mpjpe_m": reconstruction.mpjpe_m,
                "root_relative_mpjpe_m": reconstruction.root_relative_mpjpe_m,
            }
        case ModelKind.G1:
            reconstruction = evaluate_g1_reconstruction(target, prediction, mask)
            causality = evaluate_g1_tokenizer_causality(
                model if isinstance(model, G1Tokenizer) else G1Tokenizer(),
                tensor,
                mask_tensor,
            )
            metric_values = {
                "root_position_ade_m": reconstruction.root_position_ade_m,
                "root_position_fde_m": reconstruction.root_position_fde_m,
                "worst_joint_error_rad": reconstruction.worst_joint_error_rad,
            }
    payload = {
        "domain": train_config.model_kind.value,
        "quality_claim": "none_checkpoint_metrics_only",
        "feature_mae": reconstruction.features.mae,
        "codebook_perplexity": codebook.perplexity,
        "used_code_count": codebook.used_code_count,
        "dead_code_ratio": codebook.dead_code_ratio,
        "changed_token_count": causality.changed_token_count,
        "max_latent_difference": causality.max_latent_difference,
        "max_decoded_difference": causality.max_decoded_difference,
        **metric_values,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _ = output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    typer.echo(json.dumps({"output": str(output)}, sort_keys=True))
