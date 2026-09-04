"""Provenance-bound motion token cache command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import typer

from hri_ttr.cache import (
    CacheExistsError,
    CacheManifest,
    TokenCache,
    write_token_cache,
)
from hri_ttr.checkpoints.io import checkpoint_sha256
from hri_ttr.checkpoints.kinds import ModelKind
from hri_ttr.commands.common import fail, load_model, load_prepared, sha256_file

app = typer.Typer(no_args_is_help=True)
ExistingFile = Annotated[Path, typer.Option(exists=True, dir_okay=False)]


@app.command("tokens")
def cache_tokens(  # noqa: PLR0913, PLR0917 - CLI flags are independent boundaries.
    prepared: ExistingFile,
    config: ExistingFile,
    checkpoint: ExistingFile,
    normalizer: ExistingFile,
    schema: ExistingFile,
    split: ExistingFile,
    output: Annotated[Path, typer.Option(help="New token-cache directory.")],
) -> None:
    """Encode prepared NPZ data and bind every artifact identity in the cache."""
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
        encoding = model.encode(
            torch.as_tensor(features, dtype=torch.float32)[None],
            torch.as_tensor(mask, dtype=torch.bool)[None],
        )
    tokens = encoding.token_ids[0].cpu().numpy().astype(np.uint16)
    valid_tokens = (motion.metadata.valid_frames + 3) // 4
    padded_frames = len(mask) - motion.metadata.valid_frames
    manifest = CacheManifest(
        tokenizer_sha256=train_config.tokenizer_config_sha256,
        checkpoint_sha256=checkpoint_sha256(checkpoint),
        normalizer_sha256=sha256_file(normalizer),
        schema_sha256=sha256_file(schema),
        split_sha256=sha256_file(split),
        valid_frame_lengths=(motion.metadata.valid_frames,),
        valid_token_lengths=(valid_tokens,),
        padded_frame_counts=(padded_frames,),
    )
    cache = TokenCache(
        tokens[:valid_tokens],
        np.asarray([0, valid_tokens], dtype=np.int64),
        (motion.metadata.sequence_id,),
        manifest,
    )
    try:
        _ = write_token_cache(output, cache)
    except CacheExistsError:
        fail("token cache already exists; refusing to overwrite it")
    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "tokens": valid_tokens,
                "minimum_token_id": int(tokens.min()),
                "maximum_token_id": int(tokens.max()),
            },
            sort_keys=True,
        )
    )
