from __future__ import annotations

# pyright: reportUnknownMemberType=false
from typing import TYPE_CHECKING

import numpy as np
import torch
from typer.testing import CliRunner

from hri_ttr.cache import read_token_cache
from hri_ttr.cli import app
from hri_ttr.commands import cache as cache_command
from hri_ttr.commands.common import load_prepared
from hri_ttr.evaluation import codebook_statistics
from hri_ttr.tokenizers import G1Tokenizer, TokenizerArchitecture
from hri_ttr.training import TrainConfig
from tests.fixtures.configs import write_tiny_config

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_cache_cli_preserves_real_partial_tail_id_but_excludes_its_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    prepared_path = tmp_path / "prepared.npz"
    g1_features = np.arange(8 * 75, dtype=np.float64).reshape(8, 75) / 100.0
    g1_features[5:] = g1_features[4]
    mask = np.array([True] * 5 + [False] * 3, dtype=np.bool_)
    np.savez(
        prepared_path,
        sequence_id=np.asarray("partial-tail"),
        human_features=np.zeros((8, 262), dtype=np.float32),
        g1_features=g1_features,
        human_space=np.zeros((8, 3), dtype=np.float32),
        g1_space=np.zeros((8, 3), dtype=np.float64),
        human_mask=mask,
        g1_mask=mask,
        anchor_origin=np.zeros(3, dtype=np.float64),
        anchor_basis=np.eye(3, dtype=np.float64),
        fps=np.asarray(20.0),
        source_fps=np.asarray(20.0),
        source_format=np.asarray("safe_npz"),
        target_fps=np.asarray(20.0),
        valid_frames=np.asarray(5, dtype=np.int64),
    )
    runner = CliRunner()
    config_path = write_tiny_config(tmp_path / "g1.json", "g1", tmp_path / "run")
    config = TrainConfig.load_json(config_path)
    prepared = load_prepared(prepared_path)
    model = G1Tokenizer(TokenizerArchitecture(width=8, code_dim=4, residual_depth=1))
    features = torch.as_tensor(prepared.g1_features, dtype=torch.float32)[None]
    with torch.no_grad():
        latents = model.encoder.forward(features)
        _ = model.quantizer.codebook.fill_(1_000_000.0)
        _ = model.quantizer.codebook[3].copy_(latents[0, 0])
        _ = model.quantizer.codebook[7].copy_(latents[0, 1])
    _ = model.eval()

    def load_fixed_model(
        config_argument: Path, checkpoint_argument: Path
    ) -> tuple[G1Tokenizer, TrainConfig]:
        _ = config_argument, checkpoint_argument
        return model, config

    monkeypatch.setattr(cache_command, "load_model", load_fixed_model)
    checkpoint = tmp_path / "checkpoint.pt"
    normalizer = tmp_path / "normalizer.json"
    schema = tmp_path / "schema.json"
    split = tmp_path / "split.json"
    for artifact in (checkpoint, normalizer, schema, split):
        _ = artifact.write_text(artifact.name, encoding="utf-8")
    destination = tmp_path / "cache"

    # When
    result = runner.invoke(
        app,
        [
            "cache",
            "tokens",
            "--prepared",
            str(prepared_path),
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--normalizer",
            str(normalizer),
            "--schema",
            str(schema),
            "--split",
            str(split),
            "--output",
            str(destination),
        ],
    )

    # Then
    assert result.exit_code == 0, result.output
    cache = read_token_cache(destination)
    assert cache.tokens.tolist() == [3, 7]
    assert cache.manifest.valid_frame_lengths == (5,)
    assert cache.manifest.padded_frame_counts == (3,)
    assert cache.statistics_token_mask.tolist() == [True, False]
    statistics = codebook_statistics(
        cache.tokens.astype(np.int64), cache.statistics_token_mask
    )
    assert statistics.histogram[3] == 1
    assert statistics.histogram[7] == 0
