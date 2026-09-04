from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from typer.testing import CliRunner

from hri_ttr.cli import app
from hri_ttr.commands.common import load_model, load_prepared
from tests.fixtures.configs import write_tiny_config
from tests.fixtures.pairs import write_synthetic_pair

if TYPE_CHECKING:
    from pathlib import Path


def test_train_cli_preserves_prepared_mask_for_ema(tmp_path: Path) -> None:
    # Given
    raw = tmp_path / "raw"
    _ = write_synthetic_pair(raw, frames=13, fps=50.0)
    prepared = tmp_path / "prepared.npz"
    cli_output = tmp_path / "cli-run"
    config_path = write_tiny_config(tmp_path / "g1.json", "g1", cli_output)
    artifacts = tuple(tmp_path / name for name in ("normalizer", "split", "source"))
    for artifact in artifacts:
        _ = artifact.write_text(artifact.name, encoding="utf-8")
    runner = CliRunner()
    prepared_result = runner.invoke(
        app,
        ["data", "prepare", "--input-dir", str(raw), "--output", str(prepared)],
    )
    assert prepared_result.exit_code == 0, prepared_result.output
    prepared_pair = load_prepared(prepared)
    assert prepared_pair.metadata.valid_frames == 5
    assert torch.equal(
        torch.as_tensor(prepared_pair.g1_mask),
        torch.tensor([True, True, True, True, True, False, False, False]),
    )

    # When
    result = runner.invoke(
        app,
        [
            "train",
            "g1-vq",
            "--config",
            str(config_path),
            "--prepared",
            str(prepared),
            "--normalizer",
            str(artifacts[0]),
            "--split",
            str(artifacts[1]),
            "--source-artifact",
            str(artifacts[2]),
        ],
    )

    # Then
    assert result.exit_code == 0, result.output
    cli_model, _ = load_model(config_path, cli_output / "last.pt")
    torch.testing.assert_close(
        cli_model.quantizer.ema_count,
        torch.ones_like(cli_model.quantizer.ema_count),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        cli_model.quantizer.ema_sum,
        cli_model.quantizer.ema_sum[:1].expand_as(cli_model.quantizer.ema_sum),
        rtol=1e-6,
        atol=1e-7,
    )
