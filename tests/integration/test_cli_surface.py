from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from hri_ttr.cli import app
from hri_ttr.commands.common import load_prepared
from tests.fixtures.pairs import write_synthetic_pair

if TYPE_CHECKING:
    from pathlib import Path


def test_cli_exposes_all_required_command_groups() -> None:
    # Given
    runner = CliRunner()

    # When
    result = runner.invoke(app, ["--help"])

    # Then
    assert result.exit_code == 0
    for command in (
        "data",
        "checkpoint",
        "train",
        "evaluate",
        "cache",
        "visualize",
        "export",
    ):
        assert command in result.stdout


def test_cli_expected_validation_error_has_no_traceback() -> None:
    # Given
    runner = CliRunner()

    # When
    result = runner.invoke(app, ["data", "audit", "--input-dir", "missing"])

    # Then
    assert result.exit_code != 0
    assert "Traceback" not in result.stdout


def test_data_prepare_resamples_shared_50_fps_timeline(tmp_path: Path) -> None:
    # Given
    raw = tmp_path / "raw"
    _ = write_synthetic_pair(raw, frames=13, fps=50.0)
    output = tmp_path / "prepared.npz"
    runner = CliRunner()

    # When
    result = runner.invoke(
        app,
        ["data", "prepare", "--input-dir", str(raw), "--output", str(output)],
    )

    # Then
    assert result.exit_code == 0, result.output
    prepared = load_prepared(output)
    assert prepared.metadata.source_fps == 50.0
    assert prepared.metadata.target_fps == 20.0
    assert prepared.metadata.valid_frames == 5
    assert prepared.human_features.shape == (8, 262)
    assert prepared.g1_features.shape == (8, 75)
    assert prepared.human_mask.tolist() == [True] * 5 + [False] * 3


def test_data_audit_reports_fps_mismatch_without_traceback(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _ = write_synthetic_pair(raw, reactor_fps=30.0)

    result = CliRunner().invoke(app, ["data", "audit", "--input-dir", str(raw)])

    assert result.exit_code == 2
    assert "actor/reactor differ" in result.output
    assert "Traceback" not in result.output
