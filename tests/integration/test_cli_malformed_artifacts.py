from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from hri_ttr.cli import app
from tests.fixtures.configs import write_tiny_config
from tests.fixtures.pairs import write_synthetic_pair

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def prepared_and_config(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    _ = write_synthetic_pair(raw)
    prepared = tmp_path / "prepared.npz"
    result = CliRunner().invoke(
        app,
        ["data", "prepare", "--input-dir", str(raw), "--output", str(prepared)],
    )
    assert result.exit_code == 0, result.output
    config = write_tiny_config(tmp_path / "g1.json", "g1", tmp_path / "run")
    return prepared, config


@pytest.mark.parametrize("command", ["evaluate", "visualize", "cache"])
def test_checkpoint_consumers_report_corrupt_checkpoint_without_traceback(
    tmp_path: Path,
    prepared_and_config: tuple[Path, Path],
    command: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    prepared, config = prepared_and_config
    corrupt = tmp_path / "corrupt.pt"
    _ = corrupt.write_bytes(b"not a checkpoint")
    output = tmp_path / "output"
    args = [
        command,
        "tokenizer" if command == "evaluate" else "reconstruction",
        "--prepared",
        str(prepared),
        "--config",
        str(config),
        "--checkpoint",
        str(corrupt),
        "--output",
        str(output),
    ]
    if command == "cache":
        args[1] = "tokens"
        for name in ("normalizer", "schema", "split"):
            artifact = tmp_path / f"{name}.json"
            _ = artifact.write_text(name, encoding="utf-8")
            args.extend([f"--{name}", str(artifact)])

    with caplog.at_level(logging.DEBUG, logger="hri_ttr.commands.common"):
        result = CliRunner().invoke(app, args)

    assert result.exit_code == 2
    assert "unreadable or malformed" in result.output
    assert "Traceback" not in result.output
    records = [record for record in caplog.records if record.exc_info is not None]
    assert len(records) == 1
    assert records[0].exc_info is not None
    assert records[0].exc_info[1] is not None


def test_export_reports_malformed_prepared_npz_without_traceback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    malformed = tmp_path / "malformed.npz"
    _ = malformed.write_bytes(b"not an npz")

    with caplog.at_level(logging.DEBUG, logger="hri_ttr.commands.common"):
        result = CliRunner().invoke(
            app,
            [
                "export",
                "sonic",
                "--prepared",
                str(malformed),
                "--output",
                str(tmp_path / "sonic.npz"),
            ],
        )

    assert result.exit_code == 2
    assert "prepared NPZ is unreadable or malformed" in result.output
    assert "Traceback" not in result.output
    records = [record for record in caplog.records if record.exc_info is not None]
    assert len(records) == 1
    assert records[0].exc_info is not None
    assert records[0].exc_info[1] is not None
