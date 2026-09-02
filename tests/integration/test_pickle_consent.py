from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, final

import pytest
from typer.testing import CliRunner
from typing_extensions import override

from hri_ttr.cli import app
from hri_ttr.commands.common import load_prepared
from tests.fixtures.pairs import write_synthetic_pair

if TYPE_CHECKING:
    from collections.abc import Callable


def _write_marker(path: str) -> None:
    _ = Path(path).write_text("executed", encoding="utf-8")


@final
class MaliciousPayload:
    marker: Path

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    @override
    def __reduce__(self) -> tuple[Callable[[str], None], tuple[str]]:
        return _write_marker, (str(self.marker),)


@pytest.mark.parametrize("command", ["audit", "prepare"])
def test_data_commands_refuse_pickle_before_malicious_reducer_executes(
    tmp_path: Path,
    command: str,
) -> None:
    # Given
    raw = tmp_path / "raw"
    sequence_id = write_synthetic_pair(raw, source_format="pkl")
    marker = tmp_path / "pickle-executed"
    actor = raw / f"{sequence_id}_actor.pkl"
    with actor.open("wb") as stream:
        pickle.dump(MaliciousPayload(marker), stream)
    runner = CliRunner()

    # When
    args = ["data", command, "--input-dir", str(raw)]
    if command == "prepare":
        args.extend(["--output", str(tmp_path / "prepared.npz")])
    result = runner.invoke(app, args)

    # Then
    assert result.exit_code == 2
    assert "pickle" in result.output.lower()
    assert not marker.exists()


def test_trusted_local_pickle_requires_explicit_cli_consent(tmp_path: Path) -> None:
    # Given
    raw = tmp_path / "raw"
    _ = write_synthetic_pair(raw, source_format="pkl")
    output = tmp_path / "prepared.npz"
    runner = CliRunner()

    # When
    refused = runner.invoke(
        app,
        ["data", "prepare", "--input-dir", str(raw), "--output", str(output)],
    )
    accepted = runner.invoke(
        app,
        [
            "data",
            "prepare",
            "--input-dir",
            str(raw),
            "--output",
            str(output),
            "--allow-trusted-pickle",
        ],
    )
    audited = runner.invoke(
        app,
        ["data", "audit", "--input-dir", str(raw), "--allow-trusted-pickle"],
    )

    # Then
    assert refused.exit_code == 2
    assert accepted.exit_code == 0, accepted.output
    assert audited.exit_code == 0, audited.output
    assert '"source_provenance": ["trusted_pickle"]' in audited.output
    assert load_prepared(output).metadata.source_format == "trusted_pickle"
