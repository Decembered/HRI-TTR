from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def test_python311_import_and_cli_help() -> None:
    # Given
    interpreter = shutil.which("python3.11")
    if interpreter is None:
        pytest.skip("Python 3.11 is not installed")
    project = Path(__file__).parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project / "src")
    command = [
        interpreter,
        "-c",
        "import hri_ttr.checkpoints.io; from hri_ttr.cli import app; app(['--help'])",
    ]

    # When
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    # Then
    assert completed.returncode == 0, completed.stderr
    assert "Usage" in completed.stdout
