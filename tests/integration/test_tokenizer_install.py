from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_project_environment_script_selects_nonhidden_directory() -> None:
    # Given
    project_root = Path(__file__).parents[2]
    script = project_root / "scripts" / "use-nonhidden-uv-env.sh"
    command = [
        "/bin/sh",
        "-c",
        '. "$1"; printf "%s" "$UV_PROJECT_ENVIRONMENT"',
        "sh",
        str(script),
    ]

    # When
    completed = subprocess.run(  # noqa: S603
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    # Then
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "venv"


def test_project_when_environment_is_synced_then_installed_import_works(
    tmp_path: Path,
) -> None:
    # Given
    environment = os.environ.copy()
    _ = environment.pop("PYTHONPATH", None)
    command = [
        sys.executable,
        "-c",
        "import hri_ttr; print(hri_ttr.__file__)",
    ]

    # When
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    # Then
    assert completed.returncode == 0, completed.stderr
    assert "hri_ttr/__init__.py" in completed.stdout
