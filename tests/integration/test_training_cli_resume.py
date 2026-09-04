from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from hri_ttr.cli import app
from hri_ttr.training import TrainConfig
from tests.fixtures.configs import write_tiny_config
from tests.fixtures.pairs import write_synthetic_pair


def _run_cli(arguments: list[str], project: Path) -> subprocess.CompletedProcess[str]:
    executable = Path(sys.executable).with_name("hri-ttr")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project / "src")
    return subprocess.run(  # noqa: S603
        [str(executable), *arguments],
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_installed_cli_resumes_and_rejects_wrong_checkpoint_identity(
    tmp_path: Path,
) -> None:
    # Given
    project = Path(__file__).parents[2]
    raw = tmp_path / "raw"
    _ = write_synthetic_pair(raw)
    prepared = tmp_path / "prepared.npz"
    prepared_result = CliRunner().invoke(
        app, ["data", "prepare", "--input-dir", str(raw), "--output", str(prepared)]
    )
    assert prepared_result.exit_code == 0, prepared_result.output
    first_output = tmp_path / "first"
    resumed_output = tmp_path / "resumed"
    g1_config = write_tiny_config(tmp_path / "g1.json", "g1", first_output)
    config_values = TrainConfig.load_json(g1_config).model_dump()
    config_values["epochs"] = 2
    _ = g1_config.write_text(
        TrainConfig.model_validate(config_values).model_dump_json(), encoding="utf-8"
    )
    incompatible_values = dict(config_values)
    incompatible_values["tokenizer_width"] = 9
    incompatible_config = tmp_path / "incompatible-g1.json"
    _ = incompatible_config.write_text(
        TrainConfig.model_validate(incompatible_values).model_dump_json(),
        encoding="utf-8",
    )
    human_config = write_tiny_config(tmp_path / "human.json", "human", tmp_path / "h")
    artifacts = tuple(tmp_path / name for name in ("normalizer", "split", "source"))
    for artifact in artifacts:
        _ = artifact.write_text(artifact.name, encoding="utf-8")
    different_source = tmp_path / "different-source"
    _ = different_source.write_text("different", encoding="utf-8")
    common = [
        "--prepared",
        str(prepared),
        "--normalizer",
        str(artifacts[0]),
        "--split",
        str(artifacts[1]),
        "--source-artifact",
        str(artifacts[2]),
    ]
    first = _run_cli(["train", "g1-vq", "--config", str(g1_config), *common], project)
    assert first.returncode == 0, first.stderr
    checkpoint = first_output / "last.pt"

    # When
    resumed = _run_cli(
        [
            "train",
            "g1-vq",
            "--config",
            str(g1_config),
            *common,
            "--resume",
            str(checkpoint),
            "--max-steps",
            "2",
            "--output-dir",
            str(resumed_output),
        ],
        project,
    )
    wrong_domain = _run_cli(
        [
            "train",
            "human-vq",
            "--config",
            str(human_config),
            *common,
            "--resume",
            str(checkpoint),
            "--max-steps",
            "2",
        ],
        project,
    )
    missing = _run_cli(
        [
            "train",
            "g1-vq",
            "--config",
            str(g1_config),
            *common,
            "--resume",
            str(tmp_path / "missing.pt"),
        ],
        project,
    )
    wrong_architecture = _run_cli(
        [
            "train",
            "g1-vq",
            "--config",
            str(incompatible_config),
            *common,
            "--resume",
            str(checkpoint),
            "--max-steps",
            "2",
        ],
        project,
    )
    wrong_artifact = _run_cli(
        [
            "train",
            "g1-vq",
            "--config",
            str(g1_config),
            *common[:-2],
            "--source-artifact",
            str(different_source),
            "--resume",
            str(checkpoint),
            "--max-steps",
            "2",
        ],
        project,
    )

    # Then
    assert json.loads(resumed.stdout)["global_step"] == 2
    assert (resumed_output / "last.pt").is_file()
    for rejected in (wrong_domain, wrong_architecture, wrong_artifact, missing):
        assert rejected.returncode == 2, rejected.stderr
        assert "Traceback" not in rejected.stderr
