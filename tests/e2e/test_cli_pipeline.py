from __future__ import annotations

from typing import TYPE_CHECKING
from zipfile import ZipFile

import torch
from typer.testing import CliRunner

from hri_ttr.cache import read_token_cache
from hri_ttr.cli import app
from hri_ttr.commands.common import load_prepared
from tests.fixtures.configs import write_tiny_config
from tests.fixtures.pairs import write_synthetic_pair

if TYPE_CHECKING:
    from pathlib import Path


def _invoke(runner: CliRunner, arguments: list[str]) -> None:
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.stdout
    assert "Traceback" not in result.stdout


def test_complete_cli_pipeline_when_pair_is_synthetic(  # noqa: PLR0915
    tmp_path: Path,
) -> None:
    # Given
    raw = tmp_path / "raw"
    _ = write_synthetic_pair(raw)
    prepared = tmp_path / "prepared.npz"
    g1_config = write_tiny_config(tmp_path / "g1.json", "g1", tmp_path / "g1-run")
    human_config = write_tiny_config(
        tmp_path / "human.json", "human", tmp_path / "human-run"
    )
    normalizer = tmp_path / "normalizer.json"
    schema = tmp_path / "schema.json"
    split = tmp_path / "split.json"
    for artifact, value in (
        (normalizer, "normalizer"),
        (schema, "schema"),
        (split, "split"),
    ):
        _ = artifact.write_text(value, encoding="utf-8")
    runner = CliRunner()

    # When
    _invoke(runner, ["data", "audit", "--input-dir", str(raw)])
    _invoke(
        runner,
        ["data", "prepare", "--input-dir", str(raw), "--output", str(prepared)],
    )
    for domain, config, run in (
        ("g1-vq", g1_config, tmp_path / "g1-run"),
        ("human-vq", human_config, tmp_path / "human-run"),
    ):
        _invoke(
            runner,
            [
                "train",
                domain,
                "--config",
                str(config),
                "--prepared",
                str(prepared),
                "--normalizer",
                str(normalizer),
                "--split",
                str(split),
                "--source-artifact",
                str(prepared),
                "--max-steps",
                "1",
            ],
        )
        assert (run / "best.pt").is_file()
    checkpoint = tmp_path / "g1-run" / "best.pt"
    cache = tmp_path / "cache"
    metrics = tmp_path / "metrics.json"
    image = tmp_path / "reconstruction.png"
    sonic = tmp_path / "sonic.npz"
    _invoke(
        runner,
        [
            "cache",
            "tokens",
            "--prepared",
            str(prepared),
            "--config",
            str(g1_config),
            "--checkpoint",
            str(checkpoint),
            "--normalizer",
            str(normalizer),
            "--schema",
            str(schema),
            "--split",
            str(split),
            "--output",
            str(cache),
        ],
    )
    stale = runner.invoke(
        app,
        [
            "cache",
            "tokens",
            "--prepared",
            str(prepared),
            "--config",
            str(g1_config),
            "--checkpoint",
            str(checkpoint),
            "--normalizer",
            str(normalizer),
            "--schema",
            str(schema),
            "--split",
            str(split),
            "--output",
            str(cache),
        ],
    )
    assert stale.exit_code == 2
    assert "already exists" in stale.stderr
    assert "Traceback" not in stale.output
    wrong_domain = runner.invoke(
        app,
        [
            "evaluate",
            "tokenizer",
            "--prepared",
            str(prepared),
            "--config",
            str(human_config),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "wrong.json"),
        ],
    )
    assert wrong_domain.exit_code == 2
    assert "domain" in wrong_domain.stderr
    assert "Traceback" not in wrong_domain.output
    _invoke(
        runner,
        [
            "evaluate",
            "tokenizer",
            "--prepared",
            str(prepared),
            "--config",
            str(g1_config),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(metrics),
        ],
    )
    _invoke(
        runner,
        [
            "visualize",
            "reconstruction",
            "--prepared",
            str(prepared),
            "--config",
            str(g1_config),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(image),
        ],
    )
    _invoke(
        runner,
        ["export", "sonic", "--prepared", str(prepared), "--output", str(sonic)],
    )
    official = tmp_path / "official.pt"
    torch.save({}, official)
    mapping = tmp_path / "mapping.json"
    _ = mapping.write_text("{}", encoding="utf-8")
    imported = tmp_path / "imported.pt"
    _invoke(
        runner,
        [
            "checkpoint",
            "import",
            "--domain",
            "human",
            "--source",
            str(official),
            "--destination",
            str(imported),
            "--config",
            str(human_config),
            "--mapping",
            str(mapping),
        ],
    )

    # Then
    prepared_data = load_prepared(prepared)
    assert prepared_data.human_features.shape == (8, 262)
    assert prepared_data.g1_features.shape == (8, 75)
    assert prepared_data.human_space.shape == (8, 3)
    assert prepared_data.g1_space.shape == (8, 3)
    token_cache = read_token_cache(cache)
    assert token_cache.tokens.size == 2
    assert all(int(token) <= 255 for token in token_cache.tokens.flat)
    manifest = token_cache.manifest
    assert all(
        value != "0" * 64
        for value in (
            manifest.tokenizer_sha256,
            manifest.checkpoint_sha256,
            manifest.normalizer_sha256,
            manifest.schema_sha256,
            manifest.split_sha256,
        )
    )
    assert metrics.stat().st_size > 0
    assert image.stat().st_size > 1000
    assert imported.is_file()
    with ZipFile(sonic) as archive:
        assert set(archive.namelist()) == {
            "root_trans_offset.npy",
            "root_rot.npy",
            "dof.npy",
            "pose_aa.npy",
            "fps.npy",
        }


def test_cli_rejects_wrong_checkpoint_domain_without_traceback(tmp_path: Path) -> None:
    # Given
    raw = tmp_path / "raw"
    _ = write_synthetic_pair(raw)
    runner = CliRunner()

    # When
    result = runner.invoke(app, ["data", "prepare", "--input-dir", str(raw)])

    # Then
    assert result.exit_code != 0
    assert "Traceback" not in result.stdout
