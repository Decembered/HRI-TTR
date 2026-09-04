from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from hri_ttr.training import TrainConfig

PROJECT_ROOT = Path(__file__).parents[2]


class DataContract(TypedDict):
    dataset_id: str
    input_fps: int
    frames_per_token: int
    split_file: str
    human_schema: str
    g1_schema: str
    sequence_count: int


PRODUCTION_CONFIGS = (
    PROJECT_ROOT / "configs/human_vq/causal_scratch_8x3090.json",
    PROJECT_ROOT / "configs/g1_vq/causal_scratch_8x3090.json",
)
SMOKE_CONFIGS = (
    PROJECT_ROOT / "configs/human_vq/causal_8gpu_smoke.json",
    PROJECT_ROOT / "configs/g1_vq/causal_8gpu_smoke.json",
)


@pytest.mark.parametrize("path", [*PRODUCTION_CONFIGS, *SMOKE_CONFIGS])
def test_8x3090_config_preserves_tokenizer_protocol(path: Path) -> None:
    # Given / When
    config = TrainConfig.load_json(path)

    # Then
    assert config.amp is True
    assert config.tokenizer_codebook_size == 256
    assert config.window_frames % 4 == 0
    assert config.window_stride % 4 == 0


@pytest.mark.parametrize("path", PRODUCTION_CONFIGS)
def test_8x3090_production_config_is_not_a_smoke_run(path: Path) -> None:
    # Given / When
    config = TrainConfig.load_json(path)

    # Then
    assert config.max_steps >= 100_000
    assert config.tokenizer_width == 512
    assert config.tokenizer_code_dim == 512


@pytest.mark.parametrize("path", SMOKE_CONFIGS)
def test_8x3090_smoke_config_is_bounded(path: Path) -> None:
    # Given / When
    config = TrainConfig.load_json(path)

    # Then
    assert config.max_steps == 2
    assert config.batch_size == 1
    assert config.tokenizer_width <= 16


@pytest.mark.parametrize(
    "script",
    [
        "scripts/deploy_8x3090.sh",
        "scripts/bootstrap_8x3090.sh",
        "scripts/launch_8x3090.sh",
    ],
)
def test_deployment_script_has_valid_shell_syntax(script: str) -> None:
    # Given
    path = PROJECT_ROOT / script

    # When
    completed = subprocess.run(  # noqa: S603
        ["/bin/bash", "-n", str(path)],
        capture_output=True,
        check=False,
        text=True,
    )

    # Then
    assert completed.returncode == 0, completed.stderr


def test_launcher_uses_eight_local_ranks_and_data_disk_environment() -> None:
    # Given / When
    launcher = (PROJECT_ROOT / "scripts/launch_8x3090.sh").read_text(encoding="utf-8")

    # Then
    assert "--nproc-per-node=8" in launcher
    assert "/data/autovla/envs/hri-ttr" in launcher
    assert "--standalone" in launcher
    assert "NCCL_SOCKET_IFNAME" not in launcher


def test_bootstrap_keeps_the_requested_python_version() -> None:
    # Given / When
    bootstrap = (PROJECT_ROOT / "scripts/bootstrap_8x3090.sh").read_text(
        encoding="utf-8"
    )

    # Then
    assert "sync --frozen --python 3.11" in bootstrap


def test_bootstrap_bounds_download_concurrency_on_remote_server() -> None:
    # Given / When
    bootstrap = (PROJECT_ROOT / "scripts/bootstrap_8x3090.sh").read_text(
        encoding="utf-8"
    )

    # Then
    assert "UV_CONCURRENT_DOWNLOADS" in bootstrap
    assert "UV_HTTP_TIMEOUT" in bootstrap
    assert "env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY" in bootstrap


def test_linux_torch_uses_explicit_cuda_index() -> None:
    # Given / When
    project = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    # Then
    assert "torch==2.7.1" in project
    assert "https://download.pytorch.org/whl/cu126" in project
    assert "sys_platform == 'linux' and platform_machine == 'x86_64'" in project


def test_data_contract_declares_20_fps_and_four_frames_per_token() -> None:
    # Given / When
    payload = TypeAdapter(DataContract).validate_json(
        (PROJECT_ROOT / "configs/data/interx_human_g1_training_v1.json").read_bytes()
    )

    # Then
    assert payload["input_fps"] == 20
    assert payload["frames_per_token"] == 4
