from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from hri_ttr.checkpoints.kinds import ModelKind
from hri_ttr.representations.g1.schema import G1_FEATURE_DIM, G1_SCHEMA_VERSION
from hri_ttr.representations.human.features import HUMAN_FEATURE_DIM
from hri_ttr.representations.human.normalizer import SCHEMA_ID as HUMAN_SCHEMA_ID
from hri_ttr.training import TrainConfig, TrainingIdentity


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/human_vq/causal_scratch.json",
        "configs/human_vq/causal_warm_start.json",
        "configs/g1_vq/causal_scratch.json",
    ],
)
def test_shipped_training_config_parses(relative_path: str) -> None:
    # Given
    path = Path(relative_path)

    # When
    config = TrainConfig.load_json(path)

    # Then
    assert config.tokenizer_config_sha256 != "0" * 64
    raw = path.read_text(encoding="utf-8")
    assert "normalizer_sha256" not in raw
    assert "split_sha256" not in raw
    assert "source_sha256" not in raw


@pytest.mark.parametrize(
    ("relative_path", "schema", "feature_dim"),
    [
        ("configs/human_vq/causal_scratch.json", HUMAN_SCHEMA_ID, HUMAN_FEATURE_DIM),
        (
            "configs/human_vq/causal_warm_start.json",
            HUMAN_SCHEMA_ID,
            HUMAN_FEATURE_DIM,
        ),
        ("configs/g1_vq/causal_scratch.json", G1_SCHEMA_VERSION, G1_FEATURE_DIM),
    ],
)
def test_training_config_uses_authoritative_representation_schema(
    relative_path: str, schema: str, feature_dim: int
) -> None:
    # When
    config = TrainConfig.load_json(Path(relative_path))

    # Then
    assert config.representation_schema == schema
    assert config.feature_dim == feature_dim


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/human_vq/causal_scratch.json",
        "configs/human_vq/causal_scratch_8x3090.json",
        "configs/human_vq/causal_warm_start.json",
        "configs/human_vq/causal_8gpu_smoke.json",
    ],
)
def test_human_training_config_uses_official_ttr_commitment_weight(
    relative_path: str,
) -> None:
    # When
    config = TrainConfig.load_json(Path(relative_path))

    # Then
    assert config.tokenizer_commitment_weight == 0.02


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/g1_vq/causal_scratch.json",
        "configs/g1_vq/causal_scratch_8x3090.json",
        "configs/g1_vq/causal_8gpu_smoke.json",
    ],
)
def test_g1_training_config_uses_ttr_style_commitment_weight(
    relative_path: str,
) -> None:
    # When
    config = TrainConfig.load_json(Path(relative_path))

    # Then
    assert config.tokenizer_commitment_weight == 0.02


def test_paired_training_data_uses_authoritative_schemas() -> None:
    # Given
    path = Path("configs/data/interx_human_g1_training_v1.json")
    adapter = TypeAdapter(dict[str, str | int])

    # When
    config = adapter.validate_json(path.read_text(encoding="utf-8"))

    # Then
    assert config["human_schema"] == HUMAN_SCHEMA_ID
    assert config["g1_schema"] == G1_SCHEMA_VERSION


def test_config_rejects_g1_legacy_warm_start(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "legacy.pt"

    # When / Then
    with pytest.raises(ValidationError):
        _ = TrainConfig(
            model_kind=ModelKind.G1,
            representation_schema=G1_SCHEMA_VERSION,
            output_dir=tmp_path,
            seed=1,
            epochs=1,
            max_steps=1,
            batch_size=1,
            window_frames=8,
            window_stride=8,
            learning_rate=1e-3,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            amp=False,
            warm_start_checkpoint=path,
        )


def test_training_identity_rejects_all_zero_hash() -> None:
    # Given
    digest = "0" * 64

    # When / Then
    with pytest.raises(ValidationError):
        _ = TrainingIdentity(
            normalizer_sha256=digest,
            split_sha256="1" * 64,
            source_sha256="2" * 64,
        )


def test_training_config_rejects_partial_wandb_identity(tmp_path: Path) -> None:
    # Given
    values = TrainConfig.load_json(
        Path("configs/g1_vq/causal_scratch.json")
    ).model_dump()
    values["output_dir"] = tmp_path
    values["wandb_project"] = "hri-ttr-causal-vq"

    # When / Then
    with pytest.raises(ValidationError):
        _ = TrainConfig.model_validate(values)
