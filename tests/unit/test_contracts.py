from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from hri_ttr.cli import app
from hri_ttr.config import TokenizerConfig, TokenizerKind
from hri_ttr.contracts import (
    MotionBatch,
    MotionSchema,
    SchemaId,
    SpaceState,
    TokenBatch,
)


def test_space_state_when_constructed_then_is_immutable_value() -> None:
    # Given
    state = SpaceState(x_m=1.0, z_m=2.0, yaw_rad=0.5)

    # When
    observed = (state.x_m, state.z_m, state.yaw_rad)

    # Then
    assert observed == (1.0, 2.0, 0.5)
    with pytest.raises(AttributeError):
        SpaceState.__setattr__(state, "x_m", 3.0)


def test_tokenizer_config_when_boundary_value_is_invalid_then_rejects_it() -> None:
    # Given
    malformed = {
        "kind": "g1",
        "schema_id": "g1-75d-v2",
        "model_id": "g1-causal-v1",
        "input_fps": 20,
        "frames_per_token": 0,
        "codebook_size": 256,
    }

    # When
    # Then
    with pytest.raises(ValidationError):
        _ = TokenizerConfig.model_validate(malformed)


def test_motion_batch_when_mask_shape_disagrees_then_rejects_it() -> None:
    # Given
    features = np.zeros((1, 4, 3), dtype=np.float32)
    malformed_mask = np.ones((1, 3), dtype=np.bool_)
    schema = MotionSchema(schema_id=SchemaId("test-3d"), feature_dim=3)

    # When
    # Then
    with pytest.raises(ValueError, match="frame mask"):
        _ = MotionBatch(features=features, frame_mask=malformed_mask, schema=schema)


def test_token_batch_when_valid_token_exceeds_codebook_then_rejects_it() -> None:
    # Given
    config = TokenizerConfig(
        kind=TokenizerKind.HUMAN,
        schema_id="human-262d-v1",
        model_id="human-causal-v1",
        input_fps=20.0,
        frames_per_token=4,
        codebook_size=256,
    )
    token_ids = np.array([[0, 256]], dtype=np.int64)
    token_mask = np.array([[True, True]], dtype=np.bool_)

    # When
    # Then
    with pytest.raises(ValueError, match="outside"):
        _ = TokenBatch(
            token_ids=token_ids,
            token_mask=token_mask,
            tokenizer=config.to_spec(),
        )


@pytest.mark.parametrize(
    ("field_name", "malformed_value"),
    [
        ("input_fps", "20.0"),
        ("frames_per_token", "4"),
        ("codebook_size", "256"),
        ("input_fps", float("nan")),
        ("input_fps", float("inf")),
        ("input_fps", float("-inf")),
    ],
)
def test_tokenizer_config_when_numeric_boundary_is_malformed_then_rejects_it(
    field_name: str,
    malformed_value: str | float,
) -> None:
    # Given
    payload: dict[str, str | float | int] = {
        "kind": "g1",
        "schema_id": "g1-75d-v2",
        "model_id": "g1-causal-v1",
        "input_fps": 20.0,
        "frames_per_token": 4,
        "codebook_size": 256,
    }
    payload[field_name] = malformed_value

    # When
    # Then
    with pytest.raises(ValidationError):
        _ = TokenizerConfig.model_validate(payload)


def test_motion_batch_when_source_arrays_mutate_then_owned_data_is_unchanged() -> None:
    # Given
    features = np.zeros((1, 4, 3), dtype=np.float32)
    frame_mask = np.ones((1, 4), dtype=np.bool_)
    schema = MotionSchema(schema_id=SchemaId("test-3d"), feature_dim=3)
    batch = MotionBatch(features=features, frame_mask=frame_mask, schema=schema)

    # When
    features[0, 0, 0] = np.nan
    frame_mask[0, 0] = False

    # Then
    assert batch.features[0, 0, 0] == 0.0
    assert tuple(int(value) for value in batch.frame_mask.flat) == (1, 1, 1, 1)
    with pytest.raises(ValueError, match="read-only"):
        batch.features[0, 0, 0] = np.float32(1.0)


def test_token_batch_when_source_arrays_mutate_then_owned_data_is_unchanged() -> None:
    # Given
    config = TokenizerConfig(
        kind=TokenizerKind.HUMAN,
        schema_id="human-262d-v1",
        model_id="human-causal-v1",
        input_fps=20.0,
        frames_per_token=4,
        codebook_size=256,
    )
    token_ids = np.array([[0, 1]], dtype=np.int64)
    token_mask = np.array([[True, True]], dtype=np.bool_)
    batch = TokenBatch(
        token_ids=token_ids,
        token_mask=token_mask,
        tokenizer=config.to_spec(),
    )

    # When
    token_ids[0, 1] = 256
    token_mask[0, 0] = False

    # Then
    assert batch.token_ids[0, 1] == 1
    assert tuple(int(value) for value in batch.token_mask.flat) == (1, 1)
    with pytest.raises(ValueError, match="read-only"):
        batch.token_ids[0, 0] = np.int64(2)


def test_cli_when_help_requested_then_exits_successfully() -> None:
    # Given
    runner = CliRunner()

    # When
    completed = runner.invoke(app, ["--help"])

    # Then
    assert completed.exit_code == 0
    assert "Usage:" in completed.stdout
