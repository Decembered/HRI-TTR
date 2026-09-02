from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest
import torch
from pydantic import ValidationError

from hri_ttr.checkpoints import (
    CheckpointBinding,
    CheckpointComponents,
    CheckpointSnapshot,
    TrainingProgress,
    load_training_checkpoint,
    save_training_checkpoint,
)
from hri_ttr.checkpoints.io import ScalerPayload
from hri_ttr.checkpoints.kinds import ModelKind
from hri_ttr.representations.g1.schema import G1_SCHEMA_VERSION
from hri_ttr.tokenizers import G1Tokenizer, TokenizerArchitecture

if TYPE_CHECKING:
    from pathlib import Path


def _binding() -> CheckpointBinding:
    return CheckpointBinding(
        format_version=1,
        model_kind=ModelKind.G1,
        representation_schema=G1_SCHEMA_VERSION,
        tokenizer_config_sha256="1" * 64,
        normalizer_sha256="2" * 64,
        split_sha256="3" * 64,
        source_sha256="4" * 64,
    )


def test_enabled_scaler_checkpoint_roundtrip(tmp_path: Path) -> None:
    # Given
    architecture = TokenizerArchitecture(width=8, code_dim=4, residual_depth=1)
    model = G1Tokenizer(architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    snapshot = CheckpointSnapshot(
        binding=_binding(),
        progress=TrainingProgress(
            epoch=0,
            batch_in_epoch=0,
            global_step=1,
            best_validation_loss=sys.float_info.max,
        ),
    )
    path = tmp_path / "amp.pt"
    save_training_checkpoint(
        path, CheckpointComponents(model, optimizer, scaler), snapshot
    )
    restored_scaler = torch.amp.GradScaler("cpu", enabled=True)

    # When
    restored = load_training_checkpoint(
        path,
        CheckpointComponents(model, optimizer, restored_scaler),
        snapshot.binding,
    )

    # Then
    assert restored.progress.global_step == 1
    assert restored_scaler.state_dict() == scaler.state_dict()


@pytest.mark.parametrize("tracker_key", ["_growth_tracker", "growth_tracker"])
def test_scaler_payload_accepts_torch_alias_and_saved_field_name(
    tracker_key: str,
) -> None:
    # Given / When
    payload = ScalerPayload.model_validate(
        {
            "scale": 65536.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            tracker_key: 7,
        }
    )

    # Then
    assert payload.model_dump(by_alias=True)["_growth_tracker"] == 7


def test_scaler_payload_rejects_unknown_state_key() -> None:
    # Given / When / Then
    with pytest.raises(ValidationError):
        _ = ScalerPayload.model_validate({"unexpected": 1})
