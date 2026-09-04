"""Strict manifests for checkpoints and baseline artifacts."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - Pydantic resolves this runtime annotation.
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hri_ttr.checkpoints.kinds import ModelKind  # noqa: TC001 - Pydantic runtime type.


class PlaceholderCheckpointHashError(ValueError):
    """Raised when an artifact binding contains a sentinel hash."""


class CheckpointBinding(BaseModel):
    """Immutable identity of one tokenizer checkpoint space."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    format_version: int = Field(ge=1)
    model_kind: ModelKind
    representation_schema: str = Field(min_length=1)
    tokenizer_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalizer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def reject_placeholders(self) -> CheckpointBinding:
        """Keep fake artifact identities out of persisted checkpoints."""
        hashes = (
            self.tokenizer_config_sha256,
            self.normalizer_sha256,
            self.split_sha256,
            self.source_sha256,
        )
        if any(value == "0" * 64 for value in hashes):
            raise PlaceholderCheckpointHashError
        return self


class TrainingProgress(BaseModel):
    """Exact point from which deterministic training continues."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    epoch: int = Field(ge=0)
    batch_in_epoch: int = Field(ge=0)
    global_step: int = Field(ge=0)
    best_validation_loss: float = Field(ge=0.0, allow_inf_nan=False)


class CheckpointSnapshot(BaseModel):
    """Pydantic-validated metadata embedded in a tensor checkpoint."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    binding: CheckpointBinding
    progress: TrainingProgress


class BaselineManifest(BaseModel):
    """Hash-only record for a checkpoint that must never be loaded partially."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    baseline_kind: str = Field(min_length=1)
    source_path: Path
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    load_policy: str = Field(pattern=r"^baseline_only_no_partial_load$")
