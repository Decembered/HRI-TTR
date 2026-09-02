"""Strict configuration boundary for tokenizer training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - Pydantic resolves this runtime annotation.
from typing import ClassVar, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hri_ttr.checkpoints.kinds import ModelKind
from hri_ttr.representations.g1.schema import G1_FEATURE_DIM, G1_SCHEMA_VERSION
from hri_ttr.representations.human.features import HUMAN_FEATURE_DIM
from hri_ttr.representations.human.normalizer import SCHEMA_ID as HUMAN_SCHEMA_ID
from hri_ttr.training.errors import TrainingError, TrainingReason

CODEBOOK_SIZE: Final = 256


class TrainConfig(BaseModel):
    """Validated, immutable training configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    model_kind: ModelKind
    representation_schema: str = Field(min_length=1)
    output_dir: Path
    seed: int = Field(ge=0)
    epochs: int = Field(gt=0)
    max_steps: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    window_frames: int = Field(gt=0)
    window_stride: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0, allow_inf_nan=False)
    weight_decay: float = Field(ge=0.0, allow_inf_nan=False)
    gradient_clip_norm: float = Field(gt=0.0, allow_inf_nan=False)
    amp: bool
    tokenizer_width: int = Field(default=512, gt=0)
    tokenizer_code_dim: int = Field(default=512, gt=0)
    tokenizer_codebook_size: int = Field(default=256, ge=2)
    tokenizer_residual_depth: int = Field(default=3, gt=0, le=3)
    tokenizer_ema_decay: float = Field(default=0.99, ge=0.0, lt=1.0)
    tokenizer_commitment_weight: float = Field(default=1.0, gt=0.0)
    warm_start_checkpoint: Path | None = None

    @property
    def feature_dim(self) -> int:
        """Return the representation-owned feature width for this model kind."""
        if self.model_kind is ModelKind.HUMAN:
            return HUMAN_FEATURE_DIM
        return G1_FEATURE_DIM

    @property
    def tokenizer_config_sha256(self) -> str:
        """Hash the architecture fields instead of trusting user-supplied identity."""
        architecture = {
            "code_dim": self.tokenizer_code_dim,
            "codebook_size": self.tokenizer_codebook_size,
            "commitment_weight": self.tokenizer_commitment_weight,
            "ema_decay": self.tokenizer_ema_decay,
            "residual_depth": self.tokenizer_residual_depth,
            "width": self.tokenizer_width,
        }
        encoded = json.dumps(architecture, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    @model_validator(mode="after")
    def validate_protocol(self) -> Self:
        """Bind each model kind to its only supported representation."""
        expected = {
            ModelKind.HUMAN: HUMAN_SCHEMA_ID,
            ModelKind.G1: G1_SCHEMA_VERSION,
        }[self.model_kind]
        if self.representation_schema != expected:
            raise TrainingError(TrainingReason.SCHEMA)
        if self.window_frames % 4 != 0 or self.window_stride % 4 != 0:
            raise TrainingError(TrainingReason.WINDOW_ALIGNMENT)
        if self.tokenizer_codebook_size != CODEBOOK_SIZE:
            raise TrainingError(TrainingReason.CODEBOOK)
        if self.model_kind is ModelKind.G1 and self.warm_start_checkpoint is not None:
            raise TrainingError(TrainingReason.G1_WARM_START)
        return self

    @classmethod
    def load_json(cls, path: Path) -> TrainConfig:
        """Parse one JSON configuration file."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class TrainingIdentity(BaseModel):
    """Runtime hashes computed from the artifacts used by one training run."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    normalizer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def reject_placeholders(self) -> Self:
        """Reject a syntactically valid hash that binds no real artifact."""
        hashes = (
            self.normalizer_sha256,
            self.split_sha256,
            self.source_sha256,
        )
        if any(value == "0" * 64 for value in hashes):
            raise TrainingError(TrainingReason.PLACEHOLDER_HASH)
        return self


@dataclass(frozen=True, slots=True)
class TrainingInvocation:
    """Configuration, runtime identity, and optional resume point."""

    config: TrainConfig
    identity: TrainingIdentity
    resume_path: Path | None = None

    def with_resume(self, path: Path, *, max_steps: int) -> TrainingInvocation:
        """Return a validated continuation without mutating the original run."""
        values = self.config.model_dump()
        values["max_steps"] = max_steps
        return TrainingInvocation(
            config=TrainConfig.model_validate(values),
            identity=self.identity,
            resume_path=path,
        )
