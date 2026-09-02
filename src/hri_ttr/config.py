"""Pydantic models for untrusted tokenizer configuration input."""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from hri_ttr.contracts import FramesPerToken, ModelId, SchemaId, TokenizerSpec


class TokenizerKind(StrEnum):
    """The two independent motion-token domains."""

    HUMAN = "human"
    G1 = "g1"


class TokenizerConfig(BaseModel):
    """Parse one tokenizer configuration file at the configuration boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    kind: TokenizerKind
    schema_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    input_fps: float = Field(gt=0.0, strict=True, allow_inf_nan=False)
    frames_per_token: int = Field(ge=1, strict=True)
    codebook_size: int = Field(ge=2, strict=True)

    def to_spec(self) -> TokenizerSpec:
        """Convert validated boundary data into an internal frozen contract."""
        return TokenizerSpec(
            kind=self.kind.value,
            schema_id=SchemaId(self.schema_id),
            model_id=ModelId(self.model_id),
            input_fps=self.input_fps,
            frames_per_token=FramesPerToken(self.frames_per_token),
            codebook_size=self.codebook_size,
        )
