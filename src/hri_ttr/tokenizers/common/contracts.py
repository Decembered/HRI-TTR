"""Frozen outputs and the explicit streaming state contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from hri_ttr.tokenizers.common.errors import (
    ArchitectureReason,
    InvalidTokenizerArchitectureError,
)

if TYPE_CHECKING:
    import torch

CODEBOOK_SIZE: Final = 256
MAX_RESIDUAL_DEPTH: Final = 3


@dataclass(frozen=True, slots=True)
class TokenizerArchitecture:
    """Small architecture definition shared as values, never model state."""

    width: int = 512
    code_dim: int = 512
    codebook_size: int = 256
    residual_depth: int = 3
    ema_decay: float = 0.99
    commitment_weight: float = 1.0

    def __post_init__(self) -> None:
        """Reject architectures that break the fixed token contract."""
        if min(self.width, self.code_dim, self.residual_depth) <= 0:
            raise InvalidTokenizerArchitectureError(ArchitectureReason.DIMENSION)
        if self.residual_depth > MAX_RESIDUAL_DEPTH:
            raise InvalidTokenizerArchitectureError(ArchitectureReason.DEPTH)
        if self.codebook_size != CODEBOOK_SIZE:
            raise InvalidTokenizerArchitectureError(ArchitectureReason.CODEBOOK)
        if not 0.0 <= self.ema_decay < 1.0:
            raise InvalidTokenizerArchitectureError(ArchitectureReason.DECAY)


@dataclass(frozen=True, slots=True)
class Encoding:
    """Encoder and quantizer output in batch-major layout."""

    token_ids: torch.Tensor
    token_mask: torch.Tensor
    latents: torch.Tensor
    quantized: torch.Tensor
    commitment_loss: torch.Tensor
    perplexity: torch.Tensor
    codebook_updated: bool


@dataclass(frozen=True, slots=True)
class TokenizerOutput:
    """Complete VQ autoencoder output."""

    reconstruction: torch.Tensor
    encoding: Encoding


class StreamState:
    """Single-use state retaining the observed prefix, never future frames."""

    owner_id: int
    batch_size: int
    features: torch.Tensor | None
    frame_mask: torch.Tensor | None
    emitted_tokens: int
    consumed: bool

    def __init__(self, owner_id: int, batch_size: int) -> None:
        """Create an active empty state owned by one tokenizer instance."""
        self.owner_id = owner_id
        self.batch_size = batch_size
        self.features = None
        self.frame_mask = None
        self.emitted_tokens = 0
        self.consumed = False
