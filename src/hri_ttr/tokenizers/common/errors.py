"""Typed tokenizer contract failures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import override


class MotionTensorReason(StrEnum):
    """Machine-readable malformed-motion reasons."""

    SHAPE = "features must have shape [B,T,F]"
    WIDTH = "feature width does not match tokenizer"
    DTYPE = "features must use floating-point dtype"
    EMPTY = "feature sequence must contain at least one frame"
    ALIGNMENT = "frame count must be divisible by four"
    MASK = "frame mask must be bool with shape [B,T]"
    FINITE = "features must be finite"
    BATCH = "stream batch size must be positive"


class ArchitectureReason(StrEnum):
    """Machine-readable invalid-architecture reasons."""

    DIMENSION = "architecture dimensions must be positive"
    DEPTH = "residual depth cannot exceed three"
    CODEBOOK = "the tokenizer vocabulary must contain 256 codes"
    DECAY = "EMA decay must be in [0, 1)"


class TokenTensorReason(StrEnum):
    """Machine-readable malformed-token reasons."""

    SHAPE = "token IDs must have shape [B,L]"
    DTYPE = "token IDs must use int64 dtype"
    MASK = "token mask must be bool with shape [B,L]"
    RANGE = "valid token ID is outside codebook"


@dataclass(frozen=True, slots=True)
class InvalidMotionTensorError(ValueError):
    """Report a malformed feature tensor or mask."""

    reason: MotionTensorReason

    @override
    def __str__(self) -> str:
        return f"invalid motion tensor: {self.reason}"


@dataclass(frozen=True, slots=True)
class InvalidTokenizerArchitectureError(ValueError):
    """Report architecture values that violate the temporal contract."""

    reason: ArchitectureReason

    @override
    def __str__(self) -> str:
        return f"invalid tokenizer architecture: {self.reason}"


@dataclass(frozen=True, slots=True)
class InvalidTokenTensorError(ValueError):
    """Report malformed token IDs or a token mask."""

    reason: TokenTensorReason

    @override
    def __str__(self) -> str:
        return f"invalid token tensor: {self.reason}"


@dataclass(frozen=True, slots=True)
class NoValidTokensError(ValueError):
    """Report a batch that cannot contribute to quantization."""

    @override
    def __str__(self) -> str:
        return "token mask contains no valid token"


@dataclass(frozen=True, slots=True)
class StaleStreamStateError(RuntimeError):
    """Report reuse of a consumed or foreign streaming state."""

    @override
    def __str__(self) -> str:
        return "stream state is stale or belongs to another tokenizer"
