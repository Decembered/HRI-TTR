"""Immutable contracts for later Human-to-G1 language training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isclose
from typing import TYPE_CHECKING, Final

from typing_extensions import override

if TYPE_CHECKING:
    from hri_ttr.contracts import TokenBatch, TokenizerSpec

EXPECTED_FPS: Final = 20.0
EXPECTED_FRAMES_PER_TOKEN: Final = 4
EXPECTED_CODEBOOK_SIZE: Final = 256


class LanguageContractReason(StrEnum):
    """Stable failures at the future motion-language boundary."""

    SEMANTIC = "semantic condition must contain non-whitespace text"
    HUMAN_DOMAIN = "human token domain is required"
    G1_DOMAIN = "G1 token domain is required"
    TIMING = "tokenizer timing must be 20 FPS and four frames per token"
    CODEBOOK = "tokenizer codebook size must be 256"
    PREFIX = "prefix must be one fully valid sequence"
    TARGET = "target must be one fully valid G1 token"
    TEACHER_TIMELINE = "offline teacher timeline must be H[0:k] + G[0:k-1] -> G[k]"
    STUDENT_TIMELINE = "online student timeline must be H[0:k] + G[0:k] -> G[k+1]"
    TOKENIZER_IDENTITY = "target G1 tokenizer must match the G1 prefix tokenizer"


@dataclass(frozen=True, slots=True)
class LanguageContractError(ValueError):
    """Reports one invalid Stage 3/4 example."""

    reason: LanguageContractReason

    @override
    def __str__(self) -> str:
        """Return a stable diagnostic for CLI and tests."""
        return self.reason


@dataclass(frozen=True, slots=True)
class SemanticCondition:
    """Unencoded semantic text carried beside motion-token histories."""

    text: str

    def __post_init__(self) -> None:
        """Reject an absent semantic condition at the input boundary."""
        if not self.text.strip():
            raise LanguageContractError(LanguageContractReason.SEMANTIC)


def _require_final_tokenizer(spec: TokenizerSpec, expected_kind: str) -> None:
    if spec.kind != expected_kind:
        reason = (
            LanguageContractReason.HUMAN_DOMAIN
            if expected_kind == "human"
            else LanguageContractReason.G1_DOMAIN
        )
        raise LanguageContractError(reason)
    if (
        not isclose(spec.input_fps, EXPECTED_FPS)
        or spec.frames_per_token != EXPECTED_FRAMES_PER_TOKEN
    ):
        raise LanguageContractError(LanguageContractReason.TIMING)
    if spec.codebook_size != EXPECTED_CODEBOOK_SIZE:
        raise LanguageContractError(LanguageContractReason.CODEBOOK)


def _require_prefix(batch: TokenBatch, expected_kind: str) -> int:
    _require_final_tokenizer(batch.tokenizer, expected_kind)
    if batch.token_ids.shape[0] != 1 or any(
        not bool(is_valid) for is_valid in batch.token_mask.flat
    ):
        raise LanguageContractError(LanguageContractReason.PREFIX)
    return len(batch.token_ids.T)


@dataclass(frozen=True, slots=True)
class OfflineTeacherExample:
    """Teacher example where G1 token ``k`` is the same-index target."""

    human_prefix: TokenBatch
    g1_prefix: TokenBatch
    target_g1: TokenBatch
    target_index: int
    semantic: SemanticCondition

    def __post_init__(self) -> None:
        """Enforce ``H[0:k] + G[0:k-1] + semantic -> G[k]``."""
        human_count = _require_prefix(self.human_prefix, "human")
        g1_count = _require_prefix(self.g1_prefix, "g1")
        target_count = _require_prefix(self.target_g1, "g1")
        if target_count != 1:
            raise LanguageContractError(LanguageContractReason.TARGET)
        if self.target_g1.tokenizer != self.g1_prefix.tokenizer:
            raise LanguageContractError(LanguageContractReason.TOKENIZER_IDENTITY)
        if (
            self.target_index < 0
            or human_count != self.target_index + 1
            or g1_count != self.target_index
        ):
            raise LanguageContractError(LanguageContractReason.TEACHER_TIMELINE)


@dataclass(frozen=True, slots=True)
class OnlineStudentExample:
    """Future student input where the unseen target is the next G1 token."""

    human_prefix: TokenBatch
    g1_prefix: TokenBatch
    target_index: int
    semantic: SemanticCondition

    def __post_init__(self) -> None:
        """Enforce ``H[0:k] + G[0:k] + semantic -> G[k+1]``."""
        human_count = _require_prefix(self.human_prefix, "human")
        g1_count = _require_prefix(self.g1_prefix, "g1")
        if (
            self.target_index < 1
            or human_count != self.target_index
            or g1_count != self.target_index
        ):
            raise LanguageContractError(LanguageContractReason.STUDENT_TIMELINE)
