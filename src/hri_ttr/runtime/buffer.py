"""Stateful causal token buffer for a future online student."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import final

import numpy as np
from typing_extensions import override

from hri_ttr.contracts import TokenBatch, TokenizerSpec
from hri_ttr.language import OnlineStudentExample, SemanticCondition


class RuntimeBufferReason(StrEnum):
    """Stable failures at the online prefix boundary."""

    HUMAN_RANGE = "Human token ID is outside its codebook"
    G1_RANGE = "G1 token ID is outside its codebook"
    EMPTY = "at least one aligned observation is required"


@dataclass(frozen=True, slots=True)
class RuntimeBufferError(ValueError):
    """Reports an invalid online observation or unavailable snapshot."""

    reason: RuntimeBufferReason

    @override
    def __str__(self) -> str:
        """Return the boundary failure detail."""
        return self.reason


@final
class CausalPrefixBuffer:
    """Accumulate aligned Human/G1 observations for next-token inference."""

    def __init__(
        self,
        human_tokenizer: TokenizerSpec,
        g1_tokenizer: TokenizerSpec,
        semantic: SemanticCondition,
    ) -> None:
        """Create an empty causal history with fixed tokenizer identities."""
        self._human_tokenizer = human_tokenizer
        self._g1_tokenizer = g1_tokenizer
        self._semantic = semantic
        self._human_ids: list[int] = []
        self._g1_ids: list[int] = []

    def append_observation(self, human_token_id: int, g1_token_id: int) -> None:
        """Atomically append one aligned token pair after range validation."""
        if not 0 <= human_token_id < self._human_tokenizer.codebook_size:
            raise RuntimeBufferError(RuntimeBufferReason.HUMAN_RANGE)
        if not 0 <= g1_token_id < self._g1_tokenizer.codebook_size:
            raise RuntimeBufferError(RuntimeBufferReason.G1_RANGE)
        self._human_ids.append(human_token_id)
        self._g1_ids.append(g1_token_id)

    def student_context(self) -> OnlineStudentExample:
        """Return an immutable prefix snapshot for predicting the next chunk."""
        if not self._human_ids:
            raise RuntimeBufferError(RuntimeBufferReason.EMPTY)
        human_ids = np.asarray([self._human_ids], dtype=np.int64)
        g1_ids = np.asarray([self._g1_ids], dtype=np.int64)
        human_mask = np.ones_like(human_ids, dtype=np.bool_)
        g1_mask = np.ones_like(g1_ids, dtype=np.bool_)
        return OnlineStudentExample(
            human_prefix=TokenBatch(
                human_ids,
                human_mask,
                self._human_tokenizer,
            ),
            g1_prefix=TokenBatch(g1_ids, g1_mask, self._g1_tokenizer),
            target_index=len(self._human_ids),
            semantic=self._semantic,
        )
