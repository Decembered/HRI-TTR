"""Mask-aware VQ codebook usage metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, fsum, log
from typing import Final

import numpy as np
import numpy.typing as npt

from hri_ttr.evaluation.errors import reject_evaluation

CODEBOOK_SIZE: Final = 256


@dataclass(frozen=True, slots=True)
class CodebookStatistics:
    """Usage statistics computed only from valid token IDs."""

    histogram: npt.NDArray[np.int64]
    perplexity: float
    used_code_count: int
    dead_code_ratio: float
    minimum_token_id: int
    maximum_token_id: int


def codebook_statistics(
    token_ids: npt.NDArray[np.int64], token_mask: npt.NDArray[np.bool_]
) -> CodebookStatistics:
    """Measure codebook use after excluding every masked token."""
    if token_ids.shape != token_mask.shape or token_mask.dtype != np.bool_:
        reject_evaluation("token IDs and boolean mask must have matching shapes")
    valid = tuple(
        int(token)
        for token, keep in zip(token_ids.flat, token_mask.flat, strict=False)
        if bool(keep)
    )
    if not valid or any(token < 0 or token >= CODEBOOK_SIZE for token in valid):
        reject_evaluation("valid token IDs must be nonempty and in [0,255]")
    histogram = np.zeros(CODEBOOK_SIZE, dtype=np.int64)
    for token in valid:
        histogram[token] += 1
    probabilities = tuple(
        int(count) / len(valid) for count in histogram.flat if int(count) > 0
    )
    perplexity = exp(
        -fsum(probability * log(probability) for probability in probabilities)
    )
    used = len(probabilities)
    return CodebookStatistics(
        histogram,
        perplexity,
        used,
        (CODEBOOK_SIZE - used) / CODEBOOK_SIZE,
        min(valid),
        max(valid),
    )
