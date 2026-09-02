"""Typed training boundary failures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import override


class TrainingReason(StrEnum):
    """Stable reasons suitable for tests and CLI presentation."""

    ARCHITECTURE = "model architecture does not match training config"
    CODEBOOK = "the token protocol requires 256 codebook entries"
    DATASET_EMPTY = "training dataset requires at least one window"
    FEATURE_DTYPE = "feature sequence must use float32"
    FEATURE_FINITE = "feature sequence must be finite"
    FEATURE_MASK = "feature mask must be non-empty bool data aligned with features"
    FEATURE_SHAPE = "feature sequence must be named, non-empty, and rank two"
    FEATURE_WIDTH = "model feature dimension does not match training kind"
    G1_WARM_START = "G1 75D training cannot warm-start from the 73D baseline"
    PLACEHOLDER_HASH = "artifact hashes cannot use an all-zero placeholder"
    RESUME_COMPLETE = "resume checkpoint has already reached max_steps"
    SCHEMA = "representation schema does not match model kind"
    UNSUPPORTED_LOSS = "unsupported reconstruction schema"
    WANDB_INIT = "W&B did not create a run"
    WANDB_IDENTITY = "W&B project, run name, and run id must be configured together"
    WARM_START_DOMAIN = "only Human training supports warm-start"
    WARM_START_HASH = "warm-start checkpoint hash does not match config"
    WINDOW_ALIGNMENT = "training windows must align to four-frame tokens"
    WINDOW_DIMENSION = "window dimensions must be positive"


@dataclass(frozen=True, slots=True)
class TrainingError(ValueError):
    """Training failure with a stable machine-readable reason."""

    reason: TrainingReason

    @override
    def __str__(self) -> str:
        """Return the stable reason text."""
        return self.reason.value
