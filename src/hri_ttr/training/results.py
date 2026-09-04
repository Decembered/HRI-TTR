"""Typed outcomes from the training process boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from hri_ttr.checkpoints import CheckpointBinding


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Artifact locations and final identity returned to callers."""

    global_step: int
    best_validation_loss: float
    last_checkpoint: Path
    best_checkpoint: Path
    binding: CheckpointBinding


@dataclass(frozen=True, slots=True)
class TrainingInterrupted:
    """Typed non-success result with its resumable checkpoint."""

    global_step: int
    best_validation_loss: float
    interrupted_checkpoint: Path
    binding: CheckpointBinding
    exit_code: int
