"""Atomic terminal transitions for a training run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hri_ttr.checkpoints import CheckpointSnapshot, save_training_checkpoint
from hri_ttr.training.results import TrainingInterrupted, TrainingResult

if TYPE_CHECKING:
    from hri_ttr.checkpoints import (
        CheckpointBinding,
        CheckpointComponents,
        TrainingProgress,
    )
    from hri_ttr.training.config import TrainConfig
    from hri_ttr.training.distributed import DistributedContext
    from hri_ttr.training.signals import StopController


@dataclass(frozen=True, slots=True)
class ArtifactState:
    """State required to persist one terminal training outcome."""

    config: TrainConfig
    context: DistributedContext
    components: CheckpointComponents
    binding: CheckpointBinding
    progress: TrainingProgress
    stop: StopController


def save_interrupted(state: ArtifactState) -> TrainingInterrupted:
    """Persist progress without claiming a partial validation result."""
    path = state.config.output_dir / "interrupted.pt"
    snapshot = CheckpointSnapshot(binding=state.binding, progress=state.progress)
    if state.context.is_primary:
        save_training_checkpoint(path, state.components, snapshot)
    return TrainingInterrupted(
        state.progress.global_step,
        state.progress.best_validation_loss,
        path,
        state.binding,
        state.stop.exit_code,
    )


def save_completed(
    state: ArtifactState, validation_loss: float
) -> TrainingResult | TrainingInterrupted:
    """Persist normal artifacts unless a stop request wins the transition."""
    if state.stop.requested:
        return save_interrupted(state)
    previous_best = state.progress.best_validation_loss
    best_loss = min(validation_loss, previous_best)
    progress = state.progress.model_copy(update={"best_validation_loss": best_loss})
    snapshot = CheckpointSnapshot(binding=state.binding, progress=progress)
    last_path = state.config.output_dir / "last.pt"
    best_path = state.config.output_dir / "best.pt"
    if state.context.is_primary:
        save_training_checkpoint(last_path, state.components, snapshot)
        if state.stop.requested:
            return save_interrupted(state)
        if validation_loss < previous_best:
            save_training_checkpoint(best_path, state.components, snapshot)
        if state.stop.requested:
            return save_interrupted(state)
    return TrainingResult(
        state.progress.global_step,
        best_loss,
        last_path,
        best_path,
        state.binding,
    )
