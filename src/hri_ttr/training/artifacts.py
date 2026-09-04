"""Atomic terminal transitions for a training run."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from hri_ttr.checkpoints import CheckpointSnapshot, save_training_checkpoint
from hri_ttr.training.results import TrainingInterrupted, TrainingResult

if TYPE_CHECKING:
    from pathlib import Path

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


def _mirror_atomic(source: Path, target: Path) -> None:
    """Create an atomic compatibility copy without serializing twice."""
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    _ = temporary.replace(target)


def save_latest(state: ArtifactState) -> Path:
    """Persist the newest checkpoint and the legacy ``last.pt`` alias."""
    path = state.config.output_dir / "latest.pt"
    snapshot = CheckpointSnapshot(binding=state.binding, progress=state.progress)
    if state.context.is_primary:
        save_training_checkpoint(path, state.components, snapshot)
        _mirror_atomic(path, state.config.output_dir / "last.pt")
    return path


def save_best(state: ArtifactState) -> Path:
    """Persist the checkpoint associated with the best validation loss."""
    path = state.config.output_dir / "best.pt"
    snapshot = CheckpointSnapshot(binding=state.binding, progress=state.progress)
    if state.context.is_primary:
        save_training_checkpoint(path, state.components, snapshot)
    return path


def save_interrupted(state: ArtifactState) -> TrainingInterrupted:
    """Persist progress without claiming a partial validation result."""
    path = state.config.output_dir / "interrupted.pt"
    snapshot = CheckpointSnapshot(binding=state.binding, progress=state.progress)
    if state.context.is_primary:
        save_training_checkpoint(path, state.components, snapshot)
        save_latest(state)
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
    completed_state = replace(state, progress=progress)
    best_path = state.config.output_dir / "best.pt"
    if state.context.is_primary:
        save_latest(completed_state)
        if state.stop.requested:
            return save_interrupted(state)
        if validation_loss < previous_best or not best_path.exists():
            save_best(completed_state)
        if state.stop.requested:
            return save_interrupted(state)
    return TrainingResult(
        state.progress.global_step,
        best_loss,
        state.config.output_dir / "last.pt",
        best_path,
        state.binding,
    )
