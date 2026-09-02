"""Report long-running training metrics from rank zero."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import torch

from hri_ttr.training.artifacts import ArtifactState, save_completed
from hri_ttr.training.errors import TrainingError, TrainingReason

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


MetricValue = float | int


class _WandbRun(Protocol):
    def log(self, data: dict[str, MetricValue], *, step: int) -> None: ...

    def finish(self) -> None: ...


class _WandbModule(Protocol):
    def init(self, **kwargs: object) -> _WandbRun | None: ...


@dataclass(frozen=True, slots=True)
class MonitorArtifacts:
    """Mutable checkpoint components owned by one training run."""

    components: CheckpointComponents
    binding: CheckpointBinding
    stop: StopController


class TrainingMonitor:
    """Persist local metrics and mirror them to W&B when configured."""

    _config: TrainConfig
    _context: DistributedContext
    _artifacts: MonitorArtifacts

    def __init__(
        self,
        config: TrainConfig,
        context: DistributedContext,
        artifacts: MonitorArtifacts,
    ) -> None:
        """Open the rank-zero metric sinks."""
        self._config = config
        self._context = context
        self._artifacts = artifacts
        self._metrics_path: Path = config.output_dir / "metrics.jsonl"
        self._run: _WandbRun | None = None
        if not context.is_primary:
            return
        config.output_dir.mkdir(parents=True, exist_ok=True)
        if config.wandb_project is None:
            return
        wandb = cast("_WandbModule", cast("object", importlib.import_module("wandb")))
        run = wandb.init(
            project=config.wandb_project,
            name=cast("str", config.wandb_run_name),
            id=cast("str", config.wandb_run_id),
            resume="allow",
            config=cast("dict[str, object]", config.model_dump(mode="json")),
            dir=str(config.output_dir),
        )
        if run is None:
            raise TrainingError(TrainingReason.WANDB_INIT)
        self._run = run

    def record_training(self, step: int, metrics: dict[str, float]) -> None:
        """Average rank metrics and write one training point."""
        names = tuple(metrics)
        values = torch.tensor(
            [metrics[name] for name in names],
            dtype=torch.float64,
            device=self._context.device,
        )
        if self._context.world_size > 1:
            _ = cast("object", torch.distributed.all_reduce(values))
            values /= self._context.world_size
        payload: dict[str, MetricValue] = {"step": step}
        payload.update(
            {
                f"train/{name}": float(value)
                for name, value in zip(names, values, strict=True)
            }
        )
        self._write(step, payload)

    def record_validation(
        self, progress: TrainingProgress, validation_loss: float
    ) -> TrainingProgress:
        """Write validation metrics and refresh resumable checkpoints."""
        state = ArtifactState(
            self._config,
            self._context,
            self._artifacts.components,
            self._artifacts.binding,
            progress,
            self._artifacts.stop,
        )
        result = save_completed(state, validation_loss)
        self.report_validation(progress.global_step, validation_loss)
        return progress.model_copy(
            update={"best_validation_loss": result.best_validation_loss}
        )

    def report_validation(self, step: int, validation_loss: float) -> None:
        """Write one validation point without saving another checkpoint."""
        payload: dict[str, MetricValue] = {
            "step": step,
            "validation/loss": validation_loss,
        }
        self._write(step, payload)

    def close(self) -> None:
        """Flush and close the optional W&B run."""
        if self._run is not None:
            self._run.finish()

    def _write(self, step: int, payload: dict[str, MetricValue]) -> None:
        if not self._context.is_primary:
            return
        with self._metrics_path.open("a", encoding="utf-8") as stream:
            _ = stream.write(json.dumps(payload, sort_keys=True) + "\n")
        if self._run is not None:
            self._run.log(payload, step=step)
