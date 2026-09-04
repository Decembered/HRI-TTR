"""Rank-zero Weights & Biases logging for tokenizer training."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping

    from hri_ttr.training.config import TrainConfig, TrainingIdentity
    from hri_ttr.training.distributed import DistributedContext
    from hri_ttr.training.engine import StepMetrics


class WandbUnavailableError(RuntimeError):
    """Raised when an enabled W&B run cannot import its client library."""


class WandbRunIDError(RuntimeError):
    """Raised when the persistent local W&B run ID is malformed."""

    def __init__(self, path: object) -> None:
        """Include the exact local metadata path in the diagnostic."""
        super().__init__(f"empty W&B run ID in {path}")


class _WandbRun(Protocol):
    """Minimal W&B run surface used by this project."""

    url: str | None
    summary: MutableMapping[str, object]

    def log(self, data: dict[str, float | int], *, step: int) -> None:
        """Append metrics at one monotonic global step."""
        ...

    def finish(self) -> None:
        """Close the run and flush its pending records."""
        ...


class _WandbUtil(Protocol):
    """Minimal W&B utility surface used for stable local run IDs."""

    def generate_id(self) -> str:
        """Generate a W&B-compatible run ID."""
        ...


class _WandbModule(Protocol):
    """Minimal dynamically imported W&B module surface."""

    init: Callable[..., _WandbRun]
    util: _WandbUtil


@dataclass(slots=True)
class WandbLogger:
    """Log one independent Human or G1 run from rank zero only."""

    run: _WandbRun | None = None
    log_every_steps: int = 1

    @classmethod
    def start(
        cls,
        config: TrainConfig,
        identity: TrainingIdentity,
        context: DistributedContext,
    ) -> WandbLogger:
        """Start W&B when enabled, leaving all non-primary ranks inert."""
        logger = cls(log_every_steps=config.log_every_steps)
        if not config.wandb_enabled or not context.is_primary:
            return logger
        try:
            module = cast("_WandbModule", importlib.import_module("wandb"))
        except ModuleNotFoundError as error:
            raise WandbUnavailableError from error

        config.output_dir.mkdir(parents=True, exist_ok=True)
        run_id_path = config.output_dir / "wandb_run_id.txt"
        if run_id_path.is_file():
            run_id = run_id_path.read_text(encoding="utf-8").strip()
        else:
            run_id = module.util.generate_id()
            run_id_path.write_text(f"{run_id}\n", encoding="utf-8")
        if not run_id:
            raise WandbRunIDError(run_id_path)

        run_config = config.model_dump(mode="json")
        run_config.update(
            {
                "identity/normalizer_sha256": identity.normalizer_sha256,
                "identity/split_sha256": identity.split_sha256,
                "identity/source_sha256": identity.source_sha256,
            }
        )
        run = module.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name or f"{config.model_kind.value}-causal-vqvae",
            id=run_id,
            resume="allow",
            mode=config.wandb_mode,
            dir=str(config.output_dir),
            config=run_config,
        )
        logger.run = run
        _ = sys.stdout.write(f"W&B run: {run.url or run_id}\n")
        _ = sys.stdout.flush()
        return logger

    def log(self, metrics: StepMetrics, validation_loss: float | None = None) -> None:
        """Log train metrics periodically and validation metrics when available."""
        if self.run is None:
            return
        if (
            metrics.global_step % self.log_every_steps != 0
            and validation_loss is None
        ):
            return
        data: dict[str, float | int] = {
            "train/total_loss": metrics.total_loss,
            "train/reconstruction_loss": metrics.reconstruction_loss,
            "train/commitment_loss": metrics.commitment_loss,
            "train/grad_norm": metrics.grad_norm,
            "train/perplexity": metrics.perplexity,
            "train/learning_rate": metrics.learning_rate,
        }
        if validation_loss is not None:
            data["val/loss"] = validation_loss
        self.run.log(data, step=metrics.global_step)

    def set_summary(self, values: dict[str, str | float | int]) -> None:
        """Expose final artifact paths and metrics in the W&B run summary."""
        if self.run is not None:
            self.run.summary.update(values)

    def finish(self) -> None:
        """Flush and close the rank-zero W&B run."""
        if self.run is not None:
            self.run.finish()
