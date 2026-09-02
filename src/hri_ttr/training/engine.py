"""Typed training engine shared by local and torchrun execution."""

from __future__ import annotations

import math
import signal
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import torch
from torch import nn

from hri_ttr.checkpoints import TrainingProgress
from hri_ttr.training.data import TrainingBatch, WindowDataset, collate_windows

if TYPE_CHECKING:
    from collections.abc import Iterator

    from hri_ttr.tokenizers.common.contracts import TokenizerOutput
    from hri_ttr.tokenizers.common.model import CausalMotionTokenizer
    from hri_ttr.training.config import TrainConfig
    from hri_ttr.training.distributed import DistributedContext
    from hri_ttr.training.losses import MaskedReconstructionLoss
    from hri_ttr.training.monitoring import TrainingMonitor
    from hri_ttr.training.signals import StopController


class DistributedTokenizer(Protocol):
    """Callable surface used from a distributed tokenizer wrapper."""

    def __call__(
        self, features: torch.Tensor, frame_mask: torch.Tensor
    ) -> TokenizerOutput:
        """Run the wrapped tokenizer while preserving its typed output."""
        ...


@dataclass(frozen=True, slots=True)
class EngineInputs:
    """Objects and immutable configuration consumed by a training loop."""

    model: CausalMotionTokenizer
    distributed_model: DistributedTokenizer | None
    training_data: WindowDataset
    validation_data: WindowDataset
    config: TrainConfig
    distributed: DistributedContext
    optimizer: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    loss: MaskedReconstructionLoss
    stop: StopController
    monitor: TrainingMonitor


@dataclass(frozen=True, slots=True)
class StepOutcome:
    """Completed progress and whether an interruption ended the loop."""

    progress: TrainingProgress
    interrupted: bool


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    """Validation loss exists only when every batch completed."""

    loss: float | None


def _epoch_indices(inputs: EngineInputs, epoch: int) -> list[int]:
    generator = torch.Generator()
    _ = generator.manual_seed(inputs.config.seed + epoch)
    order = [
        int(value.item())
        for value in torch.randperm(len(inputs.training_data), generator=generator)
    ]
    if inputs.distributed.world_size == 1:
        return order
    usable = len(order) // inputs.distributed.world_size * inputs.distributed.world_size
    return order[inputs.distributed.rank : usable : inputs.distributed.world_size]


def _batches(
    dataset: WindowDataset, indices: list[int], batch_size: int
) -> Iterator[TrainingBatch]:
    for start in range(0, len(indices), batch_size):
        selected = [dataset[index] for index in indices[start : start + batch_size]]
        yield collate_windows(selected)


def _forward(
    inputs: EngineInputs, features: torch.Tensor, frame_mask: torch.Tensor
) -> TokenizerOutput:
    if inputs.distributed_model is None:
        return inputs.model.forward(features, frame_mask)
    return inputs.distributed_model(features, frame_mask)


def _train_batch(inputs: EngineInputs, batch: TrainingBatch) -> dict[str, float] | None:
    try:
        started_at = time.perf_counter()
        features = batch.features.to(inputs.distributed.device)
        frame_mask = batch.frame_mask.to(inputs.distributed.device)
        inputs.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=inputs.distributed.device.type,
            enabled=inputs.config.amp,
            dtype=(
                torch.float16
                if inputs.distributed.device.type == "cuda"
                else torch.bfloat16
            ),
        ):
            output = _forward(inputs, features, frame_mask)
            reconstruction = inputs.loss.forward(
                output.reconstruction, features, frame_mask
            )
            total = reconstruction + output.encoding.commitment_loss
        _ = inputs.scaler.scale(total).backward()
        inputs.scaler.unscale_(inputs.optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(
            inputs.model.parameters(), inputs.config.gradient_clip_norm
        )
        _ = inputs.scaler.step(inputs.optimizer)
        inputs.scaler.update()
    except KeyboardInterrupt:
        inputs.stop.request(signal.SIGINT)
        return None
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    memory_gib = 0.0
    if inputs.distributed.device.type == "cuda":
        memory_gib = torch.cuda.max_memory_allocated(inputs.distributed.device) / (
            1024**3
        )
        torch.cuda.reset_peak_memory_stats(inputs.distributed.device)
    return {
        "total_loss": float(total.detach().item()),
        "reconstruction_loss": float(reconstruction.detach().item()),
        "commitment_loss": float(output.encoding.commitment_loss.detach().item()),
        "codebook_perplexity": float(output.encoding.perplexity.detach().item()),
        "gradient_norm": float(gradient_norm.detach().item()),
        "learning_rate": inputs.config.learning_rate,
        "samples_per_second": (
            features.shape[0] * inputs.distributed.world_size / elapsed
        ),
        "cuda_peak_memory_gib": memory_gib,
    }


def _periodic_validation(
    inputs: EngineInputs, progress: TrainingProgress
) -> TrainingProgress | None:
    validation = validate(inputs)
    if validation.loss is None:
        return None
    updated = inputs.monitor.record_validation(progress, validation.loss)
    _ = inputs.model.train()
    return updated


def _after_step(
    inputs: EngineInputs,
    progress: TrainingProgress,
    metrics: dict[str, float],
) -> TrainingProgress | None:
    step = progress.global_step
    if step % inputs.config.log_every_steps == 0:
        inputs.monitor.record_training(step, metrics)
    if step % inputs.config.validation_every_steps == 0:
        return _periodic_validation(inputs, progress)
    return progress


def run_steps(inputs: EngineInputs, progress: TrainingProgress) -> StepOutcome:
    """Continue from an exact epoch/batch position to the configured limit."""
    epoch = progress.epoch
    batch_in_epoch = progress.batch_in_epoch
    step = progress.global_step
    best_validation_loss = progress.best_validation_loss
    while epoch < inputs.config.epochs and step < inputs.config.max_steps:
        _ = inputs.model.train()
        indices = _epoch_indices(inputs, epoch)
        batch_count = math.ceil(len(indices) / inputs.config.batch_size)
        for index, batch in enumerate(
            _batches(inputs.training_data, indices, inputs.config.batch_size)
        ):
            if inputs.stop.requested:
                break
            if index < batch_in_epoch:
                continue
            metrics = _train_batch(inputs, batch)
            if metrics is None:
                break
            step += 1
            batch_in_epoch = index + 1
            current = TrainingProgress(
                epoch=epoch,
                batch_in_epoch=batch_in_epoch,
                global_step=step,
                best_validation_loss=best_validation_loss,
            )
            updated = _after_step(inputs, current, metrics)
            if updated is None:
                break
            best_validation_loss = updated.best_validation_loss
            if step >= inputs.config.max_steps or inputs.stop.requested:
                break
        if inputs.stop.requested:
            break
        if batch_in_epoch >= batch_count:
            epoch += 1
            batch_in_epoch = 0
    return StepOutcome(
        progress=TrainingProgress(
            epoch=epoch,
            batch_in_epoch=batch_in_epoch,
            global_step=step,
            best_validation_loss=best_validation_loss,
        ),
        interrupted=inputs.stop.requested,
    )


def validate(inputs: EngineInputs) -> ValidationOutcome:
    """Evaluate all validation windows and average across torchrun ranks."""
    _ = inputs.model.eval()
    total = torch.zeros((), device=inputs.distributed.device)
    batches = torch.zeros((), device=inputs.distributed.device)
    indices = list(
        range(
            inputs.distributed.rank,
            len(inputs.validation_data),
            inputs.distributed.world_size,
        )
    )
    with torch.no_grad():
        for batch in _batches(
            inputs.validation_data, indices, inputs.config.batch_size
        ):
            if inputs.stop.requested:
                return ValidationOutcome(None)
            try:
                features = batch.features.to(inputs.distributed.device)
                frame_mask = batch.frame_mask.to(inputs.distributed.device)
                output = _forward(inputs, features, frame_mask)
                batch_loss = inputs.loss.forward(
                    output.reconstruction, features, frame_mask
                )
            except KeyboardInterrupt:
                inputs.stop.request(signal.SIGINT)
                return ValidationOutcome(None)
            total += batch_loss
            batches += 1
            if inputs.stop.requested:
                return ValidationOutcome(None)
    if inputs.stop.requested:
        return ValidationOutcome(None)
    if inputs.distributed.world_size > 1:
        _ = cast("object", torch.distributed.all_reduce(total))
        _ = cast("object", torch.distributed.all_reduce(batches))
    return ValidationOutcome(float((total / batches).item()))
