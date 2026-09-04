"""Typed training engine shared by local and torchrun execution."""

from __future__ import annotations

import math
import signal
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
from torch import distributed, nn

from hri_ttr.checkpoints import TrainingProgress
from hri_ttr.training.data import TrainingBatch, WindowDataset, collate_windows

if TYPE_CHECKING:
    from collections.abc import Iterator

    from hri_ttr.tokenizers.common.contracts import TokenizerOutput
    from hri_ttr.tokenizers.common.model import CausalMotionTokenizer
    from hri_ttr.training.config import TrainConfig
    from hri_ttr.training.distributed import DistributedContext
    from hri_ttr.training.losses import MaskedReconstructionLoss
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


@dataclass(frozen=True, slots=True)
class StepOutcome:
    """Completed progress and whether an interruption ended the loop."""

    progress: TrainingProgress
    interrupted: bool


@dataclass(frozen=True, slots=True)
class StepMetrics:
    """Global-mean metrics produced after one optimizer step."""

    global_step: int
    total_loss: float
    reconstruction_loss: float
    commitment_loss: float
    grad_norm: float
    perplexity: float
    learning_rate: float


StepHook = Callable[[StepMetrics, TrainingProgress], TrainingProgress]


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


def _forward(inputs: EngineInputs, batch: TrainingBatch) -> TokenizerOutput:
    features = batch.features.to(inputs.distributed.device)
    frame_mask = batch.frame_mask.to(inputs.distributed.device)
    if inputs.distributed_model is None:
        return inputs.model.forward(features, frame_mask)
    return inputs.distributed_model(features, frame_mask)


def _global_mean(value: torch.Tensor, inputs: EngineInputs) -> float:
    """Return a scalar mean across all ranks without building a graph."""
    reduced = value.detach().float()
    if (
        inputs.distributed.world_size > 1
        and distributed.is_available()
        and distributed.is_initialized()
    ):
        distributed.all_reduce(reduced, op=distributed.ReduceOp.SUM)
        reduced = reduced / inputs.distributed.world_size
    return float(reduced.item())


def run_steps(
    inputs: EngineInputs,
    progress: TrainingProgress,
    on_step: StepHook | None = None,
) -> StepOutcome:
    """Continue from an exact epoch/batch position to the configured limit."""
    epoch = progress.epoch
    batch_in_epoch = progress.batch_in_epoch
    step = progress.global_step
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
            try:
                features = batch.features.to(inputs.distributed.device)
                frame_mask = batch.frame_mask.to(inputs.distributed.device)
                inputs.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=inputs.distributed.device.type,
                    enabled=inputs.config.amp,
                    dtype=torch.float16
                    if inputs.distributed.device.type == "cuda"
                    else torch.bfloat16,
                ):
                    output = _forward(inputs, batch)
                    reconstruction = inputs.loss.forward(
                        output.reconstruction, features, frame_mask
                    )
                    total = reconstruction + output.encoding.commitment_loss
                scaled = inputs.scaler.scale(total)
                torch.autograd.backward(scaled)
                inputs.scaler.unscale_(inputs.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    inputs.model.parameters(), inputs.config.gradient_clip_norm
                )
                _ = inputs.scaler.step(inputs.optimizer)
                inputs.scaler.update()
            except KeyboardInterrupt:
                inputs.stop.request(signal.SIGINT)
                break
            step += 1
            batch_in_epoch = index + 1
            current = TrainingProgress(
                epoch=epoch,
                batch_in_epoch=batch_in_epoch,
                global_step=step,
                best_validation_loss=progress.best_validation_loss,
            )
            if on_step is not None:
                current = on_step(
                    StepMetrics(
                        global_step=step,
                        total_loss=_global_mean(total, inputs),
                        reconstruction_loss=_global_mean(reconstruction, inputs),
                        commitment_loss=_global_mean(
                            output.encoding.commitment_loss, inputs
                        ),
                        grad_norm=_global_mean(grad_norm, inputs),
                        perplexity=_global_mean(output.encoding.perplexity, inputs),
                        learning_rate=float(inputs.optimizer.param_groups[0]["lr"]),
                    ),
                    current,
                )
                epoch = current.epoch
                batch_in_epoch = current.batch_in_epoch
                step = current.global_step
                progress = current
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
            best_validation_loss=progress.best_validation_loss,
        ),
        interrupted=inputs.stop.requested,
    )


def validate(inputs: EngineInputs) -> ValidationOutcome:
    """Evaluate all validation windows and average across torchrun ranks."""
    _ = inputs.model.eval()
    total = torch.zeros((), device=inputs.distributed.device)
    batches = torch.zeros((), device=inputs.distributed.device)
    indices = list(range(len(inputs.validation_data)))
    with torch.no_grad():
        for batch in _batches(
            inputs.validation_data, indices, inputs.config.batch_size
        ):
            if inputs.stop.requested:
                return ValidationOutcome(None)
            try:
                features = batch.features.to(inputs.distributed.device)
                frame_mask = batch.frame_mask.to(inputs.distributed.device)
                output = _forward(inputs, batch)
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
    if (
        inputs.distributed.world_size > 1
        and distributed.is_available()
        and distributed.is_initialized()
    ):
        distributed.all_reduce(total, op=distributed.ReduceOp.SUM)
        distributed.all_reduce(batches, op=distributed.ReduceOp.SUM)
    return ValidationOutcome(float((total / batches).item()))
