"""Atomic checkpoint persistence with exact training-state restoration."""

from __future__ import annotations

import hashlib
import pickle
import random
import struct
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, NotRequired, Protocol

import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from torch import nn
from typing_extensions import TypedDict, override

from hri_ttr.checkpoints.schema import CheckpointBinding, CheckpointSnapshot

if TYPE_CHECKING:
    from pathlib import Path


class AdamState(TypedDict):
    """Per-parameter AdamW state emitted by PyTorch."""

    step: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    max_exp_avg_sq: NotRequired[torch.Tensor]


class AdamParamGroup(TypedDict):
    """AdamW parameter-group state required for exact resume."""

    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool
    maximize: bool
    foreach: bool | None
    capturable: bool
    differentiable: bool
    fused: bool | None
    decoupled_weight_decay: NotRequired[bool]
    initial_lr: NotRequired[float]
    params: list[int]


class AdamOptimizerState(TypedDict):
    """Complete AdamW state supported by this trainer."""

    state: dict[int, AdamState]
    param_groups: list[AdamParamGroup]


class ScalerPayload(BaseModel):
    """Pydantic boundary for optional CUDA scaler state."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", populate_by_name=True
    )

    scale: float | None = None
    growth_factor: float | None = None
    backoff_factor: float | None = None
    growth_interval: int | None = None
    growth_tracker: int | None = Field(default=None, alias="_growth_tracker")


class AdamOptimizerPayload(BaseModel):
    """Pydantic boundary for tensor-bearing optimizer state."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid"
    )

    state: dict[int, AdamState]
    param_groups: list[AdamParamGroup]


class CheckpointPayload(BaseModel):
    """Safe tensor-only payload accepted by weights-only loading."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid"
    )

    snapshot: str
    model: dict[str, torch.Tensor]
    optimizer: AdamOptimizerPayload
    scaler: ScalerPayload | None
    python_rng: tuple[int, tuple[int, ...], float | None]
    torch_rng: torch.Tensor
    cuda_rng: tuple[torch.Tensor, ...]


class CudaRngCapability(Protocol):
    """Capture and restore CUDA RNG without requiring CUDA in CPU tests."""

    def available(self) -> bool:
        """Return whether CUDA RNG state exists in this process."""
        ...

    def capture(self) -> tuple[torch.Tensor, ...]:
        """Return device-independent CUDA RNG byte tensors."""
        ...

    def restore(self, states: tuple[torch.Tensor, ...]) -> None:
        """Restore all CUDA generator states in device order."""
        ...


class TorchCudaRng:
    """Production CUDA RNG capability."""

    def available(self) -> bool:
        """Return the runtime CUDA availability."""
        return torch.cuda.is_available()

    def capture(self) -> tuple[torch.Tensor, ...]:
        """Copy every CUDA generator state to CPU storage."""
        return tuple(state.cpu() for state in torch.cuda.get_rng_state_all())

    def restore(self, states: tuple[torch.Tensor, ...]) -> None:
        """Restore CPU-stored bytes through PyTorch's CUDA boundary."""
        torch.cuda.set_rng_state_all(list(states))


class CheckpointMismatchError(ValueError):
    """Reports the expected and stored immutable tokenizer identities."""

    def __init__(
        self, expected: CheckpointBinding, stored: CheckpointBinding
    ) -> None:
        """Keep the bindings available without freezing exception internals."""
        super().__init__()
        self.expected = expected
        self.stored = stored

    @override
    def __str__(self) -> str:
        """Return a stable message without leaking artifact paths."""
        return "checkpoint binding does not match the requested tokenizer space"


class MalformedCheckpointError(ValueError):
    """Raised when a file is not an HRI-TTR training checkpoint."""


@dataclass(frozen=True, slots=True)
class CheckpointComponents:
    """Mutable PyTorch objects whose states are persisted together."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    scaler: torch.amp.GradScaler | None
    cuda_rng: CudaRngCapability = field(default_factory=TorchCudaRng)


def checkpoint_sha256(path: Path) -> str:
    """Return the lowercase SHA256 of a file without loading it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _rng_payload() -> tuple[tuple[int, tuple[int, ...], float | None], torch.Tensor]:
    return random.getstate(), torch.get_rng_state()


def save_training_checkpoint(
    path: Path,
    components: CheckpointComponents,
    snapshot: CheckpointSnapshot,
) -> None:
    """Atomically save model, optimizer, scaler, progress, binding, and RNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    python_rng, torch_rng = _rng_payload()
    optimizer_state = AdamOptimizerPayload.model_validate(
        components.optimizer.state_dict()
    )
    scaler_state = None
    if components.scaler is not None and components.scaler.is_enabled():
        scaler_state = ScalerPayload.model_validate(components.scaler.state_dict())
    payload = CheckpointPayload(
        snapshot=snapshot.model_dump_json(),
        model=dict(components.model.state_dict()),
        optimizer=optimizer_state,
        scaler=scaler_state,
        python_rng=python_rng,
        torch_rng=torch_rng,
        cuda_rng=(
            components.cuda_rng.capture() if components.cuda_rng.available() else ()
        ),
    )
    torch.save(payload.model_dump(), temporary)
    _ = temporary.replace(path)


def _restore_rng(payload: CheckpointPayload, capability: CudaRngCapability) -> None:
    random.setstate(payload.python_rng)
    torch.set_rng_state(payload.torch_rng.cpu())
    if payload.cuda_rng and capability.available():
        capability.restore(tuple(state.cpu() for state in payload.cuda_rng))


def load_training_checkpoint(
    path: Path,
    components: CheckpointComponents,
    expected: CheckpointBinding,
) -> CheckpointSnapshot:
    """Validate binding before mutating any supplied training component."""
    try:
        payload = CheckpointPayload.model_validate(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        snapshot = CheckpointSnapshot.model_validate_json(payload.snapshot)
    except (
        KeyError,
        EOFError,
        OSError,
        pickle.UnpicklingError,
        RuntimeError,
        struct.error,
        TypeError,
        ValidationError,
    ) as error:
        raise MalformedCheckpointError from error
    if snapshot.binding != expected:
        raise CheckpointMismatchError(expected, snapshot.binding)
    _ = components.model.load_state_dict(payload.model, strict=True)
    components.optimizer.load_state_dict(payload.optimizer.model_dump())
    scaler_state = payload.scaler
    if components.scaler is not None and scaler_state is not None:
        restored_scaler = scaler_state.model_dump(by_alias=True, exclude_none=True)
        components.scaler.load_state_dict(restored_scaler)
    _restore_rng(payload, components.cuda_rng)
    return snapshot
