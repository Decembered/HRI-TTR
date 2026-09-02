"""Small torchrun environment adapter used by the training loop."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import distributed


@dataclass(frozen=True, slots=True)
class DistributedContext:
    """Resolved rank, world size, device, and process-group ownership."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    owns_process_group: bool

    @property
    def is_primary(self) -> bool:
        """Return whether this process owns persistent artifacts."""
        return self.rank == 0


def initialize_distributed() -> DistributedContext:
    """Initialize from torchrun variables, or use a single local process."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    owns_group = False
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not distributed.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        distributed.init_process_group(backend=backend)
        owns_group = True
    return DistributedContext(rank, world_size, local_rank, device, owns_group)


def finalize_distributed(context: DistributedContext) -> None:
    """Close only a process group created by this training invocation."""
    if context.owns_process_group:
        distributed.destroy_process_group()
