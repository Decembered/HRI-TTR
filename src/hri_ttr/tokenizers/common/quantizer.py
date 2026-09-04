"""Mask-aware exponential-moving-average vector quantization."""

from __future__ import annotations

import torch
from torch import distributed, nn
from torch.nn import functional
from typing_extensions import override

from hri_ttr.tokenizers.common.contracts import Encoding
from hri_ttr.tokenizers.common.errors import NoValidTokensError


class EMAVectorQuantizer(nn.Module):
    """Own serialized codebook and EMA state for exactly one token domain."""

    codebook: torch.Tensor
    ema_sum: torch.Tensor
    ema_count: torch.Tensor
    initialized: torch.Tensor
    codebook_size: int
    decay: float
    commitment_weight: float
    _distributed: bool

    def __init__(
        self,
        codebook_size: int,
        code_dim: int,
        decay: float,
        commitment_weight: float,
    ) -> None:
        """Initialize one independent serialized EMA codebook."""
        super().__init__()
        initial = torch.randn(codebook_size, code_dim)
        initial = functional.normalize(initial, dim=1)
        self.register_buffer("codebook", initial)
        self.register_buffer("ema_sum", initial.clone())
        self.register_buffer("ema_count", torch.zeros(codebook_size))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        self.codebook = self.get_buffer("codebook")
        self.ema_sum = self.get_buffer("ema_sum")
        self.ema_count = self.get_buffer("ema_count")
        self.initialized = self.get_buffer("initialized")
        self.codebook_size = codebook_size
        self.decay = decay
        self.commitment_weight = commitment_weight
        self._distributed = False

    def set_distributed(self, enabled: bool) -> None:
        """Enable global EMA statistics when the model runs under DDP."""
        self._distributed = enabled

    def _distributed_active(self) -> bool:
        return (
            self._distributed
            and distributed.is_available()
            and distributed.is_initialized()
            and distributed.get_world_size() > 1
        )

    def _initialize(self, candidates: torch.Tensor) -> None:
        repeats = (self.codebook_size + candidates.shape[0] - 1) // candidates.shape[0]
        candidates = candidates.repeat(repeats, 1)[: self.codebook_size]
        _ = self.codebook.copy_(candidates)
        _ = self.ema_sum.copy_(candidates)
        _ = self.ema_count.fill_(1.0)
        initialized = torch.ones((), dtype=torch.bool, device=self.initialized.device)
        _ = self.initialized.copy_(initialized)

    @torch.no_grad()
    def _update(
        self,
        valid: torch.Tensor,
        valid_ids: torch.Tensor,
        candidates: torch.Tensor,
    ) -> bool:
        one_hot = functional.one_hot(valid_ids, self.codebook_size).to(torch.float32)
        counts = one_hot.sum(dim=0)
        sums = one_hot.transpose(0, 1) @ valid.float()
        if self._distributed_active():
            distributed.all_reduce(counts, op=distributed.ReduceOp.SUM)
            distributed.all_reduce(sums, op=distributed.ReduceOp.SUM)
        if not bool(counts.sum().item() > 0.0):
            return False
        _ = self.ema_count.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
        _ = self.ema_sum.mul_(self.decay).add_(sums, alpha=1.0 - self.decay)
        if candidates.shape[0] > 0:
            repeats = (
                self.codebook_size + candidates.shape[0] - 1
            ) // candidates.shape[0]
            replacement = candidates.repeat(repeats, 1)[: self.codebook_size]
            dead = self.ema_count < 1.0
            _ = self.ema_sum.copy_(
                torch.where(dead.unsqueeze(1), replacement, self.ema_sum)
            )
            _ = self.ema_count.copy_(
                torch.where(dead, torch.ones_like(self.ema_count), self.ema_count)
            )
        normalized = self.ema_sum / self.ema_count.clamp_min(1e-5).unsqueeze(1)
        _ = self.codebook.copy_(normalized)
        return True

    @torch.no_grad()
    def _candidate_bank(self, valid: torch.Tensor) -> torch.Tensor:
        """Make one identical dead-code candidate bank on every DDP rank."""
        candidates = valid.detach().float()
        if not self._distributed_active():
            return candidates
        world_size = distributed.get_world_size()
        rank = distributed.get_rank()
        owner = torch.tensor(
            rank if candidates.shape[0] > 0 else world_size,
            device=valid.device,
            dtype=torch.long,
        )
        distributed.all_reduce(owner, op=distributed.ReduceOp.MIN)
        owner_rank = int(owner.item())
        if owner_rank == world_size:
            return candidates.new_empty((0, candidates.shape[-1]))
        count = torch.tensor(
            min(candidates.shape[0], self.codebook_size) if rank == owner_rank else 0,
            device=valid.device,
            dtype=torch.long,
        )
        distributed.broadcast(count, src=owner_rank)
        count_value = int(count.item())
        bank = torch.zeros(
            (self.codebook_size, candidates.shape[-1]),
            dtype=torch.float32,
            device=valid.device,
        )
        if rank == owner_rank:
            bank[:count_value].copy_(candidates[:count_value])
        distributed.broadcast(bank, src=owner_rank)
        return bank[:count_value]

    @override
    def forward(
        self,
        latents: torch.Tensor,
        token_mask: torch.Tensor,
        assignment_mask: torch.Tensor,
    ) -> Encoding:
        """Assign every observed group while updating only complete groups."""
        assigned = latents[assignment_mask]
        distributed_active = self._distributed_active()
        if distributed_active:
            assigned_count = torch.tensor(
                assigned.shape[0], device=latents.device, dtype=torch.long
            )
            distributed.all_reduce(assigned_count, op=distributed.ReduceOp.SUM)
            has_assigned = bool(assigned_count.item() > 0)
        else:
            has_assigned = assigned.shape[0] > 0
        if not has_assigned:
            raise NoValidTokensError
        valid = latents[token_mask]
        candidates = valid.detach().float()
        if self.training and distributed_active:
            candidates = self._candidate_bank(valid)
        if (
            self.training
            and candidates.shape[0] > 0
            and not bool(self.initialized.item())
        ):
            with torch.no_grad():
                self._initialize(candidates)
        distances = (
            assigned.square().sum(dim=1, keepdim=True)
            + self.codebook.square().sum(dim=1).unsqueeze(0)
            - 2.0 * assigned @ self.codebook.transpose(0, 1)
        )
        assigned_ids = distances.argmin(dim=1)
        token_ids = torch.zeros(
            token_mask.shape, dtype=torch.long, device=latents.device
        )
        token_ids[assignment_mask] = assigned_ids
        code_vectors = functional.embedding(token_ids, self.codebook)
        quantized = latents + (code_vectors - latents).detach()
        valid_ids = token_ids[token_mask]
        if valid.shape[0] > 0:
            commitment = functional.mse_loss(valid, self.codebook[valid_ids].detach())
            probabilities = (
                functional.one_hot(valid_ids, self.codebook_size).float().mean(0)
            )
            perplexity = torch.exp(
                -(probabilities * torch.log(probabilities + 1e-10)).sum()
            )
        else:
            commitment = latents.new_zeros(())
            perplexity = latents.new_zeros(())
        updated = False
        if self.training and (valid.shape[0] > 0 or distributed_active):
            with torch.no_grad():
                updated = self._update(valid.detach(), valid_ids, candidates)
        return Encoding(
            token_ids=token_ids,
            token_mask=token_mask,
            latents=latents,
            quantized=quantized,
            commitment_loss=commitment * self.commitment_weight,
            perplexity=perplexity,
            codebook_updated=updated,
        )
