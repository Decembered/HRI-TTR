"""Mask-aware exponential-moving-average vector quantization."""

from __future__ import annotations

import torch
from torch import nn
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

    def _initialize(self, valid: torch.Tensor) -> None:
        repeats = (self.codebook_size + valid.shape[0] - 1) // valid.shape[0]
        candidates = valid.repeat(repeats, 1)[: self.codebook_size]
        _ = self.codebook.copy_(candidates)
        _ = self.ema_sum.copy_(candidates)
        _ = self.ema_count.fill_(1.0)
        initialized = torch.ones((), dtype=torch.bool, device=self.initialized.device)
        _ = self.initialized.copy_(initialized)

    @torch.no_grad()
    def _update(self, valid: torch.Tensor, valid_ids: torch.Tensor) -> None:
        one_hot = functional.one_hot(valid_ids, self.codebook_size).to(valid.dtype)
        counts = one_hot.sum(dim=0)
        sums = one_hot.transpose(0, 1) @ valid
        _ = self.ema_count.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
        _ = self.ema_sum.mul_(self.decay).add_(sums, alpha=1.0 - self.decay)
        repeats = (self.codebook_size + valid.shape[0] - 1) // valid.shape[0]
        candidates = valid.repeat(repeats, 1)[: self.codebook_size]
        dead = self.ema_count < 1.0
        _ = self.ema_sum.copy_(torch.where(dead.unsqueeze(1), candidates, self.ema_sum))
        _ = self.ema_count.copy_(
            torch.where(dead, torch.ones_like(self.ema_count), self.ema_count)
        )
        normalized = self.ema_sum / self.ema_count.clamp_min(1e-5).unsqueeze(1)
        _ = self.codebook.copy_(normalized)

    @override
    def forward(
        self,
        latents: torch.Tensor,
        token_mask: torch.Tensor,
        assignment_mask: torch.Tensor,
    ) -> Encoding:
        """Assign every observed group while updating only complete groups."""
        assigned = latents[assignment_mask]
        if assigned.shape[0] == 0:
            raise NoValidTokensError
        valid = latents[token_mask]
        if self.training and valid.shape[0] > 0 and not bool(self.initialized.item()):
            with torch.no_grad():
                self._initialize(valid.detach())
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
        updated = self.training and valid.shape[0] > 0
        if updated:
            with torch.no_grad():
                self._update(valid.detach(), valid_ids)
        return Encoding(
            token_ids=token_ids,
            token_mask=token_mask,
            latents=latents,
            quantized=quantized,
            commitment_loss=commitment * self.commitment_weight,
            perplexity=perplexity,
            codebook_updated=updated,
        )
