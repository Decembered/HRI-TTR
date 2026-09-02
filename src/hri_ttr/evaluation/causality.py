"""Strict prefix and future-perturbation tokenizer diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import torch

from hri_ttr.evaluation.errors import reject_evaluation

if TYPE_CHECKING:
    from hri_ttr.tokenizers import G1Tokenizer, HumanTokenizer

FEATURE_TENSOR_NDIM: Final = 3
MINIMUM_FRAMES: Final = 8
FRAMES_PER_TOKEN: Final = 4


@dataclass(frozen=True, slots=True)
class CausalityDiagnostic:
    """Binary token stability and maximum numeric deviations from causal checks."""

    changed_token_count: int
    max_latent_difference: float
    max_decoded_difference: float
    future_perturbation_changed_token_count: int
    future_perturbation_max_latent_difference: float
    future_perturbation_max_decoded_difference: float


def _maximum_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).item()) if left.numel() else 0.0


def _evaluate_without_updates(
    tokenizer: HumanTokenizer | G1Tokenizer,
    features: torch.Tensor,
    frame_mask: torch.Tensor,
) -> CausalityDiagnostic:
    if (
        features.ndim != FEATURE_TENSOR_NDIM
        or features.shape[1] < MINIMUM_FRAMES
        or features.shape[1] % FRAMES_PER_TOKEN
    ):
        reject_evaluation("causality diagnostic needs at least eight aligned frames")
    if frame_mask.shape != features.shape[:2] or frame_mask.dtype is not torch.bool:
        reject_evaluation("causality diagnostic requires a matching boolean mask")
    with torch.no_grad():
        full = tokenizer.encode(features, frame_mask)
        full_decoded = tokenizer.decode(full.token_ids, full.token_mask)
        prefix_ids: list[torch.Tensor] = []
        prefix_latents: list[torch.Tensor] = []
        decoded_difference = 0.0
        for frames in range(FRAMES_PER_TOKEN, features.shape[1] + 1, FRAMES_PER_TOKEN):
            prefix = tokenizer.encode(features[:, :frames], frame_mask[:, :frames])
            prefix_ids.append(prefix.token_ids[:, -1:])
            prefix_latents.append(prefix.latents[:, -1:])
            prefix_decoded = tokenizer.decode(prefix.token_ids, prefix.token_mask)
            decoded_difference = max(
                decoded_difference,
                _maximum_difference(prefix_decoded, full_decoded[:, :frames]),
            )
        incremental_ids = torch.cat(prefix_ids, dim=1)
        incremental_latents = torch.cat(prefix_latents, dim=1)
        cutoff_tokens = max(1, full.token_ids.shape[1] // 2)
        cutoff_frames = cutoff_tokens * FRAMES_PER_TOKEN
        perturbed_features = features.clone()
        perturbed_features[:, cutoff_frames:] += 17.0
        perturbed = tokenizer.encode(perturbed_features, frame_mask)
        perturbed_decoded = tokenizer.decode(perturbed.token_ids, perturbed.token_mask)
    return CausalityDiagnostic(
        int(torch.count_nonzero(full.token_ids != incremental_ids).item()),
        _maximum_difference(full.latents, incremental_latents),
        decoded_difference,
        int(
            torch.count_nonzero(
                full.token_ids[:, :cutoff_tokens]
                != perturbed.token_ids[:, :cutoff_tokens]
            ).item()
        ),
        _maximum_difference(
            full.latents[:, :cutoff_tokens], perturbed.latents[:, :cutoff_tokens]
        ),
        _maximum_difference(
            full_decoded[:, :cutoff_frames], perturbed_decoded[:, :cutoff_frames]
        ),
    )


def _evaluate(
    tokenizer: HumanTokenizer | G1Tokenizer,
    features: torch.Tensor,
    frame_mask: torch.Tensor,
) -> CausalityDiagnostic:
    modules = tuple(tokenizer.modules())
    training_modes = tuple(module.training for module in modules)
    _ = tokenizer.eval()
    try:
        return _evaluate_without_updates(tokenizer, features, frame_mask)
    finally:
        for module, training in zip(modules, training_modes, strict=True):
            module.training = training


def evaluate_human_tokenizer_causality(
    tokenizer: HumanTokenizer,
    features: torch.Tensor,
    frame_mask: torch.Tensor,
) -> CausalityDiagnostic:
    """Run prefix and future-perturbation checks on a Human tokenizer."""
    return _evaluate(tokenizer, features, frame_mask)


def evaluate_g1_tokenizer_causality(
    tokenizer: G1Tokenizer,
    features: torch.Tensor,
    frame_mask: torch.Tensor,
) -> CausalityDiagnostic:
    """Run prefix and future-perturbation checks on a G1 tokenizer."""
    return _evaluate(tokenizer, features, frame_mask)
