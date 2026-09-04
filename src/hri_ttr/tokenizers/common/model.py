"""Shared execution contract for independent causal tokenizers."""

from __future__ import annotations

from itertools import count
from typing import TYPE_CHECKING, Final

import torch
from torch import nn
from typing_extensions import override

from hri_ttr.tokenizers.common.contracts import (
    Encoding,
    StreamState,
    TokenizerArchitecture,
    TokenizerOutput,
)
from hri_ttr.tokenizers.common.errors import (
    InvalidMotionTensorError,
    InvalidTokenTensorError,
    MotionTensorReason,
    StaleStreamStateError,
    TokenTensorReason,
)
from hri_ttr.tokenizers.common.quantizer import EMAVectorQuantizer

if TYPE_CHECKING:
    from hri_ttr.tokenizers.common.layers import CausalDecoder, CausalEncoder

_STREAM_IDS = count(1)
FEATURE_NDIM: Final = 3
MASK_NDIM: Final = 2
RESIDUAL_DILATIONS: Final = (1, 3, 9)


class CausalMotionTokenizer(nn.Module):
    """Implement batch, prefix, and single-use-state streaming execution."""

    frames_per_token: Final[int] = 4
    input_fps: Final[int] = 20
    token_rate_hz: Final[int] = 5
    stream_retains_full_prefix: Final[bool] = True
    feature_dim: int
    architecture: TokenizerArchitecture
    encoder: CausalEncoder
    decoder: CausalDecoder
    quantizer: EMAVectorQuantizer
    _stream_owner_id: int

    def __init__(
        self,
        feature_dim: int,
        architecture: TokenizerArchitecture,
        encoder: CausalEncoder,
        decoder: CausalDecoder,
    ) -> None:
        """Assemble one domain-owned encoder, decoder, and quantizer."""
        super().__init__()
        self.feature_dim = feature_dim
        self.architecture = architecture
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = EMAVectorQuantizer(
            architecture.codebook_size,
            architecture.code_dim,
            architecture.ema_decay,
            architecture.commitment_weight,
        )
        self._stream_owner_id = next(_STREAM_IDS)

    @property
    def encoder_receptive_field_frames(self) -> int:
        """Return the exact number of input frames affecting one latent."""
        dilation_sum = sum(RESIDUAL_DILATIONS[: self.architecture.residual_depth])
        return 20 + 24 * dilation_sum

    def _validate_motion(self, features: torch.Tensor, mask: torch.Tensor) -> None:
        if features.ndim != FEATURE_NDIM:
            raise InvalidMotionTensorError(MotionTensorReason.SHAPE)
        if features.shape[2] != self.feature_dim:
            raise InvalidMotionTensorError(MotionTensorReason.WIDTH)
        if not features.is_floating_point():
            raise InvalidMotionTensorError(MotionTensorReason.DTYPE)
        if features.shape[1] == 0:
            raise InvalidMotionTensorError(MotionTensorReason.EMPTY)
        if features.shape[1] % self.frames_per_token != 0:
            raise InvalidMotionTensorError(MotionTensorReason.ALIGNMENT)
        if mask.ndim != MASK_NDIM or mask.shape != features.shape[:2]:
            raise InvalidMotionTensorError(MotionTensorReason.MASK)
        if mask.dtype is not torch.bool:
            raise InvalidMotionTensorError(MotionTensorReason.MASK)
        if not bool(torch.isfinite(features).all().item()):
            raise InvalidMotionTensorError(MotionTensorReason.FINITE)

    def _token_masks(
        self, frame_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames = frame_mask.shape
        grouped = frame_mask.reshape(batch, frames // self.frames_per_token, 4)
        return grouped.all(dim=2), grouped.any(dim=2)

    def encode(self, features: torch.Tensor, frame_mask: torch.Tensor) -> Encoding:
        """Encode an aligned observed prefix without access to future frames."""
        self._validate_motion(features, frame_mask)
        latents = self.encoder.forward(features)
        statistics_mask, assignment_mask = self._token_masks(frame_mask)
        return self.quantizer.forward(latents, statistics_mask, assignment_mask)

    def set_distributed(self, enabled: bool) -> None:
        """Configure domain-owned EMA updates for a DDP training process."""
        self.quantizer.set_distributed(enabled)

    def _validate_tokens(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> None:
        if token_ids.ndim != MASK_NDIM:
            raise InvalidTokenTensorError(TokenTensorReason.SHAPE)
        if token_ids.dtype is not torch.long:
            raise InvalidTokenTensorError(TokenTensorReason.DTYPE)
        if token_mask.dtype is not torch.bool or token_mask.shape != token_ids.shape:
            raise InvalidTokenTensorError(TokenTensorReason.MASK)
        valid_ids = token_ids[token_mask]
        if valid_ids.numel() > 0 and bool(
            ((valid_ids < 0) | (valid_ids >= self.architecture.codebook_size)).any()
        ):
            raise InvalidTokenTensorError(TokenTensorReason.RANGE)

    def decode(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode token IDs while keeping every past frame future-independent."""
        effective_mask = (
            torch.ones_like(token_ids, dtype=torch.bool)
            if token_mask is None
            else token_mask
        )
        self._validate_tokens(token_ids, effective_mask)
        safe_ids = torch.where(effective_mask, token_ids, torch.zeros_like(token_ids))
        return self.decoder.forward(self.quantizer.codebook[safe_ids])

    @override
    def forward(
        self, features: torch.Tensor, frame_mask: torch.Tensor
    ) -> TokenizerOutput:
        """Run the full causal VQ autoencoder."""
        encoding = self.encode(features, frame_mask)
        return TokenizerOutput(
            reconstruction=self.decoder.forward(encoding.quantized),
            encoding=encoding,
        )

    def new_stream(self, batch_size: int) -> StreamState:
        """Create empty state for one uninterrupted sequence."""
        if batch_size <= 0:
            raise InvalidMotionTensorError(MotionTensorReason.BATCH)
        return StreamState(self._stream_owner_id, batch_size)

    def _validate_stream(self, state: StreamState) -> None:
        if state.consumed or state.owner_id != self._stream_owner_id:
            raise StaleStreamStateError

    def stream_encode(
        self,
        features: torch.Tensor,
        frame_mask: torch.Tensor,
        state: StreamState,
    ) -> tuple[Encoding, StreamState]:
        """Consume a chunk and emit only newly completed four-frame tokens."""
        self._validate_stream(state)
        if features.ndim != FEATURE_NDIM or features.shape[0] != state.batch_size:
            raise InvalidMotionTensorError(MotionTensorReason.SHAPE)
        if features.shape[2] != self.feature_dim:
            raise InvalidMotionTensorError(MotionTensorReason.WIDTH)
        if not features.is_floating_point():
            raise InvalidMotionTensorError(MotionTensorReason.DTYPE)
        if frame_mask.shape != features.shape[:2] or frame_mask.dtype is not torch.bool:
            raise InvalidMotionTensorError(MotionTensorReason.MASK)
        all_features = (
            features
            if state.features is None
            else torch.cat((state.features, features), 1)
        )
        all_mask = (
            frame_mask
            if state.frame_mask is None
            else torch.cat((state.frame_mask, frame_mask), 1)
        )
        aligned_frames = (
            all_features.shape[1] // self.frames_per_token * self.frames_per_token
        )
        next_state = StreamState(self._stream_owner_id, state.batch_size)
        next_state.features = all_features
        next_state.frame_mask = all_mask
        next_state.emitted_tokens = aligned_frames // self.frames_per_token
        state.consumed = True
        if aligned_frames == 0:
            return self._empty_encoding(features), next_state
        complete = self.encode(
            all_features[:, :aligned_frames], all_mask[:, :aligned_frames]
        )
        start = state.emitted_tokens
        return self._slice_encoding(complete, start), next_state

    def _empty_encoding(self, features: torch.Tensor) -> Encoding:
        shape = (features.shape[0], 0)
        scalar = features.new_zeros(())
        return Encoding(
            token_ids=torch.zeros(shape, dtype=torch.long, device=features.device),
            token_mask=torch.zeros(shape, dtype=torch.bool, device=features.device),
            latents=features.new_zeros((*shape, self.architecture.code_dim)),
            quantized=features.new_zeros((*shape, self.architecture.code_dim)),
            commitment_loss=scalar,
            perplexity=scalar,
            codebook_updated=False,
        )

    @staticmethod
    def _slice_encoding(encoding: Encoding, start: int) -> Encoding:
        return Encoding(
            token_ids=encoding.token_ids[:, start:],
            token_mask=encoding.token_mask[:, start:],
            latents=encoding.latents[:, start:],
            quantized=encoding.quantized[:, start:],
            commitment_loss=encoding.commitment_loss,
            perplexity=encoding.perplexity,
            codebook_updated=encoding.codebook_updated,
        )

    def reset_stream(self, state: StreamState) -> StreamState:
        """Invalidate interrupted state and return a clean sequence state."""
        self._validate_stream(state)
        state.consumed = True
        return self.new_stream(state.batch_size)
