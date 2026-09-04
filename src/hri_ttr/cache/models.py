"""Strict token arrays and provenance manifest models."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Annotated, ClassVar, Final, Literal, NoReturn, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from hri_ttr.cache.errors import CacheValidationError

SHA256_PATTERN: Final = r"^[0-9a-f]{64}$"
TOKEN_LIMIT: Final = 255
FRAMES_PER_TOKEN: Final = 4
INPUT_FPS: Final = 20
TOKEN_RATE_HZ: Final = 5
Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]


def _reject(detail: str) -> NoReturn:
    raise CacheValidationError(detail)


class CacheManifest(BaseModel):
    """Strict provenance and sequence-shape contract for one token cache."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", strict=True
    )

    format_version: Literal[2] = 2
    tokenizer_sha256: Sha256
    checkpoint_sha256: Sha256
    normalizer_sha256: Sha256
    schema_sha256: Sha256
    split_sha256: Sha256
    input_fps: Literal[20] = INPUT_FPS
    frames_per_token: Literal[4] = FRAMES_PER_TOKEN
    token_rate_hz: Literal[5] = TOKEN_RATE_HZ
    codebook_size: Literal[256] = 256
    valid_frame_lengths: tuple[int, ...]
    valid_token_lengths: tuple[int, ...]
    padded_frame_counts: tuple[int, ...]

    @model_validator(mode="after")
    def validate_lengths(self) -> Self:
        """Require exactly one four-frame-rate token length per sequence."""
        if not (
            len(self.valid_frame_lengths)
            == len(self.valid_token_lengths)
            == len(self.padded_frame_counts)
        ):
            _reject("frame, token, and padding length counts differ")
        for frames, tokens, padding in zip(
            self.valid_frame_lengths,
            self.valid_token_lengths,
            self.padded_frame_counts,
            strict=True,
        ):
            if (
                frames <= 0
                or tokens != (frames + FRAMES_PER_TOKEN - 1) // FRAMES_PER_TOKEN
                or padding != tokens * FRAMES_PER_TOKEN - frames
            ):
                _reject("tokens and padding must exactly cover valid frames")
        return self


class SequenceIds(RootModel[tuple[str, ...]]):
    """Strict JSON boundary for ordered sequence identifiers."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, strict=True)


@dataclass(frozen=True, slots=True)
class TokenCache:
    """Validated concatenated uint16 tokens and their sequence offsets."""

    tokens: npt.NDArray[np.uint16]
    offsets: npt.NDArray[np.int64]
    sequence_ids: tuple[str, ...]
    manifest: CacheManifest

    def __post_init__(self) -> None:
        """Reject arrays that could desynchronize sequence boundaries."""
        tokens = self.tokens.copy()
        offsets = self.offsets.copy()
        tokens.setflags(write=False)
        offsets.setflags(write=False)
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "offsets", offsets)
        sequence_count = len(self.sequence_ids)
        if self.tokens.ndim != 1 or self.tokens.dtype != np.uint16:
            _reject("tokens must be a one-dimensional uint16 array")
        if self.offsets.ndim != 1 or self.offsets.dtype != np.int64:
            _reject("offsets must be a one-dimensional int64 array")
        if len(self.offsets) != sequence_count + 1:
            _reject("offset count must equal sequence count plus one")
        if len(set(self.sequence_ids)) != sequence_count or any(
            not sequence_id for sequence_id in self.sequence_ids
        ):
            _reject("sequence IDs must be nonempty and unique")
        if self.offsets[0] != 0 or self.offsets[-1] != len(self.tokens):
            _reject("offsets must span the complete token array")
        offset_values = tuple(int(value) for value in self.offsets.flat)
        if any(right < left for left, right in itertools.pairwise(offset_values)):
            _reject("offsets must be nondecreasing")
        if any(int(value) > TOKEN_LIMIT for value in self.tokens.flat):
            _reject("token IDs must be in [0,255]")
        lengths = tuple(
            right - left for left, right in itertools.pairwise(offset_values)
        )
        if lengths != self.manifest.valid_token_lengths:
            _reject("offsets disagree with manifest token lengths")
        if sequence_count != len(self.manifest.valid_frame_lengths):
            _reject("manifest sequence count differs from cache")

    @property
    def statistics_token_mask(self) -> npt.NDArray[np.bool_]:
        """Mark only complete four-frame groups for codebook statistics."""
        mask = np.zeros(len(self.tokens), dtype=np.bool_)
        offset_values = tuple(int(value) for value in self.offsets.flat)
        for offset, frames in zip(
            offset_values[:-1], self.manifest.valid_frame_lengths, strict=True
        ):
            mask[offset : offset + frames // FRAMES_PER_TOKEN] = True
        mask.setflags(write=False)
        return mask
