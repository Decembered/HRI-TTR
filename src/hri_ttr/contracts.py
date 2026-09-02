"""Frozen domain contracts shared by every HRI-TTR subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Final, NewType, TypeAlias

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

ModelId = NewType("ModelId", str)
SchemaId = NewType("SchemaId", str)
SequenceId = NewType("SequenceId", str)
FramesPerToken = NewType("FramesPerToken", int)

Float32Array: TypeAlias = NDArray[np.float32]
Int64Array: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
MOTION_BATCH_NDIM: Final = 3
TOKEN_BATCH_NDIM: Final = 2
MIN_CODEBOOK_SIZE: Final = 2


class ContractContext(StrEnum):
    """Stable locations where a typed contract can be violated."""

    SPACE_STATE = "space state"
    MOTION_SCHEMA = "motion schema"
    TOKENIZER_SPEC = "tokenizer spec"
    MOTION_BATCH = "motion batch"
    TOKEN_BATCH = "token batch"


class ContractReason(StrEnum):
    """Stable diagnostic reasons shared by all contract errors."""

    CODEBOOK_SIZE = "codebook size must be at least two"
    FEATURE_DIMENSION = "feature dimension must be positive"
    FEATURE_DTYPE = "features must use float32"
    FEATURE_SHAPE = "features must have shape [batch, frame, feature]"
    FEATURE_WIDTH = "feature width does not match schema"
    FINITE = "coordinates or features must be finite"
    FRAME_MASK_DTYPE = "frame mask must use bool"
    FRAME_MASK_SHAPE = "frame mask must match batch and frame axes"
    FRAME_MASK_NDIM = "frame mask must have shape [batch, frame]"
    FPS = "input FPS must be finite and positive"
    FRAMES_PER_TOKEN = "frames per token must be positive"
    KIND = "kind must be 'human' or 'g1'"
    TOKEN_DTYPE = "token IDs must use int64"
    TOKEN_MASK_DTYPE = "token mask must use bool"
    TOKEN_MASK_SHAPE = "token mask must match token IDs"
    TOKEN_RANGE = "valid token ID is outside codebook range"
    TOKEN_SHAPE = "token IDs must have shape [batch, token]"


@dataclass(frozen=True, slots=True)
class ContractViolationError(ValueError):
    """Reports a violated typed boundary or internal contract."""

    context: ContractContext
    reason: ContractReason

    @override
    def __str__(self) -> str:
        """Render a stable diagnostic for callers and test assertions."""
        return f"{self.context}: {self.reason}"


@dataclass(frozen=True, slots=True)
class SpaceState:
    """Planar interaction state expressed in the EpisodeFrame."""

    x_m: float
    z_m: float
    yaw_rad: float

    def __post_init__(self) -> None:
        """Reject non-finite coordinates at the contract boundary."""
        if not all(isfinite(value) for value in (self.x_m, self.z_m, self.yaw_rad)):
            raise ContractViolationError(
                ContractContext.SPACE_STATE,
                ContractReason.FINITE,
            )


@dataclass(frozen=True, slots=True)
class MotionSchema:
    """Names a fixed-width canonical motion representation."""

    schema_id: SchemaId
    feature_dim: int

    def __post_init__(self) -> None:
        """Reject non-positive feature dimensions."""
        if self.feature_dim <= 0:
            raise ContractViolationError(
                ContractContext.MOTION_SCHEMA,
                ContractReason.FEATURE_DIMENSION,
            )


@dataclass(frozen=True, slots=True)
class TokenizerSpec:
    """Identifies one immutable tokenizer token space."""

    kind: str
    schema_id: SchemaId
    model_id: ModelId
    input_fps: float
    frames_per_token: FramesPerToken
    codebook_size: int

    def __post_init__(self) -> None:
        """Validate values needed by all tokenizer implementations."""
        if self.kind not in {"human", "g1"}:
            raise ContractViolationError(
                ContractContext.TOKENIZER_SPEC,
                ContractReason.KIND,
            )
        if not isfinite(self.input_fps) or self.input_fps <= 0.0:
            raise ContractViolationError(
                ContractContext.TOKENIZER_SPEC,
                ContractReason.FPS,
            )
        if self.frames_per_token <= 0:
            raise ContractViolationError(
                ContractContext.TOKENIZER_SPEC,
                ContractReason.FRAMES_PER_TOKEN,
            )
        if self.codebook_size < MIN_CODEBOOK_SIZE:
            raise ContractViolationError(
                ContractContext.TOKENIZER_SPEC,
                ContractReason.CODEBOOK_SIZE,
            )

    @property
    def token_rate_hz(self) -> float:
        """Return the emitted token frequency."""
        return self.input_fps / self.frames_per_token


@dataclass(frozen=True, slots=True)
class MotionBatch:
    """Batch of canonical motion frames and their valid-frame mask."""

    features: Float32Array
    frame_mask: BoolArray
    schema: MotionSchema

    def __post_init__(self) -> None:
        """Ensure the motion tensor matches its schema and mask."""
        if self.features.ndim != MOTION_BATCH_NDIM:
            raise ContractViolationError(
                ContractContext.MOTION_BATCH,
                ContractReason.FEATURE_SHAPE,
            )
        if self.frame_mask.ndim != TOKEN_BATCH_NDIM:
            raise ContractViolationError(
                ContractContext.MOTION_BATCH,
                ContractReason.FRAME_MASK_NDIM,
            )
        if self.features.shape[:2] != self.frame_mask.shape:
            raise ContractViolationError(
                ContractContext.MOTION_BATCH,
                ContractReason.FRAME_MASK_SHAPE,
            )
        if self.features.shape[2] != self.schema.feature_dim:
            raise ContractViolationError(
                ContractContext.MOTION_BATCH,
                ContractReason.FEATURE_WIDTH,
            )
        if self.features.dtype != np.dtype(np.float32):
            raise ContractViolationError(
                ContractContext.MOTION_BATCH,
                ContractReason.FEATURE_DTYPE,
            )
        if self.frame_mask.dtype != np.dtype(np.bool_):
            raise ContractViolationError(
                ContractContext.MOTION_BATCH,
                ContractReason.FRAME_MASK_DTYPE,
            )
        if any(not isfinite(float(value)) for value in self.features.flat):
            raise ContractViolationError(
                ContractContext.MOTION_BATCH,
                ContractReason.FINITE,
            )
        feature_copy = self.features.copy()
        frame_mask_copy = self.frame_mask.copy()
        feature_copy.setflags(write=False)
        frame_mask_copy.setflags(write=False)
        object.__setattr__(self, "features", feature_copy)
        object.__setattr__(self, "frame_mask", frame_mask_copy)


@dataclass(frozen=True, slots=True)
class TokenBatch:
    """Batch of codebook IDs and their valid-token mask."""

    token_ids: Int64Array
    token_mask: BoolArray
    tokenizer: TokenizerSpec

    def __post_init__(self) -> None:
        """Ensure valid codebook IDs stay in the tokenizer vocabulary."""
        if self.token_ids.ndim != TOKEN_BATCH_NDIM:
            raise ContractViolationError(
                ContractContext.TOKEN_BATCH,
                ContractReason.TOKEN_SHAPE,
            )
        if self.token_mask.shape != self.token_ids.shape:
            raise ContractViolationError(
                ContractContext.TOKEN_BATCH,
                ContractReason.TOKEN_MASK_SHAPE,
            )
        if self.token_ids.dtype != np.dtype(np.int64):
            raise ContractViolationError(
                ContractContext.TOKEN_BATCH,
                ContractReason.TOKEN_DTYPE,
            )
        if self.token_mask.dtype != np.dtype(np.bool_):
            raise ContractViolationError(
                ContractContext.TOKEN_BATCH,
                ContractReason.TOKEN_MASK_DTYPE,
            )
        for token_id, is_valid in zip(
            self.token_ids.flat,
            self.token_mask.flat,
            strict=True,
        ):
            if bool(is_valid) and (
                int(token_id) < 0 or int(token_id) >= self.tokenizer.codebook_size
            ):
                raise ContractViolationError(
                    ContractContext.TOKEN_BATCH,
                    ContractReason.TOKEN_RANGE,
                )
        token_ids_copy = self.token_ids.copy()
        token_mask_copy = self.token_mask.copy()
        token_ids_copy.setflags(write=False)
        token_mask_copy.setflags(write=False)
        object.__setattr__(self, "token_ids", token_ids_copy)
        object.__setattr__(self, "token_mask", token_mask_copy)
