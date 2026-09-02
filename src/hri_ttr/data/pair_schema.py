"""Validated array schemas for retargeted Human/G1 pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Final, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from typing_extensions import override

Float32Array = NDArray[np.float32]
MINIMUM_FRAMES: Final = 2
QUATERNION_SQUARED_NORM_TOLERANCE: Final = 2e-4


@dataclass(slots=True)
class PairSchemaError(ValueError):
    """Reports an invalid shape or numeric value in one motion field."""

    field: str
    detail: str

    @override
    def __str__(self) -> str:
        return f"{self.field}: {self.detail}"


@dataclass(slots=True)
class PairAlignmentError(ValueError):
    """Reports unequal actor and reactor frame timelines."""

    sequence_id: str
    actor_frames: int
    reactor_frames: int
    actor_fps: float
    reactor_fps: float

    @override
    def __str__(self) -> str:
        return (
            f"{self.sequence_id}: actor/reactor differ "
            f"({self.actor_frames}@{self.actor_fps}, "
            f"{self.reactor_frames}@{self.reactor_fps})"
        )


def _array(value: NDArray[np.float32]) -> Float32Array:
    return np.asarray(value, dtype=np.float32)


def _require_motion(array: Float32Array, tail: tuple[int, ...], field: str) -> None:
    if array.ndim != len(tail) + 1 or array.shape[1:] != tail:
        raise PairSchemaError(
            field, f"expected [T,{','.join(map(str, tail))}], got {array.shape}"
        )
    if array.shape[0] < MINIMUM_FRAMES or not np.isfinite(array).all():
        raise PairSchemaError(field, "requires at least two finite frames")


def _require_quaternions(array: Float32Array, field: str) -> None:
    squared_norms: Float32Array = (array * array).sum(axis=-1)
    if any(
        abs(float(value) - 1.0) > QUATERNION_SQUARED_NORM_TOLERANCE
        for value in squared_norms.flat
    ):
        raise PairSchemaError(field, "expected unit xyzw quaternions")


class ActorMotion(BaseModel):
    """Validated Human actor arrays from the retargeted pair dataset."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    fps: float
    root_pos: Float32Array
    root_rot: Float32Array
    pose_body: Float32Array
    joints_pos: Float32Array
    betas: Float32Array
    gender: str

    @field_validator(
        "root_pos", "root_rot", "pose_body", "joints_pos", "betas", mode="before"
    )
    @classmethod
    def parse_arrays(cls, value: NDArray[np.float32]) -> Float32Array:
        """Own all numeric inputs as float32 arrays."""
        return _array(value)

    @field_validator("gender", mode="before")
    @classmethod
    def parse_gender(cls, value: str | NDArray[np.str_]) -> str:
        """Accept the one-element NumPy strings used by older pair files."""
        if isinstance(value, str):
            return value
        parsed = np.asarray(value, dtype=np.str_).reshape(-1)
        if len(parsed) != 1:
            field = "actor.gender"
            raise PairSchemaError(field, "expected one string value")
        return str(next(iter(parsed)))

    @model_validator(mode="after")
    def verify_contract(self) -> Self:
        """Require the established 24-joint Human actor schema."""
        _require_motion(self.root_pos, (3,), "actor.root_pos")
        _require_motion(self.root_rot, (4,), "actor.root_rot")
        _require_motion(self.pose_body, (69,), "actor.pose_body")
        _require_motion(self.joints_pos, (24, 3), "actor.joints_pos")
        if (
            self.betas.shape not in {(10,), (1, 10)}
            or not np.isfinite(self.betas).all()
        ):
            field = "actor.betas"
            raise PairSchemaError(field, "expected finite [10] or [1,10]")
        frames = len(self.root_pos)
        if any(
            array.shape[0] != frames
            for array in (self.root_rot, self.pose_body, self.joints_pos)
        ):
            field = "actor.frames"
            raise PairSchemaError(field, "all time axes must match")
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            field = "actor.fps"
            raise PairSchemaError(field, "must be finite and positive")
        _require_quaternions(self.root_rot, "actor.root_rot")
        return self


class ReactorMotion(BaseModel):
    """Validated G1 reactor arrays from the retargeted pair dataset."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, frozen=True
    )

    fps: float
    root_pos: Float32Array
    root_rot: Float32Array
    dof_pos: Float32Array
    joints_pos: Float32Array

    @field_validator("root_pos", "root_rot", "dof_pos", "joints_pos", mode="before")
    @classmethod
    def parse_arrays(cls, value: NDArray[np.float32]) -> Float32Array:
        """Own all numeric inputs as float32 arrays."""
        return _array(value)

    @model_validator(mode="after")
    def verify_contract(self) -> Self:
        """Require the established 29DoF and 20-joint G1 schema."""
        _require_motion(self.root_pos, (3,), "reactor.root_pos")
        _require_motion(self.root_rot, (4,), "reactor.root_rot")
        _require_motion(self.dof_pos, (29,), "reactor.dof_pos")
        _require_motion(self.joints_pos, (20, 3), "reactor.joints_pos")
        frames = len(self.root_pos)
        if any(
            array.shape[0] != frames
            for array in (self.root_rot, self.dof_pos, self.joints_pos)
        ):
            field = "reactor.frames"
            raise PairSchemaError(field, "all time axes must match")
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            field = "reactor.fps"
            raise PairSchemaError(field, "must be finite and positive")
        _require_quaternions(self.root_rot, "reactor.root_rot")
        return self
