"""Versioned Human 262D feature normalizer with contact passthrough."""

# pyright: reportAny=false

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from hri_ttr.representations.human.features import HUMAN_262_LAYOUT

if TYPE_CHECKING:
    from pathlib import Path

Float32Array = NDArray[np.float32]
SCHEMA_ID: Final = "human_intergen_262_v1"


@dataclass(frozen=True, slots=True)
class NormalizerError(ValueError):
    """Reports invalid statistics or feature arrays."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class FeatureField:
    """One named and unit-bearing range in the normalizer schema."""

    name: str
    start: int
    stop: int
    unit: str
    normalized: bool


HUMAN_FIELDS: Final = (
    FeatureField("joint_position", 0, 66, "m", normalized=True),
    FeatureField("joint_displacement", 66, 132, "m/frame", normalized=True),
    FeatureField("joint_rotation_6d", 132, 258, "unitless", normalized=True),
    FeatureField("foot_contact", 258, 262, "binary", normalized=False),
)


@dataclass(frozen=True, slots=True)
class HumanFeatureNormalizer:
    """Immutable statistics and provenance for Human 262D normalization."""

    mean: Float32Array
    std: Float32Array
    source_dataset: str
    fps: float
    schema_id: str = SCHEMA_ID
    schema_version: int = 1
    fields: tuple[FeatureField, ...] = HUMAN_FIELDS

    @classmethod
    def create(
        cls,
        mean: Float32Array,
        std: Float32Array,
        *,
        source_dataset: str,
        fps: float,
    ) -> HumanFeatureNormalizer:
        """Parse normalizer statistics and force contacts to passthrough values."""
        parsed_mean = np.asarray(mean, dtype=np.float32).copy()
        parsed_std = np.asarray(std, dtype=np.float32).copy()
        if parsed_mean.shape != (262,) or parsed_std.shape != (262,):
            detail = "mean and std must each have shape [262]"
            raise NormalizerError(detail)
        if not np.isfinite(parsed_mean).all() or not np.isfinite(parsed_std).all():
            detail = "mean and std must be finite"
            raise NormalizerError(detail)
        if (
            any(float(value) <= 0.0 for value in parsed_std[:258].flat)
            or not source_dataset
            or not np.isfinite(fps)
            or fps <= 0.0
        ):
            detail = "non-contact std, source dataset, and FPS must be positive"
            raise NormalizerError(detail)
        parsed_mean[HUMAN_262_LAYOUT.contact] = 0.0
        parsed_std[HUMAN_262_LAYOUT.contact] = 1.0
        return cls(parsed_mean, parsed_std, source_dataset, fps)

    def normalize(self, features: Float32Array) -> Float32Array:
        """Apply per-dimension z-score while preserving contact bits."""
        values = self._features(features)
        return ((values - self.mean) / self.std).astype(np.float32)

    def denormalize(self, features: Float32Array) -> Float32Array:
        """Reverse the stored per-dimension transform."""
        values = self._features(features)
        return (values * self.std + self.mean).astype(np.float32)

    def save(self, path: Path) -> None:
        """Persist the schema-bound Human normalization statistics."""
        payload = {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "source_dataset": self.source_dataset,
            "fps": self.fps,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> HumanFeatureNormalizer:
        """Load only statistics matching the Human262 schema."""
        parsed: object = json.loads(path.read_text(encoding="utf-8"))
        value = cast("dict[str, object]", parsed) if isinstance(parsed, dict) else {}
        if value.get("schema_id") != SCHEMA_ID or value.get("schema_version") != 1:
            detail = f"normalizer schema mismatch: {path}"
            raise NormalizerError(detail)
        return cls.create(
            np.asarray(value.get("mean"), dtype=np.float32),
            np.asarray(value.get("std"), dtype=np.float32),
            source_dataset=str(value.get("source_dataset", "")),
            fps=_number(value.get("fps")),
        )

    @staticmethod
    def _features(features: Float32Array) -> Float32Array:
        values = np.asarray(features, dtype=np.float32)
        if values.shape[-1:] != (262,) or not np.isfinite(values).all():
            detail = "features must be finite with final dimension 262"
            raise NormalizerError(detail)
        return values


def _number(value: object) -> float:
    if not isinstance(value, int | float):
        detail = "normalizer FPS must be numeric"
        raise NormalizerError(detail)
    return float(value)
