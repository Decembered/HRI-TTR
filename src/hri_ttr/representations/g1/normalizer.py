"""Schema-bound normalizer for G1 canonical features."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Final

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from typing_extensions import override

if TYPE_CHECKING:
    from pathlib import Path

from hri_ttr.representations.g1.schema import (
    G1_FEATURE_DIM,
    G1_FEATURE_FIELDS,
    G1_FEATURE_SLICES,
    G1_SCHEMA_VERSION,
)

FEATURE_MATRIX_NDIM: Final = 2


class NormalizerInputReason(StrEnum):
    """Stable reasons for rejected normalizer input."""

    FEATURE_SHAPE = "features must have shape [N,75]"
    MASK = "valid_mask must be bool with shape [N]"
    FINITE = "features must be finite with at least one valid frame"
    PARAMETERS = "fps and std_floor must be finite and positive"
    NONFINITE = "features must be finite"


class NormalizerArtifact(BaseModel):
    """Validated JSON artifact persisted for one G1 normalizer."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    schema_hash: str
    content_hash: str
    source_dataset: str
    fps: float
    feature_names: list[str]
    feature_slices: list[list[int]]
    feature_units: list[str]
    mean: tuple[float, ...]
    std: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class NormalizerInputError(ValueError):
    """Reports malformed arrays at the normalizer boundary."""

    reason: NormalizerInputReason

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class NormalizerSchemaError(ValueError):
    """Reports incompatible or corrupted persisted metadata."""

    path: Path

    @override
    def __str__(self) -> str:
        return f"normalizer schema or content mismatch: {self.path}"


@dataclass(frozen=True, slots=True)
class G1FeatureNormalizer:
    """Z-score non-contact fields while preserving binary contacts."""

    mean: npt.NDArray[np.float64]
    std: npt.NDArray[np.float64]
    source_dataset: str
    fps: float
    schema_version: str = G1_SCHEMA_VERSION

    @property
    def schema_hash(self) -> str:
        """Return a stable digest of names, slices, units, and version."""
        payload: dict[str, str | list[str] | list[list[int]]] = {
            "feature_names": [field.name for field in G1_FEATURE_FIELDS],
            "feature_slices": [
                [field.start, field.stop] for field in G1_FEATURE_FIELDS
            ],
            "feature_units": [field.unit for field in G1_FEATURE_FIELDS],
            "schema_version": self.schema_version,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def content_hash(self) -> str:
        """Return a digest binding statistics and dataset provenance."""
        digest = hashlib.sha256(self.schema_hash.encode())
        digest.update(self.mean.tobytes())
        digest.update(self.std.tobytes())
        digest.update(self.source_dataset.encode())
        digest.update(np.asarray([self.fps], dtype=np.float64).tobytes())
        return digest.hexdigest()

    @classmethod
    def fit(
        cls,
        features: npt.NDArray[np.float64],
        valid_mask: npt.NDArray[np.bool_],
        *,
        source_dataset: str,
        fps: float,
        std_floor: float = 1e-4,
    ) -> G1FeatureNormalizer:
        """Fit statistics using real frames only."""
        if features.ndim != FEATURE_MATRIX_NDIM or features.shape[1] != G1_FEATURE_DIM:
            raise NormalizerInputError(NormalizerInputReason.FEATURE_SHAPE)
        if valid_mask.shape != (len(features),) or valid_mask.dtype != np.dtype(
            np.bool_
        ):
            raise NormalizerInputError(NormalizerInputReason.MASK)
        if not np.isfinite(features).all() or not valid_mask.any():
            raise NormalizerInputError(NormalizerInputReason.FINITE)
        if not np.isfinite(fps) or fps <= 0.0 or std_floor <= 0.0:
            raise NormalizerInputError(NormalizerInputReason.PARAMETERS)
        valid = features[valid_mask]
        mean = np.mean(valid, axis=0, dtype=np.float64)
        std = np.maximum(np.std(valid, axis=0, dtype=np.float64), std_floor)
        contact = G1_FEATURE_SLICES["foot_contact_lr"]
        mean[contact] = 0.0
        std[contact] = 1.0
        return cls(mean, std, source_dataset, fps)

    def normalize(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Normalize one unbatched 75D feature sequence."""
        self._require_features(features)
        return (features - self.mean) / self.std

    def denormalize(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Restore one normalized 75D feature sequence."""
        self._require_features(features)
        return features * self.std + self.mean

    def save(self, path: Path) -> None:
        """Persist statistics and self-verifying schema metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(self._artifact().model_dump_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> G1FeatureNormalizer:
        """Load only an exact G1 75D normalizer artifact."""
        artifact = NormalizerArtifact.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        normalizer = cls(
            np.asarray(artifact.mean, dtype=np.float64),
            np.asarray(artifact.std, dtype=np.float64),
            artifact.source_dataset,
            artifact.fps,
            artifact.schema_version,
        )
        if (
            artifact != normalizer._artifact()
            or artifact.schema_version != G1_SCHEMA_VERSION
        ):
            raise NormalizerSchemaError(path)
        return normalizer

    def _artifact(self) -> NormalizerArtifact:
        mean_values = struct.unpack(f"{self.mean.size}d", self.mean.tobytes())
        std_values = struct.unpack(f"{self.std.size}d", self.std.tobytes())
        return NormalizerArtifact(
            schema_version=self.schema_version,
            schema_hash=self.schema_hash,
            content_hash=self.content_hash,
            source_dataset=self.source_dataset,
            fps=self.fps,
            feature_names=[field.name for field in G1_FEATURE_FIELDS],
            feature_slices=[[field.start, field.stop] for field in G1_FEATURE_FIELDS],
            feature_units=[field.unit for field in G1_FEATURE_FIELDS],
            mean=mean_values,
            std=std_values,
        )

    @staticmethod
    def _require_features(features: npt.NDArray[np.float64]) -> None:
        if features.ndim != FEATURE_MATRIX_NDIM or features.shape[1] != G1_FEATURE_DIM:
            raise NormalizerInputError(NormalizerInputReason.FEATURE_SHAPE)
        if not np.isfinite(features).all():
            raise NormalizerInputError(NormalizerInputReason.NONFINITE)
