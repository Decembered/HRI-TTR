"""Reader for immutable sequence-ID train, validation, and test splits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing_extensions import override

if TYPE_CHECKING:
    from pathlib import Path


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _SplitGroups(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]


class _SplitSourcePayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    sequence_count: int = Field(gt=0)
    sequence_ids_sha256: Sha256
    split_indices_sha256: Sha256


class _SplitPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["interx_human_g1_split"]
    schema_version: Literal[1]
    source: _SplitSourcePayload
    splits: _SplitGroups


@dataclass(frozen=True, slots=True)
class SplitSource:
    """Provenance binding a split to its source files and sequence universe."""

    dataset: str
    sequence_count: int
    sequence_ids_sha256: str
    split_indices_sha256: str


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    """A fixed split expressed with stable sequence IDs, never row indices."""

    schema_id: str
    schema_version: int
    source: SplitSource
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SplitFileError(ValueError):
    """Reports invalid JSON, duplicate IDs, or split overlap."""

    path: Path
    detail: str

    @override
    def __str__(self) -> str:
        return f"{self.path}: {self.detail}"


def read_fixed_splits(path: Path) -> DatasetSplits:
    """Parse a committed split file and reject unstable or overlapping IDs."""
    try:
        payload = _SplitPayload.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError, json.JSONDecodeError) as error:
        raise SplitFileError(path, str(error)) from error
    groups = (payload.splits.train, payload.splits.val, payload.splits.test)
    all_ids = tuple(sequence_id for group in groups for sequence_id in group)
    if any(not sequence_id for sequence_id in all_ids):
        raise SplitFileError(path, "sequence IDs must be non-empty")
    if len(set(all_ids)) != len(all_ids):
        raise SplitFileError(path, "sequence IDs must be unique across all splits")
    if len(all_ids) != payload.source.sequence_count:
        raise SplitFileError(path, "sequence count does not match source metadata")
    source = SplitSource(
        dataset=payload.source.dataset,
        sequence_count=payload.source.sequence_count,
        sequence_ids_sha256=payload.source.sequence_ids_sha256,
        split_indices_sha256=payload.source.split_indices_sha256,
    )
    return DatasetSplits(
        schema_id=payload.schema_id,
        schema_version=payload.schema_version,
        source=source,
        train=payload.splits.train,
        val=payload.splits.val,
        test=payload.splits.test,
    )
