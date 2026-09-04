"""Offline official-Human checkpoint reader with explicit key mapping."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path  # noqa: TC003 - Pydantic resolves this runtime annotation.
from typing import ClassVar

import torch
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from torch import nn

from hri_ttr.checkpoints.io import checkpoint_sha256


class OfficialHumanImportSpec(BaseModel):
    """Untrusted import request parsed before reading an external checkpoint."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    source: Path
    destination: Path
    key_mapping: dict[str, str]


class ImportStatus(StrEnum):
    """Result of one import mapping decision."""

    COPIED = "copied"
    SKIPPED = "skipped"


class ImportReason(StrEnum):
    """Reason for one import mapping decision."""

    EXACT_MATCH = "exact_match"
    NO_MAPPING = "no_mapping"
    SOURCE_MISSING = "source_missing"
    TARGET_MISSING = "target_missing"
    SHAPE_MISMATCH = "shape_mismatch"


class OfficialHumanKeyRecord(BaseModel):
    """Typed audit record for one source state key."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    source_key: str
    target_key: str | None
    source_shape: tuple[int, ...] | None
    target_shape: tuple[int, ...] | None
    status: ImportStatus
    reason: ImportReason


class OfficialHumanImportReport(BaseModel):
    """Machine-readable accounting of every external checkpoint key."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    records: tuple[OfficialHumanKeyRecord, ...]


class OfficialEnvelope(BaseModel):
    """Supported Lightning-style external checkpoint envelope."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, extra="allow"
    )

    state_dict: dict[str, torch.Tensor]


_PLAIN_STATE_ADAPTER = TypeAdapter(
    dict[str, torch.Tensor], config=ConfigDict(arbitrary_types_allowed=True)
)


def _load_official_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        envelope = OfficialEnvelope.model_validate(
            torch.load(path, map_location="cpu", weights_only=True)
        )
    except ValidationError:
        return _PLAIN_STATE_ADAPTER.validate_python(
            torch.load(path, map_location="cpu", weights_only=True)
        )
    return envelope.state_dict


def import_official_human_checkpoint(
    target: nn.Module, spec: OfficialHumanImportSpec
) -> OfficialHumanImportReport:
    """Copy only explicitly mapped keys whose tensor shapes exactly match."""
    source_state = _load_official_state(spec.source)
    target_state = _PLAIN_STATE_ADAPTER.validate_python(target.state_dict())
    records: list[OfficialHumanKeyRecord] = []
    for source_key in sorted(set(source_state) | set(spec.key_mapping)):
        target_key = spec.key_mapping.get(source_key)
        source_tensor = source_state.get(source_key)
        target_tensor = target_state.get(target_key) if target_key is not None else None
        source_shape = (
            tuple(int(dimension) for dimension in source_tensor.shape)
            if source_tensor is not None
            else None
        )
        target_shape = (
            tuple(int(dimension) for dimension in target_tensor.shape)
            if target_tensor is not None
            else None
        )
        status = ImportStatus.SKIPPED
        if source_tensor is None:
            reason = ImportReason.SOURCE_MISSING
        elif target_key is None:
            reason = ImportReason.NO_MAPPING
        elif target_tensor is None:
            reason = ImportReason.TARGET_MISSING
        elif source_shape != target_shape:
            reason = ImportReason.SHAPE_MISMATCH
        else:
            target_state[target_key] = source_tensor
            status = ImportStatus.COPIED
            reason = ImportReason.EXACT_MATCH
        records.append(
            OfficialHumanKeyRecord(
                source_key=source_key,
                target_key=target_key,
                source_shape=source_shape,
                target_shape=target_shape,
                status=status,
                reason=reason,
            )
        )
    _ = target.load_state_dict(target_state, strict=True)
    spec.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(target.state_dict(), spec.destination)
    report = OfficialHumanImportReport(
        source_sha256=checkpoint_sha256(spec.source),
        records=tuple(records),
    )
    report_path = spec.destination.with_suffix(spec.destination.suffix + ".import.json")
    _ = report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report
