"""Identity, crop, split, and serialization rules for same-motion data."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

Split = Literal["train", "val", "test"]
_SPLIT_PRIORITY: dict[str, int] = {"train": 0, "val": 1, "test": 2}
_TRAIN_PERCENT = 90
_VALIDATION_PERCENT = 95


@dataclass(frozen=True, slots=True)
class EmberClip:
    """A half-open clip range on Ember's actual sampling grid."""

    start: int
    end: int
    effective_fps: float


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def canonical_amass_key(source_path: str) -> tuple[str, str]:
    """Normalize a HumanML source path to an Ember lookup key."""
    normalized = source_path.replace("\\", "/").removeprefix("./")
    normalized = normalized.removeprefix("pose_data/")
    path = PurePosixPath(normalized)
    directory = "/".join(_compact(part) for part in path.parent.parts)
    return directory, _compact(path.stem)


def ember_key(path: Path, root: Path) -> tuple[tuple[str, str], int] | None:
    """Parse the source identity and source fps from an Ember path."""
    relative = path.relative_to(root)
    match = re.fullmatch(r"(.+)_([0-9]+)_jpos", path.stem)
    if match is None:
        return None
    directory = "/".join(_compact(part) for part in relative.parts[:-1])
    return (directory, _compact(match.group(1))), int(match.group(2))


def aligned_clip_bounds(
    *,
    start_20hz: int,
    end_20hz: int,
    human_source_frames: int,
    g1_source_frames: int,
) -> EmberClip:
    """Map a HumanML crop using the measured full Human/G1 frame ratio."""
    start = start_20hz * g1_source_frames // human_source_frames
    end = (end_20hz * g1_source_frames + human_source_frames - 1) // human_source_frames
    end = min(end, g1_source_frames)
    return EmberClip(
        start=start,
        end=end,
        effective_fps=20 * g1_source_frames / human_source_frames,
    )


def choose_group_split(memberships: set[str]) -> Split:
    """Select one leakage-safe split for every member of a source group."""
    if not memberships or not memberships <= _SPLIT_PRIORITY.keys():
        detail = f"invalid split memberships: {sorted(memberships)}"
        raise ValueError(detail)
    return cast("Split", max(memberships, key=_SPLIT_PRIORITY.__getitem__))


def stable_split(group_id: str) -> Split:
    """Assign an unsplit source group with a stable 90/5/5 partition."""
    bucket = int.from_bytes(hashlib.sha256(group_id.encode()).digest()[:8], "big") % 100
    if bucket < _TRAIN_PERCENT:
        return "train"
    if bucket < _VALIDATION_PERCENT:
        return "val"
    return "test"


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    """Write JSONL atomically and return its row count."""
    _ = path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            _ = stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            _ = stream.write("\n")
            count += 1
    _ = temporary.replace(path)
    return count
