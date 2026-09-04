"""Typed loading boundary for safe NPZ and explicitly trusted pickle pairs."""

# pyright: reportAny=false, reportUnknownMemberType=false

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

import joblib
import numpy as np
from typing_extensions import override

from hri_ttr.data.pair_schema import (
    ActorMotion,
    PairAlignmentError,
    PairSchemaError,
    ReactorMotion,
)

__all__ = [
    "ActorMotion",
    "PairAlignmentError",
    "PairSchemaError",
    "ReactorMotion",
]

if TYPE_CHECKING:
    from pathlib import Path

PAIR_PATTERN = re.compile(r"^(G\d{3}T\d{3}A\d{3}R\d{3})_(actor|reactor)\.(npz|pkl)$")
ACTOR_NPZ_FIELDS: Final = {
    "fps",
    "root_pos",
    "root_rot",
    "pose_body",
    "joints_pos",
    "betas",
    "gender",
}
REACTOR_NPZ_FIELDS: Final = {"fps", "root_pos", "root_rot", "dof_pos", "joints_pos"}


class PairSourceFormat(StrEnum):
    """Serialization and trust provenance for one pair."""

    SAFE_NPZ = "safe_npz"
    TRUSTED_PICKLE = "trusted_pickle"


class PickleTrustPolicy(StrEnum):
    """Explicit consent required before executable pickle deserialization."""

    DENY = "deny"
    TRUSTED_LOCAL = "trusted_local"


@dataclass(slots=True)
class PickleConsentRequiredError(ValueError):
    """Reject pickle before deserialization unless local trust was declared."""

    path: Path

    @override
    def __str__(self) -> str:
        return (
            "pickle can execute code; only a trusted local corpus may be loaded "
            "with explicit consent"
        )


@dataclass(frozen=True, slots=True)
class PairPaths:
    """Paths for one matched actor/reactor sequence ID."""

    sequence_id: str
    actor_path: Path
    reactor_path: Path
    source_format: PairSourceFormat


@dataclass(frozen=True, slots=True)
class PairDiscovery:
    """Matched pairs plus IDs missing either role."""

    pairs: tuple[PairPaths, ...]
    actor_only: tuple[str, ...]
    reactor_only: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetargetedPair:
    """One timeline-aligned Human actor and G1 reactor pair."""

    sequence_id: str
    actor: ActorMotion
    reactor: ReactorMotion
    source_format: PairSourceFormat


def discover_pairs(root: Path) -> PairDiscovery:
    """Discover deterministic pair filenames without opening their contents."""
    actors: dict[tuple[str, str], Path] = {}
    reactors: dict[tuple[str, str], Path] = {}
    for path in sorted(root.iterdir()):
        match = PAIR_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        sequence_id, role, extension = match.groups()
        destination = actors if role == "actor" else reactors
        destination[(sequence_id, extension)] = path
    paired = sorted(actors.keys() & reactors.keys())
    return PairDiscovery(
        tuple(
            PairPaths(
                sequence_id,
                actors[(sequence_id, extension)],
                reactors[(sequence_id, extension)],
                PairSourceFormat.SAFE_NPZ
                if extension == "npz"
                else PairSourceFormat.TRUSTED_PICKLE,
            )
            for sequence_id, extension in paired
        ),
        tuple(
            sorted(sequence_id for sequence_id, _ in actors.keys() - reactors.keys())
        ),
        tuple(
            sorted(sequence_id for sequence_id, _ in reactors.keys() - actors.keys())
        ),
    )


def _load_safe_npz(paths: PairPaths) -> tuple[ActorMotion, ReactorMotion]:
    with np.load(paths.actor_path, allow_pickle=False) as actor_archive:
        if set(actor_archive.files) != ACTOR_NPZ_FIELDS:
            field = "actor.npz"
            raise PairSchemaError(field, "unexpected or missing fields")
        actor = ActorMotion(
            fps=float(actor_archive["fps"].item()),
            root_pos=actor_archive["root_pos"],
            root_rot=actor_archive["root_rot"],
            pose_body=actor_archive["pose_body"],
            joints_pos=actor_archive["joints_pos"],
            betas=actor_archive["betas"],
            gender=str(actor_archive["gender"].item()),
        )
    with np.load(paths.reactor_path, allow_pickle=False) as reactor_archive:
        if set(reactor_archive.files) != REACTOR_NPZ_FIELDS:
            field = "reactor.npz"
            raise PairSchemaError(field, "unexpected or missing fields")
        reactor = ReactorMotion(
            fps=float(reactor_archive["fps"].item()),
            root_pos=reactor_archive["root_pos"],
            root_rot=reactor_archive["root_rot"],
            dof_pos=reactor_archive["dof_pos"],
            joints_pos=reactor_archive["joints_pos"],
        )
    return actor, reactor


def load_retargeted_pair(
    paths: PairPaths,
    *,
    pickle_policy: PickleTrustPolicy = PickleTrustPolicy.DENY,
) -> RetargetedPair:
    """Load, parse, and require exact frame/FPS alignment for one pair."""
    match paths.source_format:
        case PairSourceFormat.SAFE_NPZ:
            actor, reactor = _load_safe_npz(paths)
        case PairSourceFormat.TRUSTED_PICKLE:
            if pickle_policy is PickleTrustPolicy.DENY:
                raise PickleConsentRequiredError(paths.actor_path)
            actor = ActorMotion.model_validate(joblib.load(paths.actor_path))
            reactor = ReactorMotion.model_validate(joblib.load(paths.reactor_path))
    actor_frames = len(actor.root_pos)
    reactor_frames = len(reactor.root_pos)
    if actor_frames != reactor_frames or actor.fps != reactor.fps:
        raise PairAlignmentError(
            paths.sequence_id,
            actor_frames,
            reactor_frames,
            actor.fps,
            reactor.fps,
        )
    return RetargetedPair(paths.sequence_id, actor, reactor, paths.source_format)
