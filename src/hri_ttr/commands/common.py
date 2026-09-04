"""Shared strict artifact boundaries for CLI commands."""

# pyright: reportAny=false, reportUnknownMemberType=false

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - Typer resolves runtime annotations.
from typing import ClassVar, Final, Never

import numpy as np
import numpy.typing as npt
import torch
import typer
from pydantic import BaseModel, ConfigDict

from hri_ttr.checkpoints.io import CheckpointPayload
from hri_ttr.checkpoints.kinds import ModelKind
from hri_ttr.checkpoints.schema import CheckpointSnapshot
from hri_ttr.tokenizers import G1Tokenizer, HumanTokenizer, TokenizerArchitecture
from hri_ttr.training import TrainConfig

PREPARED_KEYS: Final = {
    "sequence_id",
    "human_features",
    "g1_features",
    "human_space",
    "g1_space",
    "human_mask",
    "g1_mask",
    "anchor_origin",
    "anchor_basis",
    "fps",
    "source_fps",
    "source_format",
    "target_fps",
    "valid_frames",
}
INPUT_FPS: Final = 20.0
Tokenizer = HumanTokenizer | G1Tokenizer
LOGGER = logging.getLogger(__name__)


class PreparedMetadata(BaseModel):
    """Parsed metadata stored alongside prepared arrays."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )
    sequence_id: str
    fps: float
    source_fps: float
    source_format: str
    target_fps: float
    valid_frames: int


@dataclass(frozen=True, slots=True)
class PreparedMotion:
    """Validated arrays emitted by ``data prepare``."""

    metadata: PreparedMetadata
    human_features: npt.NDArray[np.float32]
    g1_features: npt.NDArray[np.float64]
    human_space: npt.NDArray[np.float32]
    g1_space: npt.NDArray[np.float64]
    human_mask: npt.NDArray[np.bool_]
    g1_mask: npt.NDArray[np.bool_]
    anchor_origin: npt.NDArray[np.float64]
    anchor_basis: npt.NDArray[np.float64]


def fail(detail: str, *, cause: BaseException | None = None) -> Never:
    """Terminate one expected CLI validation failure without a traceback."""
    typer.echo(f"Error: {detail}", err=True)
    if cause is None:
        raise typer.Exit(code=2)
    LOGGER.debug("CLI validation failure", exc_info=cause)
    raise typer.Exit(code=2) from cause


def sha256_file(path: Path) -> str:
    """Hash one existing file without loading its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_prepared(path: Path) -> PreparedMotion:
    """Parse the exact non-pickle NPZ contract emitted by data preparation."""
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != PREPARED_KEYS:
            fail("prepared NPZ has missing or unexpected fields")
        sequence_id = str(archive["sequence_id"].item())
        fps = float(archive["fps"].item())
        source_fps = float(archive["source_fps"].item())
        source_format = str(archive["source_format"].item())
        target_fps = float(archive["target_fps"].item())
        valid_frames = int(archive["valid_frames"].item())
        human = np.asarray(archive["human_features"], dtype=np.float32)
        g1 = np.asarray(archive["g1_features"], dtype=np.float64)
        human_space = np.asarray(archive["human_space"], dtype=np.float32)
        g1_space = np.asarray(archive["g1_space"], dtype=np.float64)
        human_mask = np.asarray(archive["human_mask"], dtype=np.bool_)
        g1_mask = np.asarray(archive["g1_mask"], dtype=np.bool_)
        origin = np.asarray(archive["anchor_origin"], dtype=np.float64)
        basis = np.asarray(archive["anchor_basis"], dtype=np.float64)
    frames = len(human)
    valid = (
        human.shape == (frames, 262)
        and g1.shape == (frames, 75)
        and human_space.shape == (frames, 3)
        and g1_space.shape == (frames, 3)
        and human_mask.shape == (frames,)
        and g1_mask.shape == (frames,)
        and human_mask.dtype == np.bool_
        and g1_mask.dtype == np.bool_
        and origin.shape == (3,)
        and basis.shape == (3, 3)
        and frames > 0
        and frames % 4 == 0
        and 0 < valid_frames <= frames
        and frames == valid_frames + (-valid_frames) % 4
        and np.isfinite(human).all()
        and np.isfinite(g1).all()
        and np.array_equal(human_mask, g1_mask)
        and bool(np.all(human_mask[:valid_frames]))
        and not bool(np.any(human_mask[valid_frames:]))
        and fps == INPUT_FPS
        and target_fps == INPUT_FPS
        and np.isfinite(source_fps)
        and source_fps > 0.0
        and source_format in {"safe_npz", "trusted_pickle"}
    )
    if not valid:
        fail("prepared NPZ violates the aligned Human/G1 contract")
    return PreparedMotion(
        PreparedMetadata(
            sequence_id=sequence_id,
            fps=fps,
            source_fps=source_fps,
            source_format=source_format,
            target_fps=target_fps,
            valid_frames=valid_frames,
        ),
        human,
        g1,
        human_space,
        g1_space,
        human_mask,
        g1_mask,
        origin,
        basis,
    )


def load_prepared(path: Path) -> PreparedMotion:
    """Load prepared motion while presenting malformed artifacts as CLI errors."""
    try:
        return _load_prepared(path)
    except typer.Exit:
        raise
    except Exception as error:  # noqa: BLE001 - public artifact safety boundary.
        fail("prepared NPZ is unreadable or malformed", cause=error)


def model_from_config(config: TrainConfig) -> Tokenizer:
    """Instantiate the independently owned model selected by a train config."""
    architecture = TokenizerArchitecture(
        width=config.tokenizer_width,
        code_dim=config.tokenizer_code_dim,
        codebook_size=config.tokenizer_codebook_size,
        residual_depth=config.tokenizer_residual_depth,
        ema_decay=config.tokenizer_ema_decay,
        commitment_weight=config.tokenizer_commitment_weight,
    )
    match config.model_kind:
        case ModelKind.HUMAN:
            return HumanTokenizer(architecture)
        case ModelKind.G1:
            return G1Tokenizer(architecture)


def load_model(
    config_path: Path, checkpoint_path: Path
) -> tuple[Tokenizer, TrainConfig]:
    """Load an HRI-TTR checkpoint and reject domain/config identity mismatches."""
    try:
        config = TrainConfig.load_json(config_path)
        payload = CheckpointPayload.model_validate(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )
        snapshot = CheckpointSnapshot.model_validate_json(payload.snapshot)
    except Exception as error:  # noqa: BLE001 - public artifact safety boundary.
        fail("checkpoint or config is unreadable or malformed", cause=error)
    if snapshot.binding.model_kind is not config.model_kind:
        fail("checkpoint domain does not match config domain")
    if snapshot.binding.tokenizer_config_sha256 != config.tokenizer_config_sha256:
        fail("checkpoint tokenizer architecture does not match config")
    model = model_from_config(config)
    _ = model.load_state_dict(payload.model, strict=True)
    _ = model.eval()
    return model, config
