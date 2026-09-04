"""Pure PyTorch training orchestration and artifact ownership."""

from __future__ import annotations

import random
import sys
from typing import TYPE_CHECKING

import torch
from pydantic import ConfigDict, TypeAdapter
from torch.nn.parallel import DistributedDataParallel

from hri_ttr.checkpoints import (
    CheckpointBinding,
    CheckpointComponents,
    TrainingProgress,
    checkpoint_sha256,
    load_training_checkpoint,
)
from hri_ttr.checkpoints.kinds import ModelKind
from hri_ttr.training.data import (
    AlignedWindowDataset,
    FeatureSequence,
    WindowConfig,
    WindowDataset,
    build_windows,
)
from hri_ttr.training.distributed import finalize_distributed, initialize_distributed
from hri_ttr.training.engine import EngineInputs, run_steps, validate
from hri_ttr.training.errors import TrainingError, TrainingReason
from hri_ttr.training.losses import MaskedReconstructionLoss
from hri_ttr.training.results import TrainingInterrupted, TrainingResult
from hri_ttr.training.signals import StopController, installed_stop_controller

if TYPE_CHECKING:
    from hri_ttr.tokenizers.common.model import CausalMotionTokenizer
    from hri_ttr.training.config import (
        TrainConfig,
        TrainingIdentity,
        TrainingInvocation,
    )
from hri_ttr.training.artifacts import ArtifactState, save_completed, save_interrupted

_STATE_ADAPTER = TypeAdapter(
    dict[str, torch.Tensor], config=ConfigDict(arbitrary_types_allowed=True)
)


def _binding(config: TrainConfig, identity: TrainingIdentity) -> CheckpointBinding:
    return CheckpointBinding(
        format_version=1,
        model_kind=config.model_kind,
        representation_schema=config.representation_schema,
        tokenizer_config_sha256=config.tokenizer_config_sha256,
        normalizer_sha256=identity.normalizer_sha256,
        split_sha256=identity.split_sha256,
        source_sha256=identity.source_sha256,
    )


def _assert_model_config(model: CausalMotionTokenizer, config: TrainConfig) -> None:
    if model.feature_dim != config.feature_dim:
        raise TrainingError(TrainingReason.FEATURE_WIDTH)
    architecture = model.architecture
    configured = (
        config.tokenizer_width,
        config.tokenizer_code_dim,
        config.tokenizer_codebook_size,
        config.tokenizer_residual_depth,
        config.tokenizer_ema_decay,
        config.tokenizer_commitment_weight,
    )
    actual = (
        architecture.width,
        architecture.code_dim,
        architecture.codebook_size,
        architecture.residual_depth,
        architecture.ema_decay,
        architecture.commitment_weight,
    )
    if configured != actual:
        raise TrainingError(TrainingReason.ARCHITECTURE)


def _load_warm_start(
    model: CausalMotionTokenizer, config: TrainConfig, identity: TrainingIdentity
) -> None:
    path = config.warm_start_checkpoint
    if path is None:
        return
    if config.model_kind is not ModelKind.HUMAN:
        raise TrainingError(TrainingReason.WARM_START_DOMAIN)
    if checkpoint_sha256(path) != identity.source_sha256:
        raise TrainingError(TrainingReason.WARM_START_HASH)
    state = _STATE_ADAPTER.validate_python(
        torch.load(path, map_location="cpu", weights_only=True)
    )
    _ = model.load_state_dict(state, strict=True)


def _dataset(
    sequences: tuple[FeatureSequence, ...], config: TrainConfig
) -> AlignedWindowDataset:
    window = WindowConfig(config.window_frames, config.window_stride)
    return AlignedWindowDataset(build_windows(sequences, window))


def _execute(
    model: CausalMotionTokenizer,
    training_data: WindowDataset,
    validation_data: WindowDataset,
    invocation: TrainingInvocation,
    stop: StopController,
) -> TrainingResult | TrainingInterrupted:
    context = initialize_distributed()
    try:
        config = invocation.config
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        _assert_model_config(model, config)
        _ = model.to(context.device)
        model.set_distributed(context.world_size > 1)
        if invocation.resume_path is None:
            _load_warm_start(model, config, invocation.identity)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=config.amp and context.device.type == "cuda"
        )
        components = CheckpointComponents(model, optimizer, scaler)
        binding = _binding(config, invocation.identity)
        progress = TrainingProgress(
            epoch=0,
            batch_in_epoch=0,
            global_step=0,
            best_validation_loss=sys.float_info.max,
        )
        if invocation.resume_path is not None:
            progress = load_training_checkpoint(
                invocation.resume_path, components, binding
            ).progress
        if progress.global_step >= config.max_steps:
            raise TrainingError(TrainingReason.RESUME_COMPLETE)
        distributed_model = None
        if context.world_size > 1:
            device_ids = [context.local_rank] if context.device.type == "cuda" else None
            distributed_model = DistributedDataParallel(model, device_ids=device_ids)
        engine = EngineInputs(
            model=model,
            distributed_model=distributed_model,
            training_data=training_data,
            validation_data=validation_data,
            config=config,
            distributed=context,
            optimizer=optimizer,
            scaler=scaler,
            loss=MaskedReconstructionLoss.for_schema(config.representation_schema),
            stop=stop,
        )
        outcome = run_steps(engine, progress)
        progress = outcome.progress
        artifact_state = ArtifactState(
            config, context, components, binding, progress, stop
        )
        if outcome.interrupted:
            return save_interrupted(artifact_state)
        validation = validate(engine)
        if validation.loss is None:
            return save_interrupted(artifact_state)
        return save_completed(artifact_state, validation.loss)
    finally:
        finalize_distributed(context)


def train(
    model: CausalMotionTokenizer,
    training_sequences: tuple[FeatureSequence, ...],
    validation_sequences: tuple[FeatureSequence, ...],
    invocation: TrainingInvocation,
) -> TrainingResult:
    """Train through a bounded step count with validation and exact resume."""
    result = _execute(
        model,
        _dataset(training_sequences, invocation.config),
        _dataset(validation_sequences, invocation.config),
        invocation,
        StopController(),
    )
    if isinstance(result, TrainingInterrupted):
        raise KeyboardInterrupt
    return result


def run_training_boundary(
    model: CausalMotionTokenizer,
    training_sequences: tuple[FeatureSequence, ...],
    validation_sequences: tuple[FeatureSequence, ...],
    invocation: TrainingInvocation,
) -> TrainingResult | TrainingInterrupted:
    """Convert SIGTERM or Ctrl-C into an atomic resumable result."""
    with installed_stop_controller() as stop:
        return _execute(
            model,
            _dataset(training_sequences, invocation.config),
            _dataset(validation_sequences, invocation.config),
            invocation,
            stop,
        )


def train_datasets(
    model: CausalMotionTokenizer,
    training_data: WindowDataset,
    validation_data: WindowDataset,
    invocation: TrainingInvocation,
) -> TrainingResult:
    """Train from lazy sequence-bounded datasets without materializing all windows."""
    result = _execute(
        model, training_data, validation_data, invocation, StopController()
    )
    if isinstance(result, TrainingInterrupted):
        raise KeyboardInterrupt
    return result
