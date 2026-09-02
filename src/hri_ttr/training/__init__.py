"""Public pure-PyTorch training API."""

from hri_ttr.checkpoints.kinds import ModelKind
from hri_ttr.training.config import TrainConfig, TrainingIdentity, TrainingInvocation
from hri_ttr.training.data import (
    AlignedWindowDataset,
    FeatureSequence,
    TrainingBatch,
    TrainingWindow,
    WindowConfig,
    build_windows,
    collate_windows,
)
from hri_ttr.training.results import TrainingInterrupted, TrainingResult
from hri_ttr.training.trainer import run_training_boundary, train

__all__ = [
    "AlignedWindowDataset",
    "FeatureSequence",
    "ModelKind",
    "TrainConfig",
    "TrainingBatch",
    "TrainingIdentity",
    "TrainingInterrupted",
    "TrainingInvocation",
    "TrainingResult",
    "TrainingWindow",
    "WindowConfig",
    "build_windows",
    "collate_windows",
    "run_training_boundary",
    "train",
]
