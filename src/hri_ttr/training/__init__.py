"""Public pure-PyTorch training API."""

from hri_ttr.checkpoints.kinds import ModelKind
from hri_ttr.training.config import TrainConfig, TrainingIdentity, TrainingInvocation
from hri_ttr.training.data import (
    AlignedWindowDataset,
    FeatureSequence,
    TrainingBatch,
    TrainingWindow,
    WindowConfig,
    WindowDataset,
    build_windows,
    collate_windows,
)
from hri_ttr.training.engine import StepMetrics
from hri_ttr.training.results import TrainingInterrupted, TrainingResult
from hri_ttr.training.trainer import run_training_boundary, train, train_datasets

__all__ = [
    "AlignedWindowDataset",
    "FeatureSequence",
    "ModelKind",
    "StepMetrics",
    "TrainConfig",
    "TrainingBatch",
    "TrainingIdentity",
    "TrainingInterrupted",
    "TrainingInvocation",
    "TrainingResult",
    "TrainingWindow",
    "WindowConfig",
    "WindowDataset",
    "build_windows",
    "collate_windows",
    "run_training_boundary",
    "train",
    "train_datasets",
]
