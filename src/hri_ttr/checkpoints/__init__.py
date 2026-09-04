"""Safe checkpoint and external-weight import API."""

from hri_ttr.checkpoints.baseline import write_g1_73d_baseline_manifest
from hri_ttr.checkpoints.import_human import (
    ImportReason,
    ImportStatus,
    OfficialHumanImportReport,
    OfficialHumanImportSpec,
    OfficialHumanKeyRecord,
    import_official_human_checkpoint,
)
from hri_ttr.checkpoints.io import (
    CheckpointComponents,
    CheckpointMismatchError,
    CudaRngCapability,
    MalformedCheckpointError,
    checkpoint_sha256,
    load_training_checkpoint,
    save_training_checkpoint,
)
from hri_ttr.checkpoints.kinds import ModelKind
from hri_ttr.checkpoints.schema import (
    BaselineManifest,
    CheckpointBinding,
    CheckpointSnapshot,
    TrainingProgress,
)

__all__ = [
    "BaselineManifest",
    "CheckpointBinding",
    "CheckpointComponents",
    "CheckpointMismatchError",
    "CheckpointSnapshot",
    "CudaRngCapability",
    "ImportReason",
    "ImportStatus",
    "MalformedCheckpointError",
    "ModelKind",
    "OfficialHumanImportReport",
    "OfficialHumanImportSpec",
    "OfficialHumanKeyRecord",
    "TrainingProgress",
    "checkpoint_sha256",
    "import_official_human_checkpoint",
    "load_training_checkpoint",
    "save_training_checkpoint",
    "write_g1_73d_baseline_manifest",
]
