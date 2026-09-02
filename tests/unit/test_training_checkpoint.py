from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest
import torch

from hri_ttr.checkpoints import (
    CheckpointBinding,
    CheckpointComponents,
    CheckpointMismatchError,
    CheckpointSnapshot,
    MalformedCheckpointError,
    OfficialHumanImportSpec,
    TrainingProgress,
    checkpoint_sha256,
    import_official_human_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
    write_g1_73d_baseline_manifest,
)
from hri_ttr.checkpoints.kinds import ModelKind
from hri_ttr.representations.g1.schema import G1_SCHEMA_VERSION
from hri_ttr.representations.human.normalizer import SCHEMA_ID as HUMAN_SCHEMA_ID
from hri_ttr.tokenizers import G1Tokenizer, HumanTokenizer, TokenizerArchitecture

if TYPE_CHECKING:
    from pathlib import Path


def _binding(kind: ModelKind) -> CheckpointBinding:
    schema = HUMAN_SCHEMA_ID if kind is ModelKind.HUMAN else G1_SCHEMA_VERSION
    return CheckpointBinding(
        format_version=1,
        model_kind=kind,
        representation_schema=schema,
        tokenizer_config_sha256="1" * 64,
        normalizer_sha256="2" * 64,
        split_sha256="3" * 64,
        source_sha256="4" * 64,
    )


def test_checkpoint_refuses_cross_domain_load(tmp_path: Path) -> None:
    # Given
    architecture = TokenizerArchitecture(width=8, code_dim=4, residual_depth=1)
    human = HumanTokenizer(architecture)
    optimizer = torch.optim.AdamW(human.parameters(), lr=1e-3)
    path = tmp_path / "human.pt"
    snapshot = CheckpointSnapshot(
        binding=_binding(ModelKind.HUMAN),
        progress=TrainingProgress(
            epoch=0,
            batch_in_epoch=0,
            global_step=0,
            best_validation_loss=sys.float_info.max,
        ),
    )
    save_training_checkpoint(
        path, CheckpointComponents(human, optimizer, None), snapshot
    )

    # When / Then
    g1 = G1Tokenizer(architecture)
    g1_optimizer = torch.optim.AdamW(g1.parameters(), lr=1e-3)
    with pytest.raises(CheckpointMismatchError):
        _ = load_training_checkpoint(
            path,
            CheckpointComponents(g1, g1_optimizer, None),
            _binding(ModelKind.G1),
        )


def test_checkpoint_hash_is_stable_file_digest(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "sample.pt"
    _ = path.write_bytes(b"checkpoint")

    # When
    first = checkpoint_sha256(path)
    second = checkpoint_sha256(path)

    # Then
    assert first == second
    assert len(first) == 64


def test_official_import_copies_only_explicit_shape_match(tmp_path: Path) -> None:
    # Given
    architecture = TokenizerArchitecture(width=8, code_dim=4, residual_depth=1)
    model = HumanTokenizer(architecture)
    source_key = "quantizer.codebook"
    target_key = source_key
    source = tmp_path / "official.ckpt"
    destination = tmp_path / "imported.pt"
    replacement = torch.full_like(model.quantizer.codebook, 3.0)
    torch.save(
        {
            "state_dict": {
                source_key: replacement,
                "bad_shape": torch.ones(1),
                "extra": torch.ones(1),
                "target_missing": torch.ones(1),
            }
        },
        source,
    )
    spec = OfficialHumanImportSpec(
        source=source,
        destination=destination,
        key_mapping={
            source_key: target_key,
            "bad_shape": target_key,
            "source_missing": target_key,
            "target_missing": "not.a.target.key",
        },
    )

    # When
    report = import_official_human_checkpoint(model, spec)

    # Then
    records = {record.source_key: record for record in report.records}
    assert records[source_key].status == "copied"
    assert records[source_key].target_key == target_key
    assert records[source_key].source_shape == tuple(replacement.shape)
    assert records[source_key].target_shape == tuple(replacement.shape)
    assert records["extra"].status == "skipped"
    assert records["extra"].target_key is None
    assert records["extra"].reason == "no_mapping"
    assert records["bad_shape"].reason == "shape_mismatch"
    assert records["source_missing"].reason == "source_missing"
    assert records["source_missing"].source_shape is None
    assert records["target_missing"].reason == "target_missing"
    assert records["target_missing"].target_shape is None
    torch.testing.assert_close(model.state_dict()[target_key], replacement)
    assert destination.is_file()


def test_g1_legacy_manifest_never_loads_invalid_payload(tmp_path: Path) -> None:
    # Given
    source = tmp_path / "legacy.ckpt"
    _ = source.write_bytes(b"not a torch checkpoint")
    destination = tmp_path / "legacy.json"

    # When
    manifest = write_g1_73d_baseline_manifest(source, destination)

    # Then
    assert manifest.load_policy == "baseline_only_no_partial_load"
    assert manifest.source_sha256 == checkpoint_sha256(source)


def test_short_checkpoint_is_wrapped_as_malformed(tmp_path: Path) -> None:
    # Given
    architecture = TokenizerArchitecture(width=8, code_dim=4, residual_depth=1)
    model = HumanTokenizer(architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "short.pt"
    _ = path.write_bytes(b"PK")

    # When / Then
    with pytest.raises(MalformedCheckpointError) as caught:
        _ = load_training_checkpoint(
            path,
            CheckpointComponents(model, optimizer, None),
            _binding(ModelKind.HUMAN),
        )
    assert caught.value.__cause__ is not None


def test_invalid_payload_is_wrapped_as_malformed(tmp_path: Path) -> None:
    # Given
    architecture = TokenizerArchitecture(width=8, code_dim=4, residual_depth=1)
    model = HumanTokenizer(architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "invalid-payload.pt"
    torch.save({"snapshot": "not-a-complete-payload"}, path)

    # When / Then
    with pytest.raises(MalformedCheckpointError) as caught:
        _ = load_training_checkpoint(
            path,
            CheckpointComponents(model, optimizer, None),
            _binding(ModelKind.HUMAN),
        )
    assert caught.value.__cause__ is not None


class FakeCudaRng:
    def __init__(self) -> None:
        self.restored: tuple[torch.Tensor, ...] = ()

    def available(self) -> bool:
        return True

    def capture(self) -> tuple[torch.Tensor, ...]:
        return (torch.tensor([7, 8], dtype=torch.uint8),)

    def restore(self, states: tuple[torch.Tensor, ...]) -> None:
        self.restored = states


def test_checkpoint_restores_cuda_rng_through_capability(tmp_path: Path) -> None:
    # Given
    architecture = TokenizerArchitecture(width=8, code_dim=4, residual_depth=1)
    model = HumanTokenizer(architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    capability = FakeCudaRng()
    components = CheckpointComponents(model, optimizer, None, capability)
    snapshot = CheckpointSnapshot(
        binding=_binding(ModelKind.HUMAN),
        progress=TrainingProgress(
            epoch=0,
            batch_in_epoch=0,
            global_step=0,
            best_validation_loss=sys.float_info.max,
        ),
    )
    path = tmp_path / "cuda-rng.pt"
    save_training_checkpoint(path, components, snapshot)

    # When
    _ = load_training_checkpoint(path, components, snapshot.binding)

    # Then
    assert len(capability.restored) == 1
    torch.testing.assert_close(
        capability.restored[0], torch.tensor([7, 8], dtype=torch.uint8)
    )
