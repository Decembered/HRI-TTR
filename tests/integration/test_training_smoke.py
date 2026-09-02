from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np

from hri_ttr.checkpoints.kinds import ModelKind
from hri_ttr.representations.g1.schema import G1_SCHEMA_VERSION
from hri_ttr.representations.human.normalizer import SCHEMA_ID as HUMAN_SCHEMA_ID
from hri_ttr.tokenizers import G1Tokenizer, HumanTokenizer, TokenizerArchitecture
from hri_ttr.training import (
    FeatureSequence,
    TrainConfig,
    TrainingIdentity,
    TrainingInvocation,
    train,
)

if TYPE_CHECKING:
    from pathlib import Path


def _invocation(tmp_path: Path, kind: ModelKind, schema: str) -> TrainingInvocation:
    identity = TrainingIdentity(
        normalizer_sha256=hashlib.sha256(b"normalizer").hexdigest(),
        split_sha256=hashlib.sha256(b"split").hexdigest(),
        source_sha256=hashlib.sha256(kind.value.encode()).hexdigest(),
    )
    config = TrainConfig(
        model_kind=kind,
        representation_schema=schema,
        output_dir=tmp_path / kind.value,
        seed=7,
        epochs=2,
        max_steps=1,
        batch_size=1,
        window_frames=8,
        window_stride=8,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        amp=False,
        tokenizer_width=8,
        tokenizer_code_dim=4,
        tokenizer_residual_depth=1,
        log_every_steps=1,
        validation_every_steps=1,
    )
    return TrainingInvocation(config=config, identity=identity)


def test_train_saves_and_resumes_independent_models(tmp_path: Path) -> None:
    # Given
    architecture = TokenizerArchitecture(width=8, code_dim=4, residual_depth=1)
    rng = np.random.default_rng(11)
    human_data = (FeatureSequence("h", rng.normal(size=(8, 262)).astype(np.float32)),)
    g1_data = (FeatureSequence("g", rng.normal(size=(8, 75)).astype(np.float32)),)
    human_invocation = _invocation(tmp_path, ModelKind.HUMAN, HUMAN_SCHEMA_ID)
    g1_invocation = _invocation(tmp_path, ModelKind.G1, G1_SCHEMA_VERSION)

    # When
    human_first = train(
        HumanTokenizer(architecture), human_data, human_data, human_invocation
    )
    g1_first = train(G1Tokenizer(architecture), g1_data, g1_data, g1_invocation)
    human_resumed = train(
        HumanTokenizer(architecture),
        human_data,
        human_data,
        human_invocation.with_resume(human_first.last_checkpoint, max_steps=2),
    )
    g1_resumed = train(
        G1Tokenizer(architecture),
        g1_data,
        g1_data,
        g1_invocation.with_resume(g1_first.last_checkpoint, max_steps=2),
    )

    # Then
    assert human_resumed.global_step == 2
    assert g1_resumed.global_step == 2
    assert human_resumed.binding.model_kind is ModelKind.HUMAN
    assert g1_resumed.binding.model_kind is ModelKind.G1
    assert human_resumed.last_checkpoint != g1_resumed.last_checkpoint


def test_training_reports_step_and_validation_metrics(tmp_path: Path) -> None:
    # Given
    invocation = _invocation(tmp_path, ModelKind.G1, G1_SCHEMA_VERSION)
    values = np.arange(8 * 75, dtype=np.float32).reshape(8, 75) / 1000.0
    data = (FeatureSequence("g", values),)

    # When
    _ = train(
        G1Tokenizer(TokenizerArchitecture(width=8, code_dim=4, residual_depth=1)),
        data,
        data,
        invocation,
    )

    # Then
    rows = [
        json.loads(line)
        for line in (tmp_path / "g1" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["step"] == 1
    assert "train/total_loss" in rows[0]
    assert rows[-1]["validation/loss"] >= 0.0
