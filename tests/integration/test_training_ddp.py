from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from torch import distributed

from hri_ttr.checkpoints.kinds import ModelKind
from hri_ttr.representations.g1.schema import G1_SCHEMA_VERSION
from hri_ttr.tokenizers import G1Tokenizer, TokenizerArchitecture
from hri_ttr.training import (
    FeatureSequence,
    TrainConfig,
    TrainingIdentity,
    TrainingInvocation,
    train,
)


def _worker(output: Path) -> None:
    _ = distributed.init_process_group("gloo")
    config = TrainConfig(
        model_kind=ModelKind.G1,
        representation_schema=G1_SCHEMA_VERSION,
        output_dir=output,
        seed=41,
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
    )
    identity = TrainingIdentity(
        normalizer_sha256=hashlib.sha256(b"normalizer").hexdigest(),
        split_sha256=hashlib.sha256(b"split").hexdigest(),
        source_sha256=hashlib.sha256(b"g1").hexdigest(),
    )
    values = np.arange(16 * 75, dtype=np.float32).reshape(16, 75) / 1000.0
    sequences = (
        FeatureSequence("first", values[:8]),
        FeatureSequence("second", values[8:]),
    )
    model = G1Tokenizer(TokenizerArchitecture(width=8, code_dim=4, residual_depth=1))
    result = train(model, sequences, sequences, TrainingInvocation(config, identity))
    if result.global_step != 1:
        raise RuntimeError
    codebooks = [torch.empty_like(model.quantizer.codebook) for _ in range(2)]
    _ = cast("object", distributed.all_gather(codebooks, model.quantizer.codebook))
    if not torch.equal(codebooks[0], codebooks[1]):
        raise RuntimeError
    distributed.destroy_process_group()


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="PyTorch 2.7 TCPStore standalone rendezvous is broken on macOS",
)
def test_two_rank_cpu_training_saves_checkpoint(tmp_path: Path) -> None:
    # Given
    project = Path(__file__).parents[2]
    output = tmp_path / "ddp"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project / "src")
    command = [
        str(Path(sys.executable).with_name("torchrun")),
        "--standalone",
        "--nproc-per-node=2",
        str(Path(__file__)),
        str(output),
    ]

    # When
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Then
    assert completed.returncode == 0, completed.stderr
    assert (output / "last.pt").is_file()
    assert (output / "best.pt").is_file()


if __name__ == "__main__":
    _worker(Path(sys.argv[1]))
