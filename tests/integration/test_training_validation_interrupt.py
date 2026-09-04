from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

from hri_ttr.checkpoints.kinds import ModelKind
from hri_ttr.representations.g1.schema import G1_SCHEMA_VERSION
from hri_ttr.tokenizers import G1Tokenizer, TokenizerArchitecture
from hri_ttr.training import (
    FeatureSequence,
    TrainConfig,
    TrainingIdentity,
    TrainingInterrupted,
    TrainingInvocation,
    run_training_boundary,
    train,
)

if TYPE_CHECKING:
    import torch

    from hri_ttr.tokenizers.common.contracts import TokenizerOutput


def _identity() -> TrainingIdentity:
    return TrainingIdentity(
        normalizer_sha256=hashlib.sha256(b"normalizer").hexdigest(),
        split_sha256=hashlib.sha256(b"split").hexdigest(),
        source_sha256=hashlib.sha256(b"g1").hexdigest(),
    )


def _invocation(
    output: Path, steps: int, resume: Path | None = None
) -> TrainingInvocation:
    config = TrainConfig(
        model_kind=ModelKind.G1,
        representation_schema=G1_SCHEMA_VERSION,
        output_dir=output,
        seed=29,
        epochs=100,
        max_steps=steps,
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
    return TrainingInvocation(config=config, identity=_identity(), resume_path=resume)


def _data() -> tuple[FeatureSequence, ...]:
    values = np.arange(8 * 75, dtype=np.float32).reshape(8, 75) / 1000.0
    return (FeatureSequence("g", values),)


class _ValidationInterruptG1(G1Tokenizer):
    @override
    def forward(
        self, features: torch.Tensor, frame_mask: torch.Tensor
    ) -> TokenizerOutput:
        if not self.training:
            raise KeyboardInterrupt
        return super().forward(features, frame_mask)


def test_keyboard_interrupt_during_validation_saves_and_resumes(tmp_path: Path) -> None:
    # Given
    architecture = TokenizerArchitecture(width=8, code_dim=4, residual_depth=1)
    invocation = _invocation(tmp_path, 1)

    # When
    result = run_training_boundary(
        _ValidationInterruptG1(architecture), _data(), _data(), invocation
    )

    # Then
    assert isinstance(result, TrainingInterrupted)
    assert result.exit_code == 128 + signal.SIGINT
    assert result.global_step == 1
    assert result.best_validation_loss == sys.float_info.max
    assert result.interrupted_checkpoint.is_file()
    assert not (tmp_path / "last.pt").exists()
    resumed = train(
        G1Tokenizer(architecture),
        _data(),
        _data(),
        _invocation(tmp_path, 2, result.interrupted_checkpoint),
    )
    assert resumed.global_step == 2


def test_sigterm_during_validation_saves_and_resumes(tmp_path: Path) -> None:
    # Given
    project = Path(__file__).parents[2]
    output = tmp_path / "run"
    script = textwrap.dedent(
        """
        import hashlib, os, sys, time
        from pathlib import Path
        import numpy as np
        from hri_ttr.checkpoints.kinds import ModelKind
        from hri_ttr.tokenizers import G1Tokenizer, TokenizerArchitecture
        from hri_ttr.training import FeatureSequence, TrainConfig, TrainingIdentity
        from hri_ttr.training import TrainingInterrupted, TrainingInvocation
        from hri_ttr.training import run_training_boundary

        class SlowValidationG1(G1Tokenizer):
            def forward(self, features, frame_mask=None):
                if not self.training:
                    os.write(int(sys.argv[2]), b"VALIDATING")
                    time.sleep(1.0)
                return super().forward(features, frame_mask)

        output = Path(sys.argv[1])
        config = TrainConfig(
            model_kind=ModelKind.G1,
            representation_schema="g1_canonical_75d_v2",
            output_dir=output, seed=29, epochs=100, max_steps=1,
            batch_size=1, window_frames=8, window_stride=8,
            learning_rate=1e-3, weight_decay=0.0, gradient_clip_norm=1.0,
            amp=False, tokenizer_width=8, tokenizer_code_dim=4,
            tokenizer_residual_depth=1,
        )
        identity = TrainingIdentity(
            normalizer_sha256=hashlib.sha256(b"normalizer").hexdigest(),
            split_sha256=hashlib.sha256(b"split").hexdigest(),
            source_sha256=hashlib.sha256(b"g1").hexdigest(),
        )
        values = np.arange(8 * 75, dtype=np.float32).reshape(8, 75) / 1000.0
        data = (FeatureSequence("g", values),)
        model = SlowValidationG1(
            TokenizerArchitecture(width=8, code_dim=4, residual_depth=1)
        )
        result = run_training_boundary(
            model, data, data, TrainingInvocation(config=config, identity=identity)
        )
        if isinstance(result, TrainingInterrupted):
            raise SystemExit(result.exit_code)
        raise SystemExit(0)
        """
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project / "src")
    read_fd, write_fd = os.pipe()
    process: subprocess.Popen[bytes] = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", script, str(output), str(write_fd)],
        cwd=project,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        pass_fds=(write_fd,),
    )
    os.close(write_fd)
    assert os.read(read_fd, len(b"VALIDATING")) == b"VALIDATING"
    os.close(read_fd)

    # When
    process.send_signal(signal.SIGTERM)
    return_code = process.wait(timeout=15)

    # Then
    assert return_code == 128 + signal.SIGTERM
    interrupted = output / "interrupted.pt"
    assert interrupted.is_file()
    assert not (output / "last.pt").exists()
    architecture = TokenizerArchitecture(width=8, code_dim=4, residual_depth=1)
    resumed = train(
        G1Tokenizer(architecture), _data(), _data(), _invocation(output, 2, interrupted)
    )
    assert resumed.global_step == 2
