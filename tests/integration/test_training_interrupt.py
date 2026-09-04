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


def _config(output: Path, max_steps: int) -> TrainConfig:
    return TrainConfig(
        model_kind=ModelKind.G1,
        representation_schema=G1_SCHEMA_VERSION,
        output_dir=output,
        seed=19,
        epochs=100,
        max_steps=max_steps,
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


class _InterruptingG1(G1Tokenizer):
    @override
    def forward(
        self, features: torch.Tensor, frame_mask: torch.Tensor | None = None
    ) -> TokenizerOutput:
        raise KeyboardInterrupt


def test_keyboard_interrupt_returns_typed_resumable_result(tmp_path: Path) -> None:
    # Given
    architecture = TokenizerArchitecture(width=8, code_dim=4, residual_depth=1)
    data = (FeatureSequence("g", np.zeros((8, 75), dtype=np.float32)),)
    invocation = TrainingInvocation(config=_config(tmp_path, 2), identity=_identity())

    # When
    result = run_training_boundary(
        _InterruptingG1(architecture), data, data, invocation
    )

    # Then
    assert isinstance(result, TrainingInterrupted)
    assert result.exit_code == 128 + signal.SIGINT
    assert result.global_step == 0
    assert result.interrupted_checkpoint.is_file()


def test_sigterm_saves_atomic_checkpoint_and_resumes(tmp_path: Path) -> None:
    # Given
    project = Path(__file__).parents[2]
    output = tmp_path / "run"
    script = textwrap.dedent(
        """
        import hashlib
        import os
        import signal
        import sys
        import time
        from pathlib import Path

        import numpy as np

        from hri_ttr.checkpoints.kinds import ModelKind
        from hri_ttr.tokenizers import G1Tokenizer, TokenizerArchitecture
        from hri_ttr.training import (
            FeatureSequence, TrainConfig, TrainingIdentity, TrainingInvocation,
            TrainingInterrupted, run_training_boundary,
        )

        class SlowG1(G1Tokenizer):
            def forward(self, features, frame_mask=None):
                os.write(int(sys.argv[2]), b"STEP_STARTED")
                time.sleep(1.0)
                return super().forward(features, frame_mask)

        output = Path(sys.argv[1])
        config = TrainConfig(
            model_kind=ModelKind.G1,
            representation_schema="g1_canonical_75d_v2",
            output_dir=output, seed=19, epochs=100, max_steps=100,
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
        data = (FeatureSequence("g", np.zeros((8, 75), dtype=np.float32)),)
        model = SlowG1(TokenizerArchitecture(width=8, code_dim=4, residual_depth=1))
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
    assert os.read(read_fd, len(b"STEP_STARTED")) == b"STEP_STARTED"
    os.close(read_fd)

    # When
    process.send_signal(signal.SIGTERM)
    return_code = process.wait(timeout=15)

    # Then
    assert return_code == 128 + signal.SIGTERM
    interrupted = output / "interrupted.pt"
    assert interrupted.is_file()
    assert not interrupted.with_suffix(".pt.tmp").exists()
    architecture = TokenizerArchitecture(width=8, code_dim=4, residual_depth=1)
    data = (FeatureSequence("g", np.zeros((8, 75), dtype=np.float32)),)
    invocation = TrainingInvocation(
        config=_config(output, 2), identity=_identity(), resume_path=interrupted
    )
    resumed = train(G1Tokenizer(architecture), data, data, invocation)
    assert resumed.global_step == 2
