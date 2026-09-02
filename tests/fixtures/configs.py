"""Tiny deterministic tokenizer configuration fixture."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

from hri_ttr.representations.g1 import G1_SCHEMA_VERSION
from hri_ttr.representations.human.normalizer import SCHEMA_ID as HUMAN_SCHEMA_ID

if TYPE_CHECKING:
    from pathlib import Path


def write_tiny_config(path: Path, domain: Literal["human", "g1"], output: Path) -> Path:
    """Write a one-step CPU-sized config with the selected domain schema."""
    feature_schema = HUMAN_SCHEMA_ID if domain == "human" else G1_SCHEMA_VERSION
    _ = path.write_text(
        json.dumps(
            {
                "model_kind": domain,
                "representation_schema": feature_schema,
                "output_dir": str(output),
                "seed": 7,
                "epochs": 1,
                "max_steps": 1,
                "batch_size": 1,
                "window_frames": 8,
                "window_stride": 8,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "gradient_clip_norm": 1.0,
                "amp": False,
                "tokenizer_width": 8,
                "tokenizer_code_dim": 4,
                "tokenizer_codebook_size": 256,
                "tokenizer_residual_depth": 1,
                "tokenizer_ema_decay": 0.99,
                "tokenizer_commitment_weight": 1.0,
            }
        ),
        encoding="utf-8",
    )
    return path
