from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

from hri_ttr.cache import CacheManifest, TokenCache, read_token_cache, write_token_cache
from hri_ttr.evaluation import codebook_statistics
from hri_ttr.visualization import render_token_histogram


def test_cache_pipeline_when_two_sequences_are_written(tmp_path: Path) -> None:
    # Given
    destination = tmp_path / "cache"
    manifest = CacheManifest(
        tokenizer_sha256="1" * 64,
        checkpoint_sha256="2" * 64,
        normalizer_sha256="3" * 64,
        schema_sha256="4" * 64,
        split_sha256="5" * 64,
        valid_frame_lengths=(5, 8),
        valid_token_lengths=(2, 2),
        padded_frame_counts=(3, 0),
    )
    cache = TokenCache(
        np.array([0, 1, 1, 255], dtype=np.uint16),
        np.array([0, 2, 4], dtype=np.int64),
        ("a", "b"),
        manifest,
    )

    # When
    _ = write_token_cache(destination, cache)
    loaded = read_token_cache(destination)
    statistics = codebook_statistics(
        loaded.tokens.astype(np.int64), np.ones(4, dtype=np.bool_)
    )
    image = render_token_histogram(statistics.histogram, tmp_path / "hist.png")

    # Then
    assert statistics.used_code_count == 3
    assert image.stat().st_size > 1000
