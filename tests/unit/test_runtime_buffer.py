"""Tests for the causal online prefix buffer contract."""

from __future__ import annotations

import numpy as np
import pytest

from hri_ttr.contracts import FramesPerToken, ModelId, SchemaId, TokenizerSpec
from hri_ttr.language import SemanticCondition
from hri_ttr.runtime import CausalPrefixBuffer, RuntimeBufferError


def _spec(kind: str, model: str) -> TokenizerSpec:
    return TokenizerSpec(
        kind=kind,
        schema_id=SchemaId(f"{kind}-schema"),
        model_id=ModelId(model),
        input_fps=20.0,
        frames_per_token=FramesPerToken(4),
        codebook_size=256,
    )


def test_runtime_buffer_builds_next_chunk_context() -> None:
    # Given: an empty online buffer with independent tokenizers.
    buffer = CausalPrefixBuffer(
        human_tokenizer=_spec("human", "human-a"),
        g1_tokenizer=_spec("g1", "g1-a"),
        semantic=SemanticCondition("mirror the arm motion"),
    )

    # When: two aligned observations are appended.
    buffer.append_observation(human_token_id=3, g1_token_id=17)
    buffer.append_observation(human_token_id=4, g1_token_id=18)
    context = buffer.student_context()

    # Then: the context predicts token 2 from observations 0 and 1.
    assert context.target_index == 2
    assert context.human_prefix.token_ids.tolist() == [[3, 4]]
    assert context.g1_prefix.token_ids.tolist() == [[17, 18]]


def test_runtime_buffer_rejects_out_of_range_token_before_mutation() -> None:
    # Given: an empty online buffer.
    buffer = CausalPrefixBuffer(
        human_tokenizer=_spec("human", "human-a"),
        g1_tokenizer=_spec("g1", "g1-a"),
        semantic=SemanticCondition("wait"),
    )

    # When / Then: an invalid G1 token does not create a partial observation.
    with pytest.raises(RuntimeBufferError, match="G1 token ID"):
        buffer.append_observation(human_token_id=2, g1_token_id=256)
    with pytest.raises(RuntimeBufferError, match="at least one"):
        _ = buffer.student_context()


def test_runtime_context_arrays_are_read_only_snapshots() -> None:
    # Given: one observed token pair.
    buffer = CausalPrefixBuffer(
        human_tokenizer=_spec("human", "human-a"),
        g1_tokenizer=_spec("g1", "g1-a"),
        semantic=SemanticCondition("wait"),
    )
    buffer.append_observation(human_token_id=2, g1_token_id=3)

    # When: a caller tries to mutate the snapshot.
    context = buffer.student_context()

    # Then: NumPy refuses the write.
    with pytest.raises(ValueError, match="read-only"):
        context.human_prefix.token_ids[0, 0] = np.int64(9)
