from __future__ import annotations

import pytest
import torch

from hri_ttr.tokenizers import G1Tokenizer, TokenizerArchitecture
from hri_ttr.tokenizers.common import Encoding, StaleStreamStateError


def _model() -> G1Tokenizer:
    return G1Tokenizer(
        TokenizerArchitecture(width=16, code_dim=8, residual_depth=1)
    ).eval()


def test_prefix_and_streaming_when_input_is_aligned_then_match_batch() -> None:
    # Given
    model = _model()
    features = torch.randn(1, 12, 75)
    mask = torch.ones(1, 12, dtype=torch.bool)
    batch = model.encode(features, mask)

    # When
    prefixes = [model.encode(features[:, :end], mask[:, :end]) for end in (4, 8, 12)]
    state = model.new_stream(batch_size=1)
    chunks: list[Encoding] = []
    for start, end in ((0, 5), (5, 8), (8, 12)):
        output, state = model.stream_encode(
            features[:, start:end],
            mask[:, start:end],
            state,
        )
        chunks.append(output)
    stream_ids = torch.cat([chunk.token_ids for chunk in chunks], dim=1)
    stream_latents = torch.cat([chunk.latents for chunk in chunks], dim=1)

    # Then
    for token_count, prefix in enumerate(prefixes, start=1):
        assert torch.equal(prefix.token_ids, batch.token_ids[:, :token_count])
        torch.testing.assert_close(
            prefix.latents,
            batch.latents[:, :token_count],
            atol=1e-6,
            rtol=0.0,
        )
    assert torch.equal(stream_ids, batch.token_ids)
    torch.testing.assert_close(stream_latents, batch.latents, atol=1e-6, rtol=0.0)


def test_stream_when_state_is_reused_then_rejects_stale_branch() -> None:
    # Given
    model = _model()
    state = model.new_stream(batch_size=1)
    frames = torch.randn(1, 4, 75)
    mask = torch.ones(1, 4, dtype=torch.bool)
    _, next_state = model.stream_encode(frames, mask, state)

    # When
    # Then
    with pytest.raises(StaleStreamStateError):
        _ = model.stream_encode(frames, mask, state)
    _ = model.reset_stream(next_state)


def test_stream_when_reset_after_interruption_then_accepts_new_sequence() -> None:
    # Given
    model = _model()
    old_state = model.new_stream(batch_size=1)
    frames = torch.randn(1, 3, 75)
    mask = torch.ones(1, 3, dtype=torch.bool)
    _, active_state = model.stream_encode(frames, mask, old_state)

    # When
    reset_state = model.reset_stream(active_state)
    output, _ = model.stream_encode(
        torch.randn(1, 4, 75),
        torch.ones(1, 4, dtype=torch.bool),
        reset_state,
    )

    # Then
    assert output.token_ids.shape == (1, 1)


def test_partial_tail_when_streamed_then_matches_batch_assignment() -> None:
    # Given
    model = _model()
    features = torch.randn(1, 8, 75)
    mask = torch.tensor([[True, True, True, True, True, False, False, False]])
    batch = model.encode(features, mask)
    state = model.new_stream(batch_size=1)

    # When
    first, state = model.stream_encode(features[:, :5], mask[:, :5], state)
    second, _ = model.stream_encode(features[:, 5:], mask[:, 5:], state)
    streamed_ids = torch.cat((first.token_ids, second.token_ids), dim=1)
    streamed_mask = torch.cat((first.token_mask, second.token_mask), dim=1)

    # Then
    assert torch.equal(streamed_ids, batch.token_ids)
    assert torch.equal(streamed_mask, torch.tensor([[True, False]]))
    assert bool(torch.all((streamed_ids >= 0) & (streamed_ids <= 255)).item())
