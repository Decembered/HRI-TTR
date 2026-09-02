from __future__ import annotations

import pytest
import torch

from hri_ttr.tokenizers import G1Tokenizer, HumanTokenizer, TokenizerArchitecture
from hri_ttr.tokenizers.common import InvalidMotionTensorError, NoValidTokensError
from hri_ttr.tokenizers.common.quantizer import EMAVectorQuantizer


def _architecture() -> TokenizerArchitecture:
    return TokenizerArchitecture(width=16, code_dim=8, residual_depth=1)


@pytest.mark.parametrize(
    ("tokenizer_type", "feature_dim"), [(HumanTokenizer, 262), (G1Tokenizer, 75)]
)
def test_tokenizer_when_aligned_input_then_has_expected_shapes(
    tokenizer_type: type[HumanTokenizer | G1Tokenizer],
    feature_dim: int,
) -> None:
    # Given
    model = tokenizer_type(_architecture()).eval()
    features = torch.randn(2, 8, feature_dim)
    mask = torch.ones(2, 8, dtype=torch.bool)

    # When
    encoded = model.encode(features, mask)
    decoded = model.decode(encoded.token_ids, encoded.token_mask)

    # Then
    assert encoded.latents.shape == (2, 2, 8)
    assert encoded.token_ids.shape == (2, 2)
    assert decoded.shape == features.shape
    assert torch.all((encoded.token_ids >= 0) & (encoded.token_ids < 256))


def test_encoder_when_future_frames_change_then_past_is_unchanged() -> None:
    # Given
    model = G1Tokenizer(_architecture()).eval()
    original = torch.randn(1, 12, 75)
    changed = original.clone()
    changed[:, 8:] = torch.randn(1, 4, 75) * 100.0
    mask = torch.ones(1, 12, dtype=torch.bool)

    # When
    first = model.encode(original, mask)
    second = model.encode(changed, mask)

    # Then
    torch.testing.assert_close(
        first.latents[:, :2], second.latents[:, :2], atol=1e-6, rtol=0.0
    )
    assert torch.equal(first.token_ids[:, :2], second.token_ids[:, :2])


def test_decoder_when_future_tokens_are_appended_then_past_is_unchanged() -> None:
    # Given
    model = G1Tokenizer(_architecture()).eval()
    prefix = torch.tensor([[3, 9]], dtype=torch.long)
    complete = torch.tensor([[3, 9, 17]], dtype=torch.long)

    # When
    prefix_motion = model.decode(prefix)
    complete_motion = model.decode(complete)

    # Then
    torch.testing.assert_close(
        prefix_motion, complete_motion[:, :8], atol=1e-6, rtol=0.0
    )


def test_tail_mask_when_only_one_tail_frame_is_real_then_tail_token_is_invalid() -> (
    None
):
    # Given
    model = HumanTokenizer(_architecture()).eval()
    features = torch.randn(1, 8, 262)
    mask = torch.tensor([[True, True, True, True, True, False, False, False]])

    # When
    encoded = model.encode(features, mask)

    # Then
    assert torch.equal(encoded.token_mask, torch.tensor([[True, False]]))


def test_quantizer_when_mask_is_all_invalid_then_raises_typed_error() -> None:
    # Given
    model = G1Tokenizer(_architecture()).train()
    features = torch.randn(1, 4, 75)
    mask = torch.zeros(1, 4, dtype=torch.bool)

    # When
    # Then
    with pytest.raises(NoValidTokensError):
        _ = model.encode(features, mask)


def test_quantizer_assigns_partial_tail_without_counting_its_code() -> None:
    # Given
    quantizer = EMAVectorQuantizer(256, 2, decay=0.5, commitment_weight=1.0).train()
    with torch.no_grad():
        _ = quantizer.codebook.fill_(100.0)
        _ = quantizer.codebook[3].copy_(torch.tensor([0.0, 0.0]))
        _ = quantizer.codebook[7].copy_(torch.tensor([5.0, 5.0]))
        _ = quantizer.ema_sum.copy_(quantizer.codebook * 10.0)
        _ = quantizer.ema_count.fill_(10.0)
        _ = quantizer.initialized.fill_(value=True)
    latents = torch.tensor([[[0.0, 0.0], [5.0, 5.0]]])
    statistics_mask = torch.tensor([[True, False]])
    assignment_mask = torch.tensor([[True, True]])

    # When
    encoded = quantizer.forward(latents, statistics_mask, assignment_mask)

    # Then
    assert torch.equal(encoded.token_ids, torch.tensor([[3, 7]]))
    assert encoded.perplexity.item() == 1.0
    assert torch.equal(encoded.token_mask, torch.tensor([[True, False]]))
    assert quantizer.ema_count[3].item() == 5.5
    assert quantizer.ema_count[7].item() == 5.0


@pytest.mark.parametrize(
    ("features", "mask"),
    [
        (torch.randn(1, 4, 74), torch.ones(1, 4, dtype=torch.bool)),
        (torch.randn(1, 5, 75), torch.ones(1, 5, dtype=torch.bool)),
        (torch.randn(1, 4, 75), torch.ones(1, 3, dtype=torch.bool)),
        (torch.randn(1, 4, 75), torch.ones(1, 4)),
        (torch.ones(1, 4, 75, dtype=torch.long), torch.ones(1, 4, dtype=torch.bool)),
        (torch.full((1, 4, 75), torch.nan), torch.ones(1, 4, dtype=torch.bool)),
        (torch.empty(1, 0, 75), torch.empty(1, 0, dtype=torch.bool)),
    ],
)
def test_encode_when_tensor_contract_is_malformed_then_raises_typed_error(
    features: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    # Given
    model = G1Tokenizer(_architecture()).eval()

    # When
    # Then
    with pytest.raises(InvalidMotionTensorError):
        _ = model.encode(features, mask)


def test_models_when_both_created_then_codebooks_are_independent() -> None:
    # Given
    human = HumanTokenizer(_architecture())
    g1 = G1Tokenizer(_architecture())
    before = g1.quantizer.codebook.clone()

    # When
    with torch.no_grad():
        _ = human.quantizer.codebook.add_(10.0)

    # Then
    assert human.quantizer is not g1.quantizer
    assert human.encoder is not g1.encoder
    assert human.decoder is not g1.decoder
    assert torch.equal(g1.quantizer.codebook, before)


def test_ema_when_saved_then_all_state_is_serialized() -> None:
    # Given
    model = G1Tokenizer(_architecture())

    # When
    state_keys = set(model.state_dict())

    # Then
    assert {
        "quantizer.codebook",
        "quantizer.ema_sum",
        "quantizer.ema_count",
        "quantizer.initialized",
    } <= state_keys


def test_metadata_when_default_architecture_then_reports_causal_history() -> None:
    # Given
    model = G1Tokenizer()

    # When
    receptive_field = model.encoder_receptive_field_frames

    # Then
    assert receptive_field == 332
    assert model.stream_retains_full_prefix is True


def test_decode_when_masked_placeholder_is_outside_vocab_then_ignores_it() -> None:
    # Given
    model = G1Tokenizer(_architecture()).eval()
    token_ids = torch.tensor([[4, 999]], dtype=torch.long)
    token_mask = torch.tensor([[True, False]])

    # When
    decoded = model.decode(token_ids, token_mask)

    # Then
    assert decoded.shape == (1, 8, 75)
