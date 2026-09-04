from __future__ import annotations

# pyright: reportAny=false, reportUnknownMemberType=false
import numpy as np
import torch

from hri_ttr.evaluation import (
    codebook_statistics,
    evaluate_g1_reconstruction,
    evaluate_g1_tokenizer_causality,
    evaluate_human_reconstruction,
    evaluate_human_tokenizer_causality,
)
from hri_ttr.representations.g1 import G1_FEATURE_SLICES
from hri_ttr.tokenizers import G1Tokenizer, HumanTokenizer, TokenizerArchitecture


def test_codebook_statistics_excludes_masked_tokens() -> None:
    # Given
    tokens = np.array([[0, 1, 255], [1, 9, 9]], dtype=np.int64)
    mask = np.array([[True, True, True], [True, False, False]])

    # When
    result = codebook_statistics(tokens, mask)

    # Then
    assert result.used_code_count == 3
    assert result.dead_code_ratio == 253 / 256
    assert result.histogram[1] == 2
    assert result.perplexity == 2.82842712474619


def test_g1_metrics_use_only_valid_frames() -> None:
    # Given
    target = np.zeros((3, 75), dtype=np.float64)
    prediction = target.copy()
    prediction[0, G1_FEATURE_SLICES["dof_pos_rad"].start] = 0.1
    prediction[2] = 100.0
    target[:, (3, 7)] = 1.0
    prediction[:, (3, 7)] = 1.0
    mask = np.array([True, True, False])

    # When
    result = evaluate_g1_reconstruction(target, prediction, mask)

    # Then
    assert result.per_joint_mae_rad[0] == 0.05
    assert result.worst_joint_index == 0
    assert result.root_position_ade_m == 0.0


def test_human_metrics_report_mpjpe_and_contacts() -> None:
    # Given
    target = np.zeros((2, 262), dtype=np.float64)
    prediction = target.copy()
    prediction[0, 3:6] = 1.0
    rotations = target[:, 132:258].reshape(2, 21, 6)
    predicted_rotations = prediction[:, 132:258].reshape(2, 21, 6)
    rotations[:, :, (0, 4)] = 1.0
    predicted_rotations[:, :, (0, 4)] = 1.0
    target[:, 258:] = 1.0
    prediction[:, 258:] = 1.0

    # When
    result = evaluate_human_reconstruction(target, prediction, np.array([True, True]))

    # Then
    assert result.mpjpe_m == np.sqrt(3.0) / 44.0
    assert result.contact.f1 == 1.0


def test_causality_diagnostic_reports_exact_prefix_stability() -> None:
    # Given
    _ = torch.manual_seed(7)
    tokenizer = G1Tokenizer(
        TokenizerArchitecture(width=16, code_dim=8, residual_depth=1)
    ).eval()
    features = torch.randn(1, 12, 75)
    mask = torch.ones((1, 12), dtype=torch.bool)

    # When
    result = evaluate_g1_tokenizer_causality(tokenizer, features, mask)

    # Then
    assert result.changed_token_count == 0
    assert result.max_latent_difference <= 1e-6
    assert result.max_decoded_difference <= 1e-6
    assert result.future_perturbation_changed_token_count == 0
    assert result.future_perturbation_max_latent_difference <= 1e-6
    assert result.future_perturbation_max_decoded_difference <= 1e-6


def test_human_causality_api_uses_independent_human_tokenizer() -> None:
    # Given
    tokenizer = HumanTokenizer(
        TokenizerArchitecture(width=16, code_dim=8, residual_depth=1)
    ).eval()
    features = torch.zeros((1, 8, 262))
    mask = torch.ones((1, 8), dtype=torch.bool)

    # When
    result = evaluate_human_tokenizer_causality(tokenizer, features, mask)

    # Then
    assert result.changed_token_count == 0
    assert result.future_perturbation_changed_token_count == 0


def test_causality_diagnostic_restores_training_mode_without_ema_mutation() -> None:
    # Given
    tokenizer = G1Tokenizer(
        TokenizerArchitecture(width=16, code_dim=8, residual_depth=1)
    ).train()
    features = torch.zeros((1, 8, 75))
    mask = torch.ones((1, 8), dtype=torch.bool)
    codebook = tokenizer.quantizer.codebook.clone()
    ema_sum = tokenizer.quantizer.ema_sum.clone()
    ema_count = tokenizer.quantizer.ema_count.clone()
    initialized = tokenizer.quantizer.initialized.clone()

    # When
    _ = evaluate_g1_tokenizer_causality(tokenizer, features, mask)

    # Then
    assert tokenizer.training
    assert torch.equal(tokenizer.quantizer.codebook, codebook)
    assert torch.equal(tokenizer.quantizer.ema_sum, ema_sum)
    assert torch.equal(tokenizer.quantizer.ema_count, ema_count)
    assert torch.equal(tokenizer.quantizer.initialized, initialized)


def test_causality_diagnostic_restores_mixed_submodule_modes() -> None:
    # Given
    tokenizer = G1Tokenizer(
        TokenizerArchitecture(width=16, code_dim=8, residual_depth=1)
    ).train()
    _ = tokenizer.encoder.eval()
    features = torch.zeros((1, 8, 75))
    mask = torch.ones((1, 8), dtype=torch.bool)
    modes_before = tuple(module.training for module in tokenizer.modules())

    # When
    _ = evaluate_g1_tokenizer_causality(tokenizer, features, mask)

    # Then
    assert tuple(module.training for module in tokenizer.modules()) == modes_before
