from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch.nn import functional

from hri_ttr.representations.g1.schema import G1_SCHEMA_VERSION
from hri_ttr.representations.human.normalizer import SCHEMA_ID as HUMAN_SCHEMA_ID
from hri_ttr.tokenizers import G1Tokenizer, TokenizerArchitecture
from hri_ttr.training.data import (
    FeatureSequence,
    WindowConfig,
    build_windows,
    collate_windows,
)
from hri_ttr.training.losses import MaskedReconstructionLoss, ZeroValidFramesError


def test_build_windows_repeats_tail_and_masks_it() -> None:
    # Given
    values = np.arange(18, dtype=np.float32).reshape(6, 3)
    sequence = FeatureSequence(sequence_id="seq", features=values)

    # When
    windows = build_windows((sequence,), WindowConfig(frames=8, stride=8))

    # Then
    assert len(windows) == 1
    assert windows[0].frame_mask.tolist() == [True] * 6 + [False, False]
    assert windows[0].token_mask.tolist() == [True, False]
    expected = np.empty((8, 3), dtype=np.float32)
    expected[:6] = values
    expected[6:] = [15.0, 16.0, 17.0]
    np.testing.assert_array_equal(windows[0].features, expected)


def test_loss_rejects_zero_valid_frames() -> None:
    # Given
    loss = MaskedReconstructionLoss.for_schema(HUMAN_SCHEMA_ID)
    values = torch.zeros((1, 4, 262))
    mask = torch.zeros((1, 4), dtype=torch.bool)

    # When / Then
    with pytest.raises(ZeroValidFramesError):
        _ = loss.forward(values, values, mask)


def test_human_loss_matches_official_ttr_reconstruction_and_velocity() -> None:
    # Given
    target = torch.zeros((1, 3, 262))
    prediction = target.clone()
    prediction[:, :, 0] = torch.tensor([0.0, 1.0, 3.0])
    prediction[:, :, 100] = 2.0
    mask = torch.ones((1, 3), dtype=torch.bool)

    # When
    actual = MaskedReconstructionLoss.for_schema(HUMAN_SCHEMA_ID).forward(
        prediction, target, mask
    )

    # Then
    reconstruction = functional.smooth_l1_loss(prediction, target)
    position_velocity = functional.smooth_l1_loss(
        prediction[:, 1:, :66] - prediction[:, :-1, :66],
        target[:, 1:, :66] - target[:, :-1, :66],
    )
    expected = reconstruction + 0.5 * position_velocity
    torch.testing.assert_close(actual, expected)


def test_human_velocity_loss_excludes_pairs_touching_padding() -> None:
    # Given
    target = torch.zeros((1, 4, 262))
    prediction = target.clone()
    prediction[:, 2:, :66] = 100.0
    mask = torch.tensor([[True, True, False, False]])

    # When
    actual = MaskedReconstructionLoss.for_schema(HUMAN_SCHEMA_ID).forward(
        prediction, target, mask
    )

    # Then
    assert float(actual.item()) == 0.0


def test_g1_loss_uses_grouped_smooth_l1_and_position_deltas() -> None:
    # Given
    target = torch.zeros((1, 3, 75))
    prediction = target.clone()
    prediction[:, :, 0] = torch.tensor([0.0, 1.0, 3.0])
    prediction[:, :, 3] = 2.0
    prediction[:, :, 9] = torch.tensor([0.0, 2.0, 5.0])
    prediction[:, :, 40] = 2.0
    prediction[:, :, 50] = 2.0
    prediction[:, :, 73] = 2.0
    mask = torch.ones((1, 3), dtype=torch.bool)

    # When
    actual = MaskedReconstructionLoss.for_schema(G1_SCHEMA_VERSION).forward(
        prediction, target, mask
    )

    # Then
    groups = (
        (0, 3, 2.0),
        (3, 9, 2.0),
        (9, 38, 1.0),
        (38, 44, 1.0),
        (44, 73, 0.5),
        (73, 75, 1.0),
    )
    grouped = (
        sum(
            functional.smooth_l1_loss(
                prediction[:, :, start:stop], target[:, :, start:stop]
            )
            * weight
            for start, stop, weight in groups
        )
        / 7.5
    )
    root_delta = functional.smooth_l1_loss(
        prediction[:, 1:, :3] - prediction[:, :-1, :3],
        target[:, 1:, :3] - target[:, :-1, :3],
    )
    dof_delta = functional.smooth_l1_loss(
        prediction[:, 1:, 9:38] - prediction[:, :-1, 9:38],
        target[:, 1:, 9:38] - target[:, :-1, 9:38],
    )
    expected = grouped + 0.5 * (root_delta + dof_delta) / 2.0
    torch.testing.assert_close(actual, expected)


def test_authoritative_padding_mask_excludes_loss_and_partial_token_ema() -> None:
    # Given
    features = np.arange(8 * 75, dtype=np.float32).reshape(8, 75) / 1000.0
    features[5:] = features[4]
    mask = np.asarray([True] * 5 + [False] * 3, dtype=np.bool_)
    sequence = FeatureSequence("padded", features, mask)
    window = build_windows((sequence,), WindowConfig(frames=8, stride=8))[0]
    batch = collate_windows([window])
    architecture = TokenizerArchitecture(width=8, code_dim=4, residual_depth=1)
    masked_model = G1Tokenizer(architecture)
    reference_model = copy.deepcopy(masked_model)
    prediction = batch.features.clone()
    prediction[:, 5:] += 1000.0

    # When
    loss = MaskedReconstructionLoss.for_schema(G1_SCHEMA_VERSION).forward(
        prediction, batch.features, batch.frame_mask
    )
    _ = masked_model.forward(batch.features, batch.frame_mask)
    _ = reference_model.forward(
        batch.features[:, :4], torch.ones((1, 4), dtype=torch.bool)
    )

    # Then
    assert float(loss.item()) == 0.0
    assert torch.equal(
        batch.frame_mask,
        torch.tensor([[True] * 5 + [False] * 3], dtype=torch.bool),
    )
    assert torch.equal(
        batch.token_mask, torch.tensor([[True, False]], dtype=torch.bool)
    )
    torch.testing.assert_close(
        masked_model.quantizer.ema_count, reference_model.quantizer.ema_count
    )
    torch.testing.assert_close(
        masked_model.quantizer.ema_sum, reference_model.quantizer.ema_sum
    )
