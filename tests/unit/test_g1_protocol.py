from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from hri_ttr.representations.g1.constants import (
    G1_DOF_AXES,
    G1_DOF_LIMITS_RAD,
    G1_DOF_NAMES,
)
from hri_ttr.representations.g1.normalizer import (
    G1FeatureNormalizer,
    NormalizerSchemaError,
)
from hri_ttr.representations.g1.schema import (
    G1_FEATURE_DIM,
    G1_FEATURE_FIELDS,
    G1_FEATURE_SLICES,
    G1_SCHEMA_VERSION,
)
from hri_ttr.representations.g1.tail import pad_frames_to_token_multiple


def test_g1_protocol_has_verified_29dof_order_and_75d_layout() -> None:
    # Given: the immutable G1 actuator and feature protocols.
    # When: their dimensions and boundary names are inspected.
    # Then: they match the verified 29-DoF and agreed 75D contracts.
    assert len(G1_DOF_NAMES) == len(G1_DOF_AXES) == len(G1_DOF_LIMITS_RAD) == 29
    assert G1_DOF_NAMES[0] == "left_hip_pitch_joint"
    assert G1_DOF_NAMES[-1] == "right_wrist_yaw_joint"
    assert G1_FEATURE_DIM == 75
    assert tuple(field.width for field in G1_FEATURE_FIELDS) == (3, 6, 29, 3, 3, 29, 2)
    assert G1_FEATURE_SLICES["foot_contact_lr"] == slice(73, 75)


def test_normalizer_excludes_padding_and_passes_contacts_through() -> None:
    # Given: two valid frames and one extreme padded frame.
    features = np.zeros((3, G1_FEATURE_DIM), dtype=np.float64)
    features[0, 0] = 1.0
    features[1, 0] = 3.0
    features[2, 0] = 1_000.0
    features[:, G1_FEATURE_SLICES["foot_contact_lr"]] = (
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    )
    mask = np.array([True, True, False], dtype=np.bool_)

    # When: the normalizer is fitted with the validity mask.
    normalizer = G1FeatureNormalizer.fit(
        features,
        mask,
        source_dataset="fixture",
        fps=20.0,
    )
    normalized = normalizer.normalize(features[:2])

    # Then: padding is ignored and contacts remain binary/unscaled.
    assert normalizer.mean[0] == 2.0
    assert normalizer.std[1] == 1e-4
    np.testing.assert_array_equal(
        normalized[:, G1_FEATURE_SLICES["foot_contact_lr"]],
        features[:2, G1_FEATURE_SLICES["foot_contact_lr"]],
    )


def test_normalizer_rejects_stale_schema(tmp_path: Path) -> None:
    # Given: a saved normalizer whose schema metadata is then made stale.
    path = tmp_path / "normalizer.json"
    values = np.zeros((2, G1_FEATURE_DIM), dtype=np.float64)
    mask = np.ones(2, dtype=np.bool_)
    normalizer = G1FeatureNormalizer.fit(
        values, mask, source_dataset="fixture", fps=20.0
    )
    normalizer.save(path)
    stale = path.read_text(encoding="utf-8").replace(
        G1_SCHEMA_VERSION,
        "g1_canonical_73d_v1",
    )
    _ = path.write_text(stale, encoding="utf-8")

    # When/Then: loading cannot silently accept a 73D-era schema.
    with pytest.raises(NormalizerSchemaError):
        _ = G1FeatureNormalizer.load(path)


def test_tail_padding_repeats_final_frame_and_marks_it_invalid() -> None:
    # Given: five frames for a four-frame token cadence.
    frames = np.arange(5 * G1_FEATURE_DIM, dtype=np.float32).reshape(5, G1_FEATURE_DIM)

    # When: the sequence is padded to the token multiple.
    padded = pad_frames_to_token_multiple(frames, frames_per_token=4)

    # Then: the last real frame is repeated and excluded by the mask.
    assert padded.features.shape == (8, G1_FEATURE_DIM)
    expected_tail = np.empty((3, G1_FEATURE_DIM), dtype=np.float32)
    expected_tail[:] = frames[-1]
    np.testing.assert_array_equal(padded.features[5:], expected_tail)
    np.testing.assert_array_equal(
        padded.valid_mask,
        np.array([True, True, True, True, True, False, False, False]),
    )
