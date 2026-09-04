from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from hri_ttr.data.padding import pad_frames_to_multiple
from hri_ttr.data.pairs import (
    PairAlignmentError,
    PickleTrustPolicy,
    discover_pairs,
    load_retargeted_pair,
)
from hri_ttr.data.splits import DatasetSplits, SplitSource, read_fixed_splits


def _actor(
    frames: int = 3,
    *,
    root_pos: np.ndarray | None = None,
) -> dict[str, float | str | np.ndarray]:
    root_rot = np.empty((frames, 4), dtype=np.float32)
    root_rot[:] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return {
        "fps": 50.0,
        "root_pos": (
            np.zeros((frames, 3), dtype=np.float32) if root_pos is None else root_pos
        ),
        "root_rot": root_rot,
        "pose_body": np.zeros((frames, 69), dtype=np.float32),
        "joints_pos": np.zeros((frames, 24, 3), dtype=np.float32),
        "betas": np.zeros(10, dtype=np.float32),
        "gender": "neutral",
    }


def _reactor(frames: int = 3) -> dict[str, float | np.ndarray]:
    root_rot = np.empty((frames, 4), dtype=np.float32)
    root_rot[:] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return {
        "fps": 50.0,
        "root_pos": np.zeros((frames, 3), dtype=np.float32),
        "root_rot": root_rot,
        "dof_pos": np.zeros((frames, 29), dtype=np.float32),
        "joints_pos": np.zeros((frames, 20, 3), dtype=np.float32),
    }


def test_discovery_reports_unpaired_files(tmp_path: Path) -> None:
    # Given
    for name in (
        "G001T001A001R001_actor.pkl",
        "G001T001A001R001_reactor.pkl",
        "G002T001A001R001_actor.pkl",
    ):
        (tmp_path / name).touch()

    # When
    result = discover_pairs(tmp_path)

    # Then
    assert len(result.pairs) == 1
    assert result.actor_only == ("G002T001A001R001",)


def test_pair_load_rejects_unequal_timelines(tmp_path: Path) -> None:
    # Given
    sequence_id = "G001T001A001R001"
    actor_path = tmp_path / f"{sequence_id}_actor.pkl"
    reactor_path = tmp_path / f"{sequence_id}_reactor.pkl"
    with actor_path.open("wb") as stream:
        pickle.dump(_actor(3), stream)
    with reactor_path.open("wb") as stream:
        pickle.dump(_reactor(4), stream)
    paths = discover_pairs(tmp_path).pairs[0]

    # When / Then
    with pytest.raises(PairAlignmentError):
        _ = load_retargeted_pair(paths, pickle_policy=PickleTrustPolicy.TRUSTED_LOCAL)


def test_actor_boundary_rejects_nan(tmp_path: Path) -> None:
    # Given
    sequence_id = "G001T001A001R001"
    root_pos = np.zeros((3, 3), dtype=np.float32)
    root_pos[0, 0] = np.nan
    actor = _actor(root_pos=root_pos)
    with (tmp_path / f"{sequence_id}_actor.pkl").open("wb") as stream:
        pickle.dump(actor, stream)
    with (tmp_path / f"{sequence_id}_reactor.pkl").open("wb") as stream:
        pickle.dump(_reactor(), stream)

    # When / Then
    with pytest.raises(ValidationError):
        _ = load_retargeted_pair(
            discover_pairs(tmp_path).pairs[0],
            pickle_policy=PickleTrustPolicy.TRUSTED_LOCAL,
        )


def test_tail_padding_repeats_last_frame_and_masks_padding() -> None:
    # Given
    frames = np.arange(15, dtype=np.float32).reshape(5, 3)

    # When
    result = pad_frames_to_multiple(frames, 4)

    # Then
    assert result.frames.shape == (8, 3)
    expected_tail = np.array(
        [[12.0, 13.0, 14.0], [12.0, 13.0, 14.0], [12.0, 13.0, 14.0]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(result.frames[5:], expected_tail)
    np.testing.assert_array_equal(result.frame_mask, [True] * 5 + [False] * 3)


def test_fixed_split_reader_preserves_sequence_ids(tmp_path: Path) -> None:
    # Given
    split_path = tmp_path / "splits.json"
    _ = split_path.write_text(
        json.dumps(
            {
                "schema_id": "interx_human_g1_split",
                "schema_version": 1,
                "source": {
                    "dataset": "synthetic",
                    "sequence_count": 4,
                    "sequence_ids_sha256": "a" * 64,
                    "split_indices_sha256": "b" * 64,
                },
                "splits": {
                    "train": ["s2", "s1"],
                    "val": ["s3"],
                    "test": ["s4"],
                },
            }
        ),
        encoding="utf-8",
    )

    # When
    result = read_fixed_splits(split_path)

    # Then
    assert result == DatasetSplits(
        schema_id="interx_human_g1_split",
        schema_version=1,
        source=SplitSource(
            dataset="synthetic",
            sequence_count=4,
            sequence_ids_sha256="a" * 64,
            split_indices_sha256="b" * 64,
        ),
        train=("s2", "s1"),
        val=("s3",),
        test=("s4",),
    )


def test_committed_interx_split_has_frozen_membership_and_provenance() -> None:
    # Given
    split_path = (
        Path(__file__).parents[2] / "configs" / "data" / "interx_human_g1_split_v1.json"
    )

    # When
    result = read_fixed_splits(split_path)

    # Then
    assert (len(result.train), len(result.val), len(result.test)) == (6825, 374, 404)
    all_ids = set(result.train) | set(result.val) | set(result.test)
    assert len(all_ids) == 7603
    assert set(result.train).isdisjoint(result.val)
    assert set(result.train).isdisjoint(result.test)
    assert set(result.val).isdisjoint(result.test)
    assert "G001T000A000R000" in result.train
    assert "G001T000A002R012" in result.val
    assert "G001T000A001R009" in result.test
    assert result.schema_id == "interx_human_g1_split"
    assert result.schema_version == 1
    assert result.source.sequence_ids_sha256 == (
        "164367bfdbc5713bf9b05e4fda5c0a127b1dffe79c72f50dce8f2ebd5efb351c"
    )
    assert result.source.split_indices_sha256 == (
        "c54128b660768d096f928c3e693d966262df572f1f86fb28ea30bf86cf080a39"
    )
