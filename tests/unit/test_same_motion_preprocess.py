from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from hri_ttr.data.corpus import CorpusWindowDataset
from hri_ttr.data.g1_kinematics import G1Kinematics
from hri_ttr.data.same_motion_preprocess import (
    common_timeline,
    wxyz_to_xyzw,
)
from hri_ttr.data.same_motion_quality import (
    QualityError,
    aligned_quality_metrics,
    validate_g1_dof,
    validate_raw_motion,
)
from hri_ttr.representations.g1.constants import G1_DOF_NAMES
from hri_ttr.representations.human.normalizer import HumanFeatureNormalizer
from hri_ttr.training.data import WindowConfig

if TYPE_CHECKING:
    from pathlib import Path


def test_common_timeline_uses_shared_time_extent() -> None:
    human, g1, target = common_timeline(51, 50.0, 31, 30.0, 20.0)

    assert human[-1] == g1[-1] == target[-1] == 1.0
    assert len(target) == 21


def test_common_timeline_accounts_for_native_frame_quantization() -> None:
    _, _, target = common_timeline(100, 10.0, 301, 29.966442953020135, 20.0)

    assert target[-1] == pytest.approx(9.9)
    _, _, bone_target = common_timeline(1770, 50.0, 1063, 30.0, 20.0)
    assert bone_target[-1] == pytest.approx(35.35)


def test_human_stature_check_is_independent_of_world_orientation() -> None:
    parents = np.asarray(
        [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    )
    joints = np.zeros((2, 22, 3), dtype=np.float64)
    for joint in range(1, 22):
        joints[:, joint, 0] = joints[:, parents[joint], 0] + 0.2

    validate_raw_motion(
        joints,
        20.0,
        np.zeros((2, 3), dtype=np.float64),
        np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
        np.zeros((2, 29), dtype=np.float64),
    )


def test_wxyz_to_xyzw_reorders_components() -> None:
    source = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)

    np.testing.assert_array_equal(
        wxyz_to_xyzw(source), np.asarray([[2.0, 3.0, 4.0, 1.0]])
    )


def test_g1_joint_limit_gate_rejects_impossible_pose() -> None:
    values = np.zeros((2, 29), dtype=np.float64)
    values[0, 5] = 0.4

    with pytest.raises(QualityError, match="g1_dof_limit"):
        validate_g1_dof(values)


def test_root_angular_velocity_gate_rejects_rotation_jump() -> None:
    human = np.zeros((2, 22, 3), dtype=np.float64)
    root = np.zeros((2, 3), dtype=np.float64)
    rotation = np.asarray(
        [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float64
    )
    dof = np.zeros((2, 29), dtype=np.float64)
    feet = np.zeros((2, 2, 3), dtype=np.float64)

    with pytest.raises(QualityError, match="g1_root_angular_speed"):
        _ = aligned_quality_metrics(human, root, rotation, dof, feet)


def test_g1_foot_penetration_gate_rejects_deep_floor_crossing() -> None:
    human = np.zeros((2, 22, 3), dtype=np.float64)
    root = np.zeros((2, 3), dtype=np.float64)
    rotation = np.tile([0.0, 0.0, 0.0, 1.0], (2, 1))
    dof = np.zeros((2, 29), dtype=np.float64)
    feet = np.zeros((2, 2, 3), dtype=np.float64)
    feet[:, :, 1] = -0.1

    with pytest.raises(QualityError, match="g1_foot_penetration"):
        _ = aligned_quality_metrics(human, root, rotation, dof, feet)


def test_g1_kinematics_reads_standard_joint_order(tmp_path: Path) -> None:
    xml = tmp_path / "g1.xml"
    bodies = "".join(
        (
            f'<body name="{name.removesuffix("_joint")}_link" '
            f'pos="{position}"><joint name="{name}" axis="0 1 0"/></body>'
        )
        for name, position in zip(
            G1_DOF_NAMES,
            (
                "0 0.1 -0.6" if name == "left_ankle_roll_joint" else "0 0 0"
                for name in G1_DOF_NAMES
            ),
            strict=True,
        )
    )
    document = (
        f'<mujoco><worldbody><body name="pelvis"><freejoint/>{bodies}'
        "</body></worldbody></mujoco>"
    )
    _ = xml.write_text(document, encoding="utf-8")
    model = G1Kinematics.from_mjcf(xml)
    root = np.zeros((1, 3), dtype=np.float64)
    rotation = np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    dof = np.zeros((1, 29), dtype=np.float64)

    positions = model.body_positions(root, rotation, dof)

    np.testing.assert_allclose(
        positions["left_ankle_roll_link"],
        np.asarray([[0.0, 0.1, -0.6]], dtype=np.float64),
    )


def test_corpus_windows_are_mmap_backed_and_never_cross_sequences(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "shards" / "train" / "shard-00000"
    shard.mkdir(parents=True)
    human = np.arange(10 * 262, dtype=np.float32).reshape(10, 262)
    g1 = np.arange(10 * 75, dtype=np.float32).reshape(10, 75)
    np.save(shard / "human.npy", human)
    np.save(shard / "g1.npy", g1)
    np.save(shard / "offsets.npy", np.asarray([[0, 3], [3, 10]], dtype=np.int64))
    rows = [{"sample_id": "a"}, {"sample_id": "b"}]
    _ = (shard / "sequences.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    dataset = CorpusWindowDataset(
        tmp_path, "train", "human", WindowConfig(frames=4, stride=4)
    )

    assert len(dataset) == 3
    assert dataset[0].sequence_id == "a"
    assert dataset[0].frame_mask.tolist() == [True, True, True, False]
    assert dataset[1].sequence_id == "b"
    np.testing.assert_array_equal(dataset[1].features[:4], human[3:7])


def test_corpus_loader_applies_train_normalizer_before_padding(tmp_path: Path) -> None:
    shard = tmp_path / "shards" / "train" / "shard-00000"
    shard.mkdir(parents=True)
    human = np.full((2, 262), 5.0, dtype=np.float32)
    human[:, 258:] = 1.0
    np.save(shard / "human.npy", human)
    np.save(shard / "offsets.npy", np.asarray([[0, 2]], dtype=np.int64))
    _ = (shard / "sequences.jsonl").write_text(
        json.dumps({"sample_id": "normalized"}) + "\n", encoding="utf-8"
    )
    normalizer = HumanFeatureNormalizer.create(
        np.ones(262, dtype=np.float32),
        np.full(262, 2.0, dtype=np.float32),
        source_dataset="train",
        fps=20.0,
    )

    dataset = CorpusWindowDataset(
        tmp_path,
        "train",
        "human",
        WindowConfig(frames=4, stride=4),
        normalizer,
    )

    np.testing.assert_array_equal(dataset[0].features[:, :258], 2.0)
    np.testing.assert_array_equal(dataset[0].features[:, 258:], 1.0)
