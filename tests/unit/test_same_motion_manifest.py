from __future__ import annotations

from hri_ttr.data.same_motion_manifest import (
    aligned_clip_bounds,
    canonical_amass_key,
    choose_group_split,
    stable_split,
)


def test_canonical_amass_key_removes_humanml_prefix_only() -> None:
    assert canonical_amass_key("./pose_data/KIT/3/jump_left02_poses.npy") == (
        "kit/3",
        "jumpleft02poses",
    )


def test_ember_clip_bounds_uses_real_integer_stride() -> None:
    clip = aligned_clip_bounds(
        start_20hz=25,
        end_20hz=219,
        human_source_frames=219,
        g1_source_frames=364,
    )
    assert clip.start == 41
    assert clip.end == 364
    assert clip.effective_fps == 20 * 364 / 219


def test_group_split_prioritizes_held_out_membership() -> None:
    assert choose_group_split({"train", "test"}) == "test"
    assert choose_group_split({"train", "val"}) == "val"
    assert choose_group_split({"train"}) == "train"


def test_stable_split_is_repeatable() -> None:
    first = stable_split("bone_seed:walk_001")
    assert first == stable_split("bone_seed:walk_001")
    assert first in {"train", "val", "test"}
