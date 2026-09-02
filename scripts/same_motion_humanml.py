"""HumanML3D-to-Ember source and crop matching."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from same_motion_builders import motion_signature, texts_from_labels

from hri_ttr.data.same_motion_manifest import (
    aligned_clip_bounds,
    canonical_amass_key,
    choose_group_split,
    ember_key,
)

Json = dict[str, Any]


def _load_index(path: Path) -> dict[str, Json]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {Path(row["new_name"]).stem: row for row in csv.DictReader(stream)}


def _failure(sequence_id: str, row: Json, reason: str) -> Json:
    return {"sequence_id": sequence_id, "source": row["feat_p"], "reason": reason}


def _source_lengths(root: Path, row: Json, g1_path: Path) -> tuple[int, int]:
    relative = str(row["feat_p"]).replace("\\", "/").removeprefix("./")
    relative = relative.removeprefix("pose_data/")
    with np.load(root / "smplx" / "pose_data" / relative, mmap_mode="r") as data:
        human_frames = len(data["joints"])
    with np.load(g1_path, mmap_mode="r") as data:
        g1_frames = len(data["dof_positions"])
    return human_frames, g1_frames


def build_humanml(
    root: Path, index_path: Path, ember_root: Path
) -> tuple[list[Json], list[Json]]:
    """Map HumanML3D clips to new Ember G1 files and exact time ranges."""
    index = _load_index(index_path)
    ember: dict[tuple[str, str], list[tuple[Path, int]]] = defaultdict(list)
    for path in ember_root.rglob("*.npz"):
        parsed = ember_key(path, ember_root)
        if parsed is not None:
            key, source_fps = parsed
            ember[key].append((path, source_fps))
    rows: list[tuple[str, Path, Json]] = []
    memberships: dict[tuple[str, str], set[str]] = defaultdict(set)
    for split in ("train", "val", "test"):
        path = root / "smplx" / "seq_data" / f"{split}.pkl"
        for row in joblib.load(path):
            source_key = canonical_amass_key(str(row["feat_p"]))
            memberships[source_key].add(split)
            rows.append((split, path, row))
    records: list[Json] = []
    failures: list[Json] = []
    for original_split, human_path, row in rows:
        sequence_id = str(row["seq_name"])
        crop = index.get(sequence_id)
        source_key = canonical_amass_key(str(row["feat_p"]))
        candidates = ember.get(source_key, [])
        if crop is None or len(candidates) != 1:
            reason = (
                "no official crop"
                if crop is None
                else ("no Ember source" if not candidates else "ambiguous Ember source")
            )
            failures.append(_failure(sequence_id, row, reason))
            continue
        g1_path, source_fps = candidates[0]
        human_source_frames, g1_frames = _source_lengths(root, row, g1_path)
        bounds = aligned_clip_bounds(
            start_20hz=int(float(crop["start_frame"])),
            end_20hz=int(float(crop["end_frame"])),
            human_source_frames=human_source_frames,
            g1_source_frames=g1_frames,
        )
        human_motion = row["motion"]
        human_frames = len(human_motion["joints"])
        if bounds.start < 0 or bounds.end > g1_frames or bounds.end <= bounds.start:
            failures.append(_failure(sequence_id, row, "Ember crop out of bounds"))
            continue
        texts = texts_from_labels(
            row.get("frame_labels", []), "single_person", "frame_labels"
        )
        records.append(
            {
                "schema_version": 1,
                "sample_id": f"humanml3d:{sequence_id}",
                "source_dataset": "HumanML3D",
                "source_sequence_id": sequence_id,
                "source_group_id": f"amass:{source_key[0]}/{source_key[1]}",
                "augmentation": "clip",
                "split_original": original_split,
                "split": choose_group_split(memberships[source_key]),
                "duration_sec": human_frames / 10.0,
                "human": {
                    "path": str(human_path),
                    "format": "pickle_rows",
                    "locator": sequence_id,
                    "fps": 10.0,
                    "frame_start": 0,
                    "frame_end": human_frames,
                    "fields": ["poses", "trans", "joints"],
                },
                "g1": {
                    "path": str(g1_path),
                    "format": "npz",
                    "locator": None,
                    "fps": bounds.effective_fps,
                    "fps_file_value": 30.0,
                    "source_fps": source_fps,
                    "human_source_frames": human_source_frames,
                    "g1_source_frames": g1_frames,
                    "frame_start": bounds.start,
                    "frame_end": bounds.end,
                    "fields": ["body_positions", "body_rotations", "dof_positions"],
                },
                "texts": texts,
                "has_text": bool(texts),
                "human_signature": motion_signature(human_motion["joints"]),
                "same_motion_evidence": (
                    "official HumanML crop mapped to original AMASS and Ember retarget"
                ),
            }
        )
    return records, failures
