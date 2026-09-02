"""Source-specific readers for the same-motion dataset manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

from hri_ttr.data.same_motion_manifest import (
    aligned_clip_bounds,
    canonical_amass_key,
    choose_group_split,
    ember_key,
    stable_split,
)

Json = dict[str, Any]


def motion_signature(joints: NDArray[np.floating[Any]]) -> str:
    """Hash a translation-invariant, bounded sample of a Human joint track."""
    values = np.asarray(joints, dtype=np.float32)
    indexes = np.linspace(0, len(values) - 1, min(64, len(values))).round().astype(int)
    sampled = values[indexes] - values[indexes, :1]
    quantized = np.rint(sampled * 1000).astype(np.int32)
    return hashlib.sha256(quantized.tobytes()).hexdigest()


def _texts(labels: Iterable[object], kind: str, source_field: str) -> list[Json]:
    output: list[Json] = []
    for label in labels:
        if isinstance(label, str):
            text = label.strip()
            item: Json = {"text": text, "kind": kind, "source_field": source_field}
        elif isinstance(label, dict):
            text = str(label.get("proc_label", "")).strip()
            item = {"text": text, "kind": kind, "source_field": source_field}
            for key in ("start_t", "end_t"):
                if key in label:
                    item[key] = float(label[key])
        else:
            continue
        if text:
            output.append(item)
    return output


def load_bone_texts(
    database: Path,
) -> tuple[dict[str, list[Json]], dict[str, list[Json]]]:
    """Read Bone-Seed sequence and temporal annotations without editing SQLite."""
    descriptions: dict[str, list[Json]] = defaultdict(list)
    events: dict[str, list[Json]] = defaultdict(list)
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    columns = (
        "natural_desc_1",
        "natural_desc_2",
        "natural_desc_3",
        "natural_desc_4",
        "technical_description",
        "short_description_1",
        "short_description_2",
    )
    query = (
        "select filename,natural_desc_1,natural_desc_2,natural_desc_3,"
        "natural_desc_4,technical_description,short_description_1,"
        "short_description_2 from motions"
    )
    for row in connection.execute(query):
        for field, value in zip(columns, row[1:], strict=True):
            if value and str(value).strip():
                descriptions[str(row[0])].append(
                    {
                        "text": str(value).strip(),
                        "kind": "single_person",
                        "source_field": field,
                    }
                )
    for filename, _, start, end, description in connection.execute(
        "select filename,event_index,start_sec,end_sec,description from events"
    ):
        if description and str(description).strip():
            events[str(filename)].append(
                {
                    "text": str(description).strip(),
                    "kind": "single_person_event",
                    "source_field": "events.description",
                    "start_t": float(start),
                    "end_t": float(end),
                }
            )
    connection.close()
    return dict(descriptions), dict(events)


def build_bone_record(
    human_path: Path,
    g1_path: Path,
    descriptions: dict[str, list[Json]],
    events: dict[str, list[Json]],
) -> tuple[Json | None, Json | None]:
    """Validate one basename-linked Bone-Seed Human/G1 pair."""
    sequence_id = human_path.stem
    try:
        human = joblib.load(human_path)
        g1_outer = joblib.load(g1_path)
        g1 = (
            g1_outer.get(sequence_id, g1_outer)
            if isinstance(g1_outer, dict)
            else g1_outer
        )
        human_frames = len(human["smpl_joints"])
        g1_frames = len(g1["dof"])
        human_fps = float(human["fps"])
        g1_fps = float(g1["fps"])
        duration = min(human_frames / human_fps, g1_frames / g1_fps)
        delta = abs(human_frames / human_fps - g1_frames / g1_fps)
    except (KeyError, OSError, TypeError, ValueError) as error:
        return None, {"sequence_id": sequence_id, "reason": str(error)}
    if delta > max(1 / human_fps, 1 / g1_fps) + 1e-6:
        return None, {
            "sequence_id": sequence_id,
            "reason": f"duration delta {delta:.6f}s",
        }
    if np.asarray(human["pose_aa"]).shape != (human_frames, 72):
        return None, {"sequence_id": sequence_id, "reason": "invalid Human pose"}
    if np.asarray(g1["dof"]).shape != (g1_frames, 29):
        return None, {"sequence_id": sequence_id, "reason": "invalid G1 dof"}
    text = descriptions.get(sequence_id, []) + events.get(sequence_id, [])
    group = re.sub(r"_M$", "", sequence_id)
    return {
        "schema_version": 1,
        "sample_id": f"bone_seed:{sequence_id}",
        "source_dataset": "Bone-Seed",
        "source_sequence_id": sequence_id,
        "source_group_id": f"bone_seed:{group}",
        "augmentation": "mirror" if sequence_id.endswith("_M") else "original",
        "split_original": None,
        "split": stable_split(f"bone_seed:{group}"),
        "duration_sec": duration,
        "human": {
            "path": str(human_path),
            "format": "joblib_pickle",
            "locator": sequence_id,
            "fps": human_fps,
            "frame_start": 0,
            "frame_end": human_frames,
            "fields": ["pose_aa", "transl", "smpl_joints"],
        },
        "g1": {
            "path": str(g1_path),
            "format": "joblib_pickle",
            "locator": sequence_id,
            "fps": g1_fps,
            "frame_start": 0,
            "frame_end": g1_frames,
            "fields": ["root_trans_offset", "root_rot", "dof", "pose_aa"],
        },
        "texts": text,
        "has_text": bool(text),
        "human_signature": motion_signature(human["smpl_joints"]),
        "same_motion_evidence": "paired release basename and duration",
    }, None


def _person_key(sequence_name: str) -> tuple[str, str]:
    base, person = sequence_name.rsplit("_P", 1)
    return base, f"P{person}"


def build_interx(root: Path) -> tuple[list[Json], list[Json]]:
    """Join filtered Inter-X G1 rows back to the matching Human person."""
    human: dict[tuple[str, str], tuple[str, Path, Json]] = {}
    for split in ("train", "val", "test"):
        path = root / "smplx" / "seq_data" / f"{split}.pkl"
        for row in joblib.load(path):
            human[_person_key(str(row["seq_name"]))] = (split, path, row)
    g1: dict[tuple[str, str], tuple[Path, Json]] = {}
    pattern = re.compile(r"^(.*)_(P[12])_P1$")
    for container_split in ("train", "val", "test"):
        path = root / "g1_filtered" / "seq_data" / f"{container_split}.pkl"
        for row in joblib.load(path):
            match = pattern.fullmatch(str(row["seq_name"]))
            if match is not None:
                g1[(match.group(1), match.group(2))] = (path, row)
    records: list[Json] = []
    failures: list[Json] = []
    for key in sorted(human.keys() & g1.keys()):
        split, human_path, human_row = human[key]
        g1_path, g1_row = g1[key]
        human_motion = human_row["motion"]
        g1_motion = g1_row["motion"]
        human_frames = len(human_motion["joints"])
        g1_frames = len(g1_motion["dof_pos"])
        fps = float(g1_motion.get("fps", 50))
        if human_frames != g1_frames:
            failures.append({"sequence_id": key, "reason": "frame count mismatch"})
            continue
        texts = _texts(g1_row.get("frame_labels", []), "single_person", "frame_labels")
        texts += _texts(
            g1_row.get("multi_person_texts", []), "interaction", "multi_person_texts"
        )
        base, person = key
        records.append(
            {
                "schema_version": 1,
                "sample_id": f"interx:{base}:{person}",
                "source_dataset": "Inter-X",
                "source_sequence_id": f"{base}_{person}",
                "source_group_id": f"interx:{base}",
                "augmentation": "original",
                "split_original": split,
                "split": split,
                "duration_sec": human_frames / fps,
                "human": {
                    "path": str(human_path),
                    "format": "pickle_rows",
                    "locator": str(human_row["seq_name"]),
                    "fps": fps,
                    "frame_start": 0,
                    "frame_end": human_frames,
                    "fields": ["poses", "trans", "joints"],
                },
                "g1": {
                    "path": str(g1_path),
                    "format": "pickle_rows",
                    "locator": str(g1_row["seq_name"]),
                    "fps": fps,
                    "frame_start": 0,
                    "frame_end": g1_frames,
                    "fields": ["root_pos", "root_rot", "dof_pos"],
                },
                "texts": texts,
                "has_text": bool(texts),
                "human_signature": motion_signature(human_motion["joints"]),
                "same_motion_evidence": "retarget source person id and equal timeline",
            }
        )
    failures.extend(
        {"sequence_id": f"{key[0]}_{key[1]}", "reason": "no filtered G1"}
        for key in sorted(human.keys() - g1.keys())
    )
    return records, failures


def load_humanml_index(path: Path) -> dict[str, Json]:
    """Load the official HumanML3D crop table."""
    with path.open(newline="", encoding="utf-8") as stream:
        return {Path(row["new_name"]).stem: row for row in csv.DictReader(stream)}


def build_humanml(
    root: Path, index_path: Path, ember_root: Path
) -> tuple[list[Json], list[Json]]:
    """Map HumanML3D clips to new Ember G1 files and exact time ranges."""
    index = load_humanml_index(index_path)
    ember: dict[tuple[str, str], list[tuple[Path, int]]] = defaultdict(list)
    for path in ember_root.rglob("*.npz"):
        parsed = ember_key(path, ember_root)
        if parsed is not None:
            key, source_fps = parsed
            ember[key].append((path, source_fps))
    rows: list[tuple[str, Path, Json]] = []
    source_memberships: dict[tuple[str, str], set[str]] = defaultdict(set)
    for split in ("train", "val", "test"):
        path = root / "smplx" / "seq_data" / f"{split}.pkl"
        for row in joblib.load(path):
            source_key = canonical_amass_key(str(row["feat_p"]))
            source_memberships[source_key].add(split)
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
            failures.append(
                {"sequence_id": sequence_id, "source": row["feat_p"], "reason": reason}
            )
            continue
        g1_path, source_fps = candidates[0]
        relative_human_source = str(row["feat_p"]).replace("\\", "/")
        relative_human_source = relative_human_source.removeprefix("./")
        relative_human_source = relative_human_source.removeprefix("pose_data/")
        full_human_path = root / "smplx" / "pose_data" / relative_human_source
        with np.load(full_human_path, mmap_mode="r") as human_source:
            human_source_frames = len(human_source["joints"])
        with np.load(g1_path, mmap_mode="r") as g1_data:
            g1_frames = len(g1_data["dof_positions"])
        bounds = aligned_clip_bounds(
            start_20hz=int(float(crop["start_frame"])),
            end_20hz=int(float(crop["end_frame"])),
            human_source_frames=human_source_frames,
            g1_source_frames=g1_frames,
        )
        human_motion = row["motion"]
        human_frames = len(human_motion["joints"])
        if bounds.start < 0 or bounds.end > g1_frames or bounds.end <= bounds.start:
            failures.append(
                {
                    "sequence_id": sequence_id,
                    "source": row["feat_p"],
                    "reason": "Ember crop out of bounds",
                }
            )
            continue
        texts = _texts(row.get("frame_labels", []), "single_person", "frame_labels")
        final_split = choose_group_split(source_memberships[source_key])
        records.append(
            {
                "schema_version": 1,
                "sample_id": f"humanml3d:{sequence_id}",
                "source_dataset": "HumanML3D",
                "source_sequence_id": sequence_id,
                "source_group_id": f"amass:{source_key[0]}/{source_key[1]}",
                "augmentation": "clip",
                "split_original": original_split,
                "split": final_split,
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


def write_pretty_json(path: Path, value: object) -> None:
    """Write a readable UTF-8 JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
