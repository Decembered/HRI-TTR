"""Build the frozen HumanML3D/Inter-X manifest from filtered pickle pairs."""

from __future__ import annotations

# pyright: reportAny=false, reportUnknownMemberType=false
import json
import re
from collections import defaultdict
from typing import TYPE_CHECKING, NoReturn, cast

import joblib
import numpy as np

from hri_ttr.data.same_motion_manifest import choose_group_split, write_jsonl

Json = dict[str, object]
if TYPE_CHECKING:
    from pathlib import Path

DATASETS: tuple[tuple[str, str], ...] = (
    ("humanl3d", "HumanML3D"),
    ("interx", "Inter-X"),
)
SPLITS = ("train", "val", "test")
MINIMUM_FRAMES = 2
TEXT_PART_COUNT = 4


def build_filtered_manifest(root: Path) -> tuple[list[Json], dict[str, list[Json]]]:
    """Return deterministic paired records and non-fatal source failures.

    The filtered dataset is already the source-of-truth release: every sequence
    has one Human pickle and one retargeted G1 pickle with the same basename.
    This function only reads those files to validate metadata and never rewrites
    or copies source motion.
    """
    records: list[Json] = []
    failures: dict[str, list[Json]] = defaultdict(list)
    for directory, dataset in DATASETS:
        records.extend(
            _build_dataset(root / directory, directory, dataset, failures[directory])
        )
    _resolve_group_splits(records)
    records.sort(key=lambda row: str(row["sample_id"]))
    return records, dict(failures)


def write_filtered_manifest(
    output: Path, records: list[Json], failures: dict[str, list[Json]]
) -> Json:
    """Write the manifest, split lists, and a compact source inventory."""
    if output.exists() and any(output.iterdir()):
        detail = f"output directory is not empty: {output}"
        _invalid(detail)
    output.mkdir(parents=True, exist_ok=True)
    records = sorted(records, key=lambda row: str(row["sample_id"]))
    write_jsonl(output / "manifest" / "same_motion.jsonl", records)
    for split in SPLITS:
        ids = [str(row["sample_id"]) for row in records if row["split"] == split]
        split_path = output / "splits" / f"{split}.txt"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    for dataset, rows in sorted(failures.items()):
        write_jsonl(output / "audits" / f"{dataset}_failures.jsonl", rows)
    summary = _summary(records, failures)
    (output / "audits").mkdir(parents=True, exist_ok=True)
    (output / "audits" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _build_dataset(  # noqa: C901 - validation is the manifest freeze boundary.
    root: Path, directory: str, dataset: str, failures: list[Json]
) -> list[Json]:
    ids_by_split: dict[str, list[str]] = {}
    seen: set[str] = set()
    for split in SPLITS:
        path = root / "splits" / f"{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        ids = [item.removesuffix(".pkl") for item in ids if item]
        if seen.intersection(ids):
            detail = f"duplicate filtered split membership in {path}"
            _invalid(detail)
        seen.update(ids)
        ids_by_split[split] = sorted(ids)

    records: list[Json] = []
    for split in SPLITS:
        for sequence_id in ids_by_split[split]:
            human_path = root / "human" / f"{sequence_id}.pkl"
            g1_path = root / "g1" / f"{sequence_id}.pkl"
            try:
                human = _load_pair_side(human_path, "human")
                g1 = _load_pair_side(g1_path, "g1")
                human_motion = _motion(human)
                g1_motion = _motion(g1)
                human_frames = _frame_count(human_motion, "joints", human_path)
                g1_frames = _frame_count(g1_motion, "dof_pos", g1_path)
                _require_shape(human_motion, "joints", (human_frames, 22, 3))
                _require_shape(human_motion, "poses", (human_frames, 66))
                _require_shape(human_motion, "trans", (human_frames, 3))
                _require_shape(g1_motion, "root_pos", (g1_frames, 3))
                _require_shape(g1_motion, "root_rot", (g1_frames, 4))
                _require_shape(g1_motion, "dof_pos", (g1_frames, 29))
                if human_frames != g1_frames:
                    detail = (
                        f"frame count mismatch: human={human_frames}, g1={g1_frames}"
                    )
                    _invalid(detail)
                human_fps = _fps(human_motion, human_path)
                g1_fps = _fps(g1_motion, g1_path)
                if human_fps != g1_fps:
                    detail = f"FPS mismatch: human={human_fps}, g1={g1_fps}"
                    _invalid(detail)
                if str(g1_motion.get("root_rot_convention")) != "wxyz":
                    _invalid("G1 root_rot_convention must be wxyz")
                if human.get("seq_name") != g1.get("seq_name"):
                    _invalid("Human/G1 seq_name mismatch")
                if human.get("feat_p") != g1.get("feat_p"):
                    _invalid("Human/G1 feat_p mismatch")
                texts = _read_texts(root / "text" / f"{sequence_id}.txt", dataset)
                group_id = _group_id(directory, sequence_id, human)
                records.append(
                    _record(
                        directory,
                        dataset,
                        sequence_id,
                        split,
                        human_path,
                        g1_path,
                        human_frames,
                        human_fps,
                        group_id,
                        texts,
                    )
                )
            except (KeyError, OSError, TypeError, ValueError) as error:
                failures.append(
                    {
                        "sequence_id": sequence_id,
                        "reason": type(error).__name__,
                        "detail": str(error),
                    }
                )
    return records


def _load_pair_side(path: Path, side: str) -> Json:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = joblib.load(path)
    if not isinstance(value, dict):
        detail = f"{side} source is not a mapping: {path}"
        _invalid(detail)
    result = cast("Json", value)
    if not isinstance(result.get("motion"), dict):
        detail = f"{side} source has no motion mapping: {path}"
        _invalid(detail)
    return result


def _motion(side: Json) -> Json:
    return cast("Json", side["motion"])


def _frame_count(motion: Json, field: str, path: Path) -> int:
    values = np.asarray(motion[field])
    if values.ndim < 1 or values.shape[0] < MINIMUM_FRAMES:
        detail = f"{path}: {field} must contain at least two frames"
        _invalid(detail)
    return int(values.shape[0])


def _require_shape(motion: Json, field: str, shape: tuple[int, ...]) -> None:
    values = np.asarray(motion[field])
    if values.shape != shape:
        detail = f"{field} has shape {values.shape}; expected {shape}"
        _invalid(detail)


def _fps(motion: Json, path: Path) -> float:
    value = motion.get("fps")
    if not isinstance(value, int | float) or not np.isfinite(float(value)):
        detail = f"{path}: invalid FPS"
        _invalid(detail)
    fps = float(value)
    if fps <= 0.0:
        detail = f"{path}: FPS must be positive"
        _invalid(detail)
    return fps


def _group_id(directory: str, sequence_id: str, human: Json) -> str:
    if directory == "humanl3d":
        source = str(human.get("feat_p", sequence_id)).replace("\\", "/")
        return f"humanml3d:{source}"
    base = re.sub(r"_P[12]$", "", sequence_id)
    return f"interx:{base}"


def _read_texts(path: Path, dataset: str) -> list[Json]:
    if not path.is_file():
        return []
    texts: list[Json] = []
    kind = "single_person" if dataset == "HumanML3D" else "interaction"
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("#")
        text = parts[0].strip()
        if not text:
            continue
        item: Json = {"text": text, "kind": kind, "source_field": "text_file"}
        if len(parts) >= TEXT_PART_COUNT:
            try:
                item["start_t"] = float(parts[-2])
                item["end_t"] = float(parts[-1])
            except ValueError:
                pass
        texts.append(item)
    return texts


def _record(  # noqa: PLR0913, PLR0917 - manifest fields are independent columns.
    directory: str,
    dataset: str,
    sequence_id: str,
    split: str,
    human_path: Path,
    g1_path: Path,
    frames: int,
    fps: float,
    group_id: str,
    texts: list[Json],
) -> Json:
    format_name = "filtered_pickle"
    return {
        "schema_version": 2,
        "sample_id": f"{directory}:{sequence_id}",
        "source_dataset": dataset,
        "source_sequence_id": sequence_id,
        "source_group_id": group_id,
        "augmentation": "original",
        "split_original": split,
        "split": split,
        "duration_sec": frames / fps,
        "human": {
            "path": str(human_path),
            "format": format_name,
            "locator": sequence_id,
            "fps": fps,
            "frame_start": 0,
            "frame_end": frames,
            "fields": ["poses", "trans", "joints"],
        },
        "g1": {
            "path": str(g1_path),
            "format": format_name,
            "locator": sequence_id,
            "fps": fps,
            "frame_start": 0,
            "frame_end": frames,
            "fields": ["root_pos", "root_rot", "dof_pos"],
        },
        "texts": texts,
        "has_text": bool(texts),
        "same_motion_evidence": "filtered Human/G1 basename pair with equal timeline",
    }


def _resolve_group_splits(records: list[Json]) -> None:
    memberships: dict[str, set[str]] = defaultdict(set)
    for row in records:
        memberships[str(row["source_group_id"])].add(str(row["split_original"]))
    for row in records:
        row["split"] = choose_group_split(memberships[str(row["source_group_id"])])
        if row["split"] != row["split_original"]:
            row["split_adjusted_for_group"] = True


def _summary(records: list[Json], failures: dict[str, list[Json]]) -> Json:
    by_dataset: dict[str, int] = defaultdict(int)
    by_split: dict[str, int] = defaultdict(int)
    for row in records:
        by_dataset[str(row["source_dataset"])] += 1
        by_split[str(row["split"])] += 1
    return {
        "schema_version": 2,
        "source_format": "HRI-Datasets/filtered/{humanl3d,interx}",
        "sequence_count": len(records),
        "duration_hours": sum(
            cast("float", row["duration_sec"]) for row in records
        ) / 3600,
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_split": dict(sorted(by_split.items())),
        "failures": {key: len(value) for key, value in sorted(failures.items())},
        "group_split_policy": "HumanML3D feat_p and Inter-X interaction group",
        "target_fps": 20.0,
    }


def _invalid(detail: str) -> NoReturn:
    raise ValueError(detail)
