"""Build the source-referenced same-motion dataset manifest."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from same_motion_builders import (
    build_bone_record,
    build_interx,
    load_bone_texts,
    write_pretty_json,
)
from same_motion_humanml import build_humanml

from hri_ttr.data.same_motion_manifest import choose_group_split, write_jsonl

Json = dict[str, Any]
MAX_LEAKAGE_EXAMPLES = 20
_BONE_DESCRIPTIONS: dict[str, list[Json]] = {}
_BONE_EVENTS: dict[str, list[Json]] = {}


def parse_args() -> argparse.Namespace:
    """Parse dataset source and output paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--humanml-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def initialize_bone_worker(
    descriptions: dict[str, list[Json]], events: dict[str, list[Json]]
) -> None:
    """Initialize read-only annotation maps in each worker process."""
    global _BONE_DESCRIPTIONS, _BONE_EVENTS  # noqa: PLW0603
    _BONE_DESCRIPTIONS = descriptions
    _BONE_EVENTS = events


def process_bone(paths: tuple[str, str]) -> tuple[Json | None, Json | None]:
    """Build one Bone-Seed pair inside a worker process."""
    return build_bone_record(
        Path(paths[0]), Path(paths[1]), _BONE_DESCRIPTIONS, _BONE_EVENTS
    )


def build_bone(root: Path, workers: int) -> tuple[list[Json], list[Json], Json]:
    """Build all basename-matched Bone-Seed pairs."""
    human_root = root / "smpl" / "smpl_filtered"
    g1_root = root / "g1" / "robot_filtered"
    human = {path.stem: path for path in human_root.glob("*.pkl")}
    g1 = {path.stem: path for path in g1_root.rglob("*.pkl")}
    matched = sorted(human.keys() & g1.keys())
    descriptions, events = load_bone_texts(
        root / "metadata" / "seed_text_annotations_v1.sqlite"
    )
    pairs = [(str(human[key]), str(g1[key])) for key in matched]
    records: list[Json] = []
    failures: list[Json] = []
    with multiprocessing.Pool(
        workers,
        initializer=initialize_bone_worker,
        initargs=(descriptions, events),
    ) as pool:
        for record, failure in pool.imap_unordered(process_bone, pairs, chunksize=64):
            if record is not None:
                records.append(record)
            if failure is not None:
                failures.append(failure)
    inventory = {
        "human_files": len(human),
        "g1_files": len(g1),
        "basename_matches": len(matched),
        "human_without_g1": len(human.keys() - g1.keys()),
        "g1_without_human": len(g1.keys() - human.keys()),
    }
    return records, failures, inventory


def resolve_signature_leakage(records: list[Json]) -> Json:
    """Keep duplicate Human motions in a single deterministic split."""
    by_signature: dict[str, list[Json]] = defaultdict(list)
    for row in records:
        by_signature[str(row["human_signature"])].append(row)
    duplicate_sets = [rows for rows in by_signature.values() if len(rows) > 1]
    cross_split_before = 0
    changed = 0
    cross_dataset = 0
    examples: list[Json] = []
    for rows in duplicate_sets:
        original_splits = {str(row["split"]) for row in rows}
        datasets = {str(row["source_dataset"]) for row in rows}
        if len(original_splits) > 1:
            cross_split_before += 1
        if len(datasets) > 1:
            cross_dataset += 1
        final_split = choose_group_split(original_splits)
        for row in rows:
            if row["split"] != final_split:
                row["split_adjusted_for_duplicate"] = True
                row["split"] = final_split
                changed += 1
        if len(examples) < MAX_LEAKAGE_EXAMPLES:
            examples.append(
                {
                    "sample_ids": [row["sample_id"] for row in rows[:10]],
                    "datasets": sorted(datasets),
                    "final_split": final_split,
                }
            )
    cross_split_after = sum(
        len({str(row["split"]) for row in rows}) > 1 for rows in duplicate_sets
    )
    return {
        "signature_method": "64-frame root-relative joints, 1mm quantization, sha256",
        "duplicate_signature_sets": len(duplicate_sets),
        "cross_dataset_signature_sets": cross_dataset,
        "cross_split_before_resolution": cross_split_before,
        "records_reassigned": changed,
        "cross_split_after_resolution": cross_split_after,
        "examples": examples,
    }


def summarize(
    records: list[Json], failures: dict[str, list[Json]], inventory: Json
) -> Json:
    """Summarize source coverage, durations, splits, and failures."""
    datasets = Counter(str(row["source_dataset"]) for row in records)
    splits = Counter(str(row["split"]) for row in records)
    captioned = sum(bool(row["has_text"]) for row in records)
    hours = sum(float(row["duration_sec"]) for row in records) / 3600
    original_splits = Counter(str(row["split_original"]) for row in records)
    moved = sum(
        row["split_original"] is not None and row["split_original"] != row["split"]
        for row in records
    )
    return {
        "schema_version": 1,
        "sequence_count": len(records),
        "duration_hours": hours,
        "captioned_sequences": captioned,
        "text_coverage": captioned / len(records) if records else 0,
        "by_dataset": dict(sorted(datasets.items())),
        "by_split": dict(sorted(splits.items())),
        "original_split_memberships": dict(sorted(original_splits.items())),
        "records_moved_to_prevent_leakage": moved,
        "failures": {key: len(value) for key, value in failures.items()},
        "bone_seed_inventory": inventory,
    }


def write_dataset(
    output: Path,
    records: list[Json],
    failures: dict[str, list[Json]],
    summary: Json,
    leakage: Json,
) -> None:
    """Write the manifest, split lists, and audit reports."""
    records.sort(key=lambda row: str(row["sample_id"]))
    write_jsonl(output / "manifest" / "same_motion.jsonl", records)
    for split in ("train", "val", "test"):
        ids = [str(row["sample_id"]) for row in records if row["split"] == split]
        (output / "splits").mkdir(parents=True, exist_ok=True)
        (output / "splits" / f"{split}.txt").write_text(
            "\n".join(ids) + "\n", encoding="utf-8"
        )
    for dataset, rows in failures.items():
        write_jsonl(output / "audits" / f"{dataset}_failures.jsonl", rows)
    write_pretty_json(output / "audits" / "summary.json", summary)
    write_pretty_json(output / "audits" / "duplicate_leakage.json", leakage)
    readme = {
        "purpose": (
            "Same-motion Human to G1 source-reference dataset; "
            "no raw files copied or modified."
        ),
        "manifest": "manifest/same_motion.jsonl",
        "frame_ranges": "Half-open [frame_start, frame_end).",
        "text_kinds": {
            "single_person": "Caption describes the selected person.",
            "interaction": (
                "Caption describes both Inter-X people; never a single-person caption."
            ),
            "single_person_event": "Bone-Seed temporal event annotation.",
        },
        "humanml3d_note": (
            "G1 fps is measured from complete Human/G1 frame counts; "
            "fps_file_value preserves Ember's nominal value."
        ),
    }
    write_pretty_json(output / "README.json", readme)


def main() -> None:
    """Build and report the complete same-motion manifest."""
    args = parse_args()
    bone_records, bone_failures, inventory = build_bone(
        args.datasets_root / "bones_seed", args.workers
    )
    interx_records, interx_failures = build_interx(args.datasets_root / "interx")
    humanml_records, humanml_failures = build_humanml(
        args.datasets_root / "humanml3d",
        args.humanml_index,
        args.datasets_root / "amass_g1_ember" / "g1",
    )
    records = bone_records + interx_records + humanml_records
    failures = {
        "bone_seed": bone_failures,
        "interx": interx_failures,
        "humanml3d": humanml_failures,
    }
    leakage = resolve_signature_leakage(records)
    summary = summarize(records, failures, inventory)
    write_dataset(args.output, records, failures, summary, leakage)
    _ = sys.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
