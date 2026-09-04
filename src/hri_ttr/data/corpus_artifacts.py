"""Deterministic JSON, JSONL, checksum, and schema corpus artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from hri_ttr.data.same_motion_preprocess import TARGET_FPS
from hri_ttr.representations.g1.constants import G1_DOF_NAMES
from hri_ttr.representations.g1.schema import G1_FEATURE_FIELDS, G1_SCHEMA_VERSION
from hri_ttr.representations.human.normalizer import HUMAN_FIELDS, SCHEMA_ID

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

Json = dict[str, object]


def schema() -> Json:
    """Return the canonical corpus representation contract."""
    return {
        "schema_version": 1,
        "target_fps": TARGET_FPS,
        "coordinates": (
            "right-handed InteractionWorld Y-up; source (x,y,z-up) maps to (x,z,-y)"
        ),
        "space": {
            "shape": "[T,3]",
            "columns": ["x_m", "z_m", "yaw_rad"],
            "coordinates": (
                "per-frame world root trajectory in InteractionWorld Y-up; "
                "yaw = atan2(forward_z, forward_x) of the horizontal body-forward "
                "axis (Human hips-forward, G1 root x-axis)"
            ),
            "files": [
                "shards/<split>/shard-*/human_space.npy",
                "shards/<split>/shard-*/g1_space.npy",
            ],
        },
        "human": {
            "dtype": "float32",
            "shape": "[T,262]",
            "schema": SCHEMA_ID,
            "fields": [
                {
                    "name": field.name,
                    "slice": [field.start, field.stop],
                    "unit": field.unit,
                    "normalized": field.normalized,
                }
                for field in HUMAN_FIELDS
            ],
        },
        "g1": {
            "dtype": "float32",
            "shape": "[T,75]",
            "schema": G1_SCHEMA_VERSION,
            "quaternion_internal": "xyzw",
            "dof_names": list(G1_DOF_NAMES),
            "fields": [
                {
                    "name": field.name,
                    "slice": [field.start, field.stop],
                    "unit": field.unit,
                }
                for field in G1_FEATURE_FIELDS
            ],
        },
        "frame_ranges": "half-open [frame_start,frame_end)",
        "normalizer_fit_split": "train",
        "source_notes": {
            "HumanML3D": (
                "filtered human/<seq>.pkl paired with filtered g1/<seq>.pkl; "
                "Human joints use the first-frame Human canonicalization"
            ),
            "Inter-X": (
                "filtered Human/G1 basename pairs; P1/P2 stay in one "
                "interaction split group"
            ),
            "G1": "filtered retarget motion uses first-frame G1 canonicalization",
            "space": "human_space and g1_space are world InteractionWorld [x,z,yaw]",
        },
    }


def write_json(path: Path, value: object) -> None:
    """Write one stable, readable JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[Json]) -> None:
    """Stream UTF-8 JSON objects without retaining serialized text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            _ = stream.write(encoded + "\n")


def write_checksums(root: Path) -> None:
    """Bind every corpus artifact with a relative-path SHA256 inventory."""
    path = root / "checksums" / "SHA256SUMS"
    path.parent.mkdir(parents=True, exist_ok=True)
    sources = (item for item in root.rglob("*") if item.is_file() and item != path)
    lines: list[str] = []
    for source in sorted(sources):
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        lines.append(f"{digest.hexdigest()}  {source.relative_to(root)}")
    _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8")
