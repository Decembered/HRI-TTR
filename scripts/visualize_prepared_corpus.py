"""Render aligned Human/G1 corpus quality samples."""

# pyright: reportAny=false

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import animation

from hri_ttr.data.corpus_artifacts import write_checksums
from hri_ttr.data.g1_kinematics import G1Kinematics
from hri_ttr.geometry.coordinates import (
    g1_z_up_to_interaction_y_up,
    interaction_y_up_to_g1_z_up,
    quaternion_xyzw_interaction_to_g1,
)
from hri_ttr.geometry.quaternion import matrix_to_xyzw
from hri_ttr.geometry.rotation import rotation_6d_to_matrix
from hri_ttr.representations.g1.episode import EpisodeFrame
from hri_ttr.representations.g1.schema import G1_FEATURE_SLICES
from hri_ttr.representations.human.features import human262_to_joints22

if TYPE_CHECKING:
    from matplotlib.axes import Axes

Json = dict[str, object]

HUMAN_CHAINS = (
    (0, 2, 5, 8, 11),
    (0, 1, 4, 7, 10),
    (0, 3, 6, 9, 12, 15),
    (9, 14, 17, 19, 21),
    (9, 13, 16, 18, 20),
)


@dataclass(frozen=True, slots=True)
class _Candidate:
    row: Json
    shard: Path
    reason: str


@dataclass(frozen=True, slots=True)
class _Options:
    corpus: Path
    g1_mjcf: Path
    random_per_source: int


def _arguments() -> _Options:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--corpus", type=Path, required=True)
    _ = parser.add_argument("--g1-mjcf", type=Path, required=True)
    _ = parser.add_argument("--random-per-source", type=int, default=2)
    parsed = parser.parse_args()
    return _Options(parsed.corpus, parsed.g1_mjcf, parsed.random_per_source)


def _rows(corpus: Path) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for path in sorted((corpus / "shards").glob("*/*/sequences.jsonl")):
        with path.open(encoding="utf-8") as stream:
            candidates.extend(
                _Candidate(_mapping(json.loads(line)), path.parent, "")
                for line in stream
            )
    return candidates


def _choose(candidates: list[_Candidate], per_source: int) -> list[_Candidate]:
    rng = random.Random(20260903)  # noqa: S311 - reproducible visual audit only.
    chosen: dict[str, _Candidate] = {}
    for dataset in sorted({str(item.row["source_dataset"]) for item in candidates}):
        source = [
            item for item in candidates if str(item.row["source_dataset"]) == dataset
        ]
        for item in rng.sample(source, min(per_source, len(source))):
            chosen[str(item.row["sample_id"])] = _Candidate(
                item.row, item.shard, "random"
            )
    boundaries = (
        ("max_g1_dof_speed_rad_s", max),
        ("g1_near_floor_slide_p95_m_s", max),
        ("energy_correlation", min),
    )
    for metric, selector in boundaries:
        item = selector(candidates, key=lambda value: _metric(value.row, metric))
        chosen[str(item.row["sample_id"])] = _Candidate(item.row, item.shard, metric)
    return list(chosen.values())


def _segment(
    candidate: _Candidate,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    start = _integer(candidate.row["frame_start"])
    stop = _integer(candidate.row["frame_end"])
    human = np.asarray(
        np.load(candidate.shard / "human.npy", mmap_mode="r")[start:stop],
        dtype=np.float32,
    )
    g1 = np.asarray(
        np.load(candidate.shard / "g1.npy", mmap_mode="r")[start:stop],
        dtype=np.float32,
    )
    return human, g1


def _g1_joints(
    features: npt.NDArray[np.float32], row: Json, model: G1Kinematics
) -> npt.NDArray[np.float64]:
    root_episode = features[:, G1_FEATURE_SLICES["root_pos_episode_m"]].astype(
        np.float64
    )
    rotation_episode = rotation_6d_to_matrix(
        features[:, G1_FEATURE_SLICES["root_rot6d_episode"]].astype(np.float64)
    )
    anchor = EpisodeFrame(
        np.asarray(row["anchor_origin"], dtype=np.float64),
        np.asarray(row["anchor_basis"], dtype=np.float64),
    )
    root_interaction = anchor.positions_to_interaction(root_episode)
    rotation_interaction = matrix_to_xyzw(
        anchor.rotations_to_interaction(rotation_episode)
    )
    root = interaction_y_up_to_g1_z_up(root_interaction)
    rotation = quaternion_xyzw_interaction_to_g1(rotation_interaction)
    dof = features[:, G1_FEATURE_SLICES["dof_pos_rad"]].astype(np.float64)
    positions = model.body_positions(root, rotation, dof)
    joints_z_up = np.stack([positions[body.name] for body in model.bodies], axis=1)
    return g1_z_up_to_interaction_y_up(joints_z_up)


def _edges(model: G1Kinematics) -> list[tuple[int, int]]:
    return [
        (body.parent, index)
        for index, body in enumerate(model.bodies)
        if body.parent >= 0
    ]


def _draw(
    axis: Axes,
    points: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    edges: list[tuple[int, int]],
    color: str,
) -> None:
    centered = points - points[0]
    centered[:, 1] -= np.min(centered[:, 1])
    for parent, child in edges:
        _ = axis.plot(
            centered[[parent, child], 0],
            centered[[parent, child], 1],
            color=color,
            linewidth=1.8,
        )
    _ = axis.scatter(centered[:, 0], centered[:, 1], s=7, color=color)
    axis.set_aspect("equal")
    _ = axis.axis("off")


def _render(candidate: _Candidate, model: G1Kinematics, output: Path) -> Json:
    human_features, g1_features = _segment(candidate)
    if human_features.shape[0] != g1_features.shape[0]:
        raise ValueError(candidate.row["sample_id"])
    human = human262_to_joints22(human_features.astype(np.float32))
    g1 = _g1_joints(g1_features, candidate.row, model)
    frames = np.linspace(0, len(human) - 1, 6).round().astype(int)
    figure, axes = plt.subplots(2, 6, figsize=(16, 5), layout="constrained")
    human_edges = [edge for chain in HUMAN_CHAINS for edge in pairwise(chain)]
    g1_edges = _edges(model)
    for column, frame in enumerate(frames):
        _draw(axes[0, column], human[frame], human_edges, "#176b87")
        _draw(axes[1, column], g1[frame], g1_edges, "#c04b31")
        axes[0, column].set_title(f"{frame / 20:.2f}s", fontsize=9)
    _ = axes[0, 0].text(-0.2, 0.5, "Human", transform=axes[0, 0].transAxes, rotation=90)
    _ = axes[1, 0].text(-0.2, 0.5, "G1", transform=axes[1, 0].transAxes, rotation=90)
    _ = figure.suptitle(
        f"{candidate.row['sample_id']} | {candidate.reason} | aligned T={len(human)}"
    )
    sample_id = str(candidate.row["sample_id"])
    path = output / f"{sample_id.replace(':', '_')}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return {
        "sample_id": candidate.row["sample_id"],
        "source_dataset": candidate.row["source_dataset"],
        "selection_reason": candidate.reason,
        "frames": len(human),
        "human_shape": list(human_features.shape),
        "g1_shape": list(g1_features.shape),
        "quality": candidate.row["quality"],
        "image": path.name,
    }


def _animate(candidate: _Candidate, model: G1Kinematics, output: Path) -> str:
    human_features, g1_features = _segment(candidate)
    human = human262_to_joints22(human_features.astype(np.float32))
    g1 = _g1_joints(g1_features, candidate.row, model)
    human_edges = [edge for chain in HUMAN_CHAINS for edge in pairwise(chain)]
    g1_edges = _edges(model)
    figure, axes = plt.subplots(1, 2, figsize=(8, 5), layout="constrained")
    sample_id = str(candidate.row["sample_id"])

    def update(frame: int) -> tuple[()]:
        for axis in axes:
            axis.clear()
            axis.set_xlim(-1.2, 1.2)
            axis.set_ylim(-0.1, 2.2)
        _draw(axes[0], human[frame], human_edges, "#176b87")
        _draw(axes[1], g1[frame], g1_edges, "#c04b31")
        _ = axes[0].set_title("Human")
        _ = axes[1].set_title("G1 retarget")
        _ = figure.suptitle(f"{sample_id} | {frame / 20:.2f}s")
        return ()

    frames = list(range(0, len(human), 2))
    if frames[-1] != len(human) - 1:
        frames.append(len(human) - 1)
    movie = animation.FuncAnimation(figure, update, frames=frames, blit=False)
    path = output / f"{sample_id.replace(':', '_')}.mp4"
    movie.save(path, writer=animation.FFMpegWriter(fps=10, bitrate=1800), dpi=120)
    plt.close(figure)
    return path.name


def _mapping(value: object) -> Json:
    if not isinstance(value, dict):
        detail = "visual sample metadata must be a JSON object"
        raise TypeError(detail)
    return cast("Json", value)


def _metric(row: Json, name: str) -> float:
    quality = _mapping(row["quality"])
    value = quality[name]
    if not isinstance(value, int | float):
        detail = f"quality metric is not numeric: {name}"
        raise TypeError(detail)
    return float(value)


def _integer(value: object) -> int:
    if not isinstance(value, int):
        detail = "frame boundary is not an integer"
        raise TypeError(detail)
    return value


def _main() -> None:
    args = _arguments()
    output = args.corpus / "quality" / "visual_samples"
    output.mkdir(parents=True, exist_ok=True)
    model = G1Kinematics.from_mjcf(args.g1_mjcf)
    report: list[Json] = []
    animated_sources: set[str] = set()
    for item in _choose(_rows(args.corpus), args.random_per_source):
        row = _render(item, model, output)
        dataset = str(item.row["source_dataset"])
        if item.reason != "random" or dataset not in animated_sources:
            row["video"] = _animate(item, model, output)
            animated_sources.add(dataset)
        report.append(row)
    _ = (output / "samples.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_checksums(args.corpus)
    _ = sys.stdout.write(
        json.dumps({"visual_samples": len(report), "output": str(output)}) + "\n"
    )


if __name__ == "__main__":
    _main()
