"""Render fixed-seed Human/G1 contact sheets from the final manifest."""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np

HUMAN_CHAINS = (
    (0, 2, 5, 8, 11),
    (0, 1, 4, 7, 10),
    (0, 3, 6, 9, 12, 15),
    (9, 14, 17, 19, 21),
    (9, 13, 16, 18, 20),
)
MINIMUM_DURATION_SECONDS = 2
MAXIMUM_DURATION_SECONDS = 12


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--per-dataset", type=int, default=2)
    return parser.parse_args()


def _choose_rows(path: Path, count: int) -> list[dict[str, Any]]:
    rng = random.Random(20260902)  # noqa: S311
    buckets: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            duration = float(row["duration_sec"])
            if MINIMUM_DURATION_SECONDS <= duration <= MAXIMUM_DURATION_SECONDS:
                buckets.setdefault(row["source_dataset"], []).append(row)
    return [
        row
        for dataset in sorted(buckets)
        for row in rng.sample(buckets[dataset], count)
    ]


def _find_pickle_row(path: Path, locator: str) -> dict[str, Any]:
    rows = joblib.load(path)
    return next(row for row in rows if str(row["seq_name"]) == locator)


def _load_human(row: dict[str, Any]) -> np.ndarray:
    spec = row["human"]
    if spec["format"] == "joblib_pickle":
        return np.asarray(joblib.load(spec["path"])["smpl_joints"], dtype=np.float32)
    if spec["format"] == "filtered_pickle":
        item = joblib.load(spec["path"])
        return np.asarray(item["motion"]["joints"], dtype=np.float32)
    item = _find_pickle_row(Path(spec["path"]), str(spec["locator"]))
    return np.asarray(item["motion"]["joints"], dtype=np.float32)


def _model_edges(model: mujoco.MjModel) -> list[tuple[int, int]]:
    return [
        (int(model.body_parentid[index]) - 1, index - 1)
        for index in range(1, model.nbody)
        if int(model.body_parentid[index]) > 0
    ]


def _forward_kinematics(
    model: mujoco.MjModel,
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    dof: np.ndarray,
    quaternion_order: str,
) -> np.ndarray:
    data = mujoco.MjData(model)
    output = np.empty((len(dof), model.nbody - 1, 3), dtype=np.float32)
    for frame in range(len(dof)):
        data.qpos[:3] = root_pos[frame]
        quaternion = root_rot[frame]
        data.qpos[3:7] = (
            quaternion[[3, 0, 1, 2]] if quaternion_order == "xyzw" else quaternion
        )
        data.qpos[7:] = dof[frame]
        mujoco.mj_forward(model, data)
        output[frame] = data.xpos[1:]
    return output


def _load_g1(
    row: dict[str, Any], model: mujoco.MjModel
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    spec = row["g1"]
    start, end = int(spec["frame_start"]), int(spec["frame_end"])
    if spec["format"] == "npz":
        with np.load(spec["path"]) as data:
            return np.asarray(data["body_positions"][start:end]), _model_edges(model)
    if spec["format"] == "joblib_pickle":
        outer = joblib.load(spec["path"])
        item = outer.get(spec["locator"], outer)
        return (
            _forward_kinematics(
                model,
                np.asarray(item["root_trans_offset"])[start:end],
                np.asarray(item["root_rot"])[start:end],
                np.asarray(item["dof"])[start:end],
                "xyzw",
            ),
            _model_edges(model),
        )
    if spec["format"] == "filtered_pickle":
        item = joblib.load(spec["path"])["motion"]
        return (
            _forward_kinematics(
                model,
                np.asarray(item["root_pos"])[start:end],
                np.asarray(item["root_rot"])[start:end],
                np.asarray(item["dof_pos"])[start:end],
                "wxyz",
            ),
            _model_edges(model),
        )
    item = _find_pickle_row(Path(spec["path"]), str(spec["locator"]))["motion"]
    return (
        _forward_kinematics(
            model,
            np.asarray(item["root_pos"])[start:end],
            np.asarray(item["root_rot"])[start:end],
            np.asarray(item["dof_pos"])[start:end],
            "wxyz",
        ),
        _model_edges(model),
    )


def _human_edges() -> list[tuple[int, int]]:
    return [
        (parent, child) for chain in HUMAN_CHAINS for parent, child in pairwise(chain)
    ]


def _skeleton_svg(
    points: np.ndarray,
    edges: list[tuple[int, int]],
    center_x: float,
    ground_y: float,
    color: str,
) -> str:
    centered = points - points[0]
    centered[:, 2] -= centered[:, 2].min()
    scale = 105
    projected = np.column_stack(
        (center_x + scale * centered[:, 0], ground_y - scale * centered[:, 2])
    )
    lines = [
        f'<line x1="{projected[a, 0]:.1f}" y1="{projected[a, 1]:.1f}" '
        f'x2="{projected[b, 0]:.1f}" y2="{projected[b, 1]:.1f}"/>'
        for a, b in edges
        if a < len(projected) and b < len(projected)
    ]
    circles = [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2"/>' for x, y in projected]
    shapes = "".join(lines + circles)
    return f'<g stroke="{color}" fill="{color}" stroke-width="3">{shapes}</g>'


def _render(row: dict[str, Any], model: mujoco.MjModel, output: Path) -> dict[str, Any]:
    human = _load_human(row)
    g1, g1_edges = _load_g1(row, model)
    steps = 6
    human_indexes = np.linspace(0, len(human) - 1, steps).round().astype(int)
    g1_indexes = np.linspace(0, len(g1) - 1, steps).round().astype(int)
    columns = np.linspace(130, 1370, steps)
    elements = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="1500" '
            'height="700" viewBox="0 0 1500 700">'
        ),
        '<rect width="1500" height="700" fill="#f7f4ee"/>',
        (
            '<text x="40" y="45" font-size="24" font-family="sans-serif">'
            f"{html.escape(row['sample_id'])}</text>"
        ),
        (
            '<text x="40" y="90" font-size="20" fill="#176b87" '
            'font-family="sans-serif">Human</text>'
        ),
        (
            '<text x="40" y="390" font-size="20" fill="#c04b31" '
            'font-family="sans-serif">G1 retarget</text>'
        ),
    ]
    for index, x in enumerate(columns):
        elements.append(
            _skeleton_svg(
                human[human_indexes[index]], _human_edges(), x, 340, "#176b87"
            )
        )
        elements.append(
            _skeleton_svg(g1[g1_indexes[index]], g1_edges, x, 640, "#c04b31")
        )
        elements.append(
            f'<text x="{x - 18:.1f}" y="675" font-size="14">{index * 20}%</text>'
        )
    caption = next((item["text"] for item in row["texts"]), "")
    elements.append(
        '<text x="40" y="68" font-size="14" font-family="sans-serif">'
        f"{html.escape(caption[:180])}</text>"
    )
    elements.append("</svg>")
    output.write_text("".join(elements), encoding="utf-8")
    return {
        "sample_id": row["sample_id"],
        "source_dataset": row["source_dataset"],
        "duration_sec": row["duration_sec"],
        "caption": caption,
        "human_frames": len(human),
        "g1_frames": len(g1),
        "visualization": str(output),
    }


def _main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(args.model))
    report = []
    for row in _choose_rows(args.manifest, args.per_dataset):
        name = row["sample_id"].replace(":", "_") + ".svg"
        report.append(_render(row, model, args.output / name))
    (args.output / "samples.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _ = sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    _main()
