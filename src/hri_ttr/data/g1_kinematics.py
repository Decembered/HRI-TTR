"""Dependency-free G1 forward kinematics for preprocessing and audits."""

# pyright: reportAny=false

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import numpy.typing as npt

from hri_ttr.geometry.quaternion import xyzw_to_matrix
from hri_ttr.representations.g1.constants import G1_DOF_NAMES

_VECTOR_WIDTH: Final = 3

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class _Body:
    name: str
    parent: int
    position: npt.NDArray[np.float64]
    rotation: npt.NDArray[np.float64]
    joint_index: int | None
    joint_axis: npt.NDArray[np.float64] | None


@dataclass(frozen=True, slots=True)
class G1Kinematics:
    """G1 body tree loaded from the authoritative MuJoCo model."""

    bodies: tuple[_Body, ...]

    @classmethod
    def from_mjcf(cls, path: Path) -> G1Kinematics:
        """Parse body transforms and enforce the project's 29-DoF ordering."""
        world = ET.parse(path).getroot().find("worldbody")  # noqa: S314
        if world is None:
            detail = "MJCF has no worldbody"
            raise ValueError(detail)
        pelvis = world.find("body[@name='pelvis']")
        if pelvis is None:
            detail = "MJCF has no pelvis body"
            raise ValueError(detail)
        bodies: list[_Body] = []
        joint_names: list[str] = []

        def visit(element: ET.Element, parent: int) -> None:
            name = element.attrib.get("name", "")
            joint = element.find("joint")
            joint_name = None if joint is None else joint.attrib.get("name")
            joint_index = None
            axis = None
            if joint_name is not None:
                if joint_name not in G1_DOF_NAMES:
                    detail = f"unknown G1 joint: {joint_name}"
                    raise ValueError(detail)
                joint_index = G1_DOF_NAMES.index(joint_name)
                joint_names.append(joint_name)
                axis = _vector(cast("ET.Element", joint).attrib.get("axis", "0 0 1"))
            bodies.append(
                _Body(
                    name,
                    parent,
                    _vector(element.attrib.get("pos", "0 0 0")),
                    _wxyz_matrix(element.attrib.get("quat", "1 0 0 0")),
                    joint_index,
                    axis,
                )
            )
            body_index = len(bodies) - 1
            for child in element.findall("body"):
                visit(child, body_index)

        visit(pelvis, -1)
        if tuple(joint_names) != G1_DOF_NAMES:
            detail = "MJCF joint traversal does not match G1 29-DoF order"
            raise ValueError(detail)
        return cls(tuple(bodies))

    def body_positions(
        self,
        root_position: npt.NDArray[np.float64],
        root_rotation_xyzw: npt.NDArray[np.float64],
        dof_position: npt.NDArray[np.float64],
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Evaluate world-space body origins for a complete motion."""
        frames = len(root_position)
        if (
            root_position.shape != (frames, 3)
            or root_rotation_xyzw.shape != (frames, 4)
            or dof_position.shape != (frames, 29)
        ):
            detail = "invalid G1 forward-kinematics input shapes"
            raise ValueError(detail)
        positions: list[npt.NDArray[np.float64]] = []
        rotations: list[npt.NDArray[np.float64]] = []
        for body in self.bodies:
            if body.parent < 0:
                positions.append(np.asarray(root_position, dtype=np.float64))
                rotations.append(xyzw_to_matrix(root_rotation_xyzw))
                continue
            parent_position = positions[body.parent]
            parent_rotation = rotations[body.parent]
            position: npt.NDArray[np.float64] = np.asarray(
                parent_position
                + np.einsum("tij,j->ti", parent_rotation, body.position),
                dtype=np.float64,
            )
            rotation: npt.NDArray[np.float64] = np.asarray(
                parent_rotation @ body.rotation, dtype=np.float64
            )
            if body.joint_index is not None and body.joint_axis is not None:
                rotation = rotation @ _axis_angle(
                    body.joint_axis, dof_position[:, body.joint_index]
                )
            positions.append(position)
            rotations.append(rotation)
        return {body.name: positions[index] for index, body in enumerate(self.bodies)}


def _vector(value: str) -> npt.NDArray[np.float64]:
    parsed = np.fromstring(value, sep=" ", dtype=np.float64)
    if parsed.shape != (_VECTOR_WIDTH,):
        detail = f"expected three-vector, got: {value}"
        raise ValueError(detail)
    return parsed


def _wxyz_matrix(value: str) -> npt.NDArray[np.float64]:
    parsed = np.fromstring(value, sep=" ", dtype=np.float64)
    if parsed.shape != (4,):
        detail = f"expected quaternion, got: {value}"
        raise ValueError(detail)
    return xyzw_to_matrix(parsed[[1, 2, 3, 0]])


def _axis_angle(
    axis: npt.NDArray[np.float64], angles: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    unit = axis / np.linalg.norm(axis)
    x, y, z = (float(value) for value in cast("list[float]", unit.tolist()))
    skew = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    outer = np.outer(unit, unit)
    identity = np.eye(3, dtype=np.float64)
    cosine = np.cos(angles)[:, None, None]
    sine = np.sin(angles)[:, None, None]
    return cosine * identity + (1.0 - cosine) * outer + sine * skew
