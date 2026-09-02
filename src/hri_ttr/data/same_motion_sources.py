"""Trusted local source adapters for the three same-motion datasets."""

# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

from __future__ import annotations

from pathlib import Path
from typing import cast

import joblib
import numpy as np
import numpy.typing as npt

from hri_ttr.data.same_motion_preprocess import RawPair, wxyz_to_xyzw
from hri_ttr.data.same_motion_quality import QualityError
from hri_ttr.representations.g1.constants import G1_DOF_NAMES

Json = dict[str, object]


class SameMotionSourceReader:
    """Decode source references while caching large row-container pickles."""

    def __init__(self) -> None:
        """Initialize an empty cache of locator-indexed pickle rows."""
        self._rows: dict[Path, dict[str, Json]] = {}

    def load(self, record: Json) -> RawPair:
        """Load one manifest record without modifying any source artifact."""
        dataset = str(record["source_dataset"])
        if dataset == "Bone-Seed":
            return self._bone(record)
        if dataset == "Inter-X":
            return self._row_pair(record, ember=False)
        if dataset == "HumanML3D":
            return self._row_pair(record, ember=True)
        reason = "source_dataset"
        raise QualityError(reason, dataset)

    def _bone(self, record: Json) -> RawPair:
        human_ref = _mapping(record["human"])
        g1_ref = _mapping(record["g1"])
        human = _mapping(joblib.load(Path(str(human_ref["path"]))))
        outer = _mapping(joblib.load(Path(str(g1_ref["path"]))))
        locator = str(g1_ref["locator"])
        g1 = _mapping(outer.get(locator, outer))
        human_joints = _float_array(human.get("smpl_joints"))
        pose = _float_array(human.get("pose_aa"))
        translation = _float_array(human.get("transl"))
        frames = len(human_joints)
        if (
            pose.shape != (frames, 72)
            or translation.shape != (frames, 3)
            or human_joints.shape != (frames, 24, 3)
        ):
            reason = "bone_seed_smpl_schema"
            raise QualityError(
                reason, str((pose.shape, translation.shape, human_joints.shape))
            )
        return RawPair(
            human_joints[:, :22],
            _number(human_ref["fps"]),
            _slice(_float_array(g1.get("root_trans_offset")), g1_ref),
            _slice(_float_array(g1.get("root_rot")), g1_ref),
            _slice(_float_array(g1.get("dof")), g1_ref),
            _number(g1_ref["fps"]),
        )

    def _row_pair(self, record: Json, *, ember: bool) -> RawPair:
        human_ref = _mapping(record["human"])
        g1_ref = _mapping(record["g1"])
        human_row = self._row(human_ref)
        human_motion = _mapping(human_row["motion"])
        human_joints = _slice(_float_array(human_motion.get("joints")), human_ref)
        if ember:
            with np.load(Path(str(g1_ref["path"])), allow_pickle=True) as archive:
                names = tuple(str(value) for value in archive["dof_names"].tolist())
                if names != G1_DOF_NAMES:
                    reason = "g1_dof_order"
                    raise QualityError(reason, repr(names))
                bodies = tuple(str(value) for value in archive["body_names"].tolist())
                body_positions = _slice(_float_array(archive["body_positions"]), g1_ref)
                body_rotation = _slice(_float_array(archive["body_rotations"]), g1_ref)
                dof = _slice(_float_array(archive["dof_positions"]), g1_ref)
            return RawPair(
                human_joints,
                _number(human_ref["fps"]),
                body_positions[:, 0],
                wxyz_to_xyzw(body_rotation[:, 0]),
                dof,
                _number(g1_ref["fps"]),
                body_positions,
                bodies,
            )
        g1_row = self._row(g1_ref)
        g1_motion = _mapping(g1_row["motion"])
        return RawPair(
            human_joints,
            _number(human_ref["fps"]),
            _slice(_float_array(g1_motion.get("root_pos")), g1_ref),
            wxyz_to_xyzw(_slice(_float_array(g1_motion.get("root_rot")), g1_ref)),
            _slice(_float_array(g1_motion.get("dof_pos")), g1_ref),
            _number(g1_ref["fps"]),
        )

    def _row(self, reference: Json) -> Json:
        path = Path(str(reference["path"]))
        if path not in self._rows:
            rows = joblib.load(path)
            if not isinstance(rows, list):
                reason = "pickle_rows"
                raise QualityError(reason, str(path))
            self._rows[path] = {
                str(_mapping(row).get("seq_name")): _mapping(row) for row in rows
            }
        locator = str(reference["locator"])
        try:
            return self._rows[path][locator]
        except KeyError as error:
            reason = "source_locator"
            raise QualityError(reason, f"{path}:{locator}") from error


def _mapping(value: object) -> Json:
    if not isinstance(value, dict):
        reason = "source_mapping"
        raise QualityError(reason, type(value).__name__)
    return cast("Json", value)


def _float_array(value: object) -> npt.NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        reason = "source_nonfinite"
        raise QualityError(reason, str(result.shape))
    return result


def _slice(values: npt.NDArray[np.float64], reference: Json) -> npt.NDArray[np.float64]:
    start = _integer(reference["frame_start"])
    stop = _integer(reference["frame_end"])
    result = values[start:stop]
    if len(result) != stop - start:
        reason = "source_frame_range"
        raise QualityError(reason, f"{start}:{stop}/{len(values)}")
    return result


def _number(value: object) -> float:
    if not isinstance(value, int | float):
        reason = "source_number"
        raise QualityError(reason, repr(value))
    return float(value)


def _integer(value: object) -> int:
    if not isinstance(value, int):
        reason = "source_integer"
        raise QualityError(reason, repr(value))
    return value
