"""Trusted local source adapter for the frozen filtered dataset pairs."""

# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

from __future__ import annotations

from pathlib import Path
from typing import cast

import joblib
import numpy as np
import numpy.typing as npt

from hri_ttr.data.same_motion_preprocess import RawPair, wxyz_to_xyzw
from hri_ttr.data.same_motion_quality import QualityError

Json = dict[str, object]


class SameMotionSourceReader:
    """Decode one direct Human/G1 pickle pair without mutating source data."""

    def __init__(self) -> None:
        """Initialize a reader with no mutable source state."""

    def load(self, record: Json) -> RawPair:
        """Load one manifest record without modifying any source artifact."""
        human_ref = _mapping(record.get("human"))
        g1_ref = _mapping(record.get("g1"))
        if (
            human_ref.get("format") != "filtered_pickle"
            or g1_ref.get("format") != "filtered_pickle"
        ):
            reason = "source_format"
            raise QualityError(reason, str((human_ref, g1_ref)))
        return self._filtered_pair(record)

    def _filtered_pair(self, record: Json) -> RawPair:
        human_ref = _mapping(record["human"])
        g1_ref = _mapping(record["g1"])
        human = _mapping(joblib.load(Path(str(human_ref["path"]))))
        g1 = _mapping(joblib.load(Path(str(g1_ref["path"]))))
        human_motion = _mapping(human.get("motion"))
        g1_motion = _mapping(g1.get("motion"))
        if str(g1_motion.get("root_rot_convention")) != "wxyz":
            reason = "g1_quaternion_convention"
            raise QualityError(reason, str(g1_motion.get("root_rot_convention")))
        return RawPair(
            _slice(_float_array(human_motion.get("joints")), human_ref),
            _number(human_motion.get("fps")),
            _slice(_float_array(g1_motion.get("root_pos")), g1_ref),
            wxyz_to_xyzw(_slice(_float_array(g1_motion.get("root_rot")), g1_ref)),
            _slice(_float_array(g1_motion.get("dof_pos")), g1_ref),
            _number(g1_motion.get("fps")),
        )


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
