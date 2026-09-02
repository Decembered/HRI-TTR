"""Immutable G1 canonical feature schema."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

from hri_ttr.contracts import MotionSchema, SchemaId

G1_SCHEMA_VERSION: Final = "g1_canonical_75d_v2"
G1_FEATURE_DIM: Final = 75


@dataclass(frozen=True, slots=True)
class FeatureField:
    """One named contiguous field in the 75D feature vector."""

    name: str
    start: int
    stop: int
    unit: str

    @property
    def width(self) -> int:
        """Return the number of scalar channels in the field."""
        return self.stop - self.start


G1_FEATURE_FIELDS: Final[tuple[FeatureField, ...]] = (
    FeatureField("root_pos_episode_m", 0, 3, "m"),
    FeatureField("root_rot6d_episode", 3, 9, "unitless"),
    FeatureField("dof_pos_rad", 9, 38, "rad"),
    FeatureField("root_linear_vel_local_m_s", 38, 41, "m/s"),
    FeatureField("root_angular_vel_local_rad_s", 41, 44, "rad/s"),
    FeatureField("dof_vel_rad_s", 44, 73, "rad/s"),
    FeatureField("foot_contact_lr", 73, 75, "binary"),
)
G1_FEATURE_SLICES: Final[Mapping[str, slice]] = MappingProxyType(
    {field.name: slice(field.start, field.stop) for field in G1_FEATURE_FIELDS}
)
G1_FEATURE_UNITS: Final[Mapping[str, str]] = MappingProxyType(
    {field.name: field.unit for field in G1_FEATURE_FIELDS}
)
G1_MOTION_SCHEMA: Final = MotionSchema(SchemaId(G1_SCHEMA_VERSION), G1_FEATURE_DIM)
