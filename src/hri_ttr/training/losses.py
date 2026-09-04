"""Named-schema reconstruction losses with strict mask handling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from typing_extensions import override

from hri_ttr.representations.g1.schema import G1_SCHEMA_VERSION
from hri_ttr.representations.human.normalizer import SCHEMA_ID as HUMAN_SCHEMA_ID
from hri_ttr.training.errors import TrainingError, TrainingReason


class ZeroValidFramesError(ValueError):
    """Raised when a batch cannot contribute a reconstruction gradient."""


@dataclass(frozen=True, slots=True)
class LossComponent:
    """One named contiguous feature range and its training weight."""

    name: str
    start: int
    stop: int
    weight: float


SCHEMA_COMPONENTS: dict[str, tuple[LossComponent, ...]] = {
    HUMAN_SCHEMA_ID: (LossComponent("human_motion", 0, 262, 1.0),),
    G1_SCHEMA_VERSION: (
        LossComponent("root_position", 0, 3, 2.0),
        LossComponent("root_rotation", 3, 9, 2.0),
        LossComponent("dof_position", 9, 38, 1.0),
        LossComponent("root_velocity", 38, 44, 1.0),
        LossComponent("dof_velocity", 44, 73, 0.5),
        LossComponent("foot_contact", 73, 75, 1.0),
    ),
}


class MaskedReconstructionLoss(nn.Module):
    """Weighted MSE over valid frames only."""

    components: tuple[LossComponent, ...]

    def __init__(self, components: tuple[LossComponent, ...]) -> None:
        """Store the schema-selected immutable component definition."""
        super().__init__()
        self.components = components

    @classmethod
    def for_schema(cls, schema: str) -> MaskedReconstructionLoss:
        """Select the immutable loss definition for a representation schema."""
        try:
            return cls(SCHEMA_COMPONENTS[schema])
        except KeyError as error:
            raise TrainingError(TrainingReason.UNSUPPORTED_LOSS) from error

    @override
    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return a normalized scalar loss while excluding padded frames."""
        valid_count = frame_mask.sum()
        if int(valid_count.item()) == 0:
            raise ZeroValidFramesError
        valid = frame_mask.unsqueeze(2)
        total = reconstruction.new_zeros(())
        total_weight = 0.0
        for component in self.components:
            error = (
                reconstruction[:, :, component.start : component.stop]
                - target[:, :, component.start : component.stop]
            ).square()
            element_count = valid_count * (component.stop - component.start)
            total = total + (error * valid).sum() / element_count * component.weight
            total_weight += component.weight
        return total / total_weight
