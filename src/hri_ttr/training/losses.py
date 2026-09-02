"""Named-schema reconstruction losses with strict mask handling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional
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


HUMAN_POSITION_STOP = 66
HUMAN_VELOCITY_WEIGHT = 0.5
G1_ROOT_POSITION_START = 0
G1_ROOT_POSITION_STOP = 3
G1_DOF_POSITION_START = 9
G1_DOF_POSITION_STOP = 38
G1_TEMPORAL_WEIGHT = 0.5


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
    """Schema-specific reconstruction loss over valid frames only."""

    components: tuple[LossComponent, ...]
    uses_ttr_human_loss: bool

    def __init__(
        self,
        components: tuple[LossComponent, ...],
        *,
        uses_ttr_human_loss: bool,
    ) -> None:
        """Store the schema-selected immutable component definition."""
        super().__init__()
        self.components = components
        self.uses_ttr_human_loss = uses_ttr_human_loss

    @classmethod
    def for_schema(cls, schema: str) -> MaskedReconstructionLoss:
        """Select the immutable loss definition for a representation schema."""
        try:
            return cls(
                SCHEMA_COMPONENTS[schema],
                uses_ttr_human_loss=schema == HUMAN_SCHEMA_ID,
            )
        except KeyError as error:
            raise TrainingError(TrainingReason.UNSUPPORTED_LOSS) from error

    @override
    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the schema loss while excluding padded frames."""
        valid_count = frame_mask.sum()
        if int(valid_count.item()) == 0:
            raise ZeroValidFramesError
        if self.uses_ttr_human_loss:
            return self._human_ttr(reconstruction, target, frame_mask, valid_count)
        return self._g1_ttr_style(reconstruction, target, frame_mask, valid_count)

    @staticmethod
    def _human_ttr(
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        frame_mask: torch.Tensor,
        valid_count: torch.Tensor,
    ) -> torch.Tensor:
        valid = frame_mask.unsqueeze(2)
        reconstruction_error = functional.smooth_l1_loss(
            reconstruction, target, reduction="none"
        )
        reconstruction_loss = (reconstruction_error * valid).sum() / (
            valid_count * reconstruction.shape[2]
        )
        velocity_mask = frame_mask[:, 1:] & frame_mask[:, :-1]
        velocity_count = velocity_mask.sum()
        if int(velocity_count.item()) == 0:
            return reconstruction_loss
        predicted_velocity = (
            reconstruction[:, 1:, :HUMAN_POSITION_STOP]
            - reconstruction[:, :-1, :HUMAN_POSITION_STOP]
        )
        target_velocity = (
            target[:, 1:, :HUMAN_POSITION_STOP] - target[:, :-1, :HUMAN_POSITION_STOP]
        )
        velocity_error = functional.smooth_l1_loss(
            predicted_velocity, target_velocity, reduction="none"
        )
        velocity_loss = (velocity_error * velocity_mask.unsqueeze(2)).sum() / (
            velocity_count * HUMAN_POSITION_STOP
        )
        return reconstruction_loss + HUMAN_VELOCITY_WEIGHT * velocity_loss

    def _g1_ttr_style(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        frame_mask: torch.Tensor,
        valid_count: torch.Tensor,
    ) -> torch.Tensor:
        valid = frame_mask.unsqueeze(2)
        total = reconstruction.new_zeros(())
        total_weight = 0.0
        for component in self.components:
            error = functional.smooth_l1_loss(
                reconstruction[:, :, component.start : component.stop],
                target[:, :, component.start : component.stop],
                reduction="none",
            )
            element_count = valid_count * (component.stop - component.start)
            total = total + (error * valid).sum() / element_count * component.weight
            total_weight += component.weight
        grouped_reconstruction = total / total_weight
        velocity_mask = frame_mask[:, 1:] & frame_mask[:, :-1]
        velocity_count = velocity_mask.sum()
        if int(velocity_count.item()) == 0:
            return grouped_reconstruction
        root_delta = self._masked_delta_loss(
            reconstruction[:, 1:, G1_ROOT_POSITION_START:G1_ROOT_POSITION_STOP]
            - reconstruction[:, :-1, G1_ROOT_POSITION_START:G1_ROOT_POSITION_STOP],
            target[:, 1:, G1_ROOT_POSITION_START:G1_ROOT_POSITION_STOP]
            - target[:, :-1, G1_ROOT_POSITION_START:G1_ROOT_POSITION_STOP],
            velocity_mask,
        )
        dof_delta = self._masked_delta_loss(
            reconstruction[:, 1:, G1_DOF_POSITION_START:G1_DOF_POSITION_STOP]
            - reconstruction[:, :-1, G1_DOF_POSITION_START:G1_DOF_POSITION_STOP],
            target[:, 1:, G1_DOF_POSITION_START:G1_DOF_POSITION_STOP]
            - target[:, :-1, G1_DOF_POSITION_START:G1_DOF_POSITION_STOP],
            velocity_mask,
        )
        temporal = (root_delta + dof_delta) / 2.0
        return grouped_reconstruction + G1_TEMPORAL_WEIGHT * temporal

    @staticmethod
    def _masked_delta_loss(
        predicted_delta: torch.Tensor,
        target_delta: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        error = functional.smooth_l1_loss(
            predicted_delta, target_delta, reduction="none"
        )
        return (error * valid_mask.unsqueeze(2)).sum() / (
            valid_mask.sum() * predicted_delta.shape[2]
        )
