"""Strictly left-causal temporal convolution components."""

from __future__ import annotations

from typing import Final

import torch
from torch import nn
from torch.nn import functional
from typing_extensions import override

SECOND_BLOCK_DEPTH: Final = 2
THIRD_BLOCK_DEPTH: Final = 3


class CausalConv1d(nn.Module):
    """A convolution whose output never reads samples to its right."""

    left_padding: Final[int]
    conv: nn.Conv1d

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        left_padding: int | None = None,
    ) -> None:
        """Create a convolution with explicit or kernel-derived left padding."""
        super().__init__()
        self.left_padding = (
            dilation * (kernel_size - 1) if left_padding is None else left_padding
        )
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
        )

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply left padding before an otherwise unpadded convolution."""
        return self.conv.forward(functional.pad(inputs, (self.left_padding, 0)))


class CausalResidualBlock(nn.Module):
    """Residual causal convolutions without time-aggregating normalization."""

    first: CausalConv1d
    second: CausalConv1d
    activation: nn.SiLU

    def __init__(self, channels: int, dilation: int) -> None:
        """Create a two-convolution causal residual block."""
        super().__init__()
        self.first = CausalConv1d(channels, channels, 3, dilation=dilation)
        self.second = CausalConv1d(channels, channels, 3, dilation=dilation)
        self.activation = nn.SiLU()

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Preserve the temporal length and causal boundary."""
        hidden = self.activation.forward(self.first.forward(inputs))
        return inputs + self.second.forward(hidden)


class CausalResidualStack(nn.Module):
    """A statically typed stack of at most three residual blocks."""

    first: CausalResidualBlock
    second: CausalResidualBlock | None
    third: CausalResidualBlock | None

    def __init__(self, channels: int, depth: int) -> None:
        """Create residual blocks with dilations one, three, and nine."""
        super().__init__()
        self.first = CausalResidualBlock(channels, 1)
        self.second = (
            CausalResidualBlock(channels, 3) if depth >= SECOND_BLOCK_DEPTH else None
        )
        self.third = (
            CausalResidualBlock(channels, 9) if depth >= THIRD_BLOCK_DEPTH else None
        )

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply configured blocks in causal order."""
        hidden = self.first.forward(inputs)
        if self.second is not None:
            hidden = self.second.forward(hidden)
        if self.third is not None:
            hidden = self.third.forward(hidden)
        return hidden


class CausalDownsampleStage(nn.Module):
    """One aligned stride-two convolution followed by causal residuals."""

    downsample: CausalConv1d
    residuals: CausalResidualStack

    def __init__(self, width: int, depth: int) -> None:
        """Create one right-edge-aligned downsampling stage."""
        super().__init__()
        self.downsample = CausalConv1d(width, width, 4, stride=2, left_padding=2)
        self.residuals = CausalResidualStack(width, depth)

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Downsample without reading beyond each output right edge."""
        return self.residuals.forward(self.downsample.forward(inputs))


class CausalEncoder(nn.Module):
    """Map four input frames to one latent with exact right-edge alignment."""

    frames_per_token: Final[int] = 4
    input_projection: CausalConv1d
    first_stage: CausalDownsampleStage
    second_stage: CausalDownsampleStage
    output_projection: CausalConv1d
    activation: nn.SiLU

    def __init__(self, feature_dim: int, width: int, code_dim: int, depth: int) -> None:
        """Create two right-edge-aligned stride-two stages."""
        super().__init__()
        self.input_projection = CausalConv1d(feature_dim, width, 3)
        self.first_stage = CausalDownsampleStage(width, depth)
        self.second_stage = CausalDownsampleStage(width, depth)
        self.output_projection = CausalConv1d(width, code_dim, 3)
        self.activation = nn.SiLU()

    @override
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Consume ``[B,T,F]`` and return ``[B,T/4,C]``."""
        hidden = self.activation.forward(
            self.input_projection.forward(features.transpose(1, 2))
        )
        hidden = self.first_stage.forward(hidden)
        hidden = self.second_stage.forward(hidden)
        return self.output_projection.forward(hidden).transpose(1, 2)


class CausalDecoder(nn.Module):
    """Decode each token into four frames without reading future tokens."""

    frames_per_token: Final[int] = 4
    input_projection: CausalConv1d
    before: CausalResidualStack
    after_first: CausalConv1d
    after_second: CausalConv1d
    output_projection: CausalConv1d
    activation: nn.SiLU

    def __init__(self, feature_dim: int, width: int, code_dim: int, depth: int) -> None:
        """Create two causal nearest-neighbor upsampling stages."""
        super().__init__()
        self.input_projection = CausalConv1d(code_dim, width, 3)
        self.before = CausalResidualStack(width, depth)
        self.after_first = CausalConv1d(width, width, 3)
        self.after_second = CausalConv1d(width, width, 3)
        self.output_projection = CausalConv1d(width, feature_dim, 3)
        self.activation = nn.SiLU()

    @override
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Consume ``[B,L,C]`` and return ``[B,4L,F]``."""
        hidden = self.before.forward(
            self.activation.forward(
                self.input_projection.forward(latents.transpose(1, 2))
            )
        )
        hidden = torch.repeat_interleave(hidden, 2, dim=2)
        hidden = self.activation.forward(self.after_first.forward(hidden))
        hidden = torch.repeat_interleave(hidden, 2, dim=2)
        hidden = self.activation.forward(self.after_second.forward(hidden))
        return self.output_projection.forward(hidden).transpose(1, 2)
