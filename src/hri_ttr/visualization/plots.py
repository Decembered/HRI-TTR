"""Noninteractive Agg rendering for tokenizer evaluation artifacts."""

# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import matplotlib as mpl
import numpy as np
import numpy.typing as npt

from hri_ttr.evaluation.errors import reject_evaluation

if TYPE_CHECKING:
    from pathlib import Path

mpl.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

PNG_DPI: Final = 120
MATRIX_NDIM: Final = 2
TRAJECTORY_DIM: Final = 3
MAX_LEGEND_CHANNELS: Final = 6


def _save(figure: Figure, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas = FigureCanvasAgg(figure)
    canvas.print_png(destination)
    return destination


def render_trajectory_comparison(
    target: npt.NDArray[np.float64],
    prediction: npt.NDArray[np.float64],
    destination: Path,
) -> Path:
    """Render ground-truth and reconstructed root trajectories to a PNG."""
    if (
        target.shape != prediction.shape
        or target.ndim != MATRIX_NDIM
        or target.shape[1] != TRAJECTORY_DIM
    ):
        reject_evaluation("trajectory arrays must have matching shape [T,3]")
    figure = Figure(figsize=(6.4, 4.8), dpi=PNG_DPI, layout="constrained")
    axes = figure.add_subplot(1, 1, 1)
    _ = axes.plot(target[:, 0], target[:, 2], label="GT", linewidth=2)
    _ = axes.plot(
        prediction[:, 0], prediction[:, 2], label="Reconstruction", linestyle="--"
    )
    _ = axes.set(
        xlabel="Episode X (m)", ylabel="Episode Z (m)", title="Root trajectory"
    )
    _ = axes.axis("equal")
    axes.grid(visible=True, alpha=0.3)
    _ = axes.legend()
    return _save(figure, destination)


def render_feature_comparison(
    target: npt.NDArray[np.float64],
    prediction: npt.NDArray[np.float64],
    destination: Path,
) -> Path:
    """Render each requested feature channel over time to a PNG."""
    if (
        target.shape != prediction.shape
        or target.ndim != MATRIX_NDIM
        or target.shape[1] == 0
    ):
        reject_evaluation("feature arrays must have matching nonempty shape [T,D]")
    figure = Figure(figsize=(8.0, 4.8), dpi=PNG_DPI, layout="constrained")
    axes = figure.add_subplot(1, 1, 1)
    frames = np.arange(target.shape[0])
    for channel in range(target.shape[1]):
        label = f"channel {channel}"
        _ = axes.plot(frames, target[:, channel], label=f"GT {label}", linewidth=1.5)
        _ = axes.plot(
            frames, prediction[:, channel], label=f"Recon {label}", linestyle="--"
        )
    _ = axes.set(xlabel="Frame", ylabel="Value", title="Feature comparison")
    axes.grid(visible=True, alpha=0.3)
    if target.shape[1] <= MAX_LEGEND_CHANNELS:
        _ = axes.legend(ncol=2, fontsize="small")
    return _save(figure, destination)


def render_token_histogram(histogram: npt.NDArray[np.int64], destination: Path) -> Path:
    """Render all 256 code counts without dropping dead codes."""
    if histogram.shape != (256,) or np.any(histogram < 0):
        reject_evaluation("token histogram must contain 256 nonnegative counts")
    figure = Figure(figsize=(8.0, 4.2), dpi=PNG_DPI, layout="constrained")
    axes = figure.add_subplot(1, 1, 1)
    _ = axes.bar(np.arange(256), histogram, width=1.0)
    _ = axes.set(
        xlabel="Token ID", ylabel="Count", title="VQ codebook usage", xlim=(-1, 256)
    )
    axes.grid(visible=True, axis="y", alpha=0.3)
    return _save(figure, destination)
