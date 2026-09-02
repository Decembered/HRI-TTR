from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

from hri_ttr.visualization import (
    render_feature_comparison,
    render_token_histogram,
    render_trajectory_comparison,
)


def test_visualizations_render_nonempty_pngs(tmp_path: Path) -> None:
    # Given
    target = np.arange(18, dtype=np.float64).reshape(6, 3) / 10.0
    prediction = target + 0.01
    histogram = np.arange(256, dtype=np.int64)
    paths = (
        tmp_path / "trajectory.png",
        tmp_path / "features.png",
        tmp_path / "tokens.png",
    )

    # When
    _ = render_trajectory_comparison(target, prediction, paths[0])
    _ = render_feature_comparison(target, prediction, paths[1])
    _ = render_token_histogram(histogram, paths[2])

    # Then
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)
    assert all(path.stat().st_size > 1000 for path in paths)
