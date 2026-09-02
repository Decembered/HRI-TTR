"""Baseline-only manifest creation for incompatible legacy artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hri_ttr.checkpoints.io import checkpoint_sha256
from hri_ttr.checkpoints.schema import BaselineManifest

if TYPE_CHECKING:
    from pathlib import Path


def write_g1_73d_baseline_manifest(source: Path, destination: Path) -> BaselineManifest:
    """Record a legacy G1 checkpoint without loading or mapping any parameter."""
    manifest = BaselineManifest(
        baseline_kind="g1_73d_noncausal",
        source_path=source,
        source_sha256=checkpoint_sha256(source),
        load_policy="baseline_only_no_partial_load",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _ = destination.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest
