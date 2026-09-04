"""Build the HRI-TTR manifest from the frozen filtered Human/G1 datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hri_ttr.data.filtered_manifest import (
    build_filtered_manifest,
    write_filtered_manifest,
)


def parse_args() -> argparse.Namespace:
    """Parse the filtered source and output paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filtered-root",
        type=Path,
        required=True,
        help="HRI-Datasets/filtered directory containing humanl3d and interx.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Build and report the deterministic two-dataset source manifest."""
    args = parse_args()
    records, failures = build_filtered_manifest(args.filtered_root)
    summary = write_filtered_manifest(args.output, records, failures)
    _ = sys.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
