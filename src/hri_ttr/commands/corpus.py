"""Same-motion corpus preparation, validation, and loader smoke commands."""

# pyright: reportAny=false

from __future__ import annotations

import json
import multiprocessing
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - Typer resolves runtime annotations.
from typing import TYPE_CHECKING, Annotated, cast

import numpy as np
import typer

from hri_ttr.data.corpus import CorpusDomain, CorpusSplit, CorpusWindowDataset
from hri_ttr.data.corpus_audit import audit_corpus as verify_corpus
from hri_ttr.data.corpus_writer import CorpusWriter
from hri_ttr.data.g1_kinematics import G1Kinematics
from hri_ttr.data.same_motion_preprocess import (
    TARGET_FPS,
    PreparedPair,
    prepare_pair,
)
from hri_ttr.data.same_motion_quality import QualityError
from hri_ttr.data.same_motion_sources import SameMotionSourceReader
from hri_ttr.representations.g1.normalizer import G1FeatureNormalizer
from hri_ttr.representations.human.normalizer import HumanFeatureNormalizer
from hri_ttr.training.data import WindowConfig, collate_windows

Json = dict[str, object]
_worker_reader: SameMotionSourceReader | None = None
_worker_kinematics: G1Kinematics | None = None

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True, slots=True)
class _WorkerResult:
    record: Json
    pair: PreparedPair | None
    error_type: str | None = None
    error_detail: str | None = None


def prepare_same_motion(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
    g1_mjcf: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    target_fps: Annotated[float, typer.Option()] = TARGET_FPS,
    workers: Annotated[int, typer.Option(min=1)] = 16,
) -> None:
    """Convert the source-reference manifest into strict aligned mmap shards."""
    if target_fps != TARGET_FPS:
        detail = "HRI-TTR canonical target FPS is exactly 20"
        raise typer.BadParameter(detail)
    writer = CorpusWriter(output)
    records = tuple(_records(manifest))
    with multiprocessing.Pool(
        workers, initializer=_initialize_worker, initargs=(g1_mjcf,)
    ) as pool:
        for index, result in enumerate(
            pool.imap(_process_worker, records, chunksize=1), start=1
        ):
            _write_result(writer, result)
            if index % 1000 == 0:
                typer.echo(json.dumps({"processed": index}))
    typer.echo(json.dumps(writer.finish(), sort_keys=True))


def audit_corpus(
    corpus: Annotated[Path, typer.Option(exists=True, file_okay=False)],
) -> None:
    """Validate schema, sequence offsets, aligned shapes, and finite shard values."""
    typer.echo(json.dumps(verify_corpus(corpus), sort_keys=True))


def smoke_load(  # noqa: PLR0913, PLR0917 - independent CLI controls.
    corpus: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    split: Annotated[CorpusSplit, typer.Option()] = "train",
    domain: Annotated[CorpusDomain, typer.Option()] = "human",
    batches: Annotated[int, typer.Option(min=1)] = 10,
    batch_size: Annotated[int, typer.Option(min=1)] = 4,
    window_frames: Annotated[int, typer.Option(min=4)] = 196,
) -> None:
    """Read normalized real windows and verify the exact training batch contract."""
    normalizer = (
        HumanFeatureNormalizer.load(corpus / "normalizers" / "human.json")
        if domain == "human"
        else G1FeatureNormalizer.load(corpus / "normalizers" / "g1.json")
    )
    dataset = CorpusWindowDataset(
        corpus,
        split,
        domain,
        WindowConfig(window_frames, window_frames),
        normalizer,
    )
    checked = 0
    for start in range(0, min(len(dataset), batches * batch_size), batch_size):
        batch = collate_windows(
            [
                dataset[index]
                for index in range(start, min(start + batch_size, len(dataset)))
            ]
        )
        if not bool(np.isfinite(batch.features.numpy()).all()):
            detail = "smoke batch contains NaN/Inf"
            raise ValueError(detail)
        checked += 1
    typer.echo(
        json.dumps(
            {
                "domain": domain,
                "split": split,
                "batches": checked,
                "windows": len(dataset),
                "feature_dim": int(dataset[0].features.shape[1]),
            },
            sort_keys=True,
        )
    )


def _initialize_worker(g1_mjcf: Path) -> None:
    global _worker_reader, _worker_kinematics  # noqa: PLW0603
    _worker_reader = SameMotionSourceReader()
    _worker_kinematics = G1Kinematics.from_mjcf(g1_mjcf)


def _process_worker(record: Json) -> _WorkerResult:
    if _worker_reader is None or _worker_kinematics is None:
        return _WorkerResult(record, None, "worker_state", "not initialized")
    try:
        pair = prepare_pair(_worker_reader.load(record), _worker_kinematics)
        return _WorkerResult(record, pair)
    except QualityError as error:
        return _WorkerResult(record, None, error.reason, error.detail)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return _WorkerResult(record, None, type(error).__name__, str(error))


def _write_result(writer: CorpusWriter, result: _WorkerResult) -> None:
    if result.pair is not None:
        writer.add(result.record, result.pair)
    else:
        reason = result.error_type or "worker_error"
        writer.quarantine(
            result.record, QualityError(reason, result.error_detail or "")
        )


def _records(path: Path) -> Iterator[Json]:
    rows: list[Json] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value: object = json.loads(line)
            if not isinstance(value, dict):
                detail = f"expected JSON object in {path}"
                raise TypeError(detail)
            rows.append(cast("Json", value))
    yield from sorted(rows, key=lambda row: str(row["sample_id"]))
