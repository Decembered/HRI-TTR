"""Bounded independent Human and G1 tokenizer training commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, cast

import torch
import typer

from hri_ttr.checkpoints.kinds import ModelKind
from hri_ttr.commands.common import (
    Tokenizer,
    fail,
    load_prepared,
    model_from_config,
    sha256_file,
)
from hri_ttr.data.corpus import CorpusDomain, CorpusWindowDataset
from hri_ttr.representations.g1.normalizer import G1FeatureNormalizer
from hri_ttr.representations.human.normalizer import HumanFeatureNormalizer
from hri_ttr.training import (
    FeatureSequence,
    TrainConfig,
    TrainingIdentity,
    TrainingInvocation,
    WindowConfig,
    train,
    train_datasets,
)

app = typer.Typer(no_args_is_help=True)


def _new_model(config: TrainConfig) -> Tokenizer:
    """Seed torch before constructing a scratch model on every rank."""
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    return model_from_config(config)


def _run(  # noqa: PLR0913, PLR0917 - independent artifact boundaries.
    domain: ModelKind,
    config_path: Path,
    prepared_path: Path,
    identity_paths: tuple[Path, Path, Path],
    max_steps: int | None,
    output_dir: Path | None,
    resume: Path | None,
) -> None:
    config = TrainConfig.load_json(config_path)
    if config.model_kind is not domain:
        fail(f"{domain.value} command requires a {domain.value} config")
    updates = config.model_dump()
    if max_steps is not None:
        updates["max_steps"] = max_steps
    if output_dir is not None:
        updates["output_dir"] = output_dir
    config = TrainConfig.model_validate(updates)
    prepared = load_prepared(prepared_path)
    features = (
        prepared.human_features
        if domain is ModelKind.HUMAN
        else prepared.g1_features.astype("float32")
    )
    frame_mask = prepared.human_mask if domain is ModelKind.HUMAN else prepared.g1_mask
    sequence = FeatureSequence(prepared.metadata.sequence_id, features, frame_mask)
    normalizer, split, source = identity_paths
    invocation = TrainingInvocation(
        config,
        TrainingIdentity(
            normalizer_sha256=sha256_file(normalizer),
            split_sha256=sha256_file(split),
            source_sha256=sha256_file(source),
        ),
        resume,
    )
    try:
        result = train(_new_model(config), (sequence,), (sequence,), invocation)
    except (OSError, RuntimeError, ValueError) as error:
        fail(str(error) or "training checkpoint is unreadable", cause=error)
    typer.echo(
        json.dumps(
            {
                "domain": domain.value,
                "global_step": result.global_step,
                "best_checkpoint": str(result.best_checkpoint),
                "quality_claim": "none_code_smoke_only",
            },
            sort_keys=True,
        )
    )


def _run_corpus(  # noqa: PLR0913, PLR0917 - independent artifact boundaries.
    domain: ModelKind,
    config_path: Path,
    corpus: Path,
    max_steps: int | None,
    output_dir: Path | None,
    resume: Path | None,
) -> None:
    config = TrainConfig.load_json(config_path)
    if config.model_kind is not domain:
        fail(f"{domain.value} command requires a {domain.value} config")
    updates = config.model_dump()
    if max_steps is not None:
        updates["max_steps"] = max_steps
    if output_dir is not None:
        updates["output_dir"] = output_dir
    config = TrainConfig.model_validate(updates)
    normalizer_path = corpus / "normalizers" / f"{domain.value}.json"
    normalizer = (
        HumanFeatureNormalizer.load(normalizer_path)
        if domain is ModelKind.HUMAN
        else G1FeatureNormalizer.load(normalizer_path)
    )
    window = WindowConfig(config.window_frames, config.window_stride)
    corpus_domain: CorpusDomain = domain.value
    training_data = CorpusWindowDataset(
        corpus, "train", corpus_domain, window, normalizer
    )
    validation_data = CorpusWindowDataset(
        corpus, "val", corpus_domain, window, normalizer
    )
    identity = TrainingIdentity(
        normalizer_sha256=sha256_file(normalizer_path),
        split_sha256=sha256_file(corpus / "manifest" / "split_assignment.jsonl"),
        source_sha256=sha256_file(corpus / "checksums" / "SHA256SUMS"),
    )
    invocation = TrainingInvocation(config, identity, resume)
    try:
        result = train_datasets(
            _new_model(config), training_data, validation_data, invocation
        )
    except (OSError, RuntimeError, ValueError) as error:
        fail(str(error) or "training checkpoint is unreadable", cause=error)
    typer.echo(
        json.dumps(
            {
                "domain": domain.value,
                "global_step": result.global_step,
                "best_checkpoint": str(result.best_checkpoint),
                "quality_claim": "strict_same_motion_corpus",
            },
            sort_keys=True,
        )
    )


ConfigOption = Annotated[Path, typer.Option(exists=True, dir_okay=False)]
PreparedOption = Annotated[
    Path | None, typer.Option("--prepared", exists=True, dir_okay=False)
]
CorpusOption = Annotated[
    Path | None, typer.Option("--corpus", exists=True, file_okay=False)
]
ArtifactOption = Annotated[Path | None, typer.Option(exists=True, dir_okay=False)]
ResumeOption = Annotated[
    Path | None,
    typer.Option(
        exists=True,
        dir_okay=False,
        help="Resume an exactly matching training checkpoint.",
    ),
]


@app.command("human-vq")
def train_human(  # noqa: PLR0913, PLR0917 - CLI flags are independent boundaries.
    config: ConfigOption,
    prepared: PreparedOption = None,
    corpus: CorpusOption = None,
    normalizer: ArtifactOption = None,
    split: ArtifactOption = None,
    source_artifact: ArtifactOption = None,
    max_steps: Annotated[int | None, typer.Option(min=1)] = None,
    output_dir: Annotated[Path | None, typer.Option()] = None,
    resume: ResumeOption = None,
) -> None:
    """Run finite Human 262D causal-VQ steps; no quality claim is implied."""
    if corpus is not None:
        if prepared is not None:
            fail("choose exactly one of --corpus or --prepared")
        _run_corpus(ModelKind.HUMAN, config, corpus, max_steps, output_dir, resume)
        return
    if prepared is None or None in (normalizer, split, source_artifact):
        fail("legacy --prepared requires --normalizer, --split, and --source-artifact")
    identity_paths = cast(
        "tuple[Path, Path, Path]", (normalizer, split, source_artifact)
    )
    _run(
        ModelKind.HUMAN,
        config,
        prepared,
        identity_paths,
        max_steps,
        output_dir,
        resume,
    )


@app.command("g1-vq")
def train_g1(  # noqa: PLR0913, PLR0917 - CLI flags are independent boundaries.
    config: ConfigOption,
    prepared: PreparedOption = None,
    corpus: CorpusOption = None,
    normalizer: ArtifactOption = None,
    split: ArtifactOption = None,
    source_artifact: ArtifactOption = None,
    max_steps: Annotated[int | None, typer.Option(min=1)] = None,
    output_dir: Annotated[Path | None, typer.Option()] = None,
    resume: ResumeOption = None,
) -> None:
    """Run finite G1 75D causal-VQ steps; no quality claim is implied."""
    if corpus is not None:
        if prepared is not None:
            fail("choose exactly one of --corpus or --prepared")
        _run_corpus(ModelKind.G1, config, corpus, max_steps, output_dir, resume)
        return
    if prepared is None or None in (normalizer, split, source_artifact):
        fail("legacy --prepared requires --normalizer, --split, and --source-artifact")
    identity_paths = cast(
        "tuple[Path, Path, Path]", (normalizer, split, source_artifact)
    )
    _run(
        ModelKind.G1,
        config,
        prepared,
        identity_paths,
        max_steps,
        output_dir,
        resume,
    )
