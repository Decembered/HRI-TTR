"""Explicit checkpoint import commands."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 - Typer resolves runtime annotations.
from typing import Annotated

import typer
from pydantic import RootModel

from hri_ttr.checkpoints.baseline import write_g1_73d_baseline_manifest
from hri_ttr.checkpoints.import_human import (
    OfficialHumanImportSpec,
    import_official_human_checkpoint,
)
from hri_ttr.checkpoints.kinds import ModelKind
from hri_ttr.commands.common import fail, model_from_config
from hri_ttr.training import TrainConfig

app = typer.Typer(no_args_is_help=True)


class KeyMapping(RootModel[dict[str, str]]):
    """Strict source-to-target tensor key mapping file."""


@app.command("import")
def import_checkpoint(
    domain: Annotated[ModelKind, typer.Option(help="Import domain: human or g1.")],
    source: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    destination: Annotated[Path, typer.Option(help="Output checkpoint or manifest.")],
    config: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Human target config JSON."),
    ] = None,
    mapping: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Human key-map JSON object."),
    ] = None,
) -> None:
    """Import mapped Human weights or register a legacy G1 baseline."""
    match domain:
        case ModelKind.HUMAN:
            if config is None or mapping is None:
                fail("human import requires --config and --mapping")
            train_config = TrainConfig.load_json(config)
            if train_config.model_kind is not ModelKind.HUMAN:
                fail("human import requires a Human tokenizer config")
            key_mapping = KeyMapping.model_validate_json(
                mapping.read_text(encoding="utf-8")
            ).root
            report = import_official_human_checkpoint(
                model_from_config(train_config),
                OfficialHumanImportSpec(
                    source=source,
                    destination=destination,
                    key_mapping=key_mapping,
                ),
            )
            copied = sum(record.status.value == "copied" for record in report.records)
            typer.echo(
                json.dumps(
                    {"destination": str(destination), "copied_keys": copied},
                    sort_keys=True,
                )
            )
        case ModelKind.G1:
            if config is not None or mapping is not None:
                fail("G1 legacy import accepts no mapping or tokenizer config")
            manifest = write_g1_73d_baseline_manifest(source, destination)
            typer.echo(manifest.model_dump_json())
