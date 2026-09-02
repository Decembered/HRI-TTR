"""Unified command-line interface for the standalone HRI-TTR project."""

from __future__ import annotations

import typer

from hri_ttr.commands import cache, checkpoint, data, evaluate, export, train, visualize

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
app.add_typer(data.app, name="data")
app.add_typer(checkpoint.app, name="checkpoint")
app.add_typer(train.app, name="train")
app.add_typer(evaluate.app, name="evaluate")
app.add_typer(cache.app, name="cache")
app.add_typer(visualize.app, name="visualize")
app.add_typer(export.app, name="export")


@app.callback()
def command_root() -> None:
    """Prepare, train, evaluate, and export causal Human/G1 motion tokens."""


def main() -> None:
    """Run the HRI-TTR command-line interface."""
    try:
        app()
    except (OSError, RuntimeError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
