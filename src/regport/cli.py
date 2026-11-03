"""Typer entry point: ``regport <command>`` (spec §10)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from regport.config import load_config
from regport.logs import configure_logging
from regport.settings import get_settings

app = typer.Typer(no_args_is_help=True, help="Regime detection and portfolio construction.")

ConfigOption = Annotated[
    Path, typer.Option("--config", help="Experiment config (YAML).", exists=True, dir_okay=False)
]
DEFAULT_CONFIG = Path("configs/base.yaml")


@app.callback()
def main() -> None:
    """Regime detection and portfolio construction, tested out of sample against 1/N."""


@app.command()
def smoke(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Data access smoke test: one FRED series, one ALFRED series, SPY from the price source."""
    from regport.smoke import run_smoke

    settings = get_settings()
    configure_logging(settings.regport_log_level)
    steps = run_smoke(load_config(config), settings)
    for step in steps:
        typer.echo(step.line())
    if not all(step.ok for step in steps):
        raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
