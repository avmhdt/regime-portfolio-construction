"""Typer entry point: ``regport <command>`` (spec §10)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from regport.config import ExperimentConfig, Universe, load_config, load_universe
from regport.logs import configure_logging
from regport.settings import Settings, get_settings

app = typer.Typer(no_args_is_help=True, help="Regime detection and portfolio construction.")
fetch_app = typer.Typer(no_args_is_help=True, help="Ingest raw data (§7e).")
data_app = typer.Typer(no_args_is_help=True, help="Build and check the point-in-time panel.")
app.add_typer(fetch_app, name="fetch")
app.add_typer(data_app, name="data")

ConfigOption = Annotated[
    Path, typer.Option("--config", help="Experiment config (YAML).", exists=True, dir_okay=False)
]
SourceOption = Annotated[
    str, typer.Option("--source", help="auto | sharadar | tiingo | stooq (D-01).")
]
SymbolsOption = Annotated[
    str | None, typer.Option("--symbols", help="Comma-separated subset, e.g. SPY,TLT.")
]
ForceOption = Annotated[bool, typer.Option("--force", help="Download again even if present.")]
DEFAULT_CONFIG = Path("configs/base.yaml")


@app.callback()
def main() -> None:
    """Regime detection and portfolio construction, tested out of sample against 1/N."""


def _setup(config: Path) -> tuple[ExperimentConfig, Settings, Universe]:
    settings = get_settings()
    configure_logging(settings.regport_log_level)
    cfg = load_config(config)
    return cfg, settings, load_universe(cfg.data.universe_file)


def _symbols(text: str | None) -> list[str] | None:
    return [s.strip().upper() for s in text.split(",") if s.strip()] if text else None


@app.command()
def smoke(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Data access smoke test: one FRED series, one ALFRED series, SPY from the price source."""
    from regport.smoke import run_smoke

    cfg, settings, _ = _setup(config)
    steps = run_smoke(cfg, settings)
    for step in steps:
        typer.echo(step.line())
    if not all(step.ok for step in steps):
        raise typer.Exit(code=1)


def _run_fetch(what: str, config: Path, source: str, symbols: str | None, force: bool) -> None:
    from regport.data import ingest
    from regport.errors import DataSourceError

    cfg, settings, universe = _setup(config)
    steps = {
        "prices": lambda: ingest.fetch_prices(
            cfg, settings, universe, source, _symbols(symbols), force
        ),
        "fred": lambda: ingest.fetch_fred(
            settings, cfg.macro.daily_series, vintages=False, force=force
        ),
        "alfred": lambda: ingest.fetch_fred(
            settings, cfg.macro.vintage_series, vintages=True, force=force
        ),
        "french": lambda: ingest.fetch_french(settings, force=force),
    }
    order = list(steps) if what == "all" else [what]
    try:
        for name in order:
            for result in steps[name]():
                typer.echo(result.line())
    except DataSourceError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from None


@fetch_app.command("prices")
def fetch_prices(
    config: ConfigOption = DEFAULT_CONFIG,
    source: SourceOption = "auto",
    symbols: SymbolsOption = None,
    force: ForceOption = False,
) -> None:
    """ETF prices from the D-01 source, plus the Stooq conversion used as the DQ-03 cross-check."""
    _run_fetch("prices", config, source, symbols, force)


@fetch_app.command("fred")
def fetch_fred(config: ConfigOption = DEFAULT_CONFIG, force: ForceOption = False) -> None:
    """Daily FRED series in ``macro.daily_series``."""
    _run_fetch("fred", config, "auto", None, force)


@fetch_app.command("alfred")
def fetch_alfred(config: ConfigOption = DEFAULT_CONFIG, force: ForceOption = False) -> None:
    """All ALFRED vintages of ``macro.vintage_series``."""
    _run_fetch("alfred", config, "auto", None, force)


@fetch_app.command("french")
def fetch_french(config: ConfigOption = DEFAULT_CONFIG, force: ForceOption = False) -> None:
    """Kenneth French FF3, 10- and 49-industry daily files."""
    _run_fetch("french", config, "auto", None, force)


@fetch_app.command("all")
def fetch_all(
    config: ConfigOption = DEFAULT_CONFIG,
    source: SourceOption = "auto",
    symbols: SymbolsOption = None,
    force: ForceOption = False,
) -> None:
    """Every fetch above, in order. AQR is a manual download (P-08) and is parsed from M6."""
    _run_fetch("all", config, source, symbols, force)


@data_app.command("build")
def data_build(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Build the point-in-time panel into ``processed/``."""
    from regport.data.panel import build_panel
    from regport.errors import DataSourceError

    cfg, settings, universe = _setup(config)
    try:
        written = build_panel(cfg, settings, universe)
    except DataSourceError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    for name, path in written.items():
        typer.echo(f"{name:<15} {path}")


@data_app.command("check")
def data_check(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Run DQ-01 to DQ-13 and AC-W-05; write ``reports/data_quality.md``; exit 1 on failure."""
    from regport.data.checks import run_checks, write_report

    cfg, settings, universe = _setup(config)
    results = run_checks(cfg, settings, universe)
    path = write_report(results, settings.reports_dir / "data_quality.md")
    for result in results.checks:
        typer.echo(result.line())
    typer.echo(f"report: {path}")
    if not results.ok:
        raise typer.Exit(code=1)


@app.command("cov-eval")
def cov_eval(
    config: ConfigOption = DEFAULT_CONFIG,
    estimators: Annotated[
        str, typer.Option("--estimators", help="Comma-separated: sample,ewma,ledoit_wolf,oas.")
    ] = "sample,ewma,ledoit_wolf,oas",
    horizons: Annotated[
        str, typer.Option("--horizons", help="Comma-separated horizons in trading days.")
    ] = "1,5,21",
) -> None:
    """Out-of-sample covariance evaluation (FR-10); writes reports/cov_eval.md and the table."""
    import polars as pl

    from regport.report.cov_eval import write_cov_eval
    from regport.risk.evaluation import run_cov_eval, summarize

    cfg, settings, _ = _setup(config)
    path = settings.data_dir / "processed" / "returns_daily.parquet"
    if not path.exists():
        typer.echo("error: run `regport data build` first", err=True)
        raise typer.Exit(code=1)
    result = run_cov_eval(
        pl.read_parquet(path),
        cfg,
        [e.strip() for e in estimators.split(",") if e.strip()],
        [int(h) for h in horizons.split(",") if h.strip()],
    )
    table = summarize(result)
    for name, out in write_cov_eval(
        result, table, settings.reports_dir, settings.results_dir
    ).items():
        typer.echo(f"{name:<7} {out}")


@app.command()
def backtest(
    config: ConfigOption = DEFAULT_CONFIG,
    optimizer: Annotated[
        str | None,
        typer.Option("--optimizer", help="min_variance | erc | hrp | mv_turnover | equal_weight."),
    ] = None,
    regime: Annotated[
        str | None, typer.Option("--regime", help="none | hmm | bocpd | hmm_bocpd.")
    ] = None,
    cost_bps: Annotated[
        float | None, typer.Option("--cost-bps", help="Uniform one-way cost override (bps).")
    ] = None,
    tag: Annotated[str, typer.Option("--tag", help="Free-text label stored with the run.")] = "",
) -> None:
    """One backtest; appends one row to the trial log and prints the run_id."""
    from regport.backtest.engine import BacktestError, run_backtest
    from regport.errors import ConstraintViolation

    cfg, settings, _ = _setup(config)
    data = cfg.model_dump()
    if optimizer is not None:
        data["optimizer"]["name"] = optimizer
    if regime is not None:
        data["regime"]["usage"] = regime
    if cost_bps is not None:
        data["costs"]["override_bps"] = cost_bps
    cfg = type(cfg).model_validate(data)
    try:
        result = run_backtest(cfg, settings, tag=tag)
    except (BacktestError, ConstraintViolation, ValueError, OSError) as exc:
        typer.echo(f"error: {exc} (logged as a failed run)", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(result.run_id)


regime_app = typer.Typer(no_args_is_help=True, help="Walk-forward regime models (FR-11 to FR-16).")
app.add_typer(regime_app, name="regime")


@regime_app.command("run")
def regime_run(
    config: ConfigOption = DEFAULT_CONFIG,
    model: Annotated[str, typer.Option("--model", help="hmm | bocpd | all.")] = "all",
    features: Annotated[
        str | None, typer.Option("--features", help="A | B (default: the config's set).")
    ] = None,
    force: ForceOption = False,
) -> None:
    """Filtered regime probabilities to results/regime/ (cached by settings and data)."""
    from regport.regime.runner import load_features, run_bocpd, run_hmm

    cfg, settings, _ = _setup(config)
    if features is not None:
        data = cfg.model_dump()
        data["regime"]["hmm"]["feature_set"] = features
        cfg = type(cfg).model_validate(data)
    processed = settings.data_dir / "processed"
    feats = load_features(cfg, processed)
    out_dir = settings.results_dir / "regime"
    proxy = int((feats["spy_source"] != "sharadar").sum())
    typer.echo(
        f"features {cfg.regime.hmm.feature_set}: {feats.height} days from "
        f"{feats['date'][0]}, {proxy} on the French market proxy for SPY"
    )
    if model in ("hmm", "all"):
        probs, transmats, switches = run_hmm(cfg, feats, out_dir, settings.regport_n_jobs, force)
        typer.echo(
            f"hmm: {len(transmats)} refits, {len(switches)} label switches, "
            f"{probs['date'].n_unique()} dates"
        )
    if model in ("bocpd", "all"):
        frame = run_bocpd(cfg, feats, out_dir, force)
        typer.echo(
            f"bocpd: {frame.height} dates, "
            f"{int((frame['cp_prob_short'] > cfg.regime.bocpd.trigger_prob).sum())} "
            "above the trigger"
        )


@regime_app.command("report")
def regime_report(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """reports/regimes.md and figures from the cached walk-forward outputs (FR-34, AC-H-09)."""
    from regport.regime.runner import hmm_paths, load_features, run_hmm
    from regport.report.regimes import write_regimes_report

    cfg, settings, _ = _setup(config)
    feats = load_features(cfg, settings.data_dir / "processed")
    out_dir = settings.results_dir / "regime"
    probs, _, _ = run_hmm(cfg, feats, out_dir, settings.regport_n_jobs)
    _, fits_path = hmm_paths(cfg, feats, out_dir)
    for name, path in write_regimes_report(
        cfg, feats, probs, fits_path, settings.reports_dir
    ).items():
        typer.echo(f"{name:<22} {path}")


if __name__ == "__main__":  # pragma: no cover
    app()
