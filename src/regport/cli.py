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


GridOption = Annotated[
    Path, typer.Option("--config", help="Grid config (YAML).", exists=True, dir_okay=False)
]
RunsOption = Annotated[
    str | None, typer.Option("--runs", help="A grid manifest path (default: the latest grid).")
]


@app.command()
def grid(config: GridOption = Path("configs/ablation.yaml")) -> None:
    """Run every cell of the ablation grid (FR-28); each cell is one trial-log row."""
    from regport.eval.grid import load_grid, run_grid

    settings = get_settings()
    configure_logging(settings.regport_log_level)
    load_config(load_grid(config).base)
    typer.echo(f"manifest {run_grid(config, settings)}")


def _manifest(settings: Settings, runs: str | None) -> Path:
    from regport.eval.grid import latest_manifest

    return Path(runs) if runs else latest_manifest(settings.results_dir)


@app.command()
def evaluate(
    config: ConfigOption = DEFAULT_CONFIG,
    runs: RunsOption = None,
    latest_grid: Annotated[bool, typer.Option("--latest-grid", help="Use the latest grid.")] = True,
    benchmark: Annotated[str, typer.Option("--benchmark")] = "equal_weight",
) -> None:
    """Metrics, Ledoit-Wolf tests against 1/N, Holm, ablation, trials and the constraint audit."""
    from regport.eval.evaluate import compare, constraint_audit, load_runs, trial_summary
    from regport.report.evaluation import write_evaluation

    cfg, settings, _ = _setup(config)
    if benchmark != cfg.evaluation.benchmark:
        raise typer.BadParameter("only the configured benchmark (equal_weight) is supported")
    manifest = _manifest(settings, runs)
    loaded, meta = load_runs(manifest)
    table, ablation = compare(loaded, cfg, settings.regport_n_jobs)
    log_path = Path(cfg.logging.experiment_log)
    trials = trial_summary(log_path, table, loaded)
    audit = constraint_audit(loaded, cfg.optimizer.max_weight, log_path)
    head = meta.get("headline") or {"optimizer": cfg.optimizer.name, "regime": "hmm_A"}
    headline = f"{head['optimizer']}+{head['regime']}"
    dates = loaded[0].returns["date"]
    for name, path in write_evaluation(
        table,
        ablation,
        trials,
        audit,
        loaded,
        headline,
        (str(dates[0]), str(dates[-1])),
        settings.reports_dir,
    ).items():
        typer.echo(f"{name:<13} {path}")


@app.command()
def stress(config: ConfigOption = DEFAULT_CONFIG, runs: RunsOption = None) -> None:
    """Drawdown and recovery statistics in the fixed stress windows (FR-29)."""
    import json

    import polars as pl

    from regport.eval.evaluate import load_runs
    from regport.regime.runner import hmm_paths, load_features, run_bocpd, run_hmm
    from regport.report.regimes import high_volatility_states, params_from_record
    from regport.report.stress import write_stress

    cfg, settings, _ = _setup(config)
    loaded, meta = load_runs(_manifest(settings, runs))
    feats = load_features(cfg, settings.data_dir / "processed")
    out_dir = settings.results_dir / "regime"
    probs, _, _ = run_hmm(cfg, feats, out_dir, settings.regport_n_jobs)
    fits = json.loads(hmm_paths(cfg, feats, out_dir)[1].read_text(encoding="utf-8"))["fits"]
    high = high_volatility_states(params_from_record(fits[str(max(int(k) for k in fits))]))
    stress_prob = (
        probs.filter(pl.col("state").is_in(high.tolist()))
        .group_by("date", maintain_order=True)
        .agg(pl.col("p_filtered").sum().alias("p_stress"))
    )
    cp = run_bocpd(cfg, feats, out_dir)
    head = meta.get("headline") or {"optimizer": cfg.optimizer.name, "regime": "hmm_A"}
    for name, path in write_stress(
        loaded, cfg, f"{head['optimizer']}+{head['regime']}", stress_prob, cp, settings.reports_dir
    ).items():
        typer.echo(f"{name:<11} {path}")


@app.command()
def robustness(
    config: ConfigOption = DEFAULT_CONFIG,
    suite: Annotated[
        str, typer.Option("--suite", help="costs | common_start | french | alfred | all.")
    ] = "all",
    report_only: Annotated[
        bool,
        typer.Option("--report-only", help="Rebuild the report from saved tables; no new runs."),
    ] = False,
) -> None:
    """Robustness suites (FR-30); every run is logged as a trial."""
    import polars as pl

    from regport.backtest.engine import Panel
    from regport.eval import robustness as rb
    from regport.report.robustness import write_robustness

    cfg, settings, _ = _setup(config)
    all_suites = ["costs", "common_start", "french", "alfred"]
    suites = all_suites if suite == "all" else [suite]
    if any(s not in all_suites for s in suites):
        raise typer.BadParameter(f"unknown suite {suite}")
    saved = settings.reports_dir / "tables"
    tables: dict[str, pl.DataFrame] = {}
    for name in all_suites:  # keep earlier suites in the report when only some are re-run
        path = saved / f"robustness_{name}.csv"
        if path.exists() and (report_only or name not in suites):
            tables[name] = pl.read_csv(path)
    notes = {
        "french": (
            "Value-weighted daily industry returns from 1990-01-02, uniform one-way cost "
            f"{rb.FRENCH_COST_BPS:g} bps, risk-free rate from the FF RF series. The regime "
            "features use the FF market factor and BAA10Y from 1986, so the HMM's first fit "
            "comes after 2,520 days; before it the hmm arm runs on the base estimator."
        ),
        "alfred": (
            "The headline optimizer (ERC) with feature set B, macro features as revised today "
            "instead of as published. Compare with `erc+hmm_B` in reports/oos_vs_1n.md (point "
            "in time)."
        ),
    }
    regime_diff = None
    if not report_only:
        panel = Panel.load(settings.data_dir / "processed")
        for name in suites:
            if name == "costs":
                tables[name] = rb.summarize(rb.costs_suite(cfg, settings, panel))
            elif name == "common_start":
                tables[name] = rb.summarize(rb.common_start_suite(cfg, settings, panel))
            elif name == "french":
                tables[name] = rb.summarize(rb.french_suite(cfg, settings))
            else:
                runs, regime_diff = rb.alfred_suite(cfg, settings, panel)
                tables[name] = rb.summarize(runs)
    if regime_diff is None and "alfred" in tables:
        regime_diff = rb.regime_difference(*rb.alfred_configs(cfg), settings)
    break_even = rb.break_even(tables["costs"]) if "costs" in tables else None
    ordered = {k: tables[k] for k in all_suites if k in tables}
    for key, path in write_robustness(
        ordered, break_even, regime_diff, notes, settings.reports_dir
    ).items():
        typer.echo(f"{key:<13} {path}")


bench_app = typer.Typer(no_args_is_help=True, help="Benchmarks (FR-31).")
app.add_typer(bench_app, name="bench")


@bench_app.command("scaling")
def bench_scaling(
    config: ConfigOption = DEFAULT_CONFIG,
    n: Annotated[str, typer.Option("--n", help="Universe sizes.")] = "12,49,100,250,500",
    repeats: Annotated[int, typer.Option("--repeats")] = 5,
) -> None:
    """Runtime, memory and conditioning from 12 to 500 assets (AC-H-05)."""
    import numpy as np
    import polars as pl

    from regport.eval.robustness import french_panel
    from regport.eval.scaling import WINDOW, scaling_table
    from regport.report.scaling import write_scaling

    cfg, settings, _ = _setup(config)
    extra = {}
    processed = settings.data_dir / "processed"
    if (processed / "french_daily.parquet").exists():
        panel = french_panel(processed, "ind49_daily")
        block = panel.returns[-WINDOW:]
        extra["49 industries"] = block[:, ~np.isnan(block).any(axis=0)]
    table, cond = scaling_table([int(v) for v in n.split(",")], repeats, cfg.seed, extra)
    hrp_vs_mv = None
    rob = settings.reports_dir / "tables" / "robustness_french.csv"
    if rob.exists():
        r = pl.read_csv(rob).filter(
            (pl.col("variant") == "ind49_daily")
            & pl.col("optimizer").is_in(["hrp", "min_variance"])
        )
        hrp_vs_mv = r.select("optimizer", "regime", "annual_volatility", "sharpe", "max_drawdown")
    for key, path in write_scaling(table, cond, hrp_vs_mv, settings.reports_dir).items():
        typer.echo(f"{key:<9} {path}")


if __name__ == "__main__":  # pragma: no cover
    app()
