"""``regport robustness``: the preregistered robustness suites (FR-30, AC-H-06, AC-H-07).

- ``costs``: the four optimizers with and without the HMM layer, and 1/N, at uniform one-way costs
  of 0, 5, 10 and 25 bps; break-even cost against 1/N by linear interpolation.
- ``common_start``: the same cells starting on the first date all 12 ETFs are eligible (D-02 a).
- ``french``: Kenneth French 10 and 49 value-weighted industries (FR-30); regime features from the
  FF market factor and BAA10Y; uniform 5 bps; risk-free rate from the FF RF series.
- ``alfred``: feature set B with ALFRED point-in-time vs latest-revised macro (AC-H-07).
Every run is a trial in the log. Tests here are HAC (Ledoit-Wolf 2008) against 1/N.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from regport.backtest.engine import Panel, run_backtest
from regport.config import ExperimentConfig
from regport.eval.grid import Cell, RegimeArm, cell_config
from regport.eval.sharpe_test import sharpe_diff_hac
from regport.exec.costs import CostModel
from regport.regime.runner import RegimeOutputs, run_hmm
from regport.settings import Settings

OPTIMIZERS = ["min_variance", "erc", "hrp", "mv_turnover"]
HMM_A = RegimeArm(name="hmm_A", usage="hmm", feature_set="A")
NONE = RegimeArm(name="none", usage="none")
COST_GRID = (0.0, 5.0, 10.0, 25.0)
FRENCH_COST_BPS = 5.0
FRENCH_OOS_START = date(1990, 1, 2)
FRENCH_FEATURE_START = date(1986, 1, 2)


@dataclass(frozen=True)
class SuiteRun:
    suite: str
    variant: str
    optimizer: str
    regime: str
    run_id: str
    returns: pl.DataFrame
    metrics: dict[str, float]


def _cells(base: ExperimentConfig, **overrides: object) -> list[Cell]:
    cells = [
        Cell(o, arm.name, cell_config(base, o, arm, **overrides))
        for o in OPTIMIZERS
        for arm in (NONE, HMM_A)
    ]
    cells.append(Cell("equal_weight", "none", cell_config(base, "equal_weight", None, **overrides)))
    return cells


def _run(
    cells: list[Cell],
    suite: str,
    variant: str,
    settings: Settings,
    panel: Panel,
    regime: RegimeOutputs | None = None,
    costs: CostModel | None = None,
    panel_label: str | None = None,
) -> list[SuiteRun]:
    out = []
    for cell in cells:
        res = run_backtest(
            cell.cfg,
            settings,
            panel=panel,
            tag=f"robustness:{suite}:{variant}:{cell.key}",
            regime=regime if cell.regime != "none" else None,
            costs=costs,
            panel_label=panel_label,
        )
        out.append(
            SuiteRun(
                suite, variant, cell.optimizer, cell.regime, res.run_id, res.returns, res.metrics
            )
        )
    return out


def first_all_eligible(processed: Path) -> date:
    r = pl.read_parquet(processed / "returns_daily.parquet")
    n = r["symbol"].n_unique()
    counts = r.group_by("date", maintain_order=True).agg(pl.col("eligible").sum()).sort("date")
    first: date = counts.filter(pl.col("eligible") == n)["date"][0]
    return first


def costs_suite(base: ExperimentConfig, settings: Settings, panel: Panel) -> list[SuiteRun]:
    out: list[SuiteRun] = []
    for bps in COST_GRID:
        out += _run(_cells(base, costs__override_bps=bps), "costs", f"{bps:g}bps", settings, panel)
    return out


def common_start_suite(base: ExperimentConfig, settings: Settings, panel: Panel) -> list[SuiteRun]:
    start = first_all_eligible(settings.data_dir / "processed")
    data = base.model_dump()
    data["sample"]["oos_start"] = start
    shifted = ExperimentConfig.model_validate(data)
    return _run(_cells(shifted), "common_start", str(start), settings, panel)


def french_panel(processed: Path, dataset: str) -> Panel:
    f = pl.read_parquet(processed / "french_daily.parquet")
    ind = f.filter(pl.col("dataset") == dataset).sort("series", "date")
    ind = (
        ind.with_columns(
            pl.col("ret").log1p().alias("logret"),
            (
                pl.col("ret").is_not_null().cast(pl.Int32).cum_sum().over("series")
                - pl.col("ret").is_not_null().cast(pl.Int32)
            ).alias("n_valid_history"),
        )
        .with_columns(
            ((pl.col("n_valid_history") >= 252) & pl.col("ret").is_not_null()).alias("eligible")
        )
        .rename({"series": "symbol"})
    )
    rf = f.filter((pl.col("dataset") == "ff3_daily") & (pl.col("series") == "RF")).select(
        "date", pl.col("ret").alias("rf")
    )
    return Panel.from_frames(
        ind.select("date", "symbol", "ret", "logret", "n_valid_history", "eligible"), rf
    )


def french_features(processed: Path) -> pl.DataFrame:
    """Set A built on the FF market factor throughout (no SPY splice) and BAA10Y."""
    from regport.regime.features import known_daily_values

    f = pl.read_parquet(processed / "french_daily.parquet").filter(pl.col("dataset") == "ff3_daily")
    mkt = (
        f.pivot(on="series", index="date", values="ret")
        .filter(pl.col("date") >= FRENCH_FEATURE_START)
        .sort("date")
        .select("date", (pl.col("Mkt-RF") + pl.col("RF")).log1p().alias("spy_logret"))
    )
    cal = mkt["date"].to_list()
    baa = known_daily_values(pl.read_parquet(processed / "fred_daily.parquet"), "BAA10Y", cal)
    out = mkt.with_columns(
        pl.Series("baa10y_diff", np.diff(baa, prepend=np.nan)),
        pl.lit("ff_market").alias("spy_source"),
    )
    return out.filter(pl.col("baa10y_diff").is_finite())


def french_suite(base: ExperimentConfig, settings: Settings) -> list[SuiteRun]:
    processed = settings.data_dir / "processed"
    data = base.model_dump()
    data["sample"]["data_start"] = FRENCH_FEATURE_START
    data["sample"]["oos_start"] = FRENCH_OOS_START
    data["sample"]["estimator_selection_end"] = date(2004, 12, 31)
    cfg = ExperimentConfig.model_validate(data)
    feats = french_features(processed)
    probs, transmats, switches = run_hmm(
        cfg, feats, settings.results_dir / "regime", settings.regport_n_jobs
    )
    regime = RegimeOutputs(probs, transmats, None, feats, switches)
    out: list[SuiteRun] = []
    for dataset in ("ind10_daily", "ind49_daily"):
        panel = french_panel(processed, dataset)
        costs = CostModel.uniform(panel.symbols, FRENCH_COST_BPS)
        out += _run(
            _cells(cfg), "french", dataset, settings, panel, regime, costs, f"french:{dataset}"
        )
    return out


def regime_difference(
    cfg_pit: ExperimentConfig, cfg_latest: ExperimentConfig, settings: Settings
) -> dict[str, float]:
    """Label-invariant comparison of two walk-forward HMM runs (AC-H-07).

    The two runs are labelled independently, so state indices needn't correspond. Reported:
    the mean absolute difference of the stress probability (total probability of each run's
    high-volatility states at its last refit) and of the per-state probabilities after matching
    the two last fits' states by the Hungarian algorithm on symmetric KL divergence.
    """
    import json

    from regport.regime.labels import align_states
    from regport.regime.runner import hmm_paths, load_features
    from regport.report.regimes import high_volatility_states, params_from_record

    out_dir = settings.results_dir / "regime"
    probs, last = [], []
    for cfg in (cfg_pit, cfg_latest):
        feats = load_features(cfg, settings.data_dir / "processed")
        p, _, _ = run_hmm(cfg, feats, out_dir, settings.regport_n_jobs)
        fits = json.loads(hmm_paths(cfg, feats, out_dir)[1].read_text(encoding="utf-8"))["fits"]
        probs.append(p)
        last.append(params_from_record(fits[str(max(int(k) for k in fits))]))
    k = last[0].n_states
    wide = [p.sort("date", "state")["p_filtered"].to_numpy().reshape(-1, k) for p in probs]
    dates = [p["date"].unique(maintain_order=True).to_list() for p in probs]
    if dates[0] != dates[1]:
        raise ValueError("the two regime runs cover different dates")
    stress = [w[:, high_volatility_states(f)].sum(axis=1) for w, f in zip(wide, last, strict=True)]
    perm = align_states(last[0], last[1])
    aligned = wide[1][:, perm]
    return {
        "stress_probability_mad": float(np.mean(np.abs(stress[0] - stress[1]))),
        "aligned_state_probability_mad": float(np.mean(np.abs(wide[0] - aligned))),
        "argmax_agreement": float(np.mean(wide[0].argmax(axis=1) == aligned.argmax(axis=1))),
    }


def alfred_configs(base: ExperimentConfig) -> tuple[ExperimentConfig, ExperimentConfig]:
    pit = cell_config(base, "erc", RegimeArm(name="hmm_B", usage="hmm", feature_set="B"))
    latest = cell_config(
        base,
        "erc",
        RegimeArm(name="hmm_B_latest", usage="hmm", feature_set="B", macro_vintage="latest"),
    )
    return pit, latest


def alfred_suite(
    base: ExperimentConfig, settings: Settings, panel: Panel
) -> tuple[list[SuiteRun], dict[str, float]]:
    """ERC with feature set B on latest-revised macro data, plus the regime comparison."""
    pit, latest = alfred_configs(base)
    runs = _run([Cell("erc", "hmm_B_latest", latest)], "alfred", "latest", settings, panel)
    return runs, regime_difference(pit, latest, settings)


def summarize(runs: list[SuiteRun]) -> pl.DataFrame:
    """Net Sharpe, CEQ, turnover and the HAC test against the 1/N run of the same variant."""
    rows = []
    for (suite, variant), group in _groups(runs).items():
        bench = next((r for r in group if r.optimizer == "equal_weight"), None)
        for r in group:
            p = d = None
            if bench is not None and r is not bench:
                a = r.returns["ret_excess"].to_numpy()
                b = bench.returns["ret_excess"].to_numpy()
                res = sharpe_diff_hac(a, b)
                p, d = res.p_value, res.diff * math.sqrt(252)
            rows.append(
                {
                    "suite": suite,
                    "variant": variant,
                    "optimizer": r.optimizer,
                    "regime": r.regime,
                    "run_id": r.run_id,
                    "sharpe": r.metrics["sharpe_excess"],
                    "delta_sharpe_vs_1n": d,
                    "p_hac": p,
                    "ceq": r.metrics["ceq"],
                    "annual_volatility": r.metrics["annual_volatility"],
                    "annual_turnover": r.metrics["annual_turnover"],
                    "max_drawdown": r.metrics["max_drawdown"],
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None)


def _groups(runs: list[SuiteRun]) -> dict[tuple[str, str], list[SuiteRun]]:
    out: dict[tuple[str, str], list[SuiteRun]] = {}
    for r in runs:
        out.setdefault((r.suite, r.variant), []).append(r)
    return out


def break_even(costs_table: pl.DataFrame) -> pl.DataFrame:
    """Cost (bps) at which each strategy's net Sharpe equals 1/N's, by linear interpolation."""
    rows = []
    t = costs_table.with_columns(
        pl.col("variant").str.replace("bps", "").cast(pl.Float64).alias("bps")
    )
    bench = t.filter(pl.col("optimizer") == "equal_weight").sort("bps")
    for (opt, reg), g in t.filter(pl.col("optimizer") != "equal_weight").group_by(
        ["optimizer", "regime"], maintain_order=True
    ):
        g = g.sort("bps")
        gap = g["sharpe"].to_numpy() - bench["sharpe"].to_numpy()
        bps = g["bps"].to_numpy()
        text: str
        if gap[0] <= 0:
            text = "below 1/N at 0 bps"
        elif gap[-1] > 0:
            text = f"above 1/N up to {bps[-1]:g} bps"
        else:
            k = int(np.argmax(gap <= 0))
            x = bps[k - 1] + gap[k - 1] / (gap[k - 1] - gap[k]) * (bps[k] - bps[k - 1])
            text = f"{x:.1f} bps"
        rows.append({"optimizer": opt, "regime": reg, "break_even_cost": text})
    return pl.DataFrame(rows)
