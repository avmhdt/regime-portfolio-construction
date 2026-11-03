"""Walk-forward backtest engine (FR-24, DQ-06, FR-25).

Timing (DQ-06): weights are decided with data through the close of decision date ``t`` (the last
trading date of each month or week), executed at the close of ``t+1``, and earn from ``t+2``. Day
``t+1``'s return is earned by the drifted old weights, and the trade's cost is charged on ``t+1``.
Between trades the weights drift with asset returns. The first decision is the last one before
the OOS start; the portfolio starts in cash, so its first return day carries only the set-up cost.

Every invocation appends exactly one row to the trial log, including failed runs (FR-25).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import yaml

from regport.config import ExperimentConfig, Universe, load_universe
from regport.errors import ConstraintViolation
from regport.eval.metrics import performance_metrics
from regport.exec.costs import CostModel
from regport.exec.rebalance import decision_dates, rebalance
from regport.logs import get_logger
from regport.opt.constraints import check_constraints, project_capped
from regport.opt.equal_weight import equal_weight
from regport.opt.erc import ERCError, erc, risk_contributions
from regport.opt.hrp import hrp
from regport.opt.minvar import SolverError, min_variance
from regport.opt.mv_tc import james_stein_mean, mean_variance_tc
from regport.risk.covariance import ANNUALIZATION, Estimator, trailing_window, wide_returns
from regport.risk.evaluation import resolve_oos_end
from regport.settings import Settings
from regport.tracking.experiment_log import TrialRow, log_failed, log_trial

log = get_logger(__name__)
F64 = npt.NDArray[np.float64]
CAPPED = ("min_variance", "mv_turnover")
MAX_FALLBACK_SHARE = 0.01  # NFR-09
PERIODS = {"monthly": 12, "weekly": 52}


class BacktestError(RuntimeError):
    """The run failed (too many solver fallbacks, a constraint violation, missing data)."""


@dataclass
class Panel:
    """Wide daily panel: ``returns`` and ``eligible`` are ``T x N``, ``rf`` is length ``T``."""

    dates: list[date]
    symbols: list[str]
    returns: F64
    eligible: npt.NDArray[np.bool_]
    rf: F64

    @classmethod
    def load(cls, processed: Path) -> Panel:
        returns = pl.read_parquet(processed / "returns_daily.parquet")
        rf = pl.read_parquet(processed / "rf_daily.parquet")
        return cls.from_frames(returns, rf)

    @classmethod
    def from_frames(cls, returns_daily: pl.DataFrame, rf_daily: pl.DataFrame) -> Panel:
        dates, symbols, r, e = wide_returns(returns_daily)
        rf_map = dict(rf_daily.select("date", "rf").iter_rows())
        rf = np.array(
            [rf_map.get(d) if rf_map.get(d) is not None else np.nan for d in dates],
            dtype=np.float64,
        )
        return cls(dates, symbols, r, e, rf)


@dataclass
class BacktestResult:
    run_id: str
    row: TrialRow
    returns: pl.DataFrame
    weights: pl.DataFrame
    metrics: dict[str, Any]
    events: dict[str, int] = field(default_factory=dict)


def strategy_family(cfg: ExperimentConfig) -> str:
    return f"regport_{cfg.optimizer.name}"


def _cost_model(cfg: ExperimentConfig, universe: Universe) -> CostModel:
    if cfg.costs.override_bps is not None:
        return CostModel.uniform(universe.symbols, cfg.costs.override_bps)
    return CostModel({a.symbol: a.cost_bps for a in universe.assets})


@dataclass
class _Decision:
    symbols: list[str]
    target: F64  # over all panel symbols
    fallback: bool
    rc_gap: float | None


class _Strategy:
    def __init__(self, cfg: ExperimentConfig, costs: CostModel) -> None:
        self.cfg = cfg
        self.costs = costs
        self.estimator = Estimator(cfg.risk.estimator, cfg.risk.ewma_half_life_days)

    def decide(self, panel: Panel, i: int, w_drift: F64) -> _Decision:
        cfg, name = self.cfg, self.cfg.optimizer.name
        asof = panel.dates[i]
        n_all = len(panel.symbols)
        if name == "equal_weight":
            cols = np.flatnonzero(panel.eligible[i])
            symbols = [panel.symbols[c] for c in cols]
            w = equal_weight(cols.size) if cols.size else np.zeros(0)
            return self._full(cols, w, n_all, symbols, False, None)
        win = trailing_window(
            panel.dates,
            panel.symbols,
            panel.returns,
            panel.eligible,
            asof,
            cfg.risk.window_days,
            count_pre_inception=False,
        )
        if win.excluded:
            log.info("excluded_assets", asof=str(asof), symbols=win.excluded)
        cols = np.array([panel.symbols.index(s) for s in win.symbols], dtype=np.int64)
        if cols.size == 0:
            return self._full(cols, np.zeros(0), n_all, [], False, None)
        cov = self.estimator.estimate(win.returns, asof)
        prev = w_drift[cols]
        fallback, rc_gap = False, None
        try:
            if name == "min_variance":
                w = min_variance(cov, cfg.optimizer.max_weight)
            elif name == "erc":
                w = erc(cov)
                rc_gap = float(np.max(np.abs(risk_contributions(w, cov) - 1.0 / cols.size)))
            elif name == "hrp":
                w = hrp(cov, cfg.optimizer.hrp.linkage)
            else:
                mean_rows = win.returns[-cfg.optimizer.mv.mean_window_days :]
                mu = james_stein_mean(mean_rows, cov / ANNUALIZATION)
                kappa = (
                    self.costs.rates(win.symbols)
                    * cfg.optimizer.mv.turnover_penalty_mult
                    * PERIODS[cfg.execution.frequency]
                )
                start = prev / prev.sum() if prev.sum() > 0 else np.full(cols.size, 1 / cols.size)
                w = mean_variance_tc(
                    mu, cov, start, cfg.optimizer.mv.risk_aversion, kappa, cfg.optimizer.max_weight
                )
        except (SolverError, ERCError) as exc:
            fallback = True
            log.warning("solver_fallback", asof=str(asof), optimizer=name, error=str(exc))
            cap = cfg.optimizer.max_weight if name in CAPPED else None
            w = project_capped(prev if prev.sum() > 0 else equal_weight(cols.size), cap)
        check_constraints(w, cfg.optimizer.max_weight if name in CAPPED else None)
        return self._full(cols, w, n_all, win.symbols, fallback, rc_gap)

    @staticmethod
    def _full(
        cols: npt.NDArray[np.int64],
        w: F64,
        n_all: int,
        symbols: list[str],
        fallback: bool,
        rc_gap: float | None,
    ) -> _Decision:
        target = np.zeros(n_all)
        target[cols] = w
        return _Decision(symbols, target, fallback, rc_gap)


def simulate(
    cfg: ExperimentConfig, panel: Panel, universe: Universe
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Run the daily loop; returns the returns frame, the trade frame and run diagnostics."""
    costs = _cost_model(cfg, universe)
    rates = costs.rates(panel.symbols)
    strategy = _Strategy(cfg, costs)
    oos_end = resolve_oos_end(panel.dates, cfg.sample.oos_end)
    start = next(i for i, d in enumerate(panel.dates) if d >= cfg.sample.oos_start)
    end = panel.dates.index(oos_end)
    if start == 0:
        raise BacktestError("no decision date before the OOS start")
    decisions = set(decision_dates(panel.dates[: end + 1], cfg.execution.frequency))
    decisions.add(panel.dates[start - 1])
    n = len(panel.symbols)
    w = np.zeros(n)
    pending: _Decision | None = None
    rows: dict[str, list[Any]] = {k: [] for k in ("date", "ret_gross", "cost", "turnover", "w_sq")}
    trades: list[dict[str, Any]] = []
    fallbacks = rebalances = trades_done = missing_held = 0
    rc_gaps: list[float] = []
    if panel.dates[start - 1] in decisions:
        pending = strategy.decide(panel, start - 1, w)
        rebalances += 1
    for i in range(start, end + 1):
        r = panel.returns[i]
        held_nan = (w > 0) & np.isnan(r)
        missing_held += int(held_nan.sum())
        r = np.where(np.isnan(r), 0.0, r)
        gross = float(w @ r)
        w = w * (1.0 + r) / (1.0 + gross) if w.sum() > 0 else w
        cost = turnover = 0.0
        if pending is not None:
            post = rebalance(w, pending.target, cfg.execution.policy, cfg.execution.threshold_abs)
            if w.sum() == 0:
                post = pending.target.copy()  # first execution from cash
            trade = post - w
            turnover = float(np.abs(trade).sum())
            cost = float(rates @ np.abs(trade))
            if turnover > 0:
                trades_done += 1
            for j, sym in enumerate(panel.symbols):
                if w[j] != 0 or pending.target[j] != 0 or post[j] != 0:
                    trades.append(
                        {
                            "date": panel.dates[i],
                            "symbol": sym,
                            "w_drift": w[j],
                            "w_target": pending.target[j],
                            "w_post": post[j],
                            "trade": trade[j],
                        }
                    )
            if pending.fallback:
                fallbacks += 1
            if pending.rc_gap is not None:
                rc_gaps.append(pending.rc_gap)
            w = post
            pending = None
        rows["date"].append(panel.dates[i])
        rows["ret_gross"].append(gross)
        rows["cost"].append(cost)
        rows["turnover"].append(turnover)
        rows["w_sq"].append(float(w @ w) if w.sum() > 0 else 1.0)
        if panel.dates[i] in decisions and i < end:
            pending = strategy.decide(panel, i, w)
            rebalances += 1
    if rebalances and fallbacks / rebalances > MAX_FALLBACK_SHARE:
        raise BacktestError(
            f"solver fallbacks on {fallbacks} of {rebalances} rebalances (NFR-09 allows 1%)"
        )
    rf = panel.rf[start : end + 1]
    frame = (
        pl.DataFrame(rows)
        .with_columns(
            (pl.col("ret_gross") - pl.col("cost")).alias("ret"),
            pl.Series("rf", np.nan_to_num(rf, nan=0.0)),
        )
        .with_columns((pl.col("ret") - pl.col("rf")).alias("ret_excess"))
    )
    diagnostics = {
        "rebalances": rebalances,
        "trades": trades_done,
        "solver_fallbacks": fallbacks,
        "held_asset_missing_returns": missing_held,
        "erc_max_rc_gap": max(rc_gaps) if rc_gaps else None,
        "rf_missing_days": int(np.isnan(rf).sum()),
    }
    trade_frame = pl.DataFrame(
        trades,
        schema={
            "date": pl.Date,
            "symbol": pl.String,
            "w_drift": pl.Float64,
            "w_target": pl.Float64,
            "w_post": pl.Float64,
            "trade": pl.Float64,
        },
    )
    return frame, trade_frame, diagnostics


def run_backtest(
    cfg: ExperimentConfig,
    settings: Settings | None = None,
    panel: Panel | None = None,
    tag: str = "",
) -> BacktestResult:
    """One backtest; always appends exactly one trial-log row, even if it fails (FR-25)."""
    log_path = Path(cfg.logging.experiment_log)
    family = strategy_family(cfg)
    params = {**cfg.to_params(), "tag": tag, "config_hash": cfg.config_hash()}
    started = time.perf_counter()
    try:
        if cfg.regime.usage != "none":
            raise BacktestError(f"regime usage {cfg.regime.usage!r} arrives in M5")
        universe = load_universe(cfg.data.universe_file)
        if panel is None:
            settings = settings or Settings()
            panel = Panel.load(settings.data_dir / "processed")
        frame, trades, diag = simulate(cfg, panel, universe)
    except (BacktestError, ConstraintViolation, ValueError, KeyError) as exc:
        row = log_failed(family, params, f"failed: {type(exc).__name__}: {exc}", path=log_path)
        log.error("backtest_failed", run_id=row.run_id, error=str(exc))
        raise
    metrics = performance_metrics(
        frame["ret"].to_numpy(),
        frame["rf"].to_numpy(),
        frame["turnover"].to_numpy(),
        frame["cost"].to_numpy(),
        frame["w_sq"].to_numpy(),
        cfg.evaluation.ceq_risk_aversion,
    )
    metrics.update(diag)
    metrics["runtime_s"] = round(time.perf_counter() - started, 3)
    extra = {c: frame[c].to_numpy() for c in ("ret_gross", "cost", "rf", "ret_excess", "turnover")}
    row = log_trial(
        family,
        params,
        frame["date"].to_list(),
        frame["ret"].to_numpy(),
        path=log_path,
        extra=extra,
        note=tag,
    )
    run_dir = log_path.parent / "runs" / row.run_id
    trades.write_parquet(run_dir / "weights.parquet")
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(params, sort_keys=True), encoding="utf-8", newline="\n"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    log.info("backtest_done", run_id=row.run_id, family=family, sharpe=row.sharpe_annual)
    return BacktestResult(
        row.run_id,
        row,
        frame.drop("w_sq"),
        trades,
        metrics,
        {k: v for k, v in diag.items() if isinstance(v, int)},
    )
