"""Out-of-sample evaluation of covariance forecasts (FR-10, D-12, AC-W-03).

For each horizon ``h`` the origins are every ``h``-th trading date from the OOS start, so the
realized windows don't overlap. At origin ``t`` each estimator forecasts the ``h``-day variance
``F = w' Σ_t w · h / 252`` of a set of portfolios fixed at ``t``. The portfolios are 1/N,
100 random long-only Dirichlet(1) portfolios (the same for every estimator), and each estimator's
own capped minimum-variance portfolio. They are scored on the next ``h`` days:

- Bias statistic: ``b = std(r / sqrt(F))`` over origins, with 95% band ``1 ± 1.96 sqrt(1/(2T))``.
- QLIKE (Patton 2011): ``mean(log F + RV / F)``, with ``RV`` the sum of squared daily portfolio
  returns over the window (lower is better).
- Realized volatility of the min-variance portfolio, annualized.
"""

from __future__ import annotations

import calendar as cal_module
import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import numpy.typing as npt
import polars as pl

from regport.config import ExperimentConfig
from regport.logs import get_logger
from regport.opt.minvar import SolverError, min_variance
from regport.risk.covariance import ANNUALIZATION, Estimator, trailing_window, wide_returns

log = get_logger(__name__)
F64 = npt.NDArray[np.float64]
N_SPECIAL = 2  # columns 0 = 1/N, 1 = own min-variance; random portfolios follow


def bias_statistic(realized: F64, forecast_vol: F64) -> tuple[float, float, float]:
    """``(b, lo95, hi95)``: std of standardized returns and its 95% band under a correct model."""
    z = np.asarray(realized, dtype=np.float64) / np.asarray(forecast_vol, dtype=np.float64)
    z = z[np.isfinite(z)]
    t = z.size
    if t < 2:
        return float("nan"), float("nan"), float("nan")
    half = 1.96 * math.sqrt(1.0 / (2.0 * t))
    return float(np.std(z, ddof=1)), 1.0 - half, 1.0 + half


def qlike(realized_var: F64, forecast_var: F64) -> float:
    """Patton (2011) QLIKE loss ``mean(log F + RV / F)``; robust to a noisy variance proxy."""
    rv = np.asarray(realized_var, dtype=np.float64)
    f = np.asarray(forecast_var, dtype=np.float64)
    ok = np.isfinite(rv) & np.isfinite(f) & (f > 0)
    return float(np.mean(np.log(f[ok]) + rv[ok] / f[ok]))


def resolve_oos_end(calendar: list[date], oos_end: date | None) -> date:
    """``oos_end``, or the last trading date of the last complete month in ``calendar`` (D-13).

    The final month of the data counts as complete only if the data reach its last weekday.
    """
    if oos_end is not None:
        return max(d for d in calendar if d <= oos_end)
    last = calendar[-1]
    days_in_month = cal_module.monthrange(last.year, last.month)[1]
    last_weekday = max(
        date(last.year, last.month, d)
        for d in range(days_in_month - 6, days_in_month + 1)
        if date(last.year, last.month, d).weekday() < 5
    )
    if last >= last_weekday:
        return last
    return max(d for d in calendar if (d.year, d.month) < (last.year, last.month))


@dataclass
class OriginScores:
    """Per-origin standardized returns and losses for one estimator and horizon."""

    origin: list[date] = field(default_factory=list)
    end: list[date] = field(default_factory=list)
    n_assets: list[int] = field(default_factory=list)
    ret: list[F64] = field(default_factory=list)  # h-day portfolio returns, one row per origin
    fvar: list[F64] = field(default_factory=list)  # forecast h-day variances
    rv: list[F64] = field(default_factory=list)  # realized variances

    def arrays(self) -> tuple[F64, F64, F64]:
        return np.vstack(self.ret), np.vstack(self.fvar), np.vstack(self.rv)


@dataclass
class CovEvalResult:
    estimators: list[str]
    horizons: list[int]
    scores: dict[tuple[str, int], OriginScores]
    excluded_events: int
    solver_failures: int
    oos_start: date
    oos_end: date
    selection_end: date


def run_cov_eval(
    returns_daily: pl.DataFrame,
    cfg: ExperimentConfig,
    estimators: list[str] | None = None,
    horizons: list[int] | None = None,
) -> CovEvalResult:
    """Score every estimator at every horizon on non-overlapping OOS windows."""
    names = estimators or ["sample", "ewma", "ledoit_wolf", "oas"]
    hs = horizons or list(cfg.risk.eval_horizons_days)
    ests = [Estimator(n, cfg.risk.ewma_half_life_days) for n in names]
    dates, symbols, r, elig = wide_returns(returns_daily)
    oos_end = resolve_oos_end(dates, cfg.sample.oos_end)
    i0 = next(i for i, d in enumerate(dates) if d >= cfg.sample.oos_start)
    i_end = dates.index(oos_end)
    origins = sorted({i0 + k * h for h in hs for k in range((i_end - i0) // h + 1)})
    n_random = cfg.risk.n_random_portfolios
    seed = cfg.seed
    cache: dict[int, tuple[list[int], list[F64], list[F64]]] = {}
    excluded = solver_failures = 0
    for i in origins:
        win = trailing_window(dates, symbols, r, elig, dates[i], cfg.risk.window_days)
        excluded += len(win.excluded)
        cols = [symbols.index(s) for s in win.symbols]
        covs, mvs = [], []
        for est in ests:
            cov = est.estimate(win.returns, dates[i])
            try:
                w = min_variance(cov, cfg.optimizer.max_weight)
            except SolverError:
                solver_failures += 1
                w = np.full(len(cols), 1.0 / len(cols))
            covs.append(cov)
            mvs.append(w)
        cache[i] = (cols, covs, mvs)
    if excluded:
        log.info("cov_eval_excluded_assets", events=excluded)
    scores: dict[tuple[str, int], OriginScores] = {
        (e.name, h): OriginScores() for e in ests for h in hs
    }
    for h in hs:
        for k in range((i_end - i0) // h + 1):
            i = i0 + k * h
            if i + h > i_end:
                break
            cols, covs, mvs = cache[i]
            block = r[i + 1 : i + h + 1][:, cols]
            if np.isnan(block).any():
                continue
            n = len(cols)
            rng = np.random.default_rng([seed, h, i])
            weights = np.vstack(
                [np.full(n, 1.0 / n), np.zeros(n), rng.dirichlet(np.ones(n), size=n_random)]
            )
            gross = np.prod(1.0 + block, axis=0) - 1.0
            for est, cov, mv in zip(ests, covs, mvs, strict=True):
                weights[1] = mv
                port_ret = weights @ gross
                daily = block @ weights.T
                fvar = np.einsum("ij,jk,ik->i", weights, cov, weights) * h / ANNUALIZATION
                s = scores[(est.name, h)]
                s.origin.append(dates[i])
                s.end.append(dates[i + h])
                s.n_assets.append(n)
                s.ret.append(port_ret)
                s.fvar.append(fvar)
                s.rv.append((daily**2).sum(axis=0))
    return CovEvalResult(
        names,
        hs,
        scores,
        excluded,
        solver_failures,
        dates[i0],
        oos_end,
        cfg.sample.estimator_selection_end,
    )


SAMPLES = ("selection", "second_half", "full")


def _mask(s: OriginScores, sample: str, selection_end: date) -> npt.NDArray[np.bool_]:
    end = np.array(s.end, dtype="datetime64[D]")
    origin = np.array(s.origin, dtype="datetime64[D]")
    cut = np.datetime64(selection_end, "D")
    if sample == "selection":
        return end <= cut
    if sample == "second_half":
        return origin > cut
    return np.ones(end.size, dtype=bool)


def summarize(result: CovEvalResult) -> pl.DataFrame:
    """One row per estimator, horizon and sample (the ``reports/tables/cov_eval.csv`` table)."""
    rows = []
    for (name, h), s in result.scores.items():
        if not s.origin:
            continue
        ret, fvar, rv = s.arrays()
        for sample in SAMPLES:
            m = _mask(s, sample, result.selection_end)
            if m.sum() < 2:
                continue
            vol = np.sqrt(fvar[m])
            b_1n, lo, hi = bias_statistic(ret[m, 0], vol[:, 0])
            b_mv, _, _ = bias_statistic(ret[m, 1], vol[:, 1])
            b_random = [
                bias_statistic(ret[m, j], vol[:, j])[0] for j in range(N_SPECIAL, ret.shape[1])
            ]
            eval_cols = [0, *range(N_SPECIAL, ret.shape[1])]
            ql = [qlike(rv[m, j], fvar[m, j]) for j in eval_cols]
            rows.append(
                {
                    "estimator": name,
                    "horizon_days": h,
                    "sample": sample,
                    "n_windows": int(m.sum()),
                    "b_equal_weight": b_1n,
                    "b_random_median": float(np.median(b_random)),
                    "b_random_p05": float(np.quantile(b_random, 0.05)),
                    "b_random_p95": float(np.quantile(b_random, 0.95)),
                    "b_min_variance": b_mv,
                    "band_lo95": lo,
                    "band_hi95": hi,
                    "qlike_mean": float(np.mean(ql)),
                    "qlike_equal_weight": ql[0],
                    "min_variance_realized_vol": float(
                        np.std(ret[m, 1], ddof=1) * math.sqrt(ANNUALIZATION / h)
                    ),
                    "min_variance_forecast_vol": float(
                        np.mean(vol[:, 1]) * math.sqrt(ANNUALIZATION / h)
                    ),
                    "mean_n_assets": float(np.mean(np.array(s.n_assets)[m])),
                }
            )
    return pl.DataFrame(rows).sort("horizon_days", "sample", "estimator")


def select_estimator(table: pl.DataFrame, horizon: int = 21) -> str:
    """D-12 rule fixed in docs/decisions.md: lowest mean QLIKE at ``horizon`` on the first half."""
    sel = table.filter((pl.col("sample") == "selection") & (pl.col("horizon_days") == horizon))
    if sel.is_empty():
        raise ValueError("no selection-period windows to choose an estimator from")
    return str(sel.sort("qlike_mean", "estimator")["estimator"][0])


def ac_w_03(table: pl.DataFrame, chosen: str, horizon: int = 21) -> tuple[bool, float, float]:
    """AC-W-03 on the full OOS: b in [0.90, 1.10] for 1/N and the median random portfolio."""
    row = table.filter(
        (pl.col("estimator") == chosen)
        & (pl.col("horizon_days") == horizon)
        & (pl.col("sample") == "full")
    )
    b_1n = float(row["b_equal_weight"][0])
    b_rand = float(row["b_random_median"][0])
    return (0.90 <= b_1n <= 1.10 and 0.90 <= b_rand <= 1.10), b_1n, b_rand
