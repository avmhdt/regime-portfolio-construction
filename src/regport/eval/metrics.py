"""Performance metrics (FR-26) of a daily return series."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
DAYS = 252


def max_drawdown(ret: F64) -> float:
    """Largest peak-to-trough fall of the compounded wealth path, as a negative fraction."""
    wealth = np.cumprod(1.0 + np.asarray(ret, dtype=np.float64))
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return float(np.min(wealth / peak - 1.0))


def performance_metrics(
    ret: F64,
    rf: F64,
    turnover: F64,
    cost: F64,
    weights_sq_sum: F64,
    ceq_risk_aversion: float = 1.0,
    benchmark: F64 | None = None,
) -> dict[str, Any]:
    """FR-26 metrics; ``ret`` is the daily net return, ``rf`` the daily risk-free rate.

    - Annual return: geometric, ``(Π(1+r))^(252/T) − 1``; volatility ``std · sqrt(252)``.
    - Sharpe and Sortino use excess returns ``r − rf``; Sortino's downside deviation is
      ``sqrt(mean(min(r − rf, 0)²))``.
    - Calmar: annual return over the absolute maximum drawdown.
    - CEQ: ``252 · (mean(r − rf) − (γ/2) var(r − rf))`` (DeMiguel, Garlappi and Uppal 2009).
    - Turnover: ``Σ|trade|`` per year (the spec's one-way turnover, fractions of NAV); cost drag
      is total cost per year.
    - Effective number of assets: mean over days of ``1 / Σ w²``.
    - Hit rate: share of days with ``r > benchmark`` (only when a benchmark is given).
    """
    r = np.asarray(ret, dtype=np.float64)
    ex = r - np.asarray(rf, dtype=np.float64)
    years = r.size / DAYS
    ann = math.expm1(float(np.log1p(r).sum()) / years)
    vol = float(np.std(r, ddof=1)) * math.sqrt(DAYS)
    ex_sd = float(np.std(ex, ddof=1))
    downside = math.sqrt(float(np.mean(np.minimum(ex, 0.0) ** 2)))
    mdd = max_drawdown(r)
    centered = ex - ex.mean()
    m2 = float(np.mean(centered**2))
    out: dict[str, Any] = {
        "n_obs": int(r.size),
        "annual_return": ann,
        "annual_volatility": vol,
        "sharpe_excess": float(ex.mean()) / ex_sd * math.sqrt(DAYS) if np.ptp(ex) > 0 else 0.0,
        "sortino": float(ex.mean()) / downside * math.sqrt(DAYS) if downside > 0 else 0.0,
        "max_drawdown": mdd,
        "calmar": ann / abs(mdd) if mdd < 0 else float("inf"),
        "ceq": DAYS * (float(ex.mean()) - ceq_risk_aversion / 2.0 * float(np.var(ex, ddof=1))),
        "skewness": float(np.mean(centered**3)) / m2**1.5 if m2 > 0 else 0.0,
        "excess_kurtosis": float(np.mean(centered**4)) / m2**2 - 3.0 if m2 > 0 else 0.0,
        "annual_turnover": float(np.sum(turnover)) / years,
        "annual_cost_drag": float(np.sum(cost)) / years,
        "effective_n_assets": float(np.mean(1.0 / weights_sq_sum)),
        "hit_rate_vs_benchmark": None,
    }
    if benchmark is not None:
        out["hit_rate_vs_benchmark"] = float(np.mean(r > np.asarray(benchmark, dtype=np.float64)))
    if not math.isfinite(out["calmar"]):
        out["calmar"] = None
    return out
