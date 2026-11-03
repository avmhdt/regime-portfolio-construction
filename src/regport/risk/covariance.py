"""Covariance estimators behind one interface (FR-09).

All take a ``T x N`` matrix of daily simple returns and return an annualized (x 252), symmetric,
PSD-floored ``N x N`` matrix. ``trailing_window`` builds that matrix from the panel and is the only
place where rows are selected by date, so it enforces DQ-01.

- Sample: unbiased sample covariance (ddof = 1).
- EWMA: RiskMetrics (J.P. Morgan/Reuters 1996) zero-mean recursion
  ``S_t = λ S_{t-1} + (1 - λ) r_t r_t'`` with ``λ = 0.5^(1/half_life)``, started at zero and divided
  by ``1 - λ^T``, so the weights over the finite window sum to one (the bias correction).
- Ledoit-Wolf (2004) and OAS (Chen et al. 2010): scikit-learn's reference implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.covariance import OAS, LedoitWolf

from regport.errors import LookaheadError
from regport.risk.psd import nearest_psd

F64 = npt.NDArray[np.float64]
ANNUALIZATION = 252.0
ESTIMATORS = ("sample", "ewma", "ledoit_wolf", "oas")


def _check(returns: F64) -> F64:
    x = np.asarray(returns, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"need a T x N matrix with T >= 2, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("returns contain NaN or inf")
    return x


def sample_cov(returns: F64) -> F64:
    """Unbiased sample covariance, annualized."""
    x = _check(returns)
    return nearest_psd(np.atleast_2d(np.cov(x, rowvar=False, ddof=1)) * ANNUALIZATION)


def ewma_weights(n_obs: int, half_life: float) -> F64:
    """RiskMetrics weights ``(1 - λ) λ^(T-1-i) / (1 - λ^T)``, oldest first; they sum to one."""
    lam = 0.5 ** (1.0 / half_life)
    powers = lam ** np.arange(n_obs - 1, -1, -1, dtype=np.float64)
    weights: F64 = (1.0 - lam) * powers / (1.0 - lam**n_obs)
    return weights


def ewma_cov(returns: F64, half_life: float) -> F64:
    """Zero-mean EWMA covariance with bias-corrected weights, annualized."""
    x = _check(returns)
    w = ewma_weights(x.shape[0], half_life)
    return nearest_psd((x * w[:, None]).T @ x * ANNUALIZATION)


def ledoit_wolf_cov(returns: F64) -> F64:
    """Ledoit and Wolf (2004) shrinkage toward a scaled identity, annualized."""
    x = _check(returns)
    return nearest_psd(LedoitWolf(assume_centered=False).fit(x).covariance_ * ANNUALIZATION)


def oas_cov(returns: F64) -> F64:
    """Oracle approximating shrinkage (Chen, Wiesel, Eldar and Hero 2010), annualized."""
    x = _check(returns)
    return nearest_psd(OAS(assume_centered=False).fit(x).covariance_ * ANNUALIZATION)


class CovarianceEstimator(Protocol):
    name: str

    def estimate(self, returns: F64, asof: date) -> F64: ...  # rows dated <= asof


@dataclass(frozen=True)
class Estimator:
    """One of :data:`ESTIMATORS`; ``half_life`` is used by EWMA only."""

    name: str
    half_life: float = 63.0

    def __post_init__(self) -> None:
        if self.name not in ESTIMATORS:
            raise ValueError(f"unknown estimator {self.name!r}; choose from {ESTIMATORS}")

    def estimate(self, returns: F64, asof: date) -> F64:
        if self.name == "sample":
            return sample_cov(returns)
        if self.name == "ewma":
            return ewma_cov(returns, self.half_life)
        if self.name == "ledoit_wolf":
            return ledoit_wolf_cov(returns)
        return oas_cov(returns)


@dataclass(frozen=True)
class ReturnWindow:
    """Trailing returns ending at ``asof``; ``returns[i, j]`` is ``symbols[j]`` on ``dates[i]``."""

    asof: date
    dates: list[date]
    symbols: list[str]
    returns: F64
    excluded: list[str]


def wide_returns(
    returns_daily: pl.DataFrame,
) -> tuple[list[date], list[str], F64, npt.NDArray[np.bool_]]:
    """``returns_daily`` as ``(dates, symbols, T x N returns with NaN, T x N eligible)``."""
    frame = returns_daily.sort("symbol", "date")
    symbols = sorted(frame["symbol"].unique().to_list())
    ret = frame.pivot(on="symbol", index="date", values="ret").sort("date")
    elig = frame.pivot(on="symbol", index="date", values="eligible").sort("date")
    dates: list[date] = ret["date"].to_list()
    r = ret.select(symbols).to_numpy().astype(np.float64)
    e = elig.select(symbols).to_numpy().astype(bool)
    return dates, symbols, np.where(np.isnan(r), np.nan, r), e


def trailing_window(
    dates: list[date],
    symbols: list[str],
    returns: F64,
    eligible: npt.NDArray[np.bool_],
    asof: date,
    window: int,
    max_null_share: float = 0.05,
) -> ReturnWindow:
    """Returns of assets eligible at ``asof`` over the ``window`` rows ending at ``asof``.

    DQ-04: an asset with more than ``max_null_share`` null returns in the window is excluded;
    rows with a null among the remaining assets are dropped. DQ-01: raises ``LookaheadError`` if
    ``asof`` is not a row date (the caller would otherwise be guessing which rows are known).
    """
    try:
        end = dates.index(asof)
    except ValueError:
        raise LookaheadError(f"{asof} is not a trading date in the panel") from None
    start = max(0, end - window + 1)
    block = returns[start : end + 1]
    if any(d > asof for d in dates[start : end + 1]):  # pragma: no cover - guarded by index
        raise LookaheadError("window contains rows after asof")
    null_share = np.isnan(block).mean(axis=0)
    keep = eligible[end] & (null_share <= max_null_share)
    excluded = [s for s, e, k in zip(symbols, eligible[end], keep, strict=True) if e and not k]
    cols = np.flatnonzero(keep)
    sub = block[:, cols]
    rows = ~np.isnan(sub).any(axis=1)
    return ReturnWindow(
        asof=asof,
        dates=[d for d, r in zip(dates[start : end + 1], rows, strict=True) if r],
        symbols=[symbols[c] for c in cols],
        returns=sub[rows],
        excluded=excluded,
    )
