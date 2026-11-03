"""Constrained mean-variance with an L1 turnover penalty (FR-20), and its mean estimator.

``max_w  μ'w − (γ/2) w'Σw − Σ_i κ_i |w_i − w_prev,i|``  s.t. ``1'w = 1, 0 <= w <= cap``,
with ``μ`` and ``Σ`` annualized. A trade costs ``c_i`` once; spread over the expected holding period
of one rebalance interval, that is ``c_i · periods_per_year`` per year, so
``κ_i = c_i · multiplier · periods_per_year`` puts the penalty on the same annual scale as ``μ``.
"""

from __future__ import annotations

from functools import lru_cache

import cvxpy as cp
import numpy as np
import numpy.typing as npt

from regport.opt.constraints import effective_cap
from regport.opt.minvar import _factor, solve_weights

F64 = npt.NDArray[np.float64]
ANNUALIZATION = 252.0


def james_stein_mean(returns: F64, cov_daily: F64) -> F64:
    """James–Stein shrinkage of the sample mean toward the cross-sectional grand mean, annualized.

    ``μ_JS = μ0 + (1 − φ)(μ̂ − μ0)`` with ``μ0`` the mean of ``μ̂`` across assets and
    ``φ = min(1, (N − 3) / (T (μ̂ − μ0)' Σ⁻¹ (μ̂ − μ0)))`` (Jorion 1986 uses the same form toward
    the minimum-variance mean). With ``N <= 3`` no shrinkage is applied.
    """
    x = np.asarray(returns, dtype=np.float64)
    t, n = x.shape
    mu = x.mean(axis=0)
    grand = float(mu.mean())
    dev = mu - grand
    phi = 0.0
    if n > 3:
        quad = float(dev @ np.linalg.solve(cov_daily, dev))
        phi = 1.0 if quad <= 0 else min(1.0, (n - 3) / (t * quad))
    out: F64 = (grand + (1.0 - phi) * dev) * ANNUALIZATION
    return out


@lru_cache(maxsize=64)
def _problem(n: int) -> tuple[cp.Problem, cp.Variable, dict[str, cp.Parameter]]:
    w = cp.Variable(n)
    p = {
        "mu": cp.Parameter(n),
        "factor": cp.Parameter((n, n)),
        "kappa": cp.Parameter(n, nonneg=True),
        "prev": cp.Parameter(n),
        "half_gamma": cp.Parameter(nonneg=True),
        "cap": cp.Parameter(nonneg=True),
    }
    objective = (
        p["mu"] @ w
        - p["half_gamma"] * cp.sum_squares(p["factor"].T @ w)
        - p["kappa"] @ cp.abs(w - p["prev"])
    )
    problem = cp.Problem(cp.Maximize(objective), [cp.sum(w) == 1, w >= 0, w <= p["cap"]])
    return problem, w, p


def mean_variance_tc(
    mu: F64,
    cov: F64,
    w_prev: F64,
    risk_aversion: float,
    kappa: F64,
    max_weight: float = 1.0,
    solvers: tuple[str, ...] = ("CLARABEL", "OSQP"),
) -> F64:
    """Mean-variance weights with turnover penalty; raises ``SolverError`` if all solvers fail."""
    cov = np.asarray(cov, dtype=np.float64)
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)
    problem, w, p = _problem(n)
    p["mu"].value = np.asarray(mu, dtype=np.float64)
    p["factor"].value = _factor(cov)
    p["kappa"].value = np.asarray(kappa, dtype=np.float64)
    p["prev"].value = np.asarray(w_prev, dtype=np.float64)
    p["half_gamma"].value = risk_aversion / 2.0
    p["cap"].value = effective_cap(max_weight, n)
    return solve_weights(problem, w, float(p["cap"].value), solvers, "mean_variance_tc")
