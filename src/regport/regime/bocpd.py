"""Bayesian online change-point detection (FR-14; Adams and MacKay 2007), own code.

Run-length recursion in log space, with constant hazard ``H = 1/λ``:

- growth: ``log P(r_t = r+1, x_1:t) = log P(r_{t−1} = r, x_1:t−1) + log π_r(x_t) + log(1 − H)``
- change point: ``log P(r_t = 0, x_1:t) =
  logsumexp_r(log P(r_{t−1} = r, x_1:t−1) + log π_r(x_t)) + log H``

``π_r`` is the Student-t predictive of a Normal–Gamma model with unknown mean and variance:
``ν = 2α``, location ``μ``, squared scale ``β(κ + 1)/(ακ)``. After each step the sufficient
statistics update as ``μ' = (κμ + x)/(κ + 1)``, ``κ' = κ + 1``, ``α' = α + ½``,
``β' = β + κ(x − μ)²/(2(κ + 1))``; ``r = 0`` restarts from the prior. Run lengths beyond
``max_run_length`` are dropped and the posterior renormalized. Each output at ``t`` uses
``x_1:t`` only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numba
import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class NormalGammaPrior:
    mu0: float
    kappa0: float
    alpha0: float
    beta0: float


@dataclass(frozen=True)
class BOCPDResult:
    """Per date: ``P(r_t < short_run)``, MAP and mean run length; optionally the full posterior."""

    cp_prob_short: F64
    rl_map: npt.NDArray[np.int32]
    rl_mean: F64
    predictive_loglik: float
    posterior: F64 | None = None


@numba.njit(cache=True)
def _run(
    x: F64,
    log_h: float,
    log_1mh: float,
    mu0: float,
    kappa0: float,
    alpha0: float,
    beta0: float,
    max_rl: int,
    short: int,
    keep: bool,
) -> tuple[F64, npt.NDArray[np.int32], F64, float, F64]:  # pragma: no cover - compiled
    t_n = x.size
    size = max_rl + 1
    log_r = np.full(size, -np.inf)
    log_r[0] = 0.0
    mu = np.full(size, mu0)
    kappa = np.full(size, kappa0)
    alpha = np.full(size, alpha0)
    beta = np.full(size, beta0)
    n_live = 1
    cp_short = np.empty(t_n)
    rl_map = np.empty(t_n, dtype=np.int32)
    rl_mean = np.empty(t_n)
    post = np.zeros((t_n if keep else 1, size))
    total = 0.0
    lp = np.empty(size)
    new = np.empty(size)
    for t in range(t_n):
        xt = x[t]
        for r in range(n_live):
            nu = 2.0 * alpha[r]
            scale2 = beta[r] * (kappa[r] + 1.0) / (alpha[r] * kappa[r])
            z = (xt - mu[r]) ** 2 / (nu * scale2)
            lp[r] = (
                math.lgamma((nu + 1.0) / 2.0)
                - math.lgamma(nu / 2.0)
                - 0.5 * math.log(nu * math.pi * scale2)
                - (nu + 1.0) / 2.0 * math.log1p(z)
            )
        joint_max = -np.inf
        for r in range(n_live):
            v = log_r[r] + lp[r]
            if v > joint_max:
                joint_max = v
        s = 0.0
        for r in range(n_live):
            s += math.exp(log_r[r] + lp[r] - joint_max)
        log_evidence_cp = joint_max + math.log(s)
        total += log_evidence_cp  # log p(x_t | x_1:t-1)
        n_new = min(n_live + 1, size)
        new[0] = log_evidence_cp + log_h
        for r in range(1, n_new):
            new[r] = log_r[r - 1] + lp[r - 1] + log_1mh
        norm_max = -np.inf
        for r in range(n_new):
            if new[r] > norm_max:
                norm_max = new[r]
        s = 0.0
        for r in range(n_new):
            s += math.exp(new[r] - norm_max)
        log_norm = norm_max + math.log(s)
        for r in range(n_new - 1, 0, -1):
            k_old = kappa[r - 1]
            dx = xt - mu[r - 1]
            mu[r] = (k_old * mu[r - 1] + xt) / (k_old + 1.0)
            beta[r] = beta[r - 1] + k_old * dx * dx / (2.0 * (k_old + 1.0))
            kappa[r] = k_old + 1.0
            alpha[r] = alpha[r - 1] + 0.5
        mu[0], kappa[0], alpha[0], beta[0] = mu0, kappa0, alpha0, beta0
        best, best_p, mean, short_p = 0, -1.0, 0.0, 0.0
        for r in range(n_new):
            log_r[r] = new[r] - log_norm
            p = math.exp(log_r[r])
            if p > best_p:
                best, best_p = r, p
            mean += r * p
            if r < short:
                short_p += p
            if keep:
                post[t, r] = p
        n_live = n_new
        cp_short[t] = short_p
        rl_map[t] = best
        rl_mean[t] = mean
    return cp_short, rl_map, rl_mean, total, post


def bocpd(
    x: F64,
    hazard_lambda: float,
    prior: NormalGammaPrior,
    max_run_length: int,
    short_run_days: int = 21,
    keep_posterior: bool = False,
) -> BOCPDResult:
    """Run BOCPD over ``x``; ``predictive_loglik`` is ``Σ_t log p(x_t | x_1:t−1)``."""
    arr = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("BOCPD input contains NaN or inf")
    h = 1.0 / hazard_lambda
    cp, rl_map, rl_mean, total, post = _run(
        arr,
        math.log(h),
        math.log1p(-h),
        float(prior.mu0),
        float(prior.kappa0),
        float(prior.alpha0),
        float(prior.beta0),
        int(max_run_length),
        int(short_run_days),
        bool(keep_posterior),
    )
    return BOCPDResult(cp, rl_map, rl_mean, float(total), post if keep_posterior else None)
