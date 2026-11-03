"""Regime-conditioned risk inputs (FR-16).

State moments come from the trailing asset returns, each day weighted by its filtered state
probability: ``μ_k = Σ_t w_tk r_t`` and ``Σ_k = Σ_t w_tk (r_t − μ_k)(r_t − μ_k)' / (1 − Σ_t w_tk²)``
with ``w_tk = p_tk / Σ_t p_tk``. Each ``Σ_k`` is shrunk toward its diagonal by ``shrinkage``. With
forward state weights ``π`` (the average of the predicted probabilities over the next ``h`` days,
``π = (1/h) Σ_{j=1..h} p_t A^j``), the mixture's moments are

``Σ = Σ_k π_k (Σ_k + μ_k μ_k') − μ̄ μ̄'``, ``μ̄ = Σ_k π_k μ_k`` (both annualized).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from regport.risk.psd import nearest_psd

F64 = npt.NDArray[np.float64]
ANNUALIZATION = 252.0
MIN_EFFECTIVE_OBS = 30.0


def forward_weights(p_filtered: F64, transmat: F64, horizon: int) -> F64:
    """``(1/h) Σ_{j=1..h} p A^j``: expected state occupancy over the next ``h`` days."""
    p = np.asarray(p_filtered, dtype=np.float64)
    total = np.zeros_like(p)
    for _ in range(horizon):
        p = p @ transmat
        total += p
    out: F64 = total / horizon
    return out / out.sum()


def regime_mixture_cov(
    returns: F64, p_filtered: F64, pi_forward: F64, shrinkage: float
) -> tuple[F64, F64]:
    """``(Σ, μ)`` of the regime mixture, annualized; ``returns`` are daily, ``T x N``.

    A state with fewer than 30 effective observations (``1/Σ w²``) in the window falls back to
    the unweighted moments, so a rarely visited state can't produce a degenerate covariance.
    """
    r = np.asarray(returns, dtype=np.float64)
    p = np.asarray(p_filtered, dtype=np.float64)
    pi = np.asarray(pi_forward, dtype=np.float64)
    if p.shape != (r.shape[0], pi.size):
        raise ValueError("p_filtered must be T x K with one row per return row")
    base_mu = r.mean(axis=0)
    base_cov = np.cov(r, rowvar=False, ddof=1)
    mus, covs = [], []
    for k in range(pi.size):
        mass = p[:, k].sum()
        w = p[:, k] / mass if mass > 0 else np.zeros(r.shape[0])
        eff = 1.0 / float(np.sum(w**2)) if mass > 0 else 0.0
        if eff < MIN_EFFECTIVE_OBS:
            mu_k, cov_k = base_mu, base_cov
        else:
            mu_k = w @ r
            d = r - mu_k
            cov_k = (d * w[:, None]).T @ d / (1.0 - float(np.sum(w**2)))
        cov_k = (1.0 - shrinkage) * cov_k + shrinkage * np.diag(np.diag(cov_k))
        mus.append(mu_k)
        covs.append(cov_k)
    mu_bar = sum(pi[k] * mus[k] for k in range(pi.size))
    second = sum(pi[k] * (covs[k] + np.outer(mus[k], mus[k])) for k in range(pi.size))
    cov = second - np.outer(mu_bar, mu_bar)
    return nearest_psd(cov * ANNUALIZATION), np.asarray(mu_bar, dtype=np.float64) * ANNUALIZATION
