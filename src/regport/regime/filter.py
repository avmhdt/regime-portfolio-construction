"""Forward filter for a Gaussian HMM in log space (FR-12, Rabiner 1989, Hamilton 1989).

``log α_1 = log π + log b(x_1)``; ``log α_t = log b(x_t) + logsumexp_j(log f_{t−1,j} + log A_jk)``,
where ``f_t = α_t / Σ α_t`` is the filtered distribution ``P(S_t | x_1:t)``. The one-step prediction
is ``P(S_{t+1} | x_1:t) = f_t A`` and the log-likelihood is ``Σ_t log Σ_k α_t,k``. Only past and
present observations enter each row, unlike hmmlearn's ``predict_proba`` (smoothed, banned).
"""

from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt

from regport.regime.hmm import HMMParams

F64 = npt.NDArray[np.float64]
LOG_2PI = float(np.log(2.0 * np.pi))


def log_emissions(z: F64, means: F64, covars: F64) -> F64:
    """``T x K`` Gaussian log densities of standardized observations ``z``."""
    t, d = z.shape
    out = np.empty((t, means.shape[0]))
    for k in range(means.shape[0]):
        chol = np.linalg.cholesky(covars[k])
        diff = np.linalg.solve(chol, (z - means[k]).T)
        log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
        out[:, k] = -0.5 * (d * LOG_2PI + log_det + np.sum(diff**2, axis=0))
    return out


def forward_filter(x: F64, params: HMMParams) -> tuple[F64, F64, float]:
    """``(filtered T x K, predicted T x K, log-likelihood)`` of raw features ``x``.

    ``x`` is standardized with the training-window mean and standard deviation stored in
    ``params`` (FR-11), never with statistics that include the filtered period.
    """
    z = (np.asarray(x, dtype=np.float64) - params.x_mean) / params.x_std
    if not np.all(np.isfinite(z)):
        raise ValueError("features contain NaN or inf")
    log_b = log_emissions(z, params.means, params.covars)
    log_a = np.log(np.maximum(params.transmat, 1e-300))
    prior = np.log(np.maximum(params.startprob, 1e-300))
    log_f, loglik = _forward(log_b, log_a, prior)
    filtered = np.exp(log_f)
    filtered /= filtered.sum(axis=1, keepdims=True)
    predicted = filtered @ params.transmat
    return filtered, predicted, float(loglik)


@numba.njit(cache=True)
def _logsumexp(v: F64) -> float:  # pragma: no cover - compiled
    m = v.max()
    return float(m + np.log(np.sum(np.exp(v - m))))


@numba.njit(cache=True)
def _forward(log_b: F64, log_a: F64, prior: F64) -> tuple[F64, float]:  # pragma: no cover
    t, k = log_b.shape
    log_f = np.empty((t, k))
    loglik = 0.0
    pred = prior.copy()
    col = np.empty(k)
    for i in range(t):
        log_alpha = pred + log_b[i]
        c = _logsumexp(log_alpha)
        loglik += c
        log_f[i] = log_alpha - c
        for j in range(k):
            for m in range(k):
                col[m] = log_f[i, m] + log_a[m, j]
            pred[j] = _logsumexp(col)
    return log_f, loglik
