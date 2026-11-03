"""Ledoit and Wolf (2008) tests of a Sharpe-ratio difference (FR-27, D-08), own code.

With ``u = (μ1, μ2, γ1, γ2)`` the first and second moments of the two (excess) return series,
``Δ = f(u) = μ1/sqrt(γ1 − μ1²) − μ2/sqrt(γ2 − μ2²)`` and, by the delta method,
``se(Δ) = sqrt(∇f' Ψ ∇f / T)`` where ``Ψ`` is the long-run covariance of
``y_t = (r1 − μ1, r2 − μ2, r1² − γ1, r2² − γ2)``.

- HAC: ``Ψ`` by a Parzen kernel with VAR(1) prewhitening and Andrews' (1991) automatic bandwidth
  (Andrews and Monahan 1992), times ``T/(T − 4)`` as in Ledoit and Wolf's code; normal p-value.
- Bootstrap (headline): studentized circular block bootstrap (Politis and Romano 1992). The
  original statistic is studentized with the Parzen HAC standard error (no prewhitening); each
  resample with the block standard error ``Ψ* = (1/(lK)) Σ_j ζ_j ζ_j'`` over its ``K`` blocks of
  length ``l``. ``p = (#{|Δ* − Δ|/se* ≥ |Δ|/se} + 1)/(B + 1)``. The block size is chosen by the
  paper's calibration (Algorithm 3.1) over {1, 2, 4, 6, 8, 10}: data are simulated from a VAR(1)
  fitted to ``y`` with ``Δ = 0`` imposed, and the block size whose empirical rejection rate at 5%
  is closest to 5% wins.

Holm (1979) adjusts the p-values across all strategies compared with 1/N.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numba
import numpy as np
import numpy.typing as npt
from scipy.stats import norm

F64 = npt.NDArray[np.float64]
BLOCK_SIZES = (1, 2, 4, 6, 8, 10)


@dataclass(frozen=True)
class SharpeTestResult:
    sr1: float
    sr2: float
    diff: float
    se: float
    p_value: float
    method: str
    block_size: int | None


def _moments(r1: F64, r2: F64) -> tuple[F64, F64]:
    mu1, mu2 = r1.mean(), r2.mean()
    g1, g2 = (r1**2).mean(), (r2**2).mean()
    u = np.array([mu1, mu2, g1, g2])
    y = np.column_stack([r1 - mu1, r2 - mu2, r1**2 - g1, r2**2 - g2])
    return u, y


def _sharpe_and_gradient(u: F64) -> tuple[float, float, F64]:
    mu1, mu2, g1, g2 = u
    v1, v2 = g1 - mu1**2, g2 - mu2**2
    grad = np.array([g1 / v1**1.5, -g2 / v2**1.5, -0.5 * mu1 / v1**1.5, 0.5 * mu2 / v2**1.5])
    return mu1 / math.sqrt(v1), mu2 / math.sqrt(v2), grad


def parzen(x: F64) -> F64:
    a = np.abs(x)
    return np.where(a <= 0.5, 1 - 6 * a**2 + 6 * a**3, np.where(a <= 1, 2 * (1 - a) ** 3, 0.0))


def _andrews_bandwidth(e: F64) -> float:
    """Andrews (1991) Parzen bandwidth ``2.6614 (α(2) T)^(1/5)`` from univariate AR(1) fits."""
    t = e.shape[0]
    num = den = 0.0
    for col in e.T:
        rho = float(col[1:] @ col[:-1] / (col[:-1] @ col[:-1]))
        rho = min(max(rho, -0.97), 0.97)
        sigma2 = float(np.mean((col[1:] - rho * col[:-1]) ** 2))
        num += 4 * rho**2 * sigma2**2 / (1 - rho) ** 8
        den += sigma2**2 / (1 - rho) ** 4
    alpha2 = num / den
    return float(2.6614 * (alpha2 * t) ** 0.2)


def _kernel_cov(e: F64, bandwidth: float) -> F64:
    t = e.shape[0]
    psi: F64 = e.T @ e / t
    lag = 1
    while lag < t:
        w = float(parzen(np.array([lag / bandwidth]))[0])
        if w == 0.0:
            break
        g = e[lag:].T @ e[:-lag] / t
        psi += w * (g + g.T)
        lag += 1
    return psi


def hac_psi(y: F64, prewhiten: bool = True) -> F64:
    """Long-run covariance of ``y`` (Parzen kernel, optional VAR(1) prewhitening)."""
    t = y.shape[0]
    if prewhiten:
        x, z = y[:-1], y[1:]
        a = np.linalg.lstsq(x, z, rcond=None)[0].T
        e = z - x @ a.T
        psi_e = _kernel_cov(e, _andrews_bandwidth(e))
        inv = np.linalg.inv(np.eye(y.shape[1]) - a)
        psi = inv @ psi_e @ inv.T
    else:
        psi = _kernel_cov(y, _andrews_bandwidth(y))
    out: F64 = psi * t / (t - 4)
    return out


def _check(r1: F64, r2: F64) -> tuple[F64, F64]:
    a, b = np.asarray(r1, dtype=np.float64), np.asarray(r2, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size < 10:
        raise ValueError("need two aligned 1-D return series of at least 10 observations")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("returns contain NaN or inf")
    return a, b


def sharpe_diff_hac(r1: F64, r2: F64, prewhiten: bool = True) -> SharpeTestResult:
    """HAC inference on ``SR1 − SR2`` (per-period Sharpe ratios); two-sided normal p-value."""
    a, b = _check(r1, r2)
    u, y = _moments(a, b)
    sr1, sr2, grad = _sharpe_and_gradient(u)
    psi = hac_psi(y, prewhiten)
    se = math.sqrt(float(grad @ psi @ grad) / a.size)
    diff = sr1 - sr2
    p = 2.0 * (1.0 - float(norm.cdf(abs(diff) / se)))
    return SharpeTestResult(sr1, sr2, diff, se, p, "hac", None)


@numba.njit(cache=True)
def _boot_stats(
    r1: F64, r2: F64, starts: npt.NDArray[np.int64], block: int
) -> tuple[F64, F64]:  # pragma: no cover
    """Δ* and se* for each row of block ``starts`` (circular blocks of length ``block``)."""
    t = r1.size
    n_boot, n_blocks = starts.shape
    length = n_blocks * block
    diffs = np.empty(n_boot)
    ses = np.empty(n_boot)
    x1 = np.empty(length)
    x2 = np.empty(length)
    for bi in range(n_boot):
        k = 0
        for j in range(n_blocks):
            s = starts[bi, j]
            for m in range(block):
                idx = (s + m) % t
                x1[k] = r1[idx]
                x2[k] = r2[idx]
                k += 1
        mu1 = x1.mean()
        mu2 = x2.mean()
        g1 = (x1 * x1).mean()
        g2 = (x2 * x2).mean()
        v1 = g1 - mu1 * mu1
        v2 = g2 - mu2 * mu2
        diffs[bi] = mu1 / math.sqrt(v1) - mu2 / math.sqrt(v2)
        grad = np.array([g1 / v1**1.5, -g2 / v2**1.5, -0.5 * mu1 / v1**1.5, 0.5 * mu2 / v2**1.5])
        psi = np.zeros((4, 4))
        zeta = np.empty(4)
        for j in range(n_blocks):
            zeta[:] = 0.0
            for m in range(block):
                i = j * block + m
                zeta[0] += x1[i] - mu1
                zeta[1] += x2[i] - mu2
                zeta[2] += x1[i] * x1[i] - g1
                zeta[3] += x2[i] * x2[i] - g2
            for p in range(4):
                for q in range(4):
                    psi[p, q] += zeta[p] * zeta[q]
        psi /= length
        var = 0.0
        for p in range(4):
            for q in range(4):
                var += grad[p] * psi[p, q] * grad[q]
        ses[bi] = math.sqrt(max(var, 1e-300) / length)
    return diffs, ses


def _bootstrap_p(
    a: F64, b: F64, block: int, n_boot: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    t = a.size
    n_blocks = math.ceil(t / block)
    starts = rng.integers(0, t, size=(n_boot, n_blocks))
    u, y = _moments(a, b)
    sr1, sr2, grad = _sharpe_and_gradient(u)
    diff = sr1 - sr2
    se = math.sqrt(float(grad @ hac_psi(y, prewhiten=False) @ grad) / t)
    diffs, ses = _boot_stats(a, b, starts, block)
    stat = abs(diff) / se
    p = (float(np.sum(np.abs(diffs - diff) / ses >= stat)) + 1.0) / (n_boot + 1.0)
    return diff, se, p


def calibrate_block_size(
    r1: F64,
    r2: F64,
    seed: int,
    n_sim: int = 200,
    n_boot: int = 499,
    candidates: tuple[int, ...] = BLOCK_SIZES,
    alpha: float = 0.05,
) -> int:
    """Ledoit-Wolf Algorithm 3.1: block size with empirical size closest to ``alpha``.

    Pseudo-series are simulated from a VAR(1) fitted to the two return series, with each mean
    reset so that both Sharpe ratios equal the pooled one (the null), and residuals resampled.
    """
    a, b = _check(r1, r2)
    x = np.column_stack([a, b])
    lhs, rhs = x[1:], x[:-1]
    design = np.column_stack([np.ones(rhs.shape[0]), rhs])
    coef = np.linalg.lstsq(design, lhs, rcond=None)[0]
    resid = lhs - design @ coef
    resid -= resid.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    sr = float(np.mean(x.mean(axis=0) / sd))
    target_mean = sr * sd
    rng = np.random.default_rng(seed)
    rates = {}
    sims = []
    for _ in range(n_sim):
        draw = resid[rng.integers(0, resid.shape[0], size=x.shape[0] + 100)]
        z = np.empty_like(draw)
        z[0] = draw[0]
        for i in range(1, z.shape[0]):
            z[i] = coef[1:].T @ z[i - 1] + draw[i]
        z = z[100:]
        z = z - z.mean(axis=0) + target_mean
        sims.append(z)
    for block in candidates:
        brng = np.random.default_rng([seed, block])
        rej = sum(_bootstrap_p(z[:, 0], z[:, 1], block, n_boot, brng)[2] <= alpha for z in sims)
        rates[block] = rej / n_sim
    return min(candidates, key=lambda blk: (abs(rates[blk] - alpha), blk))


def sharpe_diff_bootstrap(
    r1: F64,
    r2: F64,
    n_boot: int,
    block_size: int | None,
    seed: int,
    calibration_sims: int = 200,
    calibration_boot: int = 499,
) -> SharpeTestResult:
    """Studentized circular block bootstrap p-value for ``SR1 − SR2`` (headline test, D-08)."""
    a, b = _check(r1, r2)
    block = block_size or calibrate_block_size(a, b, seed, calibration_sims, calibration_boot)
    diff, se, p = _bootstrap_p(a, b, block, n_boot, np.random.default_rng([seed, 0, block]))
    u, _ = _moments(a, b)
    sr1, sr2, _ = _sharpe_and_gradient(u)
    return SharpeTestResult(sr1, sr2, diff, se, p, "bootstrap", block)


def holm(p_values: F64) -> F64:
    """Holm (1979) step-down adjusted p-values, in the input order."""
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted
