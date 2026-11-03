"""Equal risk contribution (FR-18): Spinu's (2013) convex formulation, own code.

``min_y  ½ y'Σy − Σ_i b_i ln y_i`` over ``y > 0`` is strictly convex; at its optimum
``y_i (Σy)_i = b_i``, so ``w = y / 1'y`` has relative risk contributions equal to the budgets
``b`` (Maillard, Roncalli and Teïletche 2010). Newton's method with backtracking solves it; cyclical
coordinate descent, whose one-dimensional steps have a closed form, is the fallback.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from regport.logs import get_logger

log = get_logger(__name__)
F64 = npt.NDArray[np.float64]
ACCEPT_TOL = 1e-6  # AC-W-02; anything worse than this raises


class ERCError(RuntimeError):
    """Neither Newton nor coordinate descent reached the AC-W-02 tolerance."""


def risk_contributions(w: F64, cov: F64) -> F64:
    """Relative risk contributions ``w_i (Σw)_i / (w'Σw)``; they sum to one."""
    w = np.asarray(w, dtype=np.float64)
    marginal = cov @ w
    out: F64 = w * marginal / float(w @ marginal)
    return out


def _gap(y: F64, cov: F64, b: F64) -> float:
    return float(np.max(np.abs(risk_contributions(y / y.sum(), cov) - b)))


def _objective(y: F64, cov: F64, b: F64) -> float:
    return float(0.5 * y @ cov @ y - b @ np.log(y))


def _newton(cov: F64, b: F64, tol: float, max_iter: int) -> F64:
    y = b / np.sqrt(np.diag(cov))
    y /= np.sqrt(float(y @ cov @ y))
    for _ in range(max_iter):
        if _gap(y, cov, b) <= tol:
            break
        grad = cov @ y - b / y
        hess = cov + np.diag(b / y**2)
        step = np.linalg.solve(hess, grad)
        t, f0, slope = 1.0, _objective(y, cov, b), float(grad @ step)
        while True:
            cand = y - t * step
            if np.all(cand > 0) and _objective(cand, cov, b) <= f0 - 1e-4 * t * slope:
                break
            t *= 0.5
            if t < 1e-16:
                return y
        y = cand
    return y


def _coordinate_descent(cov: F64, b: F64, y: F64, tol: float, max_iter: int) -> F64:
    y = y.copy()
    diag = np.diag(cov)
    for _ in range(max_iter):
        for i in range(y.size):
            a = float(cov[i] @ y - diag[i] * y[i])
            y[i] = (-a + np.sqrt(a * a + 4.0 * diag[i] * b[i])) / (2.0 * diag[i])
        if _gap(y, cov, b) <= tol:
            break
    return y


def erc(cov: F64, budgets: F64 | None = None, tol: float = 1e-10, max_iter: int = 500) -> F64:
    """Long-only weights whose relative risk contributions equal ``budgets`` (default 1/N).

    Stops when ``max_i |RC_i − b_i| <= tol``. A result above ``tol`` but within 1e-6 is returned
    with a logged ``erc_tolerance`` event; anything worse raises ``ERCError``.
    """
    cov = np.asarray(cov, dtype=np.float64)
    n = cov.shape[0]
    b = np.full(n, 1.0 / n) if budgets is None else np.asarray(budgets, dtype=np.float64)
    if b.shape != (n,) or np.any(b <= 0):
        raise ValueError("budgets must be positive, one per asset")
    b = b / b.sum()
    if n == 1:
        return np.ones(1)
    scaled = cov / float(np.mean(np.diag(cov)))
    y = _newton(scaled, b, tol, max_iter)
    if _gap(y, scaled, b) > tol:
        y = _coordinate_descent(scaled, b, y, tol, max_iter * 10)
    gap = _gap(y, scaled, b)
    if gap > ACCEPT_TOL:
        raise ERCError(f"ERC did not converge: max |RC - b| = {gap:.2e}")
    if gap > tol:
        log.warning("erc_tolerance", gap=gap, tol=tol)
    w: F64 = y / y.sum()
    return w
