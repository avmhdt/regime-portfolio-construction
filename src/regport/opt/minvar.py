"""Long-only minimum variance with a per-asset cap (FR-17).

``min w' Σ w  s.t.  1'w = 1, 0 <= w <= cap``, solved with CLARABEL and, if that fails, OSQP.
The problem is a cached cvxpy ``Parameter`` problem per asset count, written as
``||L' w||^2`` with ``Σ = L L'`` so it stays DPP and re-solves cheaply.

Every solve starts a fresh solver (``warm_start=False``): cvxpy otherwise reuses the previous
CLARABEL instance with its data updated in place, and the answer then depends on what the
process solved before, by up to 1e-6 (NFR-02).

The interior-point answer can still sit up to about 1e-5 from the optimum along flat
directions, so min-variance weights are polished: on the solver's active set (weights at 0 and
at the cap) the KKT system is solved exactly, and the result is kept only if it satisfies every
KKT condition. For a positive definite Σ that point is the unique optimum.
"""

from __future__ import annotations

from functools import lru_cache

import cvxpy as cp
import numpy as np
import numpy.typing as npt

from regport.errors import ConstraintViolation
from regport.opt.constraints import clean_weights, effective_cap

F64 = npt.NDArray[np.float64]


class SolverError(RuntimeError):
    """Every configured solver failed; the caller decides the fallback (FR-17)."""


# Tight tolerances: NFR-03 only allows clipping weights within 1e-9 of their bounds.
SOLVER_OPTIONS: dict[str, dict[str, float | int | bool]] = {
    "CLARABEL": {"tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10, "tol_feas": 1e-10, "max_iter": 500},
    "OSQP": {"eps_abs": 1e-10, "eps_rel": 1e-10, "max_iter": 200_000, "polish": True},
}


def solve_weights(
    problem: cp.Problem, w: cp.Variable, cap: float, solvers: tuple[str, ...], what: str
) -> F64:
    """Solve with each solver in turn; clean and return the first acceptable weights."""
    errors = []
    for solver in solvers:
        try:
            problem.solve(solver=solver, warm_start=False, **SOLVER_OPTIONS.get(solver, {}))
        except cp.error.SolverError as exc:
            errors.append(f"{solver}: {exc}")
            continue
        if problem.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            try:
                return clean_weights(np.asarray(w.value, dtype=np.float64), cap)
            except ConstraintViolation as exc:
                errors.append(f"{solver}: {exc}")
                continue
        errors.append(f"{solver}: status {problem.status}")
    raise SolverError(f"{what} failed ({'; '.join(errors)})")


@lru_cache(maxsize=64)
def _problem(n: int) -> tuple[cp.Problem, cp.Variable, cp.Parameter, cp.Parameter]:
    w = cp.Variable(n)
    factor = cp.Parameter((n, n))
    cap = cp.Parameter(nonneg=True)
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(factor.T @ w)), [cp.sum(w) == 1, w >= 0, w <= cap]
    )
    return problem, w, factor, cap


def _factor(cov: F64) -> F64:
    values, vectors = np.linalg.eigh(0.5 * (cov + cov.T))
    out: F64 = vectors * np.sqrt(np.clip(values, 0.0, None))
    return out


def _kkt_point(sigma: F64, w: F64, cap: float, tol: float) -> F64 | None:
    """The exact optimum on ``w``'s active set at ``tol``, or None if it fails a KKT check."""
    at_cap = w >= cap - tol
    free = (w > tol) & ~at_cap
    n_cap = int(at_cap.sum())
    if not free.any():
        return np.where(at_cap, cap, 0.0) if abs(cap * n_cap - 1.0) < 1e-12 else None
    capped = np.full(n_cap, cap)
    rhs = np.column_stack([np.ones(int(free.sum())), sigma[np.ix_(free, at_cap)] @ capped])
    try:
        x, y = np.linalg.solve(sigma[np.ix_(free, free)], rhs).T
    except np.linalg.LinAlgError:
        return None
    half_nu = (1.0 - cap * n_cap + y.sum()) / x.sum()  # Σ_FF w_F + Σ_FC w_C = (ν/2) 1
    out = np.where(at_cap, cap, 0.0)
    out[free] = half_nu * x - y
    grad = sigma @ out
    slack = 1e-6 * abs(half_nu)
    ok = (
        bool(np.all(np.isfinite(out)))
        and out[free].min() >= -1e-12
        and out[free].max() <= cap + 1e-12
        and bool(np.all(grad[~free & ~at_cap] >= half_nu - slack))
        and bool(np.all(grad[at_cap] <= half_nu + slack))
    )
    return out if ok else None


def polish(cov: F64, w: F64, cap: float) -> F64:
    """Replace solver weights by the exact KKT point on their active set, when one checks out."""
    sigma = 0.5 * (cov + cov.T) / float(np.mean(np.diag(cov)))
    for tol in (1e-7, 1e-6, 1e-5, 1e-4):
        out = _kkt_point(sigma, w, cap, tol)
        if out is not None:
            return clean_weights(out, cap)
    return w


def min_variance(
    cov: F64, max_weight: float = 1.0, solvers: tuple[str, ...] = ("CLARABEL", "OSQP")
) -> F64:
    """Minimum-variance weights; raises ``SolverError`` if every solver fails."""
    cov = np.asarray(cov, dtype=np.float64)
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)
    problem, w, factor, cap = _problem(n)
    scale = float(np.mean(np.diag(cov)))
    factor.value = _factor(cov / scale)
    cap.value = effective_cap(max_weight, n)
    weights = solve_weights(problem, w, float(cap.value), solvers, "min_variance")
    return polish(cov, weights, float(cap.value))
