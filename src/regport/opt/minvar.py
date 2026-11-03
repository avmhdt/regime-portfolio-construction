"""Long-only minimum variance with a per-asset cap (FR-17).

``min w' Σ w  s.t.  1'w = 1, 0 <= w <= cap``, solved with CLARABEL and, if that fails, OSQP.
The problem is a cached cvxpy ``Parameter`` problem per asset count, written as
``||L' w||^2`` with ``Σ = L L'`` so it stays DPP and re-solves cheaply.
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
            problem.solve(solver=solver, **SOLVER_OPTIONS.get(solver, {}))
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
    return solve_weights(problem, w, float(cap.value), solvers, "min_variance")
