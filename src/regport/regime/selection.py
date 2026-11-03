"""Regime-model hyperparameters chosen on pre-OOS data only (D-05, FR-14).

The rules were fixed in ``docs/decisions.md`` before they were run:
- K: lowest BIC over ``candidate_states`` on feature set A, 1993-02 to 2004-12.
- BOCPD: highest one-step predictive log-likelihood of the SPY (proxy) log return over a hazard and
  prior-strength grid, with ``β0 = (α0 − 1) s²`` from the same window.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt

from regport.regime.bocpd import NormalGammaPrior, bocpd
from regport.regime.hmm import bic, fit_hmm

F64 = npt.NDArray[np.float64]
HAZARD_GRID = (100.0, 250.0, 500.0, 1000.0)
ALPHA_GRID = (2.0, 5.0, 10.0, 20.0)


@dataclass(frozen=True)
class KChoice:
    k: int
    table: list[dict[str, float]]


@dataclass(frozen=True)
class BOCPDChoice:
    hazard_lambda: float
    prior: NormalGammaPrior
    table: list[dict[str, float]]


def _fit_k(args: tuple[F64, int, int, int, date, int, float]) -> tuple[int, float]:
    x, k, n_init, seed, end, max_iter, tol = args
    params = fit_hmm(x, k, n_init, seed, end, max_iter, tol)
    return k, params.loglik


def choose_k(
    x: F64,
    end: date,
    candidates: list[int],
    n_init: int,
    seed: int,
    max_iter: int,
    tol: float,
    n_jobs: int = 1,
) -> KChoice:
    """Lowest BIC among ``candidates`` on the standardized pre-OOS features ``x``."""
    jobs = [(x, k, n_init, seed + k, end, max_iter, tol) for k in candidates]
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(jobs))) as pool:
            results = list(pool.map(_fit_k, jobs))
    else:
        results = [_fit_k(j) for j in jobs]
    table = [
        {"k": float(k), "loglik": ll, "bic": bic(ll, k, x.shape[1], x.shape[0])}
        for k, ll in results
    ]
    best = min(table, key=lambda r: (r["bic"], r["k"]))
    return KChoice(int(best["k"]), table)


def choose_bocpd(x: F64, max_run_length: int, short_run_days: int) -> BOCPDChoice:
    """Highest predictive log-likelihood over :data:`HAZARD_GRID` x :data:`ALPHA_GRID`."""
    s2 = float(np.var(x, ddof=1))
    table = []
    for lam in HAZARD_GRID:
        for a0 in ALPHA_GRID:
            prior = NormalGammaPrior(0.0, 1.0, a0, (a0 - 1.0) * s2)
            res = bocpd(x, lam, prior, max_run_length, short_run_days)
            table.append(
                {
                    "hazard_lambda": lam,
                    "alpha0": a0,
                    "beta0": prior.beta0,
                    "predictive_loglik": res.predictive_loglik,
                }
            )
    best = max(table, key=lambda r: (r["predictive_loglik"], -r["hazard_lambda"]))
    prior = NormalGammaPrior(0.0, 1.0, best["alpha0"], best["beta0"])
    return BOCPDChoice(best["hazard_lambda"], prior, table)
