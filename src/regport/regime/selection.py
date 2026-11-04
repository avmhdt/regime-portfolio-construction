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
from regport.resources import wait_for_memory

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
        wait_for_memory("regime selection")
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


# Constrained reselection (owner ruling 2026-11-03; rule in docs/preregistration.md).
SEGMENT = 400
N_SHIFTS = 5
REPLICATIONS = 20
DETECT_WINDOW = 10
STATIONARY_DAYS = 10_000
BURN_IN = 100


@dataclass(frozen=True)
class ValidationResult:
    hazard_lambda: float
    alpha0: float
    detect_pass_reps: int  # replications with >= 4 of 5 shifts flagged (std dev x3)
    detect_pass_reps_variance: int  # the same under the literal "variance x3" reading
    false_alarms_per_1000: float  # mean over replications

    replications: int = REPLICATIONS

    @property
    def passes(self) -> bool:
        """>= 90% of replications detect >= 4 of 5 shifts, and <= 1 false alarm per 1,000."""
        return (
            self.detect_pass_reps >= 0.9 * self.replications and self.false_alarms_per_1000 <= 1.0
        )


def _shift_series(rng: np.random.Generator, sigma: float, ratio: float) -> F64:
    return np.concatenate(
        [rng.normal(0.0, sigma * (ratio if i % 2 else 1.0), SEGMENT) for i in range(N_SHIFTS + 1)]
    )


def _flagged(cp_short: F64, level: float = 0.5) -> int:
    starts = [SEGMENT * i for i in range(1, N_SHIFTS + 1)]
    return sum(bool((cp_short[s : s + DETECT_WINDOW + 1] > level).any()) for s in starts)


def validate_setting(
    args: tuple[float, float, float, int, int, int],
    replications: int = REPLICATIONS,
) -> ValidationResult:
    """Validation item (3) for one hazard and prior strength, with fixed seeds."""
    lam, a0, sigma, max_rl, short, seed = args
    prior = NormalGammaPrior(0.0, 1.0, a0, (a0 - 1.0) * sigma**2)
    detect = detect_var = 0
    alarms = []
    for rep in range(replications):
        rng = np.random.default_rng([seed, rep])
        res = bocpd(_shift_series(rng, sigma, 3.0), lam, prior, max_rl, short)
        detect += _flagged(res.cp_prob_short) >= 4
        res_v = bocpd(_shift_series(rng, sigma, float(np.sqrt(3.0))), lam, prior, max_rl, short)
        detect_var += _flagged(res_v.cp_prob_short) >= 4
        p = bocpd(rng.normal(0.0, sigma, STATIONARY_DAYS), lam, prior, max_rl, short).cp_prob_short[
            BURN_IN:
        ]
        onsets = int(((p[1:] > 0.5) & (p[:-1] <= 0.5)).sum())
        alarms.append(onsets * 1000.0 / (p.size - 1))
    return ValidationResult(lam, a0, detect, detect_var, float(np.mean(alarms)), replications)


def choose_bocpd_constrained(
    x: F64, max_run_length: int, short_run_days: int, seed: int, n_jobs: int = 1
) -> tuple[BOCPDChoice | None, list[ValidationResult]]:
    """Highest pre-2005 predictive log-likelihood among settings passing validation item (3)."""
    sigma = float(np.std(x, ddof=1))
    base = choose_bocpd(x, max_run_length, short_run_days)
    jobs = [
        (lam, a0, sigma, max_run_length, short_run_days, seed)
        for lam in HAZARD_GRID
        for a0 in ALPHA_GRID
    ]
    if n_jobs > 1:
        wait_for_memory("regime selection")
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            checks = list(pool.map(validate_setting, jobs))
    else:
        checks = [validate_setting(j) for j in jobs]
    passing = {(c.hazard_lambda, c.alpha0) for c in checks if c.passes}
    candidates = [r for r in base.table if (r["hazard_lambda"], r["alpha0"]) in passing]
    if not candidates:
        return None, checks
    best = max(candidates, key=lambda r: (r["predictive_loglik"], -r["hazard_lambda"]))
    prior = NormalGammaPrior(0.0, 1.0, best["alpha0"], best["beta0"])
    return BOCPDChoice(best["hazard_lambda"], prior, base.table), checks
