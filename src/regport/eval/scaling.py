"""``regport bench scaling``: how the pipeline scales to 500 assets (FR-31, AC-H-05).

Synthetic daily returns come from a 5-factor model (factor volatilities 0.5% to 1.5% a day, unit
loadings scaled to a common market factor, idiosyncratic volatility 1%). For each N the median
runtime over ``repeats`` and the peak traced memory are measured for every estimator and optimizer,
with min-variance and mean-variance also fed the PCA factor covariance (k = 10). Condition numbers
of the sample, Ledoit-Wolf and factor covariances are reported against N/T.
"""

from __future__ import annotations

import statistics
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
import numpy.typing as npt
import polars as pl

from regport.opt.erc import erc
from regport.opt.hrp import hrp
from regport.opt.minvar import min_variance
from regport.opt.mv_tc import james_stein_mean, mean_variance_tc
from regport.risk.covariance import ewma_cov, ledoit_wolf_cov, oas_cov, sample_cov
from regport.risk.factor import factor_cov

F64 = npt.NDArray[np.float64]
WINDOW = 504


def factor_returns(n: int, t: int, seed: int) -> F64:
    rng = np.random.default_rng([seed, n])
    vols = np.linspace(0.005, 0.015, 5)
    f = rng.normal(0, 1, (t, 5)) * vols
    b = rng.normal(0, 0.5, (n, 5))
    b[:, 0] = rng.uniform(0.6, 1.4, n)
    out: F64 = f @ b.T + rng.normal(0, 0.01, (t, n))
    return out


@dataclass(frozen=True)
class Timing:
    seconds: float
    peak_mib: float


def measure(fn: Callable[[], object], repeats: int) -> Timing:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return Timing(statistics.median(times), peak / 2**20)


def scaling_table(
    sizes: list[int], repeats: int, seed: int, extra: dict[str, F64] | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Runtime/memory rows and condition-number rows for each universe size."""
    rows, cond = [], []
    universes = {str(n): factor_returns(n, WINDOW, seed) for n in sizes}
    universes.update(extra or {})
    for name, x in universes.items():
        n = x.shape[1]
        estimators: dict[str, Callable[[], F64]] = {
            "sample": partial(sample_cov, x),
            "ewma": partial(ewma_cov, x, 63.0),
            "ledoit_wolf": partial(ledoit_wolf_cov, x),
            "oas": partial(oas_cov, x),
            "factor_pca10": partial(factor_cov, x, 10),
        }
        covs = {}
        for label, fn in estimators.items():
            t = measure(fn, repeats)
            covs[label] = fn()
            rows.append(
                {
                    "universe": name,
                    "n_assets": n,
                    "stage": "estimator",
                    "method": label,
                    "median_seconds": t.seconds,
                    "peak_mib": t.peak_mib,
                }
            )
        mu = james_stein_mean(x[-252:], covs["sample"] / 252)
        prev = np.full(n, 1.0 / n)
        kappa = np.full(n, 2e-4 * 12)
        lw, fac = covs["ledoit_wolf"], covs["factor_pca10"]
        optimizers: dict[str, Callable[[], F64]] = {
            "min_variance (Ledoit-Wolf)": partial(min_variance, lw, 0.40),
            "min_variance (factor)": partial(min_variance, fac, 0.40),
            "erc (Ledoit-Wolf)": partial(erc, lw),
            "hrp (Ledoit-Wolf)": partial(hrp, lw),
            "mv_turnover (Ledoit-Wolf)": partial(mean_variance_tc, mu, lw, prev, 5.0, kappa, 0.40),
            "mv_turnover (factor)": partial(mean_variance_tc, mu, fac, prev, 5.0, kappa, 0.40),
        }
        for label, fn in optimizers.items():
            t = measure(fn, repeats)
            rows.append(
                {
                    "universe": name,
                    "n_assets": n,
                    "stage": "optimizer",
                    "method": label,
                    "median_seconds": t.seconds,
                    "peak_mib": t.peak_mib,
                }
            )
        for label in ("sample", "ledoit_wolf", "factor_pca10"):
            cond.append(
                {
                    "universe": name,
                    "n_assets": n,
                    "n_over_t": n / x.shape[0],
                    "estimator": label,
                    "condition_number": float(np.linalg.cond(covs[label])),
                }
            )
    return pl.DataFrame(rows), pl.DataFrame(cond)
