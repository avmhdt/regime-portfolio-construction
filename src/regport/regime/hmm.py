"""Gaussian HMM fitting with hmmlearn's Baum-Welch (FR-11); everything else is own code.

hmmlearn is used only for ``GaussianHMM.fit`` and its in-sample ``score`` (a forward pass over the
training window). ``predict``, ``predict_proba``, ``decode`` and ``score_samples`` condition on the
whole sequence and are banned in ``src/`` (FR-15, enforced by ``regport.regime.guard``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from datetime import date

import numpy as np
import numpy.typing as npt
from hmmlearn.hmm import GaussianHMM

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class HMMParams:
    """Fitted parameters in standardized units, with the standardization that produced them."""

    startprob: F64
    transmat: F64
    means: F64
    covars: F64
    fit_end: date
    n_obs: int
    loglik: float
    seed: int
    x_mean: F64 = field(default_factory=lambda: np.zeros(1))
    x_std: F64 = field(default_factory=lambda: np.ones(1))

    @property
    def n_states(self) -> int:
        return int(self.means.shape[0])

    def raw_means(self) -> F64:
        """State means in the features' original units."""
        out: F64 = self.means * self.x_std + self.x_mean
        return out

    def raw_covars(self) -> F64:
        """State covariances in the features' original units."""
        scale = np.outer(self.x_std, self.x_std)
        out: F64 = self.covars * scale
        return out

    def permute(self, perm: npt.NDArray[np.int64]) -> HMMParams:
        """Relabel states: new state ``i`` is old state ``perm[i]``."""
        return replace(
            self,
            startprob=self.startprob[perm],
            transmat=self.transmat[np.ix_(perm, perm)],
            means=self.means[perm],
            covars=self.covars[perm],
        )


def n_free_parameters(n_states: int, n_features: int) -> int:
    k, d = n_states, n_features
    return (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) // 2


def bic(loglik: float, n_states: int, n_features: int, n_obs: int) -> float:
    """``−2 log L + p log T``; lower is better (D-05)."""
    return -2.0 * loglik + n_free_parameters(n_states, n_features) * float(np.log(n_obs))


def _fit_once(z: F64, n_states: int, seed: int, max_iter: int, tol: float) -> GaussianHMM:
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=max_iter,
        tol=tol,
        random_state=seed,
        implementation="scaling",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(z)
    return model


def fit_hmm(
    x: F64,
    n_states: int,
    n_init: int,
    seed: int,
    fit_end: date,
    max_iter: int = 500,
    tol: float = 1e-6,
    min_occupancy: float = 0.05,
) -> HMMParams:
    """Best of ``n_init`` seeded Baum-Welch fits on standardized ``x`` (rows up to ``fit_end``).

    Restarts whose states don't each hold at least ``min_occupancy`` of the training days
    (filtered probability mass, R-03) are passed over while any restart passes.
    """
    from regport.regime.filter import forward_filter  # local: filter imports HMMParams

    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=1)
    z = (x - mean) / std
    seeds = np.random.SeedSequence(seed).generate_state(n_init)
    best: HMMParams | None = None
    best_ok = False
    for s in seeds:
        try:
            model = _fit_once(z, n_states, int(s), max_iter, tol)
            params = HMMParams(
                startprob=np.asarray(model.startprob_, dtype=np.float64),
                transmat=np.asarray(model.transmat_, dtype=np.float64),
                means=np.asarray(model.means_, dtype=np.float64),
                covars=np.asarray(model.covars_, dtype=np.float64),
                fit_end=fit_end,
                n_obs=int(z.shape[0]),
                loglik=float(model.score(z)),
                seed=int(s),
                x_mean=mean,
                x_std=std,
            )
            filtered, _, _ = forward_filter(x, params)
        except (np.linalg.LinAlgError, ValueError):
            continue  # a degenerate restart (singular emission covariance): try the next seed
        ok = bool(filtered.mean(axis=0).min() >= min_occupancy)
        if best is None or (ok, params.loglik) > (best_ok, best.loglik):
            best, best_ok = params, ok
    if best is None:
        raise ValueError(f"all {n_init} HMM restarts were degenerate (fit ending {fit_end})")
    return best
