"""AC-W-07: validation items (1) and (2), and label alignment."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM
from hypothesis import given, settings
from hypothesis import strategies as st

from regport.regime.filter import forward_filter
from regport.regime.hmm import HMMParams, bic, fit_hmm, n_free_parameters
from regport.regime.labels import align_states, canonical_order, relabel


def _simulate(t: int, seed: int) -> np.ndarray:
    """Two states: means 0.05 / -0.10, sds 0.8 / 2.0, persistence 0.99 / 0.97."""
    rng = np.random.default_rng(seed)
    a = np.array([[0.99, 0.01], [0.03, 0.97]])
    s = np.empty(t, dtype=int)
    s[0] = 0
    for i in range(1, t):
        s[i] = rng.choice(2, p=a[s[i - 1]])
    mu, sd = np.array([0.05, -0.10]), np.array([0.8, 2.0])
    return (mu[s] + sd[s] * rng.normal(size=t))[:, None]


def _hmmlearn(params: HMMParams) -> GaussianHMM:
    model = GaussianHMM(params.n_states, "full")
    model.startprob_, model.transmat_ = params.startprob, params.transmat
    model.means_, model.covars_ = params.means, params.covars
    return model


def test_filter_matches_hmmlearn() -> None:
    """Item (1): log-likelihood within 1e-8 relative; last filtered row = last smoothed row."""
    x = np.column_stack([_simulate(3000, 1)[:, 0], np.random.default_rng(2).normal(size=3000)])
    params = fit_hmm(x, 3, 3, 11, date(2020, 1, 1), max_iter=100)
    filtered, predicted, loglik = forward_filter(x, params)
    model = _hmmlearn(params)
    z = (x - params.x_mean) / params.x_std
    reference = model.score(z)
    assert abs(loglik - reference) <= 1e-8 * abs(reference)
    assert np.max(np.abs(filtered[-1] - model.predict_proba(z)[-1])) <= 1e-10
    assert np.allclose(predicted, filtered @ params.transmat, atol=1e-15)


def test_parameter_recovery() -> None:
    """Item (2): T = 5,000; means within 0.1, sds within 10%, diagonal of A within 0.02."""
    x = _simulate(5000, 7)
    params = fit_hmm(x, 2, 5, 3, date(2020, 1, 1), max_iter=300)
    params = params.permute(canonical_order(params))
    means = params.raw_means()[:, 0]
    sds = np.sqrt(params.raw_covars()[:, 0, 0])
    assert np.all(np.abs(means - [0.05, -0.10]) <= 0.1)
    assert np.all(np.abs(sds / [0.8, 2.0] - 1) <= 0.10)
    assert np.all(np.abs(np.diag(params.transmat) - [0.99, 0.97]) <= 0.02)


def test_bic_and_parameter_count() -> None:
    assert n_free_parameters(2, 2) == 1 + 2 + 4 + 6
    assert bic(-100.0, 2, 2, 1000) == pytest.approx(200 + 13 * np.log(1000))


def _params(k: int, seed: int) -> HMMParams:
    rng = np.random.default_rng(seed)
    means = np.column_stack([rng.normal(0, 3, k), rng.normal(0, 3, k)])
    covars = np.array([np.diag([0.5 + i, 1.0 + 0.3 * i]) for i in range(k)])
    trans = rng.dirichlet(np.ones(k) * 5, size=k)
    return HMMParams(
        np.full(k, 1 / k),
        trans,
        means,
        covars,
        date(2020, 1, 1),
        100,
        0.0,
        0,
        np.zeros(2),
        np.ones(2),
    )


@settings(max_examples=50, deadline=None)
@given(st.integers(2, 5), st.integers(0, 10_000), st.randoms())
def test_align_states_recovers_any_permutation(k: int, seed: int, rnd: object) -> None:
    params = _params(k, seed)
    perm = np.array(rnd.sample(range(k), k))  # type: ignore[attr-defined]
    shuffled = params.permute(perm)
    back = align_states(params, shuffled)
    assert np.array_equal(shuffled.permute(back).means, params.means)


def test_alignment_with_estimation_noise() -> None:
    """100 of 100 simulated refits with separated states: the true permutation is recovered."""
    rng = np.random.default_rng(0)
    hits = 0
    for trial in range(100):
        params = _params(3, trial)
        perm = rng.permutation(3)
        noisy = params.permute(perm)
        noisy = HMMParams(
            noisy.startprob,
            noisy.transmat,
            noisy.means + rng.normal(0, 0.05, noisy.means.shape),
            noisy.covars * rng.uniform(0.95, 1.05, (3, 1, 1)),
            noisy.fit_end,
            noisy.n_obs,
            0.0,
            0,
            noisy.x_mean,
            noisy.x_std,
        )
        back = align_states(params, noisy)
        hits += np.array_equal(perm[back], np.arange(3))
    assert hits == 100


def test_relabel_returns_the_alignment() -> None:
    params = _params(3, 1)
    first, perm = relabel(None, params)
    assert np.array_equal(perm, np.arange(3))
    assert np.all(np.diff(first.covars[:, 0, 0]) >= 0)
    # Swap two variances so the canonical order disagrees with the matching.
    swapped = HMMParams(
        first.startprob,
        first.transmat,
        first.means,
        first.covars[[1, 0, 2]],
        first.fit_end,
        1,
        0.0,
        0,
        first.x_mean,
        first.x_std,
    )
    aligned, perm = relabel(first, swapped)
    assert not np.array_equal(perm, np.arange(3))
    assert np.allclose(aligned.means, first.means)
    # The same refit again keeps the same mapping: not a new switch.
    _, again = relabel(aligned, swapped)
    assert np.array_equal(again, perm)
