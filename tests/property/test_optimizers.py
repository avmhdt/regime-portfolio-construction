"""AC-W-02 property tests on random positive-definite matrices (N = 2..50, condition up to 1e6)."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from regport.opt.constraints import check_constraints
from regport.opt.erc import erc, risk_contributions
from regport.opt.hrp import hrp
from regport.opt.minvar import min_variance
from regport.opt.mv_tc import mean_variance_tc


@st.composite
def pd_matrices(draw: st.DrawFn) -> np.ndarray:
    n = draw(st.integers(min_value=2, max_value=50))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    log_cond = draw(st.floats(min_value=0.0, max_value=6.0))
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eig = np.logspace(0, -log_cond, n) * 0.04
    cov = (q * eig) @ q.T
    return 0.5 * (cov + cov.T)


@settings(max_examples=60, deadline=None)
@given(pd_matrices())
def test_erc_equal_risk_contributions(cov: np.ndarray) -> None:
    w = erc(cov)
    check_constraints(w, None)
    rc = risk_contributions(w, cov)
    assert np.max(np.abs(rc - 1.0 / cov.shape[0])) <= 1e-6
    assert abs(rc.sum() - 1.0) <= 1e-12


@settings(max_examples=60, deadline=None)
@given(pd_matrices())
def test_hrp_long_only_fully_invested(cov: np.ndarray) -> None:
    w = hrp(cov)
    assert np.all(w >= 0) and abs(w.sum() - 1.0) <= 1e-12
    check_constraints(w, None)


@settings(max_examples=40, deadline=None)
@given(pd_matrices())
def test_min_variance_beats_feasible_portfolios(cov: np.ndarray) -> None:
    n = cov.shape[0]
    w = min_variance(cov, 0.40)
    check_constraints(w, 0.40)
    var = float(w @ cov @ w)
    ones = np.full(n, 1.0 / n)
    tol = 1e-7 * float(np.mean(np.diag(cov)))
    assert var <= float(ones @ cov @ ones) + tol
    rng = np.random.default_rng(n)
    cap = max(0.40, 1.0 / n)
    for _ in range(100):
        other = rng.dirichlet(np.ones(n))
        if other.max() <= cap:
            assert var <= float(other @ cov @ other) + tol


@settings(max_examples=30, deadline=None)
@given(pd_matrices(), st.floats(min_value=0.0, max_value=0.01))
def test_mean_variance_constraints(cov: np.ndarray, kappa: float) -> None:
    n = cov.shape[0]
    rng = np.random.default_rng(n)
    w = mean_variance_tc(
        rng.normal(0.05, 0.05, n), cov, np.full(n, 1.0 / n), 5.0, np.full(n, kappa), 0.40
    )
    check_constraints(w, 0.40)
