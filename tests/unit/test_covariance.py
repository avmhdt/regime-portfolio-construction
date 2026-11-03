"""FR-09 estimators, PSD flooring, trailing windows (DQ-01, DQ-04) and their perturbation test."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.covariance import OAS, LedoitWolf

from regport.errors import LookaheadError
from regport.risk.covariance import (
    ANNUALIZATION,
    Estimator,
    ewma_cov,
    ewma_weights,
    ledoit_wolf_cov,
    oas_cov,
    sample_cov,
    trailing_window,
    wide_returns,
)
from regport.risk.psd import nearest_psd


def _returns(t: int = 300, n: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return rng.multivariate_normal(np.zeros(n), a @ a.T * 1e-4, size=t)


def _is_psd(m: np.ndarray) -> bool:
    return bool(np.allclose(m, m.T, atol=0) and np.linalg.eigvalsh(m).min() >= 0)


def test_sample_matches_numpy() -> None:
    x = _returns()
    assert np.allclose(sample_cov(x), np.cov(x, rowvar=False) * ANNUALIZATION, rtol=1e-12)


def test_shrinkage_matches_sklearn() -> None:
    x = _returns()
    assert np.allclose(ledoit_wolf_cov(x), LedoitWolf().fit(x).covariance_ * 252, rtol=1e-12)
    assert np.allclose(oas_cov(x), OAS().fit(x).covariance_ * 252, rtol=1e-12)


def test_ewma_equals_explicit_recursion() -> None:
    """Validation item (5): EWMA equals the RiskMetrics recursion within 1e-12."""
    x = _returns(t=504, n=6, seed=3)
    half_life = 63.0
    lam = 0.5 ** (1 / half_life)
    s = np.zeros((6, 6))
    for row in x:
        s = lam * s + (1 - lam) * np.outer(row, row)
    expected = s / (1 - lam ** x.shape[0]) * ANNUALIZATION
    got = ewma_cov(x, half_life)
    assert np.max(np.abs(got - expected)) <= 1e-12 * np.max(np.abs(expected))
    w = ewma_weights(504, half_life)
    assert w.sum() == pytest.approx(1.0, abs=1e-14) and w[-1] > w[0]


def test_outputs_symmetric_psd_and_annualized() -> None:
    x = _returns(t=40, n=30, seed=5)  # T close to N: the sample covariance is near-singular
    for name in ("sample", "ewma", "ledoit_wolf", "oas"):
        cov = Estimator(name).estimate(x, date(2024, 1, 2))
        assert cov.shape == (30, 30) and _is_psd(cov), name


def test_nearest_psd_floor() -> None:
    m = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1
    fixed = nearest_psd(m)
    values = np.linalg.eigvalsh(fixed)
    assert values.min() >= 1e-10 * np.mean(np.abs(np.linalg.eigvalsh(m))) * 0.999
    good = np.array([[2.0, 0.5], [0.5, 1.0]])
    assert np.array_equal(nearest_psd(good), good)


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="T x N"):
        sample_cov(np.ones(5))
    with pytest.raises(ValueError, match="NaN"):
        sample_cov(np.array([[0.1, np.nan], [0.2, 0.3]]))
    with pytest.raises(ValueError, match="unknown estimator"):
        Estimator("garch")


def _panel(n_days: int = 700, late_start: int = 400) -> pl.DataFrame:
    rng = np.random.default_rng(9)
    dates = [date(2010, 1, 4) + timedelta(days=i) for i in range(n_days)]
    rows = []
    for sym, start in (("AAA", 1), ("BBB", 1), ("CCC", late_start)):
        hist = 0
        for i, d in enumerate(dates):
            ret = rng.normal(0, 0.01) if i >= start else None
            rows.append({"date": d, "symbol": sym, "ret": ret, "eligible": hist >= 252})
            hist += ret is not None
    return pl.DataFrame(rows)


def test_trailing_window_excludes_sparse_assets() -> None:
    dates, symbols, r, e = wide_returns(_panel())
    asof = dates[690]
    win = trailing_window(dates, symbols, r, e, asof, 504)
    # CCC has 291 returns in the last 504 rows: eligible, but more than 5% null, so excluded.
    assert win.symbols == ["AAA", "BBB"] and win.excluded == ["CCC"]
    assert win.returns.shape == (504, 2) and win.dates[-1] == asof
    early = trailing_window(dates, symbols, r, e, dates[100], 504)
    assert early.symbols == [] and early.excluded == []  # nothing is eligible yet
    with pytest.raises(LookaheadError):
        trailing_window(dates, symbols, r, e, date(1999, 1, 1), 504)


def test_covariance_inputs_ignore_data_after_asof() -> None:
    """DQ-01 perturbation test: replacing data after t changes nothing at t."""
    dates, symbols, r, e = wide_returns(_panel(late_start=1))
    rng = np.random.default_rng(1)
    for t in rng.choice(np.arange(300, 690), size=20, replace=False):
        asof = dates[int(t)]
        base = trailing_window(dates, symbols, r, e, asof, 504)
        for fill in (rng.normal(0, 0.5, size=r.shape), np.full(r.shape, np.nan)):
            noisy = r.copy()
            noisy[int(t) + 1 :] = fill[int(t) + 1 :]
            other = trailing_window(dates, symbols, noisy, e, asof, 504)
            assert other.symbols == base.symbols
            for name in ("sample", "ewma", "ledoit_wolf", "oas"):
                a = Estimator(name).estimate(base.returns, asof)
                b = Estimator(name).estimate(other.returns, asof)
                assert np.max(np.abs(a - b)) <= 1e-12
