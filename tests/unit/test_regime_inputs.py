"""FR-16 regime mixture, forward weights, and the feature builder's point-in-time rules."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from regport.regime.features import (
    PROXY_LABEL,
    build_features,
    known_daily_values,
    macro_change,
)
from regport.risk.regime_cov import forward_weights, regime_mixture_cov


def test_forward_weights() -> None:
    a = np.array([[0.9, 0.1], [0.2, 0.8]])
    pi = forward_weights(np.array([1.0, 0.0]), a, 21)
    expected = np.mean(
        [np.array([1.0, 0.0]) @ np.linalg.matrix_power(a, j) for j in range(1, 22)], axis=0
    )
    assert np.allclose(pi, expected) and pi.sum() == pytest.approx(1.0)


def test_mixture_reduces_to_weighted_moments() -> None:
    rng = np.random.default_rng(0)
    calm = rng.multivariate_normal([0, 0], [[1e-4, 2e-5], [2e-5, 1e-4]], 300)
    wild = rng.multivariate_normal([0, 0], [[9e-4, 6e-4], [6e-4, 9e-4]], 300)
    r = np.vstack([calm, wild])
    p = np.vstack([np.tile([1.0, 0.0], (300, 1)), np.tile([0.0, 1.0], (300, 1))])
    cov_calm, mu_calm = regime_mixture_cov(r, p, np.array([1.0, 0.0]), 0.0)
    assert np.allclose(cov_calm, np.cov(calm, rowvar=False) * 252, rtol=1e-10)
    assert np.allclose(mu_calm, calm.mean(axis=0) * 252)
    cov_mix, _ = regime_mixture_cov(r, p, np.array([0.5, 0.5]), 0.0)
    assert cov_calm[0, 0] < cov_mix[0, 0] < np.cov(wild, rowvar=False)[0, 0] * 252
    shrunk, _ = regime_mixture_cov(r, p, np.array([1.0, 0.0]), 1.0)
    assert shrunk[0, 1] == pytest.approx(0.0, abs=1e-15)


def test_rare_state_falls_back_to_unweighted_moments() -> None:
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, (200, 3))
    p = np.tile([1.0, 0.0], (200, 1))
    p[0] = [0.0, 1.0]  # one effective observation for state 1
    cov, _ = regime_mixture_cov(r, p, np.array([0.0, 1.0]), 0.0)
    assert np.allclose(cov, np.cov(r, rowvar=False) * 252)
    with pytest.raises(ValueError, match="T x K"):
        regime_mixture_cov(r, p[:10], np.array([0.5, 0.5]), 0.0)


def _days(n: int, start: date) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_known_daily_values_respect_the_lag() -> None:
    cal = _days(5, date(2024, 1, 1))  # Mon 1 .. Fri 5
    fred = pl.DataFrame(
        {
            "series_id": "BAA10Y",
            "obs_date": [cal[0], cal[2]],
            "value": [2.0, 2.5],
            "available_date": [None, None],
        }
    )
    vals = known_daily_values(fred, "BAA10Y", cal)
    assert np.isnan(vals[0]) and vals.tolist()[1:] == [2.0, 2.0, 2.5, 2.5]


def test_features_splice_the_french_proxy() -> None:
    cal = _days(60, date(1997, 11, 3))
    first_spy = date(1998, 1, 2)
    returns = pl.DataFrame(
        {"date": [d for d in cal if d >= first_spy], "symbol": "SPY", "ret": 0.001}
    )
    french = pl.DataFrame(
        {
            "dataset": "ff3_daily",
            "date": [d for d in cal for _ in range(2)],
            "series": ["Mkt-RF", "RF"] * len(cal),
            "ret": [0.002, 0.0001] * len(cal),
        }
    )
    fred = pl.DataFrame(
        {
            "series_id": "BAA10Y",
            "obs_date": cal,
            "value": np.linspace(2, 3, len(cal)),
            "available_date": cal,
        }
    )
    feats = build_features(returns, french, fred, None, date(1997, 11, 1), "A")
    before = feats.filter(pl.col("date") < first_spy)
    assert set(before["spy_source"]) == {PROXY_LABEL}
    assert np.allclose(before["spy_logret"], np.log1p(0.0021))
    assert set(feats.filter(pl.col("date") >= first_spy)["spy_source"]) == {"sharadar"}
    assert feats["baa10y_diff"].is_finite().all()
    with pytest.raises(ValueError, match="vintages"):
        build_features(returns, french, fred, None, date(1997, 11, 1), "B")


def test_macro_change_uses_one_vintage() -> None:
    # The March value is first published at 4.0 then revised to 4.4; the 3-month change as
    # known in May uses May's vintage for both ends.
    rows = [
        ("2024-01-01", 3.0, "2024-02-02", "9999-12-31"),
        ("2024-02-01", 3.5, "2024-03-08", "9999-12-31"),
        ("2024-03-01", 4.0, "2024-04-05", "2024-05-02"),
        ("2024-03-01", 4.4, "2024-05-03", "9999-12-31"),
        ("2024-04-01", 3.9, "2024-05-03", "9999-12-31"),
    ]
    vintages = pl.DataFrame(
        {
            "series_id": "X",
            "obs_date": [date.fromisoformat(r[0]) for r in rows],
            "value": [r[1] for r in rows],
            "realtime_start": [date.fromisoformat(r[2]) for r in rows],
            "realtime_end": [date.fromisoformat(r[3]) for r in rows],
        }
    )
    cal = [date(2024, 4, 10), date(2024, 5, 7)]
    pit = macro_change(vintages, "X", cal, 2, False, "pit")
    assert pit.tolist() == pytest.approx([4.0 - 3.0, 3.9 - 3.5])
    latest = macro_change(vintages, "X", cal, 2, False, "latest")
    assert latest.tolist() == pytest.approx([4.4 - 3.0, 3.9 - 3.5])
