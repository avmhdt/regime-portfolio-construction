"""Validation items (4) and (6), plus closed forms for ERC and the M4 building blocks."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from regport.exec.costs import CostModel
from regport.exec.rebalance import decision_dates, rebalance
from regport.opt.equal_weight import equal_weight
from regport.opt.erc import ERCError, erc, risk_contributions
from regport.opt.hrp import correlation_distance, hrp, quasi_diagonal_order
from regport.opt.mv_tc import james_stein_mean, mean_variance_tc


def test_hrp_hand_computed_block_diagonal() -> None:
    """Item (6): two blocks {A, B} and {C, D}, correlation 0.5 inside, 0 across."""
    sd = np.array([1.0, 2.0, 1.0, 1.0])
    corr = np.array([[1, 0.5, 0, 0], [0.5, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0.5, 1]], float)
    cov = corr * np.outer(sd, sd)
    # Left cluster IVP (0.8, 0.2): var = 0.64 + 0.04*4 + 2*0.16*1 = 1.12; right (0.5, 0.5): 0.75.
    left = 0.75 / 1.87
    expected = np.array([left * 0.8, left * 0.2, (1 - left) / 2, (1 - left) / 2])
    assert np.max(np.abs(hrp(cov) - expected)) <= 1e-12
    order = quasi_diagonal_order(cov).tolist()
    assert {frozenset(order[:2]), frozenset(order[2:])} == {frozenset({0, 1}), frozenset({2, 3})}
    d = correlation_distance(cov)
    assert d[0, 1] == pytest.approx(0.5) and d[0, 2] == pytest.approx(math.sqrt(0.5))


def test_erc_two_assets_closed_form() -> None:
    """For two assets ERC is inverse volatility, whatever the correlation."""
    for rho in (-0.5, 0.0, 0.7):
        cov = np.array([[0.04, rho * 0.2 * 0.1], [rho * 0.2 * 0.1, 0.01]])
        assert np.allclose(erc(cov), [1 / 3, 2 / 3], atol=1e-12)


def test_erc_budgets_and_errors() -> None:
    cov = np.diag([0.01, 0.04, 0.09]) + 0.002
    w = erc(cov, budgets=np.array([0.5, 0.3, 0.2]))
    assert np.allclose(risk_contributions(w, cov), [0.5, 0.3, 0.2], atol=1e-10)
    assert erc(np.array([[0.04]])).tolist() == [1.0]
    with pytest.raises(ValueError, match="budgets"):
        erc(cov, budgets=np.array([1.0, -1.0, 1.0]))


def test_erc_fallback_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import regport.opt.erc as erc_mod

    cov = np.diag([0.01, 0.04, 0.09]) + 0.002
    monkeypatch.setattr(erc_mod, "_newton", lambda c, b, t, m: np.ones(3))
    w = erc(cov)  # coordinate descent from a poor start still converges
    assert np.max(np.abs(risk_contributions(w, cov) - 1 / 3)) <= 1e-10
    monkeypatch.setattr(erc_mod, "_coordinate_descent", lambda c, b, y, t, m: np.array([1, 5, 9.0]))
    with pytest.raises(ERCError):
        erc(cov)


def test_james_stein_shrinks_toward_grand_mean() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.0005, 0.01, size=(252, 8))
    cov = np.cov(x, rowvar=False)
    js = james_stein_mean(x, cov)
    raw = x.mean(axis=0) * 252
    assert np.all(np.abs(js - raw.mean()) <= np.abs(raw - raw.mean()) + 1e-15)
    three = james_stein_mean(x[:, :3], cov[:3, :3])
    assert np.allclose(three, x[:, :3].mean(axis=0) * 252)  # N <= 3: no shrinkage


def test_turnover_penalty_holds_previous_weights() -> None:
    cov = np.diag([0.04, 0.05, 0.06])
    prev = np.array([0.2, 0.3, 0.5])
    mu = np.array([0.10, 0.02, 0.01])
    stuck = mean_variance_tc(mu, cov, prev, 5.0, np.full(3, 10.0), 1.0)
    assert np.allclose(stuck, prev, atol=1e-6)
    free = mean_variance_tc(mu, cov, prev, 5.0, np.zeros(3), 1.0)
    assert free[0] > prev[0]


def test_costs_rebalance_and_calendar() -> None:
    model = CostModel({"A": 1.0, "B": 5.0})
    assert model.cost(np.array([0.1, -0.2]), ["A", "B"]) == pytest.approx(0.1e-4 + 0.2 * 5e-4)
    assert CostModel.uniform(["A", "B"], 10.0).rates(["B"]).tolist() == [1e-3]
    with pytest.raises(ValueError):
        CostModel({"A": -1.0})
    drift, target = np.array([0.51, 0.49]), np.array([0.5, 0.5])
    assert rebalance(drift, target, "threshold", 0.02).tolist() == drift.tolist()
    assert rebalance(drift, target, "calendar", 0.02).tolist() == target.tolist()
    assert rebalance(np.array([0.53, 0.47]), target, "threshold", 0.02).tolist() == [0.5, 0.5]
    with pytest.raises(ValueError):
        rebalance(drift, target, "sometimes", 0.02)  # type: ignore[arg-type]
    cal = [
        date(2024, 1, 30),
        date(2024, 1, 31),
        date(2024, 2, 1),
        date(2024, 2, 2),
        date(2024, 2, 5),
    ]
    assert decision_dates(cal, "monthly") == [date(2024, 1, 31)]
    assert decision_dates(cal, "weekly") == [date(2024, 2, 2)]
    with pytest.raises(ValueError):
        decision_dates(cal, "daily")  # type: ignore[arg-type]
    assert equal_weight(4).tolist() == [0.25] * 4
    with pytest.raises(ValueError):
        equal_weight(0)
