"""FR-26 metrics on hand-checkable series, and the fallback projection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from regport.eval.metrics import max_drawdown, performance_metrics
from regport.opt.constraints import check_constraints, project_capped


def test_max_drawdown() -> None:
    assert max_drawdown(np.array([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
    assert max_drawdown(np.array([-0.2, 0.1])) == pytest.approx(-0.2)
    assert max_drawdown(np.array([0.01, 0.02])) == 0.0


def test_performance_metrics_known_values() -> None:
    ret = np.array([0.01, -0.01] * 126)
    rf = np.zeros(252)
    m = performance_metrics(
        ret, rf, np.full(252, 0.1), np.full(252, 1e-5), np.full(252, 0.25), benchmark=np.zeros(252)
    )
    assert m["annual_return"] == pytest.approx((1.01 * 0.99) ** 126 - 1)
    assert m["annual_volatility"] == pytest.approx(np.std(ret, ddof=1) * math.sqrt(252))
    assert m["sharpe_excess"] == pytest.approx(0.0, abs=1e-12)
    assert m["annual_turnover"] == pytest.approx(25.2) and m["annual_cost_drag"] == pytest.approx(
        252e-5
    )
    assert m["effective_n_assets"] == pytest.approx(4.0)
    assert m["hit_rate_vs_benchmark"] == pytest.approx(0.5)
    assert m["ceq"] == pytest.approx(252 * (0 - 0.5 * np.var(ret, ddof=1)))
    assert abs(m["skewness"]) < 1e-9 and m["excess_kurtosis"] == pytest.approx(-2.0)
    up = performance_metrics(np.full(252, 0.001), rf, rf, rf, np.ones(252))
    assert up["calmar"] is None and up["sortino"] == 0.0 and up["sharpe_excess"] == 0.0


def test_project_capped() -> None:
    w = project_capped(np.array([0.5, 0.3, 0.2]), 0.40)
    check_constraints(w, 0.40)
    assert w[0] == pytest.approx(0.40) and w[1] / w[2] == pytest.approx(1.5)
    assert np.allclose(project_capped(np.array([0.0, 0.0]), None), [0.5, 0.5])
    many = project_capped(np.array([0.45, 0.45, 0.05, 0.05]), 0.40)
    check_constraints(many, 0.40)
