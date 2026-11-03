"""FR-17 minimum variance and the constraint checks used after every optimization."""

from __future__ import annotations

import numpy as np
import pytest

from regport.errors import ConstraintViolation
from regport.opt import minvar
from regport.opt.constraints import check_constraints, clean_weights, effective_cap
from regport.opt.minvar import SolverError, min_variance


def _cov(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return a @ a.T / n + np.eye(n) * 0.1


def test_closed_form_when_no_bound_binds() -> None:
    """Validation item (4): equals Σ⁻¹1 / (1'Σ⁻¹1) within 1e-6 when no bound binds."""
    cov = np.diag([0.04, 0.05, 0.06, 0.08]) + 0.005
    inv = np.linalg.solve(cov, np.ones(4))
    expected = inv / inv.sum()
    assert expected.min() > 0 and expected.max() < 1
    assert np.max(np.abs(min_variance(cov, 1.0) - expected)) <= 1e-6


def test_cap_binds_and_variance_is_minimal() -> None:
    rng = np.random.default_rng(2)
    for seed in range(5):
        cov = _cov(8, seed)
        w = min_variance(cov, 0.40)
        check_constraints(w, 0.40)
        var = w @ cov @ w
        assert var <= np.full(8, 1 / 8) @ cov @ np.full(8, 1 / 8) + 1e-12
        for _ in range(20):
            other = rng.dirichlet(np.ones(8))
            if other.max() <= 0.40:
                assert var <= other @ cov @ other + 1e-10


def test_small_universes() -> None:
    assert min_variance(np.array([[0.04]]), 0.4).tolist() == [1.0]
    w = min_variance(np.array([[0.04, 0.0], [0.0, 0.09]]), 0.40)  # cap raised to 1/2
    assert np.allclose(w, [0.5, 0.5], atol=1e-7)
    assert effective_cap(0.4, 2) == 0.5 and effective_cap(None, 3) == 1.0


def test_solver_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken:
        status = "infeasible"

        def solve(self, solver: str) -> None:
            if solver == "OSQP":
                raise minvar.cp.error.SolverError("boom")

    variable = type("V", (), {"value": None})()
    param = type("P", (), {"value": None})()
    monkeypatch.setattr(minvar, "_problem", lambda n: (Broken(), variable, param, param))
    with pytest.raises(SolverError, match="OSQP: boom"):
        min_variance(np.eye(3) * 0.04, 0.5)


def test_check_constraints() -> None:
    check_constraints(np.array([0.4, 0.4, 0.2]), 0.40)
    check_constraints(np.array([-5e-10, 0.6, 0.4 + 5e-10]), None)
    for bad, match in (
        (np.array([0.5, 0.6]), "sum"),
        (np.array([1.1, -0.1]), "negative"),
        (np.array([0.45, 0.3, 0.25]), "above cap"),
        (np.array([np.nan, 1.0]), "NaN"),
    ):
        with pytest.raises(ConstraintViolation, match=match):
            check_constraints(bad, 0.40)


def test_clean_weights() -> None:
    out = clean_weights(np.array([-1e-10, 0.5, 0.5]))
    assert out.min() == 0.0 and out.sum() == pytest.approx(1.0, abs=1e-15)
    with pytest.raises(ConstraintViolation, match="clip tolerance"):
        clean_weights(np.array([-1e-6, 0.5, 0.5]))
