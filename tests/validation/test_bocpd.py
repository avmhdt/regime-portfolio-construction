"""AC-W-07 item (3) for BOCPD, plus exactness against a plain reference implementation.

The hyperparameters are the ones chosen on pre-2005 data (``configs/base.yaml``: hazard 100,
alpha0 2), with ``beta0 = (alpha0 - 1) sigma^2`` for the synthetic base volatility. "Variance
shifts (x3)" is read as the standard deviation tripling; see docs/decisions.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import t as student

from regport.config import load_config
from regport.regime.bocpd import NormalGammaPrior, bocpd

ROOT = Path(__file__).resolve().parents[2]
CFG = load_config(ROOT / "configs/base.yaml").regime.bocpd
SIGMA = 0.01
SEG = 400
SHIFTS = [SEG * i for i in range(1, 6)]
PRIOR = NormalGammaPrior(0.0, 1.0, CFG.prior.alpha0, (CFG.prior.alpha0 - 1.0) * SIGMA**2)


def _reference(x: np.ndarray, lam: float, p: NormalGammaPrior) -> np.ndarray:
    """Textbook Adams-MacKay in probability space, no truncation; returns the posterior."""
    h = 1.0 / lam
    post = np.zeros((x.size + 1, x.size + 1))
    post[0, 0] = 1.0
    mu, k, a, b = (np.array([v]) for v in (p.mu0, p.kappa0, p.alpha0, p.beta0))
    for t, xt in enumerate(x):
        pred = student.pdf(xt, 2 * a, loc=mu, scale=np.sqrt(b * (k + 1) / (a * k)))
        post[t + 1, 1 : t + 2] = post[t, : t + 1] * pred * (1 - h)
        post[t + 1, 0] = np.sum(post[t, : t + 1] * pred * h)
        post[t + 1] /= post[t + 1].sum()
        mu, b = (
            np.concatenate([[p.mu0], (k * mu + xt) / (k + 1)]),
            np.concatenate([[p.beta0], b + k * (xt - mu) ** 2 / (2 * (k + 1))]),
        )
        k, a = np.concatenate([[p.kappa0], k + 1]), np.concatenate([[p.alpha0], a + 0.5])
    return post[1:]


def test_matches_reference_implementation() -> None:
    x = np.random.default_rng(1).normal(0, SIGMA, 300)
    res = bocpd(x, CFG.hazard_lambda_days, PRIOR, 1500, 21, keep_posterior=True)
    ref = _reference(x, CFG.hazard_lambda_days, PRIOR)
    assert res.posterior is not None
    assert np.max(np.abs(res.posterior[:, : ref.shape[1]] - ref)) <= 1e-12


@settings(max_examples=25, deadline=None)
@given(st.integers(0, 10_000), st.integers(5, 60))
def test_run_length_posterior_sums_to_one(seed: int, max_rl: int) -> None:
    x = np.random.default_rng(seed).normal(0, 1, 200)
    res = bocpd(x, 50.0, NormalGammaPrior(0, 1, 2, 1), max_rl, 5, keep_posterior=True)
    assert res.posterior is not None
    assert np.max(np.abs(res.posterior.sum(axis=1) - 1.0)) <= 1e-10


def _shifted(rng: np.random.Generator) -> np.ndarray:
    return np.concatenate([rng.normal(0, [SIGMA, 3 * SIGMA][i % 2], SEG) for i in range(6)])


def test_detects_variance_shifts() -> None:
    """Item (3a): at least 4 of 5 shifts flagged (cp_prob_short > 0.5) within 10 observations."""
    rng = np.random.default_rng(20261030)
    for _ in range(5):
        res = bocpd(
            _shifted(rng), CFG.hazard_lambda_days, PRIOR, CFG.max_run_length, CFG.short_run_days
        )
        flagged = sum(bool((res.cp_prob_short[s : s + 11] > 0.5).any()) for s in SHIFTS)
        assert flagged >= 4


def _false_alarms(lam: float, n: int = 10_000, seed: int = 4) -> float:
    x = np.random.default_rng(seed).normal(0, SIGMA, n)
    p = bocpd(x, lam, PRIOR, CFG.max_run_length, CFG.short_run_days).cp_prob_short[100:]
    return int(((p[1:] > 0.5) & (p[:-1] <= 0.5)).sum()) * 1000 / (p.size - 1)


@pytest.mark.xfail(
    strict=True,
    reason="Documented failure pending an owner ruling: at the "
    "hazard chosen on pre-2005 data (100 days) BOCPD raises about 6 false alarms "
    "per 1,000 stationary days, against <= 1 (docs/decisions.md, M5)",
)
def test_false_alarm_rate_at_chosen_hyperparameters() -> None:
    """Item (3b): at most 1 false alarm per 1,000 stationary observations."""
    assert _false_alarms(CFG.hazard_lambda_days) <= 1.0


def test_false_alarm_rate_depends_on_the_hazard() -> None:
    """Diagnostic for the ruling: the same code meets item (3b) at a 500-day hazard."""
    at_500 = np.mean([_false_alarms(500.0, seed=s) for s in range(5)])
    at_chosen = np.mean([_false_alarms(CFG.hazard_lambda_days, seed=s) for s in range(5)])
    assert at_500 <= 1.0 < at_chosen


def test_rejects_non_finite_input() -> None:
    with pytest.raises(ValueError, match="NaN"):
        bocpd(np.array([0.0, np.nan]), 100.0, PRIOR, 10)
