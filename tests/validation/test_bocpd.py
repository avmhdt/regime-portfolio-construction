"""AC-W-07 item (3) for BOCPD, plus exactness against a plain reference implementation.

The hyperparameters are the configured ones (``configs/base.yaml``: hazard 500 after the
constrained reselection, alpha0 2), with ``beta0 = (alpha0 - 1) sigma^2`` for the synthetic base
volatility. "Variance shifts (x3)" is read as the standard deviation tripling (owner ruling);
detection under the literal variance reading is reported, not asserted.
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
from regport.regime.selection import ValidationResult, validate_setting

ROOT = Path(__file__).resolve().parents[2]
CFG = load_config(ROOT / "configs/base.yaml").regime.bocpd
SIGMA = 0.01
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


def _check(lam: float, replications: int) -> ValidationResult:
    return validate_setting(
        (lam, CFG.prior.alpha0, SIGMA, CFG.max_run_length, CFG.short_run_days, 20261030),
        replications,
    )


def test_item_3_at_the_configured_hazard() -> None:
    """Item (3) at the reselected hazard (fast: 5 replications; the full run is marked slow)."""
    res = _check(CFG.hazard_lambda_days, 5)
    assert res.detect_pass_reps >= 0.9 * 5  # (3a): >= 4 of 5 shifts within 10 observations
    assert res.false_alarms_per_1000 <= 1.0  # (3b)


@pytest.mark.slow
def test_item_3_full_replications() -> None:
    """The reselection's own check: 20 replications at the configured hazard."""
    assert _check(CFG.hazard_lambda_days, 20).passes


@pytest.mark.slow
def test_original_hazard_failed_item_3b() -> None:
    """On record (preregistration deviation): hazard 1/100 failed the false-alarm bound."""
    res = _check(100.0, 20)
    assert res.false_alarms_per_1000 > 1.0 and not res.passes


def test_rejects_non_finite_input() -> None:
    with pytest.raises(ValueError, match="NaN"):
        bocpd(np.array([0.0, np.nan]), 100.0, PRIOR, 10)
