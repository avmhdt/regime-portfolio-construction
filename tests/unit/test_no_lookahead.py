"""AC-W-01: regime probabilities at t never use data after t (no smoothed-state leakage).

For 50 random t the data after t are replaced with noise, or cut off (the NaN case: nothing after
t exists), and the forward filter, the full walk-forward HMM with its refits, and BOCPD are re-run.
Every output dated <= t must be unchanged (max abs diff <= 1e-12). The contrast test shows that
hmmlearn's smoothed posteriors do change, which is why they are banned in ``src/``.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from hmmlearn.hmm import GaussianHMM

from regport.config import HMMConfig
from regport.data.panel import build_returns
from regport.errors import LookaheadError
from regport.regime.bocpd import NormalGammaPrior, bocpd
from regport.regime.filter import forward_filter
from regport.regime.hmm import fit_hmm
from regport.regime.walkforward import WalkForwardHMM

T = 420
RNG = np.random.default_rng(20261030)
TIMES = sorted(RNG.choice(np.arange(160, T - 5), size=50, replace=False).tolist())


def _series(seed: int = 1) -> tuple[np.ndarray, list[date]]:
    rng = np.random.default_rng(seed)
    state = (np.sin(np.arange(T) / 25.0) > 0.3).astype(int)
    x = np.column_stack(
        [rng.normal(0, 1, T) * np.where(state, 2.5, 0.8), rng.normal(0, 1, T) * 0.5 + state * 0.4]
    )
    dates = [date(2000, 1, 3) + timedelta(days=i) for i in range(T)]
    return x, dates


def _perturbed(x: np.ndarray, t: int, seed: int) -> np.ndarray:
    noisy = x.copy()
    noisy[t + 1 :] = np.random.default_rng(seed).normal(0, 5, size=noisy[t + 1 :].shape)
    return noisy


def test_forward_filter_ignores_the_future() -> None:
    x, dates = _series()
    params = fit_hmm(x[:200], 2, 2, 7, dates[199], max_iter=50)
    base_f, base_p, _ = forward_filter(x, params)
    for t in TIMES:
        for other in (_perturbed(x, t, t), x[: t + 1]):
            f, p, _ = forward_filter(other, params)
            assert np.max(np.abs(f[: t + 1] - base_f[: t + 1])) <= 1e-12
            assert np.max(np.abs(p[: t + 1] - base_p[: t + 1])) <= 1e-12
            assert np.allclose(f.sum(axis=1), 1.0, atol=1e-12)


def test_walk_forward_hmm_ignores_the_future() -> None:
    cfg = HMMConfig(min_train_days=150, refit_every_days=40, n_init=2, max_iter=40, tol=1e-4)
    x, dates = _series()
    model = WalkForwardHMM(cfg, n_states=2, seed=3)
    base = model.run(x, dates).probs
    assert base.filter(pl.col("params_fit_end") > pl.col("date")).is_empty()
    for t in TIMES:
        cut = dates[t]
        for other_x, other_dates in ((_perturbed(x, t, t), dates), (x[: t + 1], dates[: t + 1])):
            other = model.run(other_x, other_dates).probs.filter(pl.col("date") <= cut)
            ref = base.filter(pl.col("date") <= cut)
            assert other.select("date", "state", "refit_id").equals(
                ref.select("date", "state", "refit_id")
            )
            for col in ("p_filtered", "p_predicted"):
                assert float((other[col] - ref[col]).abs().max()) <= 1e-12


def test_bocpd_ignores_the_future() -> None:
    x, _ = _series()
    prior = NormalGammaPrior(0.0, 1.0, 2.0, 1.0)
    base = bocpd(x[:, 0], 100.0, prior, 300, 21)
    for t in TIMES:
        for other_x in (_perturbed(x, t, t)[:, 0], x[: t + 1, 0]):
            out = bocpd(other_x, 100.0, prior, 300, 21)
            for a, b in ((out.cp_prob_short, base.cp_prob_short), (out.rl_mean, base.rl_mean)):
                assert np.max(np.abs(a[: t + 1] - b[: t + 1])) <= 1e-12
            assert np.array_equal(out.rl_map[: t + 1], base.rl_map[: t + 1])


def test_smoothed_posteriors_do_look_ahead() -> None:
    """Contrast: hmmlearn's forward-backward posteriors at t change when the future changes."""
    x, dates = _series()
    params = fit_hmm(x[:200], 2, 2, 7, dates[199], max_iter=50)
    model = GaussianHMM(2, "full")
    model.startprob_, model.transmat_ = params.startprob, params.transmat
    model.means_, model.covars_ = params.means, params.covars
    z = (x - params.x_mean) / params.x_std
    t = 300
    smoothed = model.predict_proba(z)  # banned in src/, allowed in tests
    other = model.predict_proba((_perturbed(x, t, 1) - params.x_mean) / params.x_std)
    assert np.max(np.abs(smoothed[: t + 1] - other[: t + 1])) > 1e-6


def test_panel_inputs_ignore_the_future() -> None:
    dates = [date(2010, 1, 4) + timedelta(days=i) for i in range(400)]
    rng = np.random.default_rng(5)
    prices = pl.DataFrame(
        {
            "date": dates * 2,
            "symbol": ["A"] * 400 + ["B"] * 400,
            "adj_close": np.concatenate(
                [np.cumprod(1 + rng.normal(0, 0.01, 400)), np.cumprod(1 + rng.normal(0, 0.01, 400))]
            )
            * 100,
        }
    )
    base = build_returns(prices, dates, ["A", "B"], 252, 3)
    for t in (260, 300, 350):
        noisy = prices.with_columns(
            pl.when(pl.col("date") > dates[t])
            .then(pl.col("adj_close") * 3)
            .otherwise(pl.col("adj_close"))
        )
        other = build_returns(noisy, dates, ["A", "B"], 252, 3)
        keep = pl.col("date") <= dates[t]
        assert other.filter(keep).equals(base.filter(keep))


def test_regime_frame_rejects_rows_from_later_fits() -> None:
    cfg = HMMConfig(min_train_days=150, refit_every_days=100, n_init=1, max_iter=20, tol=1e-3)
    x, dates = _series()
    frame = WalkForwardHMM(cfg, n_states=2, seed=3).run(x, dates)
    assert frame.at(dates[200]).height == 2
    frame.probs = frame.probs.with_columns(pl.lit(dates[-1]).alias("params_fit_end"))
    with pytest.raises(LookaheadError):
        frame.at(dates[200])
