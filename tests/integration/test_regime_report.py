"""FR-34 / AC-H-09 regime report on a small synthetic walk-forward run."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from hmmlearn.hmm import GaussianHMM

from regport.config import ExperimentConfig
from regport.regime.hmm import fit_hmm
from regport.regime.runner import hmm_paths, run_hmm
from regport.report.figures import smoothed_for_illustration
from regport.report.regimes import first_crossing, stationary, write_regimes_report


def _features(n: int = 900) -> pl.DataFrame:
    rng = np.random.default_rng(8)
    dates = pl.date_range(date(2005, 1, 3), date(2012, 12, 31), "1d", eager=True)
    dates = dates.filter(dates.dt.weekday() <= 5)[:n]
    wild = (np.arange(n) % 300) > 230
    return pl.DataFrame(
        {
            "date": dates,
            "spy_logret": rng.normal(0, 1, n) * np.where(wild, 0.03, 0.008),
            "baa10y_diff": rng.normal(0, 0.03, n) + wild * 0.02,
            "spy_source": ["french_mkt_proxy"] * 50 + ["sharadar"] * (n - 50),
        }
    )


def test_smoother_matches_hmmlearn() -> None:
    x = _features().select("spy_logret", "baa10y_diff").to_numpy()
    params = fit_hmm(x, 2, 2, 3, date(2012, 1, 1), max_iter=50)
    model = GaussianHMM(2, "full")
    model.startprob_, model.transmat_ = params.startprob, params.transmat
    model.means_, model.covars_ = params.means, params.covars
    ours = smoothed_for_illustration(x, params)
    theirs = model.predict_proba((x - params.x_mean) / params.x_std)
    assert np.max(np.abs(ours - theirs)) <= 1e-10


def test_regimes_report(tmp_path: Path) -> None:
    feats = _features()
    cfg = ExperimentConfig.model_validate(
        {
            "sample": {
                "data_start": "2005-01-03",
                "oos_start": "2006-06-01",
                "estimator_selection_end": "2008-12-31",
            },
            "regime": {
                "hmm": {
                    "n_states": 2,
                    "min_train_days": 300,
                    "refit_every_days": 150,
                    "n_init": 2,
                    "max_iter": 40,
                    "tol": 1e-3,
                },
                "bocpd": {"max_run_length": 300, "prior": {"beta0": 1e-4}},
            },
            "evaluation": {
                "stress_windows": {
                    "gfc_2008": ["2007-10-09", "2008-03-09"],
                    "covid_2020": ["2008-06-02", "2008-09-30"],
                    "rates_2022": ["2009-01-05", "2009-06-30"],
                }
            },
        }
    )
    probs, _, _ = run_hmm(cfg, feats, tmp_path / "regime")
    _, fits = hmm_paths(cfg, feats, tmp_path / "regime")
    out = write_regimes_report(cfg, feats, probs, fits, tmp_path / "reports")
    text = out["report"].read_text(encoding="utf-8")
    assert "K = 2" in text and "Proxy" in text and "50 days" in text
    assert "| gfc_2008 |" in text
    assert out["filtered_vs_smoothed"].stat().st_size > 1000
    assert out["bocpd"].stat().st_size > 1000


def test_high_volatility_states() -> None:
    from regport.report.regimes import high_volatility_states

    params = fit_hmm(
        _features().select("spy_logret", "baa10y_diff").to_numpy(),
        2,
        2,
        3,
        date(2012, 1, 1),
        max_iter=50,
    )
    high = high_volatility_states(params)
    assert high.size == 1 and high[0] == int(np.argmax(params.covars[:, 0, 0]))


def test_helpers() -> None:
    a = np.array([[0.9, 0.1], [0.3, 0.7]])
    pi = stationary(a)
    assert np.allclose(pi @ a, pi) and abs(pi.sum() - 1.0) < 1e-12
    dates = [date(2020, 1, d) for d in range(1, 6)]
    p = np.array([0.1, 0.2, 0.6, 0.7, 0.2])
    assert first_crossing(dates, p, date(2020, 1, 4), date(2020, 1, 5), pad_days=2) == date(
        2020, 1, 3
    )
    assert first_crossing(dates, p * 0, date(2020, 1, 4), date(2020, 1, 5)) is None
