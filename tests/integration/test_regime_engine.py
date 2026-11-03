"""Regime modes in the engine (FR-16) and the regime runner and CLI, on synthetic inputs."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import yaml
from typer.testing import CliRunner

from regport.backtest.engine import BacktestError, Panel, run_backtest
from regport.cli import app
from regport.config import ExperimentConfig
from regport.data.panel import write_parquet
from regport.regime.runner import RegimeOutputs, run_bocpd, run_hmm

K = 2


def _cfg(tmp_path: Path, universe: Path, usage: str, name: str = "erc") -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "data": {"universe_file": str(universe), "calendar_symbol": "AEQ"},
            "sample": {
                "data_start": "2001-01-02",
                "oos_start": "2003-01-02",
                "estimator_selection_end": "2006-12-29",
            },
            "regime": {"usage": usage, "hmm": {"n_states": K}},
            "risk": {"estimator": "ewma"},
            "optimizer": {"name": name},
            "logging": {"experiment_log": str(tmp_path / "results" / "trial_log.csv")},
        }
    )


def _fake_outputs(panel: Panel) -> RegimeOutputs:
    dates = panel.dates
    n = len(dates)
    calm = (np.sin(np.arange(n) / 180.0) <= 0.6).astype(float)
    p = np.column_stack([0.1 + 0.8 * calm, 0.9 - 0.8 * calm])
    hmm = pl.DataFrame(
        {
            "date": np.repeat(np.array(dates, dtype="datetime64[D]"), K),
            "model": "hmm",
            "state": np.tile(np.arange(K, dtype=np.int8), n),
            "p_filtered": p.ravel(),
            "p_predicted": p.ravel(),
            "params_fit_end": np.repeat(np.array(dates, dtype="datetime64[D]"), K),
            "refit_id": np.int32(0),
        }
    ).with_columns(pl.col("date", "params_fit_end").cast(pl.Date))
    cp = np.where(np.arange(n) % 40 < 5, 0.9, 0.1)
    bocpd = pl.DataFrame({"date": dates, "cp_prob_short": cp})
    return RegimeOutputs(
        hmm, {0: np.array([[0.95, 0.05], [0.1, 0.9]])}, bocpd, pl.DataFrame({"date": dates}), []
    )


@pytest.fixture
def panel(synthetic_panel: tuple[pl.DataFrame, pl.DataFrame]) -> Panel:
    return Panel.from_frames(*synthetic_panel)


@pytest.mark.parametrize(
    ("usage", "name"),
    [("hmm", "erc"), ("bocpd", "min_variance"), ("hmm_bocpd", "hrp"), ("hmm", "mv_turnover")],
)
def test_regime_modes(
    usage: str, name: str, tmp_path: Path, universe_file: Path, panel: Panel
) -> None:
    cfg = _cfg(tmp_path, universe_file, usage, name)
    result = run_backtest(cfg, panel=panel, regime=_fake_outputs(panel))
    m = result.metrics
    if "hmm" in usage:
        assert m["hmm_mixtures"] > 0
    if "bocpd" in usage:
        assert m["bocpd_triggers"] > 0
    base = run_backtest(_cfg(tmp_path, universe_file, "none", name), panel=panel)
    assert not np.allclose(result.returns["ret"].to_numpy(), base.returns["ret"].to_numpy())


def test_regime_rows_from_later_fits_are_refused(
    tmp_path: Path, universe_file: Path, panel: Panel
) -> None:
    out = _fake_outputs(panel)
    assert out.hmm is not None
    out.hmm = out.hmm.with_columns(pl.lit(panel.dates[-1]).alias("params_fit_end"))
    with pytest.raises(BacktestError, match="uses a fit"):
        run_backtest(_cfg(tmp_path, universe_file, "hmm"), panel=panel, regime=out)


def _features(n: int = 700) -> pl.DataFrame:
    rng = np.random.default_rng(3)
    calm = np.sin(np.arange(n) / 40.0) <= 0.4
    dates = pl.date_range(date(2000, 1, 3), date(2010, 1, 1), "1d", eager=True)
    dates = dates.filter(dates.dt.weekday() <= 5)[:n]
    return pl.DataFrame(
        {
            "date": dates,
            "spy_logret": rng.normal(0, 1, n) * np.where(calm, 0.008, 0.02),
            "baa10y_diff": rng.normal(0, 0.03, n),
            "spy_source": "sharadar",
        }
    )


def _small_cfg(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "regime": {
                "hmm": {
                    "n_states": 2,
                    "min_train_days": 300,
                    "refit_every_days": 100,
                    "n_init": 2,
                    "max_iter": 30,
                    "tol": 1e-3,
                },
                "bocpd": {"max_run_length": 200},
            },
            "logging": {"experiment_log": str(tmp_path / "trial_log.csv")},
        }
    )


def test_runner_caches(tmp_path: Path) -> None:
    cfg = _small_cfg(tmp_path)
    feats = _features()
    probs, transmats, _ = run_hmm(cfg, feats, tmp_path / "regime")
    assert probs["date"].n_unique() == 700 - 299 and len(transmats) == 5
    again, _, _ = run_hmm(cfg, feats, tmp_path / "regime")
    assert again.equals(probs)
    frame = run_bocpd(cfg, feats, tmp_path / "regime")
    assert frame.height == 700 and set(frame["spy_source"]) == {"sharadar"}
    assert run_bocpd(cfg, feats, tmp_path / "regime").equals(frame)
    assert len(list((tmp_path / "regime").iterdir())) == 3
    with pytest.raises(ValueError, match="auto"):
        auto = ExperimentConfig.model_validate({"regime": {"hmm": {"n_states": "auto"}}})
        run_hmm(auto, feats, tmp_path / "other")


def test_cli_regime_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    feats = _features()
    dates = feats["date"].to_list()
    proc = tmp_path / "data" / "processed"
    write_parquet(
        pl.DataFrame(
            {"date": dates, "symbol": "SPY", "ret": np.expm1(feats["spy_logret"].to_numpy())}
        ),
        proc / "returns_daily.parquet",
    )
    write_parquet(
        pl.DataFrame(
            {
                "dataset": "ff3_daily",
                "date": [dates[0]] * 2,
                "series": ["Mkt-RF", "RF"],
                "ret": [0.0, 0.0],
            }
        ),
        proc / "french_daily.parquet",
    )
    baa = np.cumsum(feats["baa10y_diff"].to_numpy()) + 2
    write_parquet(
        pl.DataFrame(
            {"series_id": "BAA10Y", "obs_date": dates, "value": baa, "available_date": dates}
        ),
        proc / "fred_daily.parquet",
    )
    params: dict[str, Any] = _small_cfg(tmp_path).to_params()
    params["sample"]["data_start"] = str(dates[0])
    params["data"]["universe_file"] = str(
        Path(__file__).resolve().parents[2] / "configs" / "universe.yaml"
    )
    config = tmp_path / "cfg.yaml"
    config.write_text(yaml.safe_dump(params), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REGPORT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("REGPORT_RESULTS_DIR", str(tmp_path / "results"))
    out = CliRunner().invoke(app, ["regime", "run", "--config", str(config), "--features", "A"])
    assert out.exit_code == 0, out.output
    assert "hmm:" in out.output and "bocpd:" in out.output
