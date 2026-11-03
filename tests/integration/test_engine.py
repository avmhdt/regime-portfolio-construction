"""AC-W-10 accounting, DQ-06 timing, FR-25 logging and NFR-09 on the synthetic panel."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import yaml
from typer.testing import CliRunner

from regport.backtest import engine
from regport.backtest.engine import BacktestError, Panel, run_backtest
from regport.cli import app
from regport.config import ExperimentConfig
from regport.data.panel import write_parquet
from regport.opt.erc import risk_contributions
from regport.opt.minvar import SolverError
from regport.tracking.experiment_log import read_trial_log

OPTIMIZERS = ["equal_weight", "min_variance", "erc", "hrp", "mv_turnover"]


def make_cfg(tmp_path: Path, universe: Path, **overrides: Any) -> ExperimentConfig:
    data: dict[str, Any] = {
        "data": {"universe_file": str(universe), "calendar_symbol": "AEQ"},
        "sample": {
            "data_start": "2001-01-02",
            "oos_start": "2003-01-02",
            "estimator_selection_end": "2006-12-29",
        },
        "regime": {"usage": "none"},
        "risk": {"estimator": "ewma"},
        "logging": {"experiment_log": str(tmp_path / "results" / "trial_log.csv")},
    }
    for key, value in overrides.items():
        section, field = key.split("__")
        data.setdefault(section, {})[field] = value
    return ExperimentConfig.model_validate(data)


@pytest.fixture
def panel(synthetic_panel: tuple[pl.DataFrame, pl.DataFrame]) -> Panel:
    return Panel.from_frames(*synthetic_panel)


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_accounting_and_constraints(
    name: str, tmp_path: Path, universe_file: Path, panel: Panel
) -> None:
    cfg = make_cfg(tmp_path, universe_file, optimizer__name=name)
    result = run_backtest(cfg, panel=panel, tag="unit")
    ret = result.returns
    assert float((ret["ret"] - (ret["ret_gross"] - ret["cost"])).abs().max()) <= 1e-12
    assert float((ret["ret_excess"] - (ret["ret"] - ret["rf"])).abs().max()) <= 1e-15
    assert ret["date"][0] == panel.dates[panel.dates.index(cfg.sample.oos_start)]
    w = result.weights
    # Targets always satisfy the constraints; post-trade weights do whenever a trade executes.
    # Without a trade, D-06's 2-point threshold lets drifted weights stay put, cap or not.
    traded = w.group_by("date", maintain_order=True).agg(
        (pl.col("trade").abs().sum() > 0).alias("traded"),
        pl.col("w_target").sum().alias("t_sum"),
        pl.col("w_target").min().alias("t_lo"),
        pl.col("w_target").max().alias("t_hi"),
        pl.col("w_post").sum().alias("p_sum"),
        pl.col("w_post").max().alias("p_hi"),
    )
    assert float((traded["t_sum"] - 1.0).abs().max()) <= 1e-8
    assert float((traded["p_sum"] - 1.0).abs().max()) <= 1e-8
    assert traded["t_lo"].min() >= -1e-9
    if name in ("min_variance", "mv_turnover"):
        assert traded["t_hi"].max() <= 0.40 + 1e-8
        assert traded.filter("traded")["p_hi"].max() <= 0.40 + 1e-8
    if name == "erc":
        assert result.metrics["erc_max_rc_gap"] <= 1e-6
    assert result.metrics["solver_fallbacks"] == 0
    rows = read_trial_log(Path(cfg.logging.experiment_log))
    assert len(rows) == 1 and rows[0]["strategy_family"] == f"regport_{name}"
    run_dir = Path(cfg.logging.experiment_log).parent / "runs" / result.run_id
    for file in ("returns.parquet", "weights.parquet", "metrics.json", "config.yaml"):
        assert (run_dir / file).exists()
    stored = pl.read_parquet(run_dir / "returns.parquet")
    assert stored.columns[:2] == ["date", "ret"] and stored.height == ret.height


def test_zero_cost_gross_equals_net(tmp_path: Path, universe_file: Path, panel: Panel) -> None:
    cfg = make_cfg(tmp_path, universe_file, optimizer__name="hrp", costs__override_bps=0.0)
    ret = run_backtest(cfg, panel=panel).returns
    assert (ret["ret"] == ret["ret_gross"]).all() and (ret["cost"] == 0).all()
    assert ret["turnover"].sum() > 0


def test_weights_drift_with_returns(tmp_path: Path, universe_file: Path, panel: Panel) -> None:
    cfg = make_cfg(
        tmp_path, universe_file, optimizer__name="equal_weight", execution__policy="calendar"
    )
    result = run_backtest(cfg, panel=panel)
    w = result.weights
    dates = sorted(w["date"].unique().to_list())
    for a, b in itertools.pairwise(dates[3:9]):
        post = w.filter(pl.col("date") == a).sort("symbol")
        drift = w.filter(pl.col("date") == b).sort("symbol")
        cols = [panel.symbols.index(s) for s in post["symbol"]]
        x = post["w_post"].to_numpy().copy()
        for i in range(panel.dates.index(a) + 1, panel.dates.index(b) + 1):
            r = panel.returns[i, cols]
            x = x * (1 + r) / (1 + x @ r)
        got = drift.filter(pl.col("symbol").is_in(post["symbol"].to_list())).sort("symbol")
        assert np.max(np.abs(got["w_drift"].to_numpy() - x)) <= 1e-12


def test_signal_on_t_earns_from_t_plus_2(
    tmp_path: Path, universe_file: Path, panel: Panel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DQ-06: weights decided at the close of t trade at t+1 and earn from t+2."""
    cfg = make_cfg(
        tmp_path, universe_file, optimizer__name="equal_weight", execution__policy="calendar"
    )
    base = run_backtest(cfg, panel=panel).returns
    signal_day = next(
        d
        for d in panel.dates
        if d.year == 2005 and d.month == 6 and panel.dates[panel.dates.index(d) + 1].month == 7
    )
    t = panel.dates.index(signal_day)
    original = engine._Strategy.decide

    def decide(self: Any, p: Panel, i: int, w: np.ndarray) -> Any:
        out = original(self, p, i, w)
        if i == t:
            out.target = np.zeros(len(p.symbols))
            out.target[0] = 1.0  # all in AEQ
        return out

    monkeypatch.setattr(engine._Strategy, "decide", decide)
    hit = run_backtest(cfg, panel=panel).returns
    day = {d: k for k, d in enumerate(hit["date"].to_list())}
    k1, k2 = day[panel.dates[t + 1]], day[panel.dates[t + 2]]
    assert np.array_equal(
        hit["ret_gross"][: k1 + 1].to_numpy(), base["ret_gross"][: k1 + 1].to_numpy()
    )
    assert hit["ret_gross"][k2] == pytest.approx(panel.returns[t + 2, 0], abs=1e-15)
    assert hit["ret_gross"][k2] != pytest.approx(base["ret_gross"][k2], abs=1e-12)


def test_failed_runs_are_logged(
    tmp_path: Path, universe_file: Path, panel: Panel, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_cfg(tmp_path, universe_file, regime__usage="hmm")
    with pytest.raises(ValueError, match="auto"):
        run_backtest(cfg, panel=panel)

    def broken(*_a: Any, **_k: Any) -> np.ndarray:
        raise SolverError("forced")

    monkeypatch.setattr(engine, "min_variance", broken)
    cfg = make_cfg(tmp_path, universe_file, optimizer__name="min_variance")
    with pytest.raises(BacktestError, match="NFR-09"):
        run_backtest(cfg, panel=panel)
    rows = read_trial_log(Path(cfg.logging.experiment_log))
    assert [r["n_obs"] for r in rows] == ["0", "0"]
    assert all(r["sharpe_annual"] == "" and r["x_note"].startswith("failed") for r in rows)


def test_trial_log_validates_with_overfitlab(
    tmp_path: Path, universe_file: Path, panel: Panel
) -> None:
    validate = pytest.importorskip("overfitlab.triallog").validate_trial_log
    for name in ("erc", "equal_weight"):
        run_backtest(make_cfg(tmp_path, universe_file, optimizer__name=name), panel=panel)
    issues = validate(tmp_path / "results" / "trial_log.csv")
    assert [i for i in issues if i.severity == "error"] == []


def test_cli_backtest(
    tmp_path: Path,
    universe_file: Path,
    synthetic_panel: tuple[pl.DataFrame, pl.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returns, rf = synthetic_panel
    data = tmp_path / "data"
    write_parquet(returns, data / "processed" / "returns_daily.parquet")
    write_parquet(rf, data / "processed" / "rf_daily.parquet")
    cfg = make_cfg(tmp_path, universe_file)
    config = tmp_path / "cfg.yaml"
    config.write_text(yaml.safe_dump(cfg.to_params()), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REGPORT_DATA_DIR", str(data))
    out = CliRunner().invoke(
        app,
        [
            "backtest",
            "--config",
            str(config),
            "--optimizer",
            "hrp",
            "--regime",
            "none",
            "--cost-bps",
            "5",
            "--tag",
            "cli",
        ],
    )
    assert out.exit_code == 0, out.output
    run_id = out.output.strip().splitlines()[-1]
    rows = read_trial_log(Path(cfg.logging.experiment_log))
    assert rows[-1]["run_id"] == run_id and rows[-1]["strategy_family"] == "regport_hrp"
    assert '"override_bps":5.0' in rows[-1]["params_json"]
    bad = CliRunner().invoke(app, ["backtest", "--config", str(config), "--regime", "hmm"])
    assert bad.exit_code == 1 and "failed run" in bad.output


def test_erc_rc_gap_recorded(tmp_path: Path, universe_file: Path, panel: Panel) -> None:
    cfg = make_cfg(tmp_path, universe_file, optimizer__name="erc", execution__policy="calendar")
    result = run_backtest(cfg, panel=panel)
    assert result.metrics["rebalances"] > 50
    # Recompute RC at one rebalance from the stored target weights.
    first = result.weights.filter(pl.col("date") == result.weights["date"].min())
    held = first.filter(pl.col("w_target") > 0)
    assert abs(held["w_target"].sum() - 1.0) <= 1e-12
    assert risk_contributions(
        held["w_target"].to_numpy(), np.eye(held.height)
    ).sum() == pytest.approx(1.0)
