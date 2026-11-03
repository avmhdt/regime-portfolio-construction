"""FR-30 robustness helpers and FR-31 scaling on synthetic inputs."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from regport.backtest.engine import Panel, run_backtest
from regport.config import ExperimentConfig
from regport.data.panel import write_parquet
from regport.eval import robustness as rb
from regport.eval.scaling import factor_returns, measure, scaling_table
from regport.exec.costs import CostModel
from regport.report.robustness import write_robustness
from regport.report.scaling import write_scaling
from regport.risk.factor import factor_cov


def _french(tmp_path: Path, n_days: int = 1500) -> Path:
    rng = np.random.default_rng(2)
    days, d = [], date(1986, 1, 2)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for name, n in (("ind10_daily", 10), ("ind49_daily", 12)):
        for j in range(n):
            ret = rng.normal(3e-4, 0.01, n_days)
            ret[:5] = np.nan if j == 0 else ret[:5]
            rows.append(
                pl.DataFrame(
                    {"dataset": name, "date": days, "series": f"I{j}", "ret": ret}
                ).with_columns(pl.col("ret").fill_nan(None))
            )
    for s, v in (("Mkt-RF", rng.normal(3e-4, 0.01, n_days)), ("RF", np.full(n_days, 1e-4))):
        rows.append(pl.DataFrame({"dataset": "ff3_daily", "date": days, "series": s, "ret": v}))
    proc = tmp_path / "processed"
    write_parquet(pl.concat(rows), proc / "french_daily.parquet")
    write_parquet(
        pl.DataFrame(
            {
                "series_id": "BAA10Y",
                "obs_date": days,
                "value": 2 + np.cumsum(rng.normal(0, 0.02, n_days)),
                "available_date": days,
            }
        ),
        proc / "fred_daily.parquet",
    )
    return proc


def test_french_panel_and_features(tmp_path: Path) -> None:
    proc = _french(tmp_path)
    panel = rb.french_panel(proc, "ind10_daily")
    assert len(panel.symbols) == 10 and np.isnan(panel.returns[:5, 0]).all()
    assert not panel.eligible[100].any() and panel.eligible[400].all()
    assert panel.rf[10] == pytest.approx(1e-4)
    feats = rb.french_features(proc)
    assert set(feats["spy_source"]) == {"ff_market"} and feats["baa10y_diff"].is_finite().all()


def test_engine_accepts_explicit_costs(tmp_path: Path) -> None:
    proc = _french(tmp_path)
    panel = rb.french_panel(proc, "ind49_daily")
    cfg = ExperimentConfig.model_validate(
        {
            "sample": {
                "data_start": "1986-01-02",
                "oos_start": "1988-01-04",
                "estimator_selection_end": "1990-12-31",
            },
            "regime": {"usage": "none"},
            "optimizer": {"name": "hrp"},
            "logging": {"experiment_log": str(tmp_path / "results" / "trial_log.csv")},
        }
    )
    res = run_backtest(cfg, panel=panel, costs=CostModel.uniform(panel.symbols, 5.0))
    assert res.metrics["annual_cost_drag"] > 0 and res.returns.height > 300


def _suite_runs() -> list[rb.SuiteRun]:
    rng = np.random.default_rng(1)
    dates = [date(2010, 1, 4) + timedelta(days=i) for i in range(400)]
    out = []
    for bps, shift in ((0.0, 0.0004), (5.0, 0.0002), (10.0, 0.0), (25.0, -0.0005)):
        for opt, mu in (("erc", shift), ("equal_weight", 0.0001)):
            ret = rng.normal(mu, 0.01, 400)
            frame = pl.DataFrame({"date": dates, "ret": ret, "ret_excess": ret})
            out.append(
                rb.SuiteRun(
                    "costs",
                    f"{bps:g}bps",
                    opt,
                    "none",
                    f"{opt}{bps}",
                    frame,
                    {
                        "sharpe_excess": float(ret.mean() / ret.std() * 15.87),
                        "ceq": 0.0,
                        "annual_volatility": 0.16,
                        "annual_turnover": 1.0,
                        "max_drawdown": -0.1,
                    },
                )
            )
    return out


def test_summarize_and_break_even(tmp_path: Path) -> None:
    table = rb.summarize(_suite_runs())
    assert table.height == 8
    assert table.filter(pl.col("optimizer") == "equal_weight")["p_hac"].null_count() == 4
    be = rb.break_even(table)
    assert be.height == 1 and isinstance(be["break_even_cost"][0], str)
    out = write_robustness({"costs": table}, be, None, {"costs": "note"}, tmp_path)
    text = out["md"].read_text(encoding="utf-8")
    assert "Break-even" in text and "note" in text


def test_break_even_text_cases() -> None:
    def table(gaps: list[float]) -> pl.DataFrame:
        rows = []
        for bps, g in zip(rb.COST_GRID, gaps, strict=True):
            rows += [
                {"variant": f"{bps:g}bps", "optimizer": "erc", "regime": "none", "sharpe": 0.5 + g},
                {
                    "variant": f"{bps:g}bps",
                    "optimizer": "equal_weight",
                    "regime": "none",
                    "sharpe": 0.5,
                },
            ]
        return pl.DataFrame(rows)

    assert rb.break_even(table([0.1, 0.05, -0.05, -0.2]))["break_even_cost"][0] == "7.5 bps"
    assert "below" in rb.break_even(table([-0.1, -0.1, -0.2, -0.3]))["break_even_cost"][0]
    assert "above" in rb.break_even(table([0.3, 0.2, 0.1, 0.05]))["break_even_cost"][0]


def test_scaling(tmp_path: Path) -> None:
    x = factor_returns(30, 504, 1)
    assert x.shape == (504, 30)
    cov = factor_cov(x, 10)
    assert np.allclose(cov, cov.T) and np.linalg.eigvalsh(cov).min() > 0
    t = measure(lambda: np.ones(1000).sum(), 2)
    assert t.seconds >= 0 and t.peak_mib >= 0
    table, cond = scaling_table([12, 30], 1, 3, {"ind": factor_returns(15, 504, 9)})
    assert set(table["universe"]) == {"12", "30", "ind"}
    assert table.filter(pl.col("stage") == "optimizer").height == 18
    sample = cond.filter((pl.col("estimator") == "sample") & (pl.col("universe") == "30"))
    lw = cond.filter((pl.col("estimator") == "ledoit_wolf") & (pl.col("universe") == "30"))
    assert lw["condition_number"][0] < sample["condition_number"][0]
    out = write_scaling(table, cond, pl.DataFrame({"optimizer": ["hrp"], "vol": [0.1]}), tmp_path)
    assert "HRP vs min-variance" in out["md"].read_text(encoding="utf-8")


def test_first_all_eligible(
    tmp_path: Path, synthetic_panel: tuple[pl.DataFrame, pl.DataFrame]
) -> None:
    returns, _ = synthetic_panel
    write_parquet(returns, tmp_path / "returns_daily.parquet")
    first = rb.first_all_eligible(tmp_path)
    counts = returns.filter(pl.col("date") == first)["eligible"]
    assert counts.all()
    _ = Panel.from_frames  # the engine path is covered elsewhere
