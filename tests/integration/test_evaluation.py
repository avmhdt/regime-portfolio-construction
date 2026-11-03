"""M6 evaluation on synthetic runs: grid, 1/N comparison, ablation, trials, audit, stress."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml
from typer.testing import CliRunner

from regport.cli import app
from regport.config import ExperimentConfig
from regport.data.panel import write_parquet
from regport.eval.evaluate import compare, constraint_audit, load_runs, sharpe_ci, trial_summary
from regport.eval.grid import Cell, cell_config, grid_cells, latest_manifest, load_grid, run_cells
from regport.eval.stress import drawdown_stats
from regport.report.evaluation import headline_sentence, write_evaluation
from regport.report.stress import write_stress
from regport.settings import Settings


def _base(tmp_path: Path, universe: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "data": {"universe_file": str(universe), "calendar_symbol": "AEQ"},
            "sample": {
                "data_start": "2001-01-02",
                "oos_start": "2003-01-02",
                "estimator_selection_end": "2006-12-29",
            },
            "regime": {"usage": "none"},
            "risk": {"estimator": "ewma"},
            "evaluation": {
                "sharpe_test": {"n_boot": 199, "block_size": 2},
                "stress_windows": {
                    "gfc_2008": ["2005-03-01", "2005-09-30"],
                    "covid_2020": ["2007-01-02", "2007-06-29"],
                    "rates_2022": ["2008-02-01", "2008-08-29"],
                },
            },
            "logging": {"experiment_log": str(tmp_path / "results" / "trial_log.csv")},
        }
    )


@pytest.fixture
def grid_run(
    tmp_path: Path, universe_file: Path, synthetic_panel: tuple[pl.DataFrame, pl.DataFrame]
) -> tuple[Path, ExperimentConfig]:
    returns, rf = synthetic_panel
    data = tmp_path / "data"
    write_parquet(returns, data / "processed" / "returns_daily.parquet")
    write_parquet(rf, data / "processed" / "rf_daily.parquet")
    settings = Settings(
        _env_file=None,
        regport_data_dir=data,  # type: ignore[call-arg]
        regport_results_dir=tmp_path / "results",
    )
    base = _base(tmp_path, universe_file)
    grid_yaml = tmp_path / "grid.yaml"
    base_yaml = tmp_path / "base.yaml"
    base_yaml.write_text(yaml.safe_dump(base.to_params()), encoding="utf-8")
    grid_yaml.write_text(
        yaml.safe_dump(
            {
                "base": str(base_yaml),
                "optimizers": ["erc", "hrp"],
                "regimes": [{"name": "none", "usage": "none"}],
                "headline": {"optimizer": "erc", "regime": "none"},
            }
        ),
        encoding="utf-8",
    )
    cells = grid_cells(load_grid(grid_yaml), base)
    assert [c.key for c in cells] == ["erc|none", "hrp|none", "equal_weight|none"]
    manifest = run_cells(
        cells, settings, "grid", "ablation", {"headline": {"optimizer": "erc", "regime": "none"}}
    )
    return manifest, base


def test_grid_compare_and_reports(grid_run: tuple[Path, ExperimentConfig], tmp_path: Path) -> None:
    manifest, cfg = grid_run
    assert latest_manifest(tmp_path / "results") == manifest
    runs, meta = load_runs(manifest)
    assert len(runs) == 3 and meta["kind"] == "ablation"
    table, ablation = compare(runs, cfg)
    assert table.height == 3 and ablation.is_empty()
    bench = table.filter(pl.col("strategy") == "1/N")
    assert bench["delta_sharpe_vs_1n"][0] == 0.0
    strategies = table.filter(pl.col("strategy") != "1/N")
    assert strategies["p_holm"].to_numpy().min() >= strategies["p_bootstrap"].to_numpy().min()
    assert (strategies["block_size"] == 2).all()
    trials = trial_summary(Path(cfg.logging.experiment_log), table, runs)
    assert trials.rows == 3 and trials.failed == 0 and trials.distinct_configs == 3
    assert 0.0 <= trials.dsr <= 1.0
    audit = constraint_audit(runs, 0.40, Path(cfg.logging.experiment_log))
    assert audit["violations"].sum() == 0
    out = write_evaluation(
        table,
        ablation,
        trials,
        audit,
        runs,
        "erc+none",
        ("2003-01-02", "2010-12-31"),
        tmp_path / "reports",
    )
    text = out["oos_md"].read_text(encoding="utf-8")
    assert "1/N" in text and "Deflated Sharpe" in text
    assert out["ablation_fig"].stat().st_size > 1000
    stress = write_stress(runs, cfg, "erc+none", None, None, tmp_path / "reports")
    assert "Stress tests" in stress["md"].read_text(encoding="utf-8")
    assert pl.read_csv(stress["csv"]).height == 9


def test_ablation_pairs(
    tmp_path: Path, universe_file: Path, synthetic_panel: tuple[pl.DataFrame, pl.DataFrame]
) -> None:
    """A second arm of the same optimizer is tested against its base arm."""
    returns, rf = synthetic_panel
    data = tmp_path / "data"
    write_parquet(returns, data / "processed" / "returns_daily.parquet")
    write_parquet(rf, data / "processed" / "rf_daily.parquet")
    settings = Settings(
        _env_file=None,
        regport_data_dir=data,  # type: ignore[call-arg]
        regport_results_dir=tmp_path / "results",
    )
    base = _base(tmp_path, universe_file)
    cells = [
        Cell("erc", "none", cell_config(base, "erc", None)),
        Cell("erc", "costly", cell_config(base, "erc", None, costs__override_bps=25.0)),
        Cell("equal_weight", "none", cell_config(base, "equal_weight", None)),
    ]
    runs, _ = load_runs(run_cells(cells, settings, "t", "ablation"))
    _, ablation = compare(runs, base)
    assert ablation.height == 1
    row = ablation.row(0, named=True)
    assert row["regime"] == "costly" and row["delta_cost_drag"] > 0 and row["delta_sharpe"] < 0


def test_headline_sentence() -> None:
    table = pl.DataFrame(
        {
            "strategy": ["a", "b", "1/N"],
            "p_holm": [0.01, 0.5, None],
            "delta_sharpe_vs_1n": [0.3, -0.1, 0.0],
        }
    )
    assert "1 of 2 strategies beat 1/N" in headline_sentence(table)
    table = table.with_columns(pl.Series("delta_sharpe_vs_1n", [-0.3, -0.1, 0.0]))
    text = headline_sentence(table)
    assert text.startswith("No strategy beats 1/N") and "significantly worse" in text


def test_drawdown_stats() -> None:
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(12)]
    ret = np.array([0.1, 0.0, -0.1, -0.1, 0.05, 0.0, 0.1, 0.1, 0.0, 0.0, -0.01, 0.0])
    st = drawdown_stats(ret, dates, dates[1], dates[9])
    assert st.peak_date == dates[1] and st.trough_date == dates[3]
    assert st.max_drawdown == pytest.approx(0.9 * 0.9 - 1)
    assert st.days_to_trough == 2 and st.recovery_date == dates[7]
    assert st.days_to_recover == 4 and "days" in st.recovery_text
    never = drawdown_stats(np.r_[0.0, -0.2, np.zeros(10)], dates, dates[0], dates[11])
    assert never.recovery_date is None and "not recovered by" in never.recovery_text
    with pytest.raises(ValueError, match="fewer than 5"):
        drawdown_stats(ret, dates, dates[0], dates[2])
    with pytest.raises(ValueError, match="length"):
        drawdown_stats(ret[:5], dates, dates[0], dates[9])


def test_sharpe_ci_brackets_the_estimate() -> None:
    x = np.random.default_rng(0).normal(0.0005, 0.01, 3000)
    lo, hi = sharpe_ci(x, 1)
    est = x.mean() / x.std(ddof=1) * np.sqrt(252)
    assert lo < est < hi


def test_cli_evaluate(
    grid_run: tuple[Path, ExperimentConfig], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, cfg = grid_run
    config = tmp_path / "cfg.yaml"
    config.write_text(yaml.safe_dump(cfg.to_params()), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REGPORT_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("REGPORT_REPORTS_DIR", str(tmp_path / "rep"))
    out = CliRunner().invoke(app, ["evaluate", "--config", str(config), "--runs", str(manifest)])
    assert out.exit_code == 0, out.output
    assert (tmp_path / "rep" / "oos_vs_1n.md").exists()
    assert json.loads(manifest.read_text(encoding="utf-8"))["headline"]["optimizer"] == "erc"
