"""FR-10 evaluation: validation item (7), statistics, selection and the cov-eval report."""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from regport.cli import app
from regport.config import ExperimentConfig
from regport.data.panel import write_parquet
from regport.report.cov_eval import write_cov_eval
from regport.risk.evaluation import (
    ac_w_03,
    bias_statistic,
    qlike,
    resolve_oos_end,
    run_cov_eval,
    select_estimator,
    summarize,
)


def test_bias_statistic_of_true_covariance() -> None:
    """Validation item (7): b of the true covariance on Gaussian returns, T = 2000."""
    rng = np.random.default_rng(20261030)
    n, t = 6, 2000
    a = rng.normal(size=(n, n))
    cov = a @ a.T * 1e-4
    w = rng.dirichlet(np.ones(n))
    returns = rng.multivariate_normal(np.zeros(n), cov, size=t) @ w
    b, lo, hi = bias_statistic(returns, np.full(t, math.sqrt(w @ cov @ w)))
    assert 0.95 <= b <= 1.05
    assert lo == pytest.approx(1 - 1.96 * math.sqrt(1 / 4000)) and hi == pytest.approx(
        1 + 1.96 * math.sqrt(1 / 4000)
    )
    assert math.isnan(bias_statistic(np.array([1.0]), np.array([1.0]))[0])


def test_qlike_prefers_the_true_variance() -> None:
    rng = np.random.default_rng(1)
    rv = rng.normal(0, 0.02, 5000) ** 2
    true = qlike(rv, np.full(5000, 0.02**2))
    assert true < qlike(rv, np.full(5000, 0.01**2)) and true < qlike(rv, np.full(5000, 0.04**2))
    assert qlike(np.array([0.0, 1.0]), np.array([1.0, 0.0])) == pytest.approx(0.0)


def test_resolve_oos_end() -> None:
    oct_days = [date(2026, 10, d) for d in (28, 29, 30)]
    assert resolve_oos_end([*oct_days, date(2026, 11, 2)], None) == date(2026, 10, 30)
    assert resolve_oos_end(oct_days, None) == date(2026, 10, 30)  # Oct 30 is the last weekday
    assert resolve_oos_end([*oct_days, date(2026, 11, 2)], date(2026, 10, 29)) == date(2026, 10, 29)


def _synthetic_returns(n_days: int = 2400) -> pl.DataFrame:
    """Three assets with known constant covariance from 2003; a fourth enters in 2006."""
    rng = np.random.default_rng(4)
    start = date(2003, 1, 2)
    dates, d = [], start
    while len(dates) < n_days:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    cov = (
        np.array(
            [[1.0, 0.3, 0.1, 0.2], [0.3, 1.5, 0.2, 0.1], [0.1, 0.2, 0.5, 0.0], [0.2, 0.1, 0.0, 2.0]]
        )
        * 1e-4
    )
    r = rng.multivariate_normal(np.zeros(4), cov, size=n_days)
    rows = []
    for j, sym in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        first = 1 if sym != "DDD" else 800
        hist = 0
        for i, day in enumerate(dates):
            ret = float(r[i, j]) if i >= first else None
            rows.append(
                {
                    "date": day,
                    "symbol": sym,
                    "ret": ret,
                    "logret": None,
                    "n_valid_history": hist,
                    "eligible": hist >= 252,
                }
            )
            hist += ret is not None
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def evaluated() -> tuple[ExperimentConfig, object, pl.DataFrame]:
    cfg = ExperimentConfig.model_validate(
        {
            "sample": {
                "data_start": "2003-01-02",
                "oos_start": "2005-01-03",
                "estimator_selection_end": "2008-12-31",
                "oos_end": "2012-03-30",
            },
            "risk": {"n_random_portfolios": 20},
        }
    )
    result = run_cov_eval(_synthetic_returns(), cfg)
    return cfg, result, summarize(result)


def test_cov_eval_on_known_covariance(
    evaluated: tuple[ExperimentConfig, object, pl.DataFrame],
) -> None:
    _cfg, result, table = evaluated
    assert set(table["estimator"]) == {"sample", "ewma", "ledoit_wolf", "oas"}
    assert set(table["sample"]) == {"selection", "second_half", "full"}
    full = table.filter((pl.col("sample") == "full") & (pl.col("estimator") == "sample"))
    for row in full.iter_rows(named=True):
        # Constant Gaussian covariance: forecasts are unbiased up to sampling noise.
        assert row["band_lo95"] - 0.1 <= row["b_equal_weight"] <= row["band_hi95"] + 0.1
    assert result.solver_failures == 0  # type: ignore[attr-defined]
    assert result.excluded_events > 0  # type: ignore[attr-defined]  # DDD while sparse
    assert table.filter(pl.col("horizon_days") == 21)["mean_n_assets"].max() > 3
    chosen = select_estimator(table)
    assert chosen in {"sample", "ewma", "ledoit_wolf", "oas"}
    ok, b_1n, b_rand = ac_w_03(table, chosen)
    assert ok == (0.9 <= b_1n <= 1.1 and 0.9 <= b_rand <= 1.1)
    with pytest.raises(ValueError, match="no selection"):
        select_estimator(table.filter(pl.col("sample") == "full"))


def test_cov_eval_is_deterministic(
    evaluated: tuple[ExperimentConfig, object, pl.DataFrame],
) -> None:
    cfg, _result, table = evaluated
    again = summarize(
        run_cov_eval(_synthetic_returns(), cfg, ["sample", "ewma", "ledoit_wolf", "oas"])
    )
    # NFR-02: tables are compared after rounding to 1e-10. Min-variance solves differ between
    # runs at about 1e-15 (cvxpy/CLARABEL), so bit-identity is not expected.
    floats = [c for c, t in table.schema.items() if t == pl.Float64]
    assert again.drop(floats).equals(table.drop(floats))
    for col in floats:
        assert np.allclose(again[col].to_numpy(), table[col].to_numpy(), rtol=0, atol=1e-10)


def test_report_and_cli(
    evaluated: tuple[ExperimentConfig, object, pl.DataFrame],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, result, table = evaluated
    paths = write_cov_eval(result, table, tmp_path / "reports", tmp_path / "results")  # type: ignore[arg-type]
    text = paths["report"].read_text(encoding="utf-8")
    assert "Chosen estimator" in text and "AC-W-03" in text
    assert paths["figure"].stat().st_size > 1000
    csv = pl.read_csv(paths["table"])
    assert csv.height == table.height

    data = tmp_path / "data"
    write_parquet(_synthetic_returns(), data / "processed" / "returns_daily.parquet")
    config = tmp_path / "cfg.yaml"
    import yaml

    params = cfg.to_params()
    config.write_text(yaml.safe_dump(params), encoding="utf-8")
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    monkeypatch.setenv("REGPORT_DATA_DIR", str(data))
    monkeypatch.setenv("REGPORT_REPORTS_DIR", str(tmp_path / "r2"))
    monkeypatch.setenv("REGPORT_RESULTS_DIR", str(tmp_path / "res2"))
    out = CliRunner().invoke(
        app, ["cov-eval", "--config", str(config), "--estimators", "sample,oas", "--horizons", "21"]
    )
    assert out.exit_code == 0, out.output
    assert (tmp_path / "r2" / "cov_eval.md").exists()
    monkeypatch.setenv("REGPORT_DATA_DIR", str(tmp_path / "empty"))
    missing = CliRunner().invoke(app, ["cov-eval", "--config", str(config)])
    assert missing.exit_code == 1 and "data build" in missing.output
