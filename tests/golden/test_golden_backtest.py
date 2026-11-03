"""Golden regression: fixture backtest metrics and weights, frozen as JSON (rounded to 1e-10).

Regenerate only on purpose, with the reason in the commit message:
``REGPORT_UPDATE_GOLDEN=1 uv run pytest tests/golden``.

Solver-based optimizers (min-variance, mean-variance) are compared within 1e-8, because
interior-point results differ in the last digits across platforms; the rest within 1e-10.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from regport.backtest.engine import Panel, run_backtest
from regport.config import ExperimentConfig

GOLDEN = Path(__file__).with_name("backtest_fixture.json")
OPTIMIZERS = ["equal_weight", "min_variance", "erc", "hrp", "mv_turnover"]
METRICS = [
    "annual_return",
    "annual_volatility",
    "sharpe_excess",
    "max_drawdown",
    "ceq",
    "annual_turnover",
    "annual_cost_drag",
    "effective_n_assets",
]
TOL = {"min_variance": 1e-8, "mv_turnover": 1e-8}


def _snapshot(name: str, tmp_path: Path, universe: Path, panel: Panel) -> dict[str, Any]:
    cfg = ExperimentConfig.model_validate(
        {
            "data": {"universe_file": str(universe), "calendar_symbol": "AEQ"},
            "sample": {
                "data_start": "2001-01-02",
                "oos_start": "2003-01-02",
                "estimator_selection_end": "2006-12-29",
            },
            "regime": {"usage": "none"},
            "risk": {"estimator": "ewma"},
            "optimizer": {"name": name},
            "logging": {"experiment_log": str(tmp_path / name / "trial_log.csv")},
        }
    )
    result = run_backtest(cfg, panel=panel)
    weights = (
        result.weights.filter(pl.col("w_post") > 1e-6)
        .sort("date", "symbol")
        .select(pl.col("date").cast(pl.String), "symbol", pl.col("w_post").round(10))
    )
    return {
        "metrics": {k: round(float(result.metrics[k]), 10) for k in METRICS},
        "n_obs": result.returns.height,
        "weights": [list(r) for r in weights.iter_rows()],
    }


def test_golden_fixture(
    tmp_path: Path, universe_file: Path, synthetic_panel: tuple[pl.DataFrame, pl.DataFrame]
) -> None:
    panel = Panel.from_frames(*synthetic_panel)
    current = {name: _snapshot(name, tmp_path, universe_file, panel) for name in OPTIMIZERS}
    if os.environ.get("REGPORT_UPDATE_GOLDEN") == "1":
        GOLDEN.write_text(
            json.dumps(current, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        pytest.skip("golden file rewritten")
    frozen = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for name in OPTIMIZERS:
        tol = TOL.get(name, 1e-10)
        got, want = current[name], frozen[name]
        assert got["n_obs"] == want["n_obs"], name
        for key in METRICS:
            assert got["metrics"][key] == pytest.approx(want["metrics"][key], abs=tol), (name, key)
        assert [w[:2] for w in got["weights"]] == [w[:2] for w in want["weights"]], name
        for g, w in zip(got["weights"], want["weights"], strict=True):
            assert g[2] == pytest.approx(w[2], abs=tol), (name, g[:2])
