"""FR-33 / AC-W-04: manifest, table checks and README blocks rendered from tables."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from regport.report.build import build_manifest, check_tables, fill_readme, write_manifest
from regport.report.readme import render_blocks
from regport.report.tables import write_csv


def _tables(root: Path) -> Path:
    t = root / "tables"
    write_csv(
        pl.DataFrame(
            {
                "strategy": ["erc+hmm_A", "erc", "erc+hmm_B", "1/N"],
                "optimizer": ["erc", "erc", "erc", "equal_weight"],
                "regime": ["hmm_A", "none", "hmm_B", "none"],
                "sharpe": [0.6, 0.62, 0.62, 0.55],
                "sharpe_ci_lo": [0.2, 0.2, 0.2, 0.15],
                "sharpe_ci_hi": [1.0, 1.0, 1.0, 0.95],
                "delta_sharpe_vs_1n": [0.05, 0.07, 0.07, 0.0],
                "p_bootstrap": [0.4, 0.3, 0.3, None],
                "p_hac": [0.41, 0.31, 0.31, None],
                "p_holm": [0.8, 0.8, 0.8, None],
                "min_detectable_delta_80pct": [0.3, 0.3, 0.3, None],
                "ceq": [0.03, 0.03, 0.03, 0.04],
                "annual_volatility": [0.05, 0.05, 0.05, 0.09],
                "max_drawdown": [-0.1, -0.1, -0.1, -0.28],
                "annual_turnover": [0.5, 0.6, 0.6, 0.2],
                "annual_cost_drag": [1e-4, 1e-4, 1e-4, 2e-5],
            }
        ),
        t / "oos_vs_1n.csv",
    )
    write_csv(
        pl.DataFrame(
            {
                "optimizer": ["erc"],
                "regime": ["hmm_A"],
                "delta_sharpe": [-0.02],
                "p_bootstrap": [0.7],
                "p_hac": [0.7],
                "block_size": [2],
                "delta_ceq": [0.001],
                "delta_turnover": [-0.1],
                "delta_cost_drag": [-1e-5],
                "delta_max_drawdown": [-0.02],
            }
        ),
        t / "ablation.csv",
    )
    write_csv(
        pl.DataFrame(
            {
                "window": ["gfc_2008"] * 3,
                "strategy": ["erc+hmm_A", "erc", "1/N"],
                "window_return": [-0.05, 0.02, -0.2],
                "max_drawdown": [-0.13, -0.09, -0.28],
                "peak": ["2008-05-19"] * 3,
                "trough": ["2008-11-20"] * 3,
                "days_to_trough": [130] * 3,
                "recovery": ["10 days (2009-01-02)"] * 3,
                "worst_5d_return": [-0.05] * 3,
                "realized_vol": [0.07] * 3,
            }
        ),
        t / "stress.csv",
    )
    costs = []
    for bps, d in ((0, 0.1), (5, 0.05), (10, -0.01), (25, -0.1)):
        costs += [
            {
                "suite": "costs",
                "variant": f"{bps}bps",
                "optimizer": "erc",
                "regime": "none",
                "sharpe": 0.5 + d,
                "delta_sharpe_vs_1n": d,
                "p_hac": 0.5,
            },
            {
                "suite": "costs",
                "variant": f"{bps}bps",
                "optimizer": "equal_weight",
                "regime": "none",
                "sharpe": 0.5,
                "delta_sharpe_vs_1n": None,
                "p_hac": None,
            },
        ]
    write_csv(pl.DataFrame(costs), t / "robustness_costs.csv")
    write_csv(
        pl.DataFrame(
            {
                "suite": "common_start",
                "variant": "2008-04-11",
                "optimizer": ["erc", "equal_weight"],
                "regime": ["hmm_A", "none"],
                "sharpe": [0.6, 0.5],
                "p_hac": [0.3, None],
            }
        ),
        t / "robustness_common_start.csv",
    )
    write_csv(
        pl.DataFrame(
            {
                "suite": "french",
                "variant": ["ind49_daily"] * 3,
                "optimizer": ["hrp", "min_variance", "equal_weight"],
                "regime": ["none", "none", "none"],
                "sharpe": [0.5, 0.52, 0.56],
                "p_hac": [0.4, 0.6, None],
                "annual_volatility": [0.15, 0.12, 0.17],
            }
        ),
        t / "robustness_french.csv",
    )
    write_csv(
        pl.DataFrame({"optimizer": ["erc"], "regime": ["hmm_B_latest"], "sharpe": [0.695]}),
        t / "robustness_alfred.csv",
    )
    write_csv(
        pl.DataFrame(
            {
                "metric": [
                    "stress_probability_mad",
                    "aligned_state_probability_mad",
                    "argmax_agreement",
                ],
                "value": [0.3, 0.28, 0.43],
            }
        ),
        t / "robustness_alfred_regime.csv",
    )
    write_csv(
        pl.DataFrame(
            {
                "estimator": ["ewma"],
                "horizon_days": [21],
                "sample": ["full"],
                "n_windows": [261],
                "b_equal_weight": [0.976],
                "b_random_median": [0.979],
                "b_min_variance": [1.136],
                "band_lo95": [0.914],
                "band_hi95": [1.086],
            }
        ),
        t / "cov_eval.csv",
    )
    write_csv(
        pl.DataFrame(
            {
                "n_states": [4],
                "feature_set": ["A"],
                "refits": [285],
                "first_fit_end": ["2003-01-31"],
                "label_switches": [18],
                "proxy_days": [1243],
                "bocpd_hazard_days": [500.0],
                "bocpd_trigger_share_oos": [0.059],
                "smoother_lead_days_gfc_2008": [2],
            }
        ),
        t / "regime_summary.csv",
    )
    write_csv(
        pl.DataFrame(
            {
                "trial_log_rows": [81],
                "failed_rows": [0],
                "distinct_configurations": [81],
                "selected": ["erc+bocpd"],
                "selected_sharpe": [0.73],
                "var_trial_sharpe_annual": [0.02],
                "expected_max_sharpe_annual": [0.38],
                "dsr": [1.0],
            }
        ),
        t / "trials.csv",
    )
    write_csv(
        pl.DataFrame(
            {
                "universe": ["12", "500"],
                "n_assets": [12, 500],
                "n_over_t": [0.02, 0.99],
                "estimator": ["sample", "sample"],
                "condition_number": [7.5, 2e7],
            }
        ),
        t / "scaling_condition.csv",
    )
    write_csv(
        pl.DataFrame(
            {
                "symbol": ["SPY"],
                "first_date": ["1997-12-31"],
                "first_eligible": ["1999-01-04"],
                "rows": [7254],
            }
        ),
        t / "data_coverage.csv",
    )
    write_csv(pl.DataFrame({"universe": ["12"], "median_seconds": [0.1]}), t / "scaling.csv")
    return t


def test_readme_blocks_render_from_tables(tmp_path: Path) -> None:
    tables = _tables(tmp_path)
    blocks = render_blocks(tables, "erc+hmm_A", "ewma")
    assert "No strategy beats 1/N" in blocks["headline"]
    assert "0.60" in blocks["headline"] and "0.55" in blocks["headline"]
    assert "largest gain" in blocks["ablation"]
    assert "gfc_2008" in blocks["stress"]
    assert "Break-even" in blocks["robustness"] and "agreeing on 43%" in blocks["robustness"]
    assert "`ewma`" in blocks["risk"] and "0.976" in blocks["risk"]
    assert "K = 4" in blocks["regime"] and "gfc_2008 2" in blocks["regime"]
    assert "81 backtests" in blocks["trials"]
    assert "2e+07" in blocks["scaling"] and "12.0% for min-variance" in blocks["scaling"]
    assert "SPY" in blocks["data"] and "eligible from 1999-01-04" in blocks["data"]
    empty = render_blocks(tmp_path / "none", "erc+hmm_A", "ewma")
    assert all(v.startswith("_Run") or v.startswith("_No") for v in empty.values())


def test_fill_readme(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "intro\n<!-- regport:headline:start -->\nold\n<!-- regport:headline:end -->\n"
        "<!-- regport:other:start -->\nkeep\n<!-- regport:other:end -->\n",
        encoding="utf-8",
    )
    assert fill_readme(readme, {"headline": "new text\n"}) == ["headline"]
    text = readme.read_text(encoding="utf-8")
    assert "new text" in text and "old" not in text and "keep" in text


def test_manifest_and_checks(tmp_path: Path) -> None:
    tables = _tables(tmp_path)
    manifest = json.loads(write_manifest(tmp_path).read_text(encoding="utf-8"))
    assert set(manifest["tables"]) == {p.name for p in tables.glob("*.csv")}
    assert build_manifest(tmp_path) == manifest
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    for p in tables.glob("*.csv"):
        (ref_dir / p.name).write_bytes(p.read_bytes())
    ok = check_tables(manifest["tables"], ref_dir, tmp_path)
    assert all(c.ok for c in ok)
    assert {c.status for c in ok} == {"hash", "uncompared"}
    # Rounding noise passes on tolerance; a real change fails; a missing table fails.
    oos = pl.read_csv(tables / "oos_vs_1n.csv")
    write_csv(oos.with_columns(pl.col("sharpe") + 5e-10), tables / "oos_vs_1n.csv")
    write_csv(oos.with_columns(pl.lit(1).alias("extra")), tables / "new_table.csv")
    stress = pl.read_csv(tables / "stress.csv")
    write_csv(stress.with_columns(pl.col("max_drawdown") * 2), tables / "stress.csv")
    (tables / "data_coverage.csv").unlink()
    status = {c.table: c.status for c in check_tables(manifest["tables"], ref_dir, tmp_path)}
    assert status["oos_vs_1n.csv"] == "tolerance"
    assert status["stress.csv"] == "differs"
    assert status["data_coverage.csv"] == "missing"
    assert status["trials.csv"] == "uncompared"
    assert status["new_table.csv"] == "new"
    assert status["scaling.csv"] == "uncompared"
    no_ref = check_tables(manifest["tables"], None, tmp_path)
    assert next(c for c in no_ref if c.table == "stress.csv").status == "differs"
