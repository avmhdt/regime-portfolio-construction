"""``regport evaluate``: out-of-sample comparison with 1/N and the regime ablation (FR-26 to FR-28).

Every strategy is tested against 1/N, and every regime arm against the same optimizer without a
regime layer, by the Ledoit-Wolf (2008) test on daily excess returns net of costs: the studentized
circular block bootstrap (headline, D-08) and HAC alongside. Holm adjusts the bootstrap p-values
across all strategies compared with 1/N. The trial count and the deflated Sharpe ratio of the best
strategy come from the whole trial log (AC-H-04).
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.stats import norm

from regport.config import ExperimentConfig
from regport.eval.dsr import deflated_sharpe
from regport.eval.sharpe_test import holm, sharpe_diff_bootstrap, sharpe_diff_hac
from regport.tracking.experiment_log import read_trial_log

F64 = npt.NDArray[np.float64]
DAYS = 252


@dataclass(frozen=True)
class Run:
    optimizer: str
    regime: str
    run_id: str
    returns: pl.DataFrame
    metrics: dict[str, Any]

    @property
    def label(self) -> str:
        return self.optimizer if self.regime == "none" else f"{self.optimizer}+{self.regime}"


def load_runs(manifest: Path) -> tuple[list[Run], dict[str, Any]]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    runs_dir = manifest.parent.parent / "runs"
    runs = []
    for r in data["runs"]:
        folder = runs_dir / r["run_id"]
        runs.append(
            Run(
                r["optimizer"],
                r["regime"],
                r["run_id"],
                pl.read_parquet(folder / "returns.parquet"),
                json.loads((folder / "metrics.json").read_text(encoding="utf-8")),
            )
        )
    return runs, data


PairResult = tuple[float, float, float, float, int | None, float]


def _pair(job: tuple[F64, F64, int, int]) -> PairResult:
    """``(Δ annual, p_boot, p_hac, se_hac annual, block, Δ per period)``, strategy vs reference."""
    a, b, n_boot, seed = job
    boot = sharpe_diff_bootstrap(a, b, n_boot, None, seed)
    hac = sharpe_diff_hac(a, b)
    root = math.sqrt(DAYS)
    return hac.diff * root, boot.p_value, hac.p_value, hac.se * root, boot.block_size, hac.diff


def sharpe_ci(
    x: F64, seed: int, n_boot: int = 2000, block: int = 10, level: float = 0.95
) -> tuple[float, float]:
    """Percentile circular block bootstrap interval for the annualized Sharpe ratio."""
    rng = np.random.default_rng(seed)
    t = x.size
    starts = rng.integers(0, t, size=(n_boot, math.ceil(t / block)))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :t] % t
    s = x[idx]
    sr = s.mean(axis=1) / s.std(axis=1, ddof=1) * math.sqrt(DAYS)
    lo, hi = np.quantile(sr, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def _excess(run: Run) -> F64:
    out: F64 = run.returns["ret_excess"].to_numpy().astype(np.float64)
    return out


def compare(
    runs: list[Run], cfg: ExperimentConfig, n_jobs: int = 1
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The 1/N table and the ablation table."""
    bench = next(r for r in runs if r.optimizer == cfg.evaluation.benchmark)
    for r in runs:
        if not r.returns["date"].equals(bench.returns["date"]):
            raise ValueError(f"{r.label} and 1/N cover different dates")
    n_boot = cfg.evaluation.sharpe_test.n_boot
    strategies = [r for r in runs if r is not bench]
    pairs = [(r, bench) for r in strategies]
    ablation = [
        (r, next(b for b in strategies if b.optimizer == r.optimizer and b.regime == "none"))
        for r in strategies
        if r.regime != "none"
    ]
    jobs = [
        (_excess(a), _excess(b), n_boot, cfg.seed + i) for i, (a, b) in enumerate(pairs + ablation)
    ]
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            results = list(pool.map(_pair, jobs))
    else:
        results = [_pair(j) for j in jobs]
    vs_bench, vs_none = results[: len(pairs)], results[len(pairs) :]
    p_holm = holm(np.array([r[1] for r in vs_bench]))
    power = norm.ppf(0.975) + norm.ppf(0.80)
    rows = []
    for k, ((run, _), res) in enumerate(zip(pairs, vs_bench, strict=True)):
        m = run.metrics
        lo, hi = sharpe_ci(_excess(run), cfg.seed + 1000 + k)
        rows.append(
            {
                "strategy": run.label,
                "optimizer": run.optimizer,
                "regime": run.regime,
                "run_id": run.run_id,
                "sharpe": m["sharpe_excess"],
                "sharpe_ci_lo": lo,
                "sharpe_ci_hi": hi,
                "delta_sharpe_vs_1n": res[0],
                "p_bootstrap": res[1],
                "p_hac": res[2],
                "p_holm": float(p_holm[k]),
                "block_size": res[4],
                "min_detectable_delta_80pct": float(power * res[3]),
                "ceq": m["ceq"],
                "annual_return": m["annual_return"],
                "annual_volatility": m["annual_volatility"],
                "max_drawdown": m["max_drawdown"],
                "annual_turnover": m["annual_turnover"],
                "annual_cost_drag": m["annual_cost_drag"],
            }
        )
    bm = bench.metrics
    lo, hi = sharpe_ci(_excess(bench), cfg.seed + 999)
    rows.append(
        {
            "strategy": "1/N",
            "optimizer": bench.optimizer,
            "regime": "none",
            "run_id": bench.run_id,
            "sharpe": bm["sharpe_excess"],
            "sharpe_ci_lo": lo,
            "sharpe_ci_hi": hi,
            "delta_sharpe_vs_1n": 0.0,
            "p_bootstrap": None,
            "p_hac": None,
            "p_holm": None,
            "block_size": None,
            "min_detectable_delta_80pct": None,
            "ceq": bm["ceq"],
            "annual_return": bm["annual_return"],
            "annual_volatility": bm["annual_volatility"],
            "max_drawdown": bm["max_drawdown"],
            "annual_turnover": bm["annual_turnover"],
            "annual_cost_drag": bm["annual_cost_drag"],
        }
    )
    table = pl.DataFrame(rows, infer_schema_length=None)
    arows = []
    for (arm, base), res in zip(ablation, vs_none, strict=True):
        a, b = arm.metrics, base.metrics
        arows.append(
            {
                "optimizer": arm.optimizer,
                "regime": arm.regime,
                "run_id": arm.run_id,
                "base_run_id": base.run_id,
                "delta_sharpe": res[0],
                "p_bootstrap": res[1],
                "p_hac": res[2],
                "block_size": res[4],
                "delta_ceq": a["ceq"] - b["ceq"],
                "delta_turnover": a["annual_turnover"] - b["annual_turnover"],
                "delta_cost_drag": a["annual_cost_drag"] - b["annual_cost_drag"],
                "delta_max_drawdown": a["max_drawdown"] - b["max_drawdown"],
            }
        )
    return table, pl.DataFrame(arows, infer_schema_length=None)


@dataclass(frozen=True)
class TrialSummary:
    rows: int
    failed: int
    distinct_configs: int
    selected: str
    selected_sharpe: float
    dsr: float
    sr0_annual: float
    var_trials_annual: float


def trial_summary(log_path: Path, table: pl.DataFrame, runs: list[Run]) -> TrialSummary:
    """AC-H-04: the honest trial count and the DSR of the best non-1/N strategy in this grid."""
    rows = read_trial_log(log_path)
    valid = [r for r in rows if int(r["n_obs"]) > 0]
    params = {
        json.dumps(
            {k: v for k, v in json.loads(r["params_json"]).items() if k not in ("tag",)},
            sort_keys=True,
        )
        for r in rows
    }
    sr = np.array([float(r["sharpe_annual"]) for r in valid]) / math.sqrt(DAYS)
    var = float(np.var(sr, ddof=1)) if sr.size > 1 else 0.0
    strategies = table.filter(pl.col("strategy") != "1/N").sort("sharpe", descending=True)
    best = strategies.row(0, named=True)
    run = next(r for r in runs if r.run_id == best["run_id"])
    x = run.returns["ret"].to_numpy()
    c = x - x.mean()
    m2 = float(np.mean(c**2))
    skew = float(np.mean(c**3)) / m2**1.5
    kurt = float(np.mean(c**4)) / m2**2
    sr_sel = float(x.mean() / x.std(ddof=1))
    dsr, sr0 = deflated_sharpe(sr_sel, x.size, skew, kurt, var, len(rows))
    return TrialSummary(
        len(rows),
        len(rows) - len(valid),
        len(params),
        best["strategy"],
        best["sharpe"],
        dsr,
        sr0 * math.sqrt(DAYS),
        var * DAYS,
    )


def constraint_audit(runs: list[Run], cap: float, log_path: Path) -> pl.DataFrame:
    """AC-W-02 per run: target and post-trade constraints, and drift above the cap."""
    out = []
    capped = ("min_variance", "mv_turnover")
    for run in runs:
        w = pl.read_parquet(log_path.parent / "runs" / run.run_id / "weights.parquet")
        per = w.group_by("date", maintain_order=True).agg(
            (pl.col("trade").abs().sum() > 0).alias("traded"),
            pl.col("w_target").sum().alias("t_sum"),
            pl.col("w_target").min().alias("t_min"),
            pl.col("w_target").max().alias("t_max"),
            pl.col("w_post").sum().alias("p_sum"),
            pl.col("w_post").max().alias("p_max"),
        )
        traded = per["traded"].to_numpy()
        t_sum_err = float(np.max(np.abs(per["t_sum"].to_numpy() - 1.0)))
        t_min = float(np.min(per["t_min"].to_numpy()))
        t_max = float(np.max(per["t_max"].to_numpy()))
        p_max = per["p_max"].to_numpy()
        traded_max = float(np.max(p_max[traded])) if traded.any() else 0.0
        lim = cap if run.optimizer in capped else 1.0
        rc_gap = run.metrics.get("erc_max_rc_gap")
        out.append(
            {
                "strategy": run.label,
                "run_id": run.run_id,
                "rebalances": per.height,
                "trades": int(traded.sum()),
                "max_abs_target_sum_error": t_sum_err,
                "min_target_weight": t_min,
                "max_target_weight": t_max,
                "max_post_trade_weight_on_trade_days": traded_max,
                "cap": lim,
                "days_post_weight_above_cap_untraded": int(np.sum(~traded & (p_max > lim + 1e-8))),
                "erc_max_rc_gap": rc_gap,
                "solver_fallbacks": run.metrics.get("solver_fallbacks"),
                "violations": int(
                    t_sum_err > 1e-8
                    or t_min < -1e-9
                    or t_max > lim + 1e-8
                    or traded_max > lim + 1e-8
                    or (rc_gap or 0.0) > 1e-6
                ),
            }
        )
    return pl.DataFrame(out, infer_schema_length=None)
