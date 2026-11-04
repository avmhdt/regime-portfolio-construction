"""``reports/oos_vs_1n.md`` and ``reports/ablation.md`` with their tables and figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from regport.eval.evaluate import Run, TrialSummary
from regport.report.tables import md_table, write_csv

REGIME_ORDER = ["none", "hmm_A", "hmm_B", "bocpd"]
OPT_ORDER = ["min_variance", "erc", "hrp", "mv_turnover"]


def _public(frame: pl.DataFrame) -> pl.DataFrame:
    """Published tables carry no run ids: they change on every rerun (see results/grids/)."""
    return frame.drop([c for c in ("run_id", "base_run_id") if c in frame.columns])


def _fmt_p(p: float | None) -> str:
    return "-" if p is None else f"{p:.3f}"


def headline_sentence(table: pl.DataFrame, alpha: float = 0.05) -> str:
    strategies = table.filter(pl.col("strategy") != "1/N")
    beat = strategies.filter((pl.col("p_holm") <= alpha) & (pl.col("delta_sharpe_vs_1n") > 0))
    lost = strategies.filter((pl.col("p_holm") <= alpha) & (pl.col("delta_sharpe_vs_1n") < 0))
    n = strategies.height
    if beat.height == 0:
        text = (
            f"No strategy beats 1/N at the 5% level after Holm's adjustment: none of the {n} "
            "strategies has a significantly higher net Sharpe ratio than 1/N."
        )
    else:
        text = (
            f"{beat.height} of {n} strategies beat 1/N at the 5% level after Holm's "
            f"adjustment: {', '.join(beat['strategy'].to_list())}."
        )
    if lost.height:
        text += (
            f" {lost.height} are significantly worse than 1/N: "
            f"{', '.join(lost['strategy'].to_list())}."
        )
    return text


def cumulative_figure(runs: list[Run], labels: list[str], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    for run in runs:
        if run.label in labels or (run.optimizer == "equal_weight" and "1/N" in labels):
            r = run.returns
            wealth = np.cumprod(1 + r["ret"].to_numpy())
            name = "1/N" if run.optimizer == "equal_weight" else run.label
            ax.plot(
                np.array(r["date"].to_list(), dtype="datetime64[D]"), wealth, label=name, lw=1.1
            )
    ax.set_yscale("log")
    ax.set_ylabel("growth of 1 (net of costs, log scale)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, metadata={"Software": None})
    plt.close(fig)
    return path


def ablation_figure(table: pl.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    regimes = [r for r in REGIME_ORDER if r in table["regime"].to_list()]
    width = 0.8 / len(regimes)
    for j, regime in enumerate(regimes):
        rows = [
            table.filter((pl.col("optimizer") == o) & (pl.col("regime") == regime))
            for o in OPT_ORDER
        ]
        xs = np.arange(len(OPT_ORDER)) + (j - (len(regimes) - 1) / 2) * width
        vals = np.array([r["sharpe"][0] if r.height else np.nan for r in rows])
        lo = np.array([r["sharpe_ci_lo"][0] if r.height else np.nan for r in rows])
        hi = np.array([r["sharpe_ci_hi"][0] if r.height else np.nan for r in rows])
        ax.errorbar(xs, vals, yerr=[vals - lo, hi - vals], fmt="o", capsize=3, label=regime)
    bench = table.filter(pl.col("strategy") == "1/N")
    if bench.height:
        ax.axhline(bench["sharpe"][0], color="0.3", lw=1, ls="--", label="1/N")
    ax.set_xticks(range(len(OPT_ORDER)), OPT_ORDER)
    ax.set_ylabel("net Sharpe ratio (excess, annual)")
    ax.set_title("Net Sharpe by optimizer and regime arm, 95% block-bootstrap intervals")
    ax.legend(fontsize=8, ncol=5, loc="lower center")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, metadata={"Software": None})
    plt.close(fig)
    return path


def write_evaluation(
    table: pl.DataFrame,
    ablation: pl.DataFrame,
    trials: TrialSummary,
    audit: pl.DataFrame,
    runs: list[Run],
    headline: str,
    period: tuple[str, str],
    reports_dir: Path,
) -> dict[str, Path]:
    tables = reports_dir / "tables"
    figs = reports_dir / "figures"
    out = {
        "oos_csv": write_csv(_public(table), tables / "oos_vs_1n.csv"),
        "ablation_csv": write_csv(_public(ablation), tables / "ablation.csv"),
        "audit_csv": write_csv(_public(audit), tables / "constraint_audit.csv"),
        "trials_csv": write_csv(
            pl.DataFrame(
                {
                    "trial_log_rows": [trials.rows],
                    "failed_rows": [trials.failed],
                    "distinct_configurations": [trials.distinct_configs],
                    "selected": [trials.selected],
                    "selected_sharpe": [trials.selected_sharpe],
                    "var_trial_sharpe_annual": [trials.var_trials_annual],
                    "expected_max_sharpe_annual": [trials.sr0_annual],
                    "dsr": [trials.dsr],
                }
            ),
            tables / "trials.csv",
        ),
        "cumulative": cumulative_figure(
            runs, [headline, "1/N", headline.split("+")[0]], figs / "cumulative_vs_1n.png"
        ),
        "ablation_fig": ablation_figure(table, figs / "ablation_sharpe.png"),
    }
    show = [
        "strategy",
        "sharpe",
        "sharpe_ci_lo",
        "sharpe_ci_hi",
        "delta_sharpe_vs_1n",
        "p_bootstrap",
        "p_hac",
        "p_holm",
        "ceq",
        "annual_turnover",
        "cost_drag_bps",
        "max_drawdown",
    ]
    # Costs are a few bps a year: show them in bps, keep fractions in the CSV.
    ordered = table.sort("sharpe", descending=True).with_columns(
        (pl.col("annual_cost_drag") * 1e4).alias("cost_drag_bps")
    )
    ablation_view = (
        ablation.with_columns((pl.col("delta_cost_drag") * 1e4).alias("delta_cost_drag_bps"))
        if not ablation.is_empty()
        else ablation
    )
    lines = [
        "# Out-of-sample comparison with 1/N",
        "",
        "Generated by `regport evaluate`; do not edit by hand.",
        "",
        f"**{headline_sentence(table)}**",
        "",
        f"- Period: {period[0]} to {period[1]}, daily returns net of costs, excess of the "
        "risk-free rate. Sharpe ratios are annualized.",
        f"- Headline (preregistered): `{headline}`.",
        "- p-values: Ledoit-Wolf (2008) studentized circular block bootstrap (5,000 resamples, "
        "block size calibrated per pair; headline test) and HAC (Parzen, VAR(1) prewhitening), "
        "two-sided, for the Sharpe difference against 1/N. Holm adjusts the bootstrap p-values "
        "across all strategies.",
        "- Intervals: 95% percentile circular block bootstrap (block 10) of each Sharpe ratio.",
        "",
        *md_table(ordered, show),
        "",
        "## Power",
        "",
        "Minimum Sharpe difference detectable at 80% power and 5% two-sided, from each pair's HAC "
        "standard error (annualized):",
        "",
        *md_table(
            ordered.filter(pl.col("strategy") != "1/N"), ["strategy", "min_detectable_delta_80pct"]
        ),
        "",
        "## Trials",
        "",
        f"- Rows in `results/trial_log.csv` when this report was built: {trials.rows} "
        f"({trials.failed} failed), {trials.distinct_configs} distinct configurations.",
        f"- Best non-1/N strategy here: `{trials.selected}`, net Sharpe "
        f"{trials.selected_sharpe:.3f}.",
        f"- Deflated Sharpe ratio (Bailey and López de Prado 2014), with N = {trials.rows} trials "
        f"and the variance of their Sharpe ratios ({trials.var_trials_annual:.4f}, annual): "
        f"expected maximum Sharpe under the null {trials.sr0_annual:.3f}; DSR = {trials.dsr:.3f}.",
        "  The DSR asks whether the best strategy's Sharpe ratio exceeds what the best of N "
        "zero-skill trials would show by luck; it is not a test against 1/N, which the table "
        "above answers.",
        "",
        "## Constraint audit (AC-W-02)",
        "",
        *md_table(audit.drop("run_id"), digits=6),
        "",
        "![Cumulative returns](figures/cumulative_vs_1n.png)",
    ]
    (reports_dir / "oos_vs_1n.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    out["oos_md"] = reports_dir / "oos_vs_1n.md"
    alines = [
        "# Ablation: what the regime layer adds",
        "",
        "Generated by `regport evaluate`; do not edit by hand.",
        "",
        "Each regime arm against the same optimizer without a regime layer, net of costs. "
        "Differences are arm minus base; p-values are the Ledoit-Wolf bootstrap and HAC tests of "
        "the paired Sharpe difference.",
        "",
        *(
            md_table(
                ablation_view.sort("optimizer", "regime"),
                [
                    "optimizer",
                    "regime",
                    "delta_sharpe",
                    "p_bootstrap",
                    "p_hac",
                    "delta_ceq",
                    "delta_turnover",
                    "delta_cost_drag_bps",
                    "delta_max_drawdown",
                ],
            )
            if not ablation_view.is_empty()
            else ["No regime arms in this grid."]
        ),
        "",
        "![Ablation Sharpe](figures/ablation_sharpe.png)",
    ]
    (reports_dir / "ablation.md").write_text(
        "\n".join(alines) + "\n", encoding="utf-8", newline="\n"
    )
    out["ablation_md"] = reports_dir / "ablation.md"
    return out
