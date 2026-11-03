"""README result blocks, rendered from ``reports/tables/`` (AC-W-04: no hand-typed numbers)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from regport.eval.robustness import break_even
from regport.report.evaluation import headline_sentence
from regport.report.tables import md_table


def _read(tables: Path, name: str) -> pl.DataFrame | None:
    path = tables / name
    return pl.read_csv(path) if path.exists() else None


def _label(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("optimizer") == "equal_weight")
        .then(pl.lit("1/N"))
        .when(pl.col("regime") == "none")
        .then(pl.col("optimizer"))
        .otherwise(pl.col("optimizer") + "+" + pl.col("regime"))
        .alias("strategy")
    )


def headline_block(tables: Path, headline: str) -> str:
    t = _read(tables, "oos_vs_1n.csv")
    if t is None:
        return "_Run `regport evaluate`._"
    head = t.filter(pl.col("strategy") == headline)
    bench = t.filter(pl.col("strategy") == "1/N")
    lines = [f"**{headline_sentence(t)}**", ""]
    if head.height and bench.height:
        h, b = head.row(0, named=True), bench.row(0, named=True)
        lines += [
            f"The preregistered headline, `{headline}`, earned a net Sharpe ratio of "
            f"{h['sharpe']:.2f} (95% interval {h['sharpe_ci_lo']:.2f} to {h['sharpe_ci_hi']:.2f}) "
            f"against {b['sharpe']:.2f} for 1/N; the difference, {h['delta_sharpe_vs_1n']:+.2f}, "
            f"has a bootstrap p-value of {h['p_bootstrap']:.2f} ({h['p_holm']:.2f} after Holm). "
            f"The smallest difference this sample could detect with 80% power is about "
            f"{h['min_detectable_delta_80pct']:.2f}.",
            "",
        ]
    keep = (pl.col("regime") == "none") | (pl.col("strategy") == headline)
    view = (
        t.filter(keep)
        .sort("sharpe", descending=True)
        .with_columns((pl.col("annual_cost_drag") * 1e4).alias("cost_drag_bps"))
    )
    lines += [
        f"Selected rows (the headline, each optimizer without regimes, and 1/N); all "
        f"{t.height - 1} strategies are in reports/oos_vs_1n.md.",
        "",
    ]
    lines += md_table(
        view,
        [
            "strategy",
            "sharpe",
            "delta_sharpe_vs_1n",
            "p_bootstrap",
            "max_drawdown",
            "annual_turnover",
            "cost_drag_bps",
        ],
        digits=2,
    )
    return "\n".join(lines)


def ablation_block(tables: Path) -> str:
    a = _read(tables, "ablation.csv")
    if a is None or a.is_empty():
        return "_Run `regport evaluate` on the ablation grid._"
    best = a.sort("delta_sharpe", descending=True).row(0, named=True)
    lines = [
        f"Of {a.height} regime arms, {int((a['p_bootstrap'] <= 0.05).sum())} change the net "
        f"Sharpe ratio significantly at 5% (unadjusted). The largest gain is "
        f"`{best['optimizer']}+{best['regime']}`, {best['delta_sharpe']:+.2f} "
        f"(p = {best['p_bootstrap']:.2f}).",
        "",
        *md_table(
            a.sort("optimizer", "regime").with_columns(
                (pl.col("delta_cost_drag") * 1e4).alias("delta_cost_drag_bps")
            ),
            [
                "optimizer",
                "regime",
                "delta_sharpe",
                "p_bootstrap",
                "delta_turnover",
                "delta_max_drawdown",
            ],
            digits=2,
        ),
    ]
    return "\n".join(lines)


def stress_block(tables: Path, headline: str) -> str:
    s = _read(tables, "stress.csv")
    if s is None:
        return "_Run `regport stress`._"
    keep = [headline, "min_variance", "1/N"]
    view = s.filter(pl.col("strategy").is_in(keep)).sort(
        "window", "max_drawdown", descending=[False, True]
    )
    return "\n".join(
        md_table(
            view,
            [
                "window",
                "strategy",
                "max_drawdown",
                "recovery",
            ],
            digits=3,
        )
    )


def robustness_block(tables: Path, headline: str) -> str:
    lines: list[str] = []
    costs = _read(tables, "robustness_costs.csv")
    if costs is not None:
        lines += [
            "**Break-even one-way cost against 1/N** (uniform cost grid):",
            "",
            *md_table(break_even(costs)),
            "",
        ]
    opt, _, regime = headline.partition("+")
    cs = _read(tables, "robustness_common_start.csv")
    if cs is not None:
        h = cs.filter((pl.col("optimizer") == opt) & (pl.col("regime") == regime))
        b = cs.filter(pl.col("optimizer") == "equal_weight")
        if h.height and b.height:
            lines += [
                f"- **Common start ({cs['variant'][0]}):** `{headline}` "
                f"{h['sharpe'][0]:.2f} vs 1/N {b['sharpe'][0]:.2f} "
                f"(HAC p = {h['p_hac'][0]:.2f})."
            ]
    fr = _read(tables, "robustness_french.csv")
    if fr is not None:
        for variant in fr["variant"].unique(maintain_order=True).to_list():
            part = _label(fr.filter(pl.col("variant") == variant)).sort("sharpe", descending=True)
            top = part.row(0, named=True)
            bench = part.filter(pl.col("optimizer") == "equal_weight")["sharpe"][0]
            sig = int((part["p_hac"].fill_null(1.0) <= 0.05).sum())
            lines += [
                f"- **Kenneth French {variant.replace('_daily', '')}:** best "
                f"`{top['strategy']}` {top['sharpe']:.2f}, 1/N {bench:.2f}; "
                f"{sig} strategies differ from 1/N at 5% (HAC)."
            ]
    al = _read(tables, "robustness_alfred.csv")
    ar = _read(tables, "robustness_alfred_regime.csv")
    oos = _read(tables, "oos_vs_1n.csv")
    if al is not None and ar is not None and oos is not None:
        pit = oos.filter(pl.col("strategy") == "erc+hmm_B")
        m = dict(ar.iter_rows())
        if pit.height:
            lines += [
                f"- **ALFRED point in time vs revised (feature set B, ERC):** net Sharpe "
                f"{pit['sharpe'][0]:.3f} vs {al['sharpe'][0]:.3f}; stress-probability mean "
                f"absolute difference {m['stress_probability_mad']:.2f}, most likely state "
                f"agreeing on {100 * m['argmax_agreement']:.0f}% of days."
            ]
    return "\n".join(lines) if lines else "_Run `regport robustness`._"


def risk_block(tables: Path, chosen: str) -> str:
    c = _read(tables, "cov_eval.csv")
    if c is None:
        return "_Run `regport cov-eval`._"
    row = c.filter(
        (pl.col("estimator") == chosen)
        & (pl.col("horizon_days") == 21)
        & (pl.col("sample") == "full")
    )
    if row.is_empty():
        return "_No cov-eval row for the chosen estimator._"
    r = row.row(0, named=True)
    return (
        f"The estimator chosen on 2005–2014 windows (D-12) is `{chosen}`. Over the full "
        f"out-of-sample period at a 21-day horizon its bias statistic is {r['b_equal_weight']:.3f} "
        f"for 1/N and {r['b_random_median']:.3f} for the median random portfolio "
        f"({r['n_windows']} windows; 95% band {r['band_lo95']:.3f} to {r['band_hi95']:.3f}). "
        f"For its own minimum-variance portfolio it is {r['b_min_variance']:.3f}: like every "
        "estimator here, it understates the risk of the portfolio it was optimized on."
    )


def regime_block(tables: Path) -> str:
    r = _read(tables, "regime_summary.csv")
    if r is None:
        return "_Run `regport regime report`._"
    row = r.row(0, named=True)
    leads = ", ".join(
        f"{k.removeprefix('smoother_lead_days_')} {v}"
        for k, v in row.items()
        if k.startswith("smoother_lead_days_")
    )
    return (
        f"K = {row['n_states']} states on feature set {row['feature_set']}, "
        f"{row['refits']} walk-forward refits (first fit ending {row['first_fit_end']}), "
        f"{row['label_switches']} label switches. {row['proxy_days']} early days use the "
        f"French market proxy for SPY. BOCPD (hazard 1/{row['bocpd_hazard_days']:g}) flags a "
        f"short run on {100 * row['bocpd_trigger_share_oos']:.1f}% of out-of-sample days. "
        f"With the same parameters, the smoother crosses 50% stress probability earlier than "
        f"the real-time filter by (days): {leads}."
    )


def trials_block(tables: Path) -> str:
    t = _read(tables, "trials.csv")
    if t is None:
        return "_Run `regport evaluate`._"
    r = t.row(0, named=True)
    return (
        f"The trial log holds {r['trial_log_rows']} backtests ({r['failed_rows']} failed) over "
        f"{r['distinct_configurations']} distinct configurations, every run included. The "
        f"best strategy, `{r['selected']}` (net Sharpe {r['selected_sharpe']:.2f}), has a "
        f"deflated Sharpe ratio of {r['dsr']:.2f} against the best of "
        f"{r['trial_log_rows']} zero-skill trials (expected maximum "
        f"{r['expected_max_sharpe_annual']:.2f}); that is a test of skill against zero, not "
        "against 1/N."
    )


def scaling_block(tables: Path) -> str:
    c = _read(tables, "scaling_condition.csv")
    fr = _read(tables, "robustness_french.csv")
    lines = []
    if c is not None:
        big = c.filter(pl.col("n_assets") == c["n_assets"].max())
        vals = {r["estimator"]: r["condition_number"] for r in big.iter_rows(named=True)}
        lines.append(
            f"At N = {big['n_assets'][0]} assets on a 504-day window "
            f"(N/T = {big['n_over_t'][0]:.2f}) the condition number is "
            f"{vals.get('sample', float('nan')):.2g} for the sample covariance, "
            f"{vals.get('ledoit_wolf', float('nan')):.2g} for Ledoit-Wolf and "
            f"{vals.get('factor_pca10', float('nan')):.2g} for a 10-factor PCA model."
        )
    if fr is not None:
        f49 = fr.filter((pl.col("variant") == "ind49_daily") & (pl.col("regime") == "none"))
        vol = dict(zip(f49["optimizer"].to_list(), f49["annual_volatility"].to_list(), strict=True))
        if "hrp" in vol and "min_variance" in vol:
            lines.append(
                f"On the 49 industries, out-of-sample volatility was "
                f"{100 * vol['min_variance']:.1f}% for min-variance and "
                f"{100 * vol['hrp']:.1f}% for HRP."
            )
    return " ".join(lines) if lines else "_Run `regport bench scaling`._"


def data_block(tables: Path) -> str:
    d = _read(tables, "data_coverage.csv")
    if d is None:
        return "_Run `regport data check`._"
    spy = d.filter(pl.col("symbol") == "SPY")
    ordered = d.sort("first_eligible")
    text = f"The panel has {d.height} ETFs. "
    if spy.height:
        text += f"SPY's Sharadar history starts on {spy['first_date'][0]}; "
    return text + (
        f"all {d.height} are eligible from {ordered['first_eligible'][-1]} (the last "
        f"to enter is {ordered['symbol'][-1]})."
    )


def render_blocks(tables: Path, headline: str, chosen_estimator: str) -> dict[str, str]:
    return {
        "headline": headline_block(tables, headline),
        "ablation": ablation_block(tables),
        "stress": stress_block(tables, headline),
        "robustness": robustness_block(tables, headline),
        "risk": risk_block(tables, chosen_estimator),
        "regime": regime_block(tables),
        "trials": trials_block(tables),
        "scaling": scaling_block(tables),
        "data": data_block(tables),
    }
