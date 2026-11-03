"""``reports/regimes.md`` and the regime figures (FR-34, AC-H-09).

The smoothed probabilities here use the final refit's parameters over the whole sample. They
illustrate leakage only and never feed a strategy (DQ-08). Emission statistics are shown for the
equity-return feature only; BAA10Y is a model input that is never tabulated (P-07 ruling).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from regport.config import ExperimentConfig
from regport.regime.bocpd import NormalGammaPrior, bocpd
from regport.regime.features import FEATURE_SETS, PROXY_LABEL
from regport.regime.filter import forward_filter
from regport.regime.hmm import HMMParams
from regport.report.figures import bocpd_heatmap, filtered_vs_smoothed, smoothed_for_illustration
from regport.report.tables import md_table, write_csv


@dataclass(frozen=True)
class SwitchTiming:
    window: str
    filtered_first: date | None
    smoothed_first: date | None
    walk_forward_first: date | None

    @property
    def lead_days(self) -> int | None:
        """Days by which the smoother (same parameters, whole sample) is ahead of the filter."""
        if self.filtered_first is None or self.smoothed_first is None:
            return None
        return (self.filtered_first - self.smoothed_first).days


def high_volatility_states(params: HMMParams) -> np.ndarray:
    """States whose equity-return variance exceeds the stationary-weighted average variance."""
    var = params.covars[:, 0, 0]
    out: np.ndarray = np.flatnonzero(var > stationary(params.transmat) @ var)
    return out


def stationary(transmat: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eig(transmat.T)
    v = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
    out: np.ndarray = v / v.sum()
    return out


def params_from_record(record: dict[str, Any]) -> HMMParams:
    """A refit's parameters in raw feature units (standardization folded in)."""
    a = np.array(record["transmat"])
    d = len(record["means"][0])
    return HMMParams(
        stationary(a),
        a,
        np.array(record["means"]),
        np.array(record["covars"]),
        date.fromisoformat(record["fit_end"]),
        int(record["n_obs"]),
        float(record["loglik"]),
        int(record["seed"]),
        np.zeros(d),
        np.ones(d),
    )


def first_crossing(
    dates: list[date], p: np.ndarray, start: date, end: date, pad_days: int = 90, level: float = 0.5
) -> date | None:
    lo = start - timedelta(days=pad_days)
    for d, v in zip(dates, p, strict=True):
        if lo <= d <= end and v > level:
            return d
    return None


def write_regimes_report(
    cfg: ExperimentConfig,
    features: pl.DataFrame,
    probs: pl.DataFrame,
    fits_path: Path,
    reports_dir: Path,
) -> dict[str, Path]:
    fits = json.loads(fits_path.read_text(encoding="utf-8"))
    records = fits["fits"]
    last = params_from_record(records[str(max(int(k) for k in records))])
    k = last.n_states
    # Labels are aligned across refits, so state indices don't follow volatility order.
    high = high_volatility_states(last)
    cols = FEATURE_SETS[cfg.regime.hmm.feature_set]
    hmm_dates = probs["date"].unique(maintain_order=True).to_list()
    walk_forward = probs.sort("date", "state")["p_filtered"].to_numpy().reshape(-1, k)
    feat = features.filter(pl.col("date").is_in(hmm_dates))
    x = feat.select(cols).to_numpy()
    # Same parameters for both, so the only difference is the information set (DQ-08).
    same_filtered, _, _ = forward_filter(x, last)
    smoothed = smoothed_for_illustration(x, last)
    p_wf = walk_forward[:, high].sum(axis=1)
    p_filt = same_filtered[:, high].sum(axis=1)
    p_smooth = smoothed[:, high].sum(axis=1)

    windows = cfg.evaluation.stress_windows
    timing = [
        SwitchTiming(
            name,
            first_crossing(hmm_dates, p_filt, s, e),
            first_crossing(hmm_dates, p_smooth, s, e),
            first_crossing(hmm_dates, p_wf, s, e),
        )
        for name, (s, e) in windows.items()
    ]
    figs = reports_dir / "figures"
    fig1 = filtered_vs_smoothed(
        hmm_dates,
        p_filt,
        p_smooth,
        {n: w for n, w in windows.items() if n != "rates_2022"},
        figs / "filtered_vs_smoothed.png",
    )
    b = cfg.regime.bocpd
    prior = NormalGammaPrior(b.prior.mu0, b.prior.kappa0, b.prior.alpha0, b.prior.beta0)
    bres = bocpd(
        features[b.series].to_numpy(),
        b.hazard_lambda_days,
        prior,
        b.max_run_length,
        b.short_run_days,
        keep_posterior=True,
    )
    assert bres.posterior is not None
    fig2 = bocpd_heatmap(
        features["date"].to_list(),
        bres.posterior,
        bres.cp_prob_short,
        figs / "bocpd_run_length.png",
    )

    oos = np.array(hmm_dates, dtype="datetime64[D]") >= np.datetime64(cfg.sample.oos_start, "D")
    argmax = walk_forward.argmax(axis=1)
    pi = stationary(last.transmat)
    states = pl.DataFrame(
        {
            "state": list(range(k)),
            "equity_return_mean_annual": last.means[:, 0] * 252,
            "equity_volatility_annual": np.sqrt(last.covars[:, 0, 0] * 252),
            "expected_duration_days": 1.0 / (1.0 - np.diag(last.transmat)),
            "stationary_probability": pi,
            "share_of_oos_days_most_likely": [float(np.mean(argmax[oos] == s)) for s in range(k)],
        }
    )
    proxy = features.filter(pl.col("spy_source") == PROXY_LABEL)
    oos_cp = bres.cp_prob_short[
        np.array(features["date"].to_list(), dtype="datetime64[D]")
        >= np.datetime64(cfg.sample.oos_start, "D")
    ]
    lines = [
        "# Regime models",
        "",
        "Generated by `regport regime report`; do not edit by hand.",
        "",
        "## Set-up",
        "",
        f"- Gaussian HMM, K = {k} (BIC on 1993-02 to 2004-12, D-05), features "
        f"{', '.join(cols)} (set {cfg.regime.hmm.feature_set}).",
        f"- Refits: {len(records)}, every {cfg.regime.hmm.refit_every_days} trading days on an "
        f"{cfg.regime.hmm.window} window of at least {cfg.regime.hmm.min_train_days} days; "
        f"first fit ends {records['0']['fit_end']}.",
        f"- **Proxy:** before 1998-01-02 the SPY return feature is the Kenneth French daily market "
        f"return (Mkt-RF + RF): {proxy.height} days, {proxy['date'].min()!s} to "
        f"{proxy['date'].max()!s}. It feeds the regime models only, never a portfolio return.",
        f"- Label switches (Hungarian matching overruled the canonical order): "
        f"{len(fits['label_switches'])}"
        + (f", on {', '.join(fits['label_switches'])}." if fits["label_switches"] else "."),
        f"- BOCPD: hazard 1/{b.hazard_lambda_days:g}, Normal-Gamma prior (alpha0 "
        f"{b.prior.alpha0:g}, beta0 {b.prior.beta0:.3g}); P(run length < {b.short_run_days}) "
        f"above {b.trigger_prob} on {100 * float(np.mean(oos_cp > b.trigger_prob)):.1f}% of OOS "
        "days.",
        "",
        "## States at the last refit (equity-return emission only)",
        "",
        *md_table(states, digits=3),
        "",
        "## Filtered vs smoothed (AC-H-09)",
        "",
        "Stress probability: the total probability of the high-volatility states (states "
        f"{', '.join(map(str, high.tolist()))} at the last refit, whose equity-return variance "
        "exceeds the stationary-weighted average). The table gives the first date it passes "
        "50%, from 90 days before each window. Filtered and smoothed use the same (final) "
        "parameters, so the only difference is that the smoother also sees the data after each "
        "date. The walk-forward column is what the strategies actually used.",
        "",
        "| window | filtered | smoothed | smoothed earlier by (days) | walk-forward filtered |",
        "|---|---|---|---|---|",
        *[
            f"| {t.window} | {t.filtered_first} | {t.smoothed_first} | {t.lead_days} | "
            f"{t.walk_forward_first} |"
            for t in timing
        ],
        "",
        "![Filtered vs smoothed](figures/filtered_vs_smoothed.png)",
        "",
        "![BOCPD run length](figures/bocpd_run_length.png)",
    ]
    summary = pl.DataFrame(
        {
            "n_states": [k],
            "feature_set": [cfg.regime.hmm.feature_set],
            "refits": [len(records)],
            "first_fit_end": [records["0"]["fit_end"]],
            "label_switches": [len(fits["label_switches"])],
            "proxy_days": [proxy.height],
            "bocpd_hazard_days": [b.hazard_lambda_days],
            "bocpd_trigger_share_oos": [float(np.mean(oos_cp > b.trigger_prob))],
            **{f"smoother_lead_days_{t.window}": [t.lead_days] for t in timing},
        }
    )
    write_csv(summary, reports_dir / "tables" / "regime_summary.csv")
    md = reports_dir / "regimes.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return {"report": md, "filtered_vs_smoothed": fig1, "bocpd": fig2}
