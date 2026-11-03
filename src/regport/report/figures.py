"""Report figures (PNG, matplotlib Agg backend)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from regport.regime.hmm import HMMParams

ESTIMATOR_LABELS = {"sample": "Sample", "ewma": "EWMA", "ledoit_wolf": "Ledoit-Wolf", "oas": "OAS"}


def cov_eval_bias(table: pl.DataFrame, path: Path, sample: str = "full") -> Path:
    """Bias statistic by estimator and horizon (1/N and median random portfolio), 95% bands."""
    data = table.filter(pl.col("sample") == sample)
    horizons = sorted(data["horizon_days"].unique().to_list())
    estimators = [e for e in ESTIMATOR_LABELS if e in data["estimator"].unique().to_list()]
    fig, axes = plt.subplots(
        1, len(horizons), figsize=(4.2 * len(horizons), 3.6), sharey=True, squeeze=False
    )
    for ax, h in zip(axes[0], horizons, strict=True):
        rows = data.filter(pl.col("horizon_days") == h)
        xs = range(len(estimators))
        for offset, col, label in (
            (-0.15, "b_equal_weight", "1/N"),
            (0.15, "b_random_median", "random (median)"),
        ):
            vals = [rows.filter(pl.col("estimator") == e)[col][0] for e in estimators]
            ax.scatter([x + offset for x in xs], vals, label=label, zorder=3)
        lo = rows["band_lo95"][0]
        hi = rows["band_hi95"][0]
        ax.axhspan(lo, hi, color="0.85", zorder=1, label="95% band")
        ax.axhline(1.0, color="0.4", lw=0.8)
        ax.set_xticks(list(xs), [ESTIMATOR_LABELS[e] for e in estimators], rotation=20)
        ax.set_title(f"h = {h} days")
    axes[0][0].set_ylabel("bias statistic b")
    axes[0][-1].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Volatility forecast bias, {sample.replace('_', ' ')} OOS sample")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, metadata={"Software": None})
    plt.close(fig)
    return path


def smoothed_for_illustration(x: np.ndarray, params: HMMParams) -> np.ndarray:
    """Forward-backward smoothed probabilities ``P(S_t | x_1:T)`` (DQ-08: illustration only).

    These condition on the whole sample, so they may never feed a strategy. They exist only to
    show how much earlier a smoother "knows" about a regime change than the real-time filter.
    """
    from scipy.special import logsumexp

    from regport.regime.filter import log_emissions

    z = (np.asarray(x, dtype=np.float64) - params.x_mean) / params.x_std
    log_b = log_emissions(z, params.means, params.covars)
    log_a = np.log(np.maximum(params.transmat, 1e-300))
    t, k = log_b.shape
    log_alpha = np.empty((t, k))
    log_alpha[0] = np.log(np.maximum(params.startprob, 1e-300)) + log_b[0]
    for i in range(1, t):
        log_alpha[i] = log_b[i] + logsumexp(log_alpha[i - 1][:, None] + log_a, axis=0)
    log_beta = np.zeros((t, k))
    for i in range(t - 2, -1, -1):
        log_beta[i] = logsumexp(log_a + (log_b[i + 1] + log_beta[i + 1])[None, :], axis=1)
    log_post = log_alpha + log_beta
    log_post -= logsumexp(log_post, axis=1, keepdims=True)
    return np.exp(log_post)


def filtered_vs_smoothed(
    dates: list[date],
    filtered: np.ndarray,
    smoothed: np.ndarray,
    windows: dict[str, tuple[date, date]],
    path: Path,
    pad_days: int = 90,
) -> Path:
    """Probability of the highest-volatility state, filtered vs smoothed, around each window."""
    fig, axes = plt.subplots(1, len(windows), figsize=(5.2 * len(windows), 3.4), squeeze=False)
    arr = np.array(dates, dtype="datetime64[D]")
    for ax, (name, (start, end)) in zip(axes[0], windows.items(), strict=True):
        lo = np.datetime64(start, "D") - np.timedelta64(pad_days, "D")
        hi = np.datetime64(end, "D") + np.timedelta64(pad_days, "D")
        m = (arr >= lo) & (arr <= hi)
        ax.plot(arr[m], filtered[m], label="filtered (real time)", lw=1.2)
        ax.plot(arr[m], smoothed[m], label="smoothed (uses the future)", lw=1.2, ls="--")
        ax.axvspan(np.datetime64(start, "D"), np.datetime64(end, "D"), color="0.9", zorder=0)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(name.replace("_", " "))
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    axes[0][0].set_ylabel("P(high-volatility states)")
    axes[0][-1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, metadata={"Software": None})
    plt.close(fig)
    return path


def bocpd_heatmap(
    dates: list[date],
    posterior: np.ndarray,
    cp_short: np.ndarray,
    path: Path,
    max_rl: int = 750,
    step: int = 5,
) -> Path:
    """Run-length posterior (log scale) over time, with ``P(r < short)`` underneath."""
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    arr = np.array(dates, dtype="datetime64[D]")[::step]
    img = np.log10(np.maximum(posterior[::step, : max_rl + 1].T, 1e-8))
    top.pcolormesh(
        arr, np.arange(img.shape[0]), img, shading="auto", cmap="viridis", vmin=-8, vmax=0
    )
    top.set_ylabel("run length (days)")
    top.set_title("BOCPD run-length posterior, log10 probability")
    bottom.plot(np.array(dates, dtype="datetime64[D]"), cp_short, lw=0.6)
    bottom.axhline(0.5, color="0.4", lw=0.8)
    bottom.set_ylabel("P(r < 21)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, metadata={"Software": None})
    plt.close(fig)
    return path
