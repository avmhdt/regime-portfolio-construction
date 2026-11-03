"""Report figures (PNG, matplotlib Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

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
