"""Shared synthetic fixtures (no real data): a 10-asset daily panel with staggered entry."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

SYMBOLS = ["AEQ", "BEQ", "CEQ", "DRT", "ERT", "FCR", "GCR", "HCM", "ICM", "JFX"]
CLASSES = [
    "equity",
    "equity",
    "equity",
    "rates",
    "rates",
    "credit",
    "credit",
    "commodity",
    "commodity",
    "fx",
]


def weekdays(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def make_panel(
    n_days: int = 2600,
    seed: int = 11,
    late: dict[str, int] | None = None,
    start: date = date(2001, 1, 2),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``returns_daily`` and ``rf_daily`` frames for :data:`SYMBOLS` (two volatility regimes)."""
    rng = np.random.default_rng(seed)
    dates = weekdays(start, n_days)
    n = len(SYMBOLS)
    a = rng.normal(size=(n, n))
    base = a @ a.T / n + np.eye(n)
    vol = np.linspace(0.6, 1.4, n) * 1e-2
    cov = base / np.sqrt(np.outer(np.diag(base), np.diag(base))) * np.outer(vol, vol)
    regime = (np.sin(np.arange(n_days) / 180.0) > 0.6).astype(float)
    scale = 1.0 + 1.5 * regime
    r = rng.multivariate_normal(np.full(n, 2e-4), cov, size=n_days) * scale[:, None]
    late = late or {"HCM": 700, "JFX": 900}
    rows = []
    for j, sym in enumerate(SYMBOLS):
        first = late.get(sym, 1)
        hist = 0
        for i, d in enumerate(dates):
            ret = float(r[i, j]) if i >= first else None
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "ret": ret,
                    "logret": None if ret is None else float(np.log1p(ret)),
                    "n_valid_history": hist,
                    "eligible": hist >= 252,
                }
            )
            hist += ret is not None
    returns = pl.DataFrame(rows, schema_overrides={"n_valid_history": pl.Int32})
    rf = pl.DataFrame({"date": dates, "rf": np.full(n_days, 0.02 / 252)})
    return returns, rf


def write_universe(path: Path, cost_bps: float = 2.0) -> Path:
    assets = [
        {
            "symbol": s,
            "asset_class": c,
            "exposure": "synthetic",
            "inception": "2001-01-02",
            "sharadar": s,
            "tiingo": s.lower(),
            "stooq": f"{s.lower()}.us",
            "cost_bps": cost_bps,
        }
        for s, c in zip(SYMBOLS, CLASSES, strict=True)
    ]
    path.write_text(yaml.safe_dump({"assets": assets}), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def synthetic_panel() -> tuple[pl.DataFrame, pl.DataFrame]:
    return make_panel()


@pytest.fixture
def universe_file(tmp_path: Path) -> Path:
    return write_universe(tmp_path / "universe.yaml")
