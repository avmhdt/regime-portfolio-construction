"""Stress-window statistics (FR-29, AC-H-03). Windows come from the config and are never tuned.

For each window ``[start, end]`` the wealth path is the run's own compounded net return. The
high-water mark entering the window is the running maximum up to the day before ``start``:

- max drawdown: the lowest ``wealth / running HWM − 1`` inside the window;
- peak and trough dates of that drawdown, and the trading days between them;
- recovery: trading days from the trough until wealth regains the peak level, or
  "not recovered by <last date>";
- worst 5-day return (compounded, inside the window) and annualized realized volatility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class DrawdownStats:
    max_drawdown: float
    peak_date: date
    trough_date: date
    days_to_trough: int
    recovery_date: date | None
    days_to_recover: int | None
    recovered_by: date
    worst_5d_return: float
    realized_vol: float
    window_return: float

    @property
    def recovery_text(self) -> str:
        if self.recovery_date is None:
            return f"not recovered by {self.recovered_by}"
        return f"{self.days_to_recover} days ({self.recovery_date})"


def drawdown_stats(ret: F64, dates: list[date], start: date, end: date) -> DrawdownStats:
    r = np.asarray(ret, dtype=np.float64)
    if r.size != len(dates):
        raise ValueError("returns and dates differ in length")
    idx = [i for i, d in enumerate(dates) if start <= d <= end]
    if len(idx) < 5:
        raise ValueError(f"window {start} to {end} has fewer than 5 return days")
    i0, i1 = idx[0], idx[-1]
    wealth = np.cumprod(1.0 + r)
    hwm = np.maximum.accumulate(wealth)
    entering = hwm[i0 - 1] if i0 > 0 else 1.0
    running = np.maximum.accumulate(np.concatenate([[entering], wealth[i0 : i1 + 1]]))[1:]
    dd = wealth[i0 : i1 + 1] / running - 1.0
    trough = i0 + int(np.argmin(dd))
    peak_level = running[trough - i0]
    peak_candidates = [i for i in range(trough + 1) if wealth[i] >= peak_level - 1e-15]
    peak = peak_candidates[-1] if peak_candidates else max(i0 - 1, 0)
    after = np.flatnonzero(wealth[trough + 1 :] >= peak_level)
    recovery = trough + 1 + int(after[0]) if after.size else None
    window = r[i0 : i1 + 1]
    five = np.array([np.prod(1 + window[k : k + 5]) - 1 for k in range(window.size - 4)])
    return DrawdownStats(
        max_drawdown=float(dd.min()),
        peak_date=dates[peak],
        trough_date=dates[trough],
        days_to_trough=trough - peak,
        recovery_date=dates[recovery] if recovery is not None else None,
        days_to_recover=(recovery - trough) if recovery is not None else None,
        recovered_by=dates[-1],
        worst_5d_return=float(five.min()),
        realized_vol=float(np.std(window, ddof=1) * math.sqrt(252)),
        window_return=float(np.prod(1 + window) - 1),
    )
