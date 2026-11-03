"""Rebalancing rule and decision calendar (FR-23, D-06)."""

from __future__ import annotations

import itertools
from datetime import date
from typing import Literal

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


def rebalance(
    w_drift: F64,
    w_target: F64,
    policy: Literal["calendar", "threshold"],
    threshold_abs: float,
) -> F64:
    """Post-trade weights: the target, or the drifted weights if the threshold rule holds still.

    ``calendar`` always trades to target. ``threshold`` trades to target only if
    ``max_i |w_drift,i − w_target,i| > threshold_abs``.
    """
    drift = np.asarray(w_drift, dtype=np.float64)
    target = np.asarray(w_target, dtype=np.float64)
    if policy == "calendar":
        return target.copy()
    if policy == "threshold":
        return target.copy() if np.max(np.abs(drift - target)) > threshold_abs else drift.copy()
    raise ValueError(f"unknown rebalance policy {policy!r}")


def decision_dates(calendar: list[date], frequency: Literal["monthly", "weekly"]) -> list[date]:
    """Last trading date of each calendar month (or ISO week) in ``calendar``.

    The final period counts only if it is complete in the data, i.e. a later trading date
    exists in another period.
    """
    out = []
    for prev, nxt in itertools.pairwise(calendar):
        if frequency == "monthly":
            boundary = (prev.year, prev.month) != (nxt.year, nxt.month)
        elif frequency == "weekly":
            boundary = prev.isocalendar()[:2] != nxt.isocalendar()[:2]
        else:
            raise ValueError(f"unknown frequency {frequency!r}")
        if boundary:
            out.append(prev)
    return out
