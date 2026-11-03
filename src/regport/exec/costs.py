"""Proportional transaction costs (FR-22, D-04)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


class CostModel:
    """``cost_t = Σ_i c_i |trade_i,t|`` with one-way costs ``c_i`` in basis points.

    Costs are an assumption (D-04), not measured spreads. ``uniform(bps)`` overrides every asset
    for the sensitivity grid.
    """

    def __init__(self, bps: Mapping[str, float]) -> None:
        if any(v < 0 for v in bps.values()):
            raise ValueError("costs must be non-negative")
        self.bps = dict(bps)

    @classmethod
    def uniform(cls, symbols: list[str], bps: float) -> CostModel:
        return cls(dict.fromkeys(symbols, bps))

    def rates(self, symbols: list[str]) -> F64:
        """One-way cost rates as fractions (bps / 10,000)."""
        return np.array([self.bps[s] for s in symbols], dtype=np.float64) / 1e4

    def cost(self, trades: F64, symbols: list[str]) -> float:
        """Cost as a fraction of NAV of ``trades`` (signed weight changes)."""
        return float(self.rates(symbols) @ np.abs(np.asarray(trades, dtype=np.float64)))
