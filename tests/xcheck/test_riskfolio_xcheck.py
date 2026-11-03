"""FR-32 / AC-W-08: ERC and HRP against riskfolio-lib within 1e-6 (tests only, marker xcheck).

riskfolio-lib is imported only here. Twenty random covariance matrices come from simulated
returns; three come from the synthetic multi-asset panel's three overlapping 504-day windows
(the real ETF panel stays out of CI). Argument names follow riskfolio-lib 7.x:
``Portfolio.rp_optimization(model="Classic", rm="MV", rf=0, b=None, hist=True)`` and
``HCPortfolio.optimization(model="HRP", codependence="pearson", rm="MV", linkage="single",
leaf_order=False)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from regport.opt.erc import erc
from regport.opt.hrp import hrp

pytestmark = pytest.mark.xcheck
rp = pytest.importorskip("riskfolio")
pd = pytest.importorskip("pandas")


def _random_returns(seed: int) -> Any:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 15))
    a = rng.normal(size=(n, n))
    cov = a @ a.T / n * 1e-4 + np.eye(n) * 1e-5
    x = rng.multivariate_normal(np.zeros(n), cov, size=600)
    return pd.DataFrame(x, columns=[f"A{i}" for i in range(n)])


def _panel_returns(synthetic_panel: tuple[pl.DataFrame, pl.DataFrame]) -> list[Any]:
    returns, _ = synthetic_panel
    wide = returns.pivot(on="symbol", index="date", values="ret")
    wide = wide.sort("date").drop("date")
    wide = wide.drop_nulls()
    return [wide[i : i + 504].to_pandas() for i in (0, 500, 1000)]


def _cases(synthetic_panel: tuple[pl.DataFrame, pl.DataFrame]) -> list[Any]:
    return [_random_returns(s) for s in range(20)] + _panel_returns(synthetic_panel)


def test_erc_matches_riskfolio(synthetic_panel: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    for x in _cases(synthetic_panel):
        port = rp.Portfolio(returns=x)
        port.assets_stats(method_mu="hist", method_cov="hist")
        theirs = port.rp_optimization(model="Classic", rm="MV", rf=0, b=None, hist=True)
        ours = erc(port.cov.to_numpy())
        assert np.max(np.abs(ours - theirs.to_numpy().ravel())) <= 1e-6


def test_hrp_matches_riskfolio(synthetic_panel: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    for x in _cases(synthetic_panel):
        port = rp.HCPortfolio(returns=x)
        theirs = port.optimization(
            model="HRP", codependence="pearson", rm="MV", rf=0, linkage="single", leaf_order=False
        )
        ours = hrp(x.cov().to_numpy(), "single")
        assert np.max(np.abs(ours - theirs.to_numpy().ravel())) <= 1e-6
