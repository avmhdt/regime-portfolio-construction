"""Hierarchical risk parity (FR-19), own code after López de Prado (2016).

1. Correlation distance ``d_ij = sqrt((1 − ρ_ij) / 2)``, clustered by ``scipy`` linkage on the
   condensed distance matrix (single linkage by default).
2. Quasi-diagonalization: the dendrogram's leaf order, without optimal leaf reordering.
3. Recursive bisection: the ordered list is split in halves; each half's variance is that of its
   inverse-variance portfolio, and the halves get ``1 − v_L / (v_L + v_R)`` and the rest.

This matches riskfolio-lib's ``HCPortfolio`` with ``model="HRP"``, ``codependence="pearson"``,
``rm="MV"`` and ``leaf_order=False`` (cross-checked in ``tests/xcheck``).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

F64 = npt.NDArray[np.float64]
Linkage = Literal["single", "complete", "average", "ward"]


def correlation_distance(cov: F64) -> F64:
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    out: F64 = 0.5 * (dist + dist.T)
    return out


def quasi_diagonal_order(cov: F64, method: Linkage = "single") -> npt.NDArray[np.int64]:
    """Leaf order of the correlation-distance dendrogram."""
    condensed = squareform(correlation_distance(cov), checks=False)
    tree = linkage(condensed, method=method, optimal_ordering=False)
    order: npt.NDArray[np.int64] = leaves_list(tree).astype(np.int64)
    return order


def _cluster_variance(cov: F64, items: npt.NDArray[np.int64]) -> float:
    sub = cov[np.ix_(items, items)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp(cov: F64, linkage: Linkage = "single") -> F64:
    """HRP weights; long-only and fully invested by construction."""
    cov = np.asarray(cov, dtype=np.float64)
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)
    weights = np.ones(n)
    clusters = [quasi_diagonal_order(cov, linkage)]
    while clusters:
        nxt = []
        for items in clusters:
            if items.size < 2:
                continue
            half = items.size // 2
            left, right = items[:half], items[half:]
            v_left, v_right = _cluster_variance(cov, left), _cluster_variance(cov, right)
            alpha = 1.0 - v_left / (v_left + v_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            nxt += [left, right]
        clusters = nxt
    out: F64 = weights / weights.sum()
    return out
