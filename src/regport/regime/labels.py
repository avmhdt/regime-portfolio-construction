"""Label switching (FR-13): canonical order, then Hungarian matching across refits.

1. Canonical order: states sorted by ascending variance of the equity-return emission
   (feature ``vol_feature``), so state 0 is the calmest.
2. Alignment: the new fit's states are matched to the previous refit's by
   ``scipy.optimize.linear_sum_assignment`` on the symmetric KL divergence between Gaussian
   emissions, in the features' original units. If the matching disagrees with the canonical
   order, the matching wins and a ``label_switch`` event is recorded (Stephens 2000).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from regport.regime.hmm import HMMParams

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


def canonical_order(params: HMMParams, vol_feature: int = 0) -> I64:
    """Permutation putting states in ascending order of the ``vol_feature`` emission variance."""
    order: I64 = np.argsort(params.covars[:, vol_feature, vol_feature], kind="stable").astype(
        np.int64
    )
    return order


def kl_gaussian(m0: F64, s0: F64, m1: F64, s1: F64) -> float:
    """``KL(N(m0, s0) || N(m1, s1))``."""
    d = m0.size
    inv1 = np.linalg.inv(s1)
    diff = m1 - m0
    _, logdet0 = np.linalg.slogdet(s0)
    _, logdet1 = np.linalg.slogdet(s1)
    return 0.5 * float(np.trace(inv1 @ s0) + diff @ inv1 @ diff - d + logdet1 - logdet0)


def symmetric_kl(m0: F64, s0: F64, m1: F64, s1: F64) -> float:
    return kl_gaussian(m0, s0, m1, s1) + kl_gaussian(m1, s1, m0, s0)


def align_states(prev: HMMParams, new: HMMParams) -> I64:
    """Permutation of ``new``'s states that best matches ``prev``: new state ``perm[i]``
    plays the role of previous state ``i``."""
    pm, pc = prev.raw_means(), prev.raw_covars()
    nm, nc = new.raw_means(), new.raw_covars()
    k = prev.n_states
    cost = np.array(
        [[symmetric_kl(pm[i], pc[i], nm[j], nc[j]) for j in range(k)] for i in range(k)]
    )
    rows, cols = linear_sum_assignment(cost)
    perm: I64 = cols[np.argsort(rows)].astype(np.int64)
    return perm


def relabel(prev: HMMParams | None, new: HMMParams, vol_feature: int = 0) -> tuple[HMMParams, I64]:
    """Canonically order ``new``, then align it with ``prev``.

    Returns the aligned parameters and the alignment permutation applied on top of the canonical
    order (identity when the matching agrees with it). A label switch is a *change* of this
    permutation between consecutive refits: once the matching has overruled the canonical order,
    later refits that keep the same mapping are not new switches.
    """
    canon = new.permute(canonical_order(new, vol_feature))
    if prev is None:
        return canon, np.arange(canon.n_states, dtype=np.int64)
    perm = align_states(prev, canon)
    return canon.permute(perm), perm
