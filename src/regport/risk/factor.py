"""Statistical factor-model covariance for the scaling study (FR-31).

``Σ = B Σ_f B' + D`` from the top ``k`` principal components of the sample covariance:
``B Σ_f B' = Σ_{j<=k} λ_j v_j v_j'`` and ``D`` the diagonal of the residual, floored at 1e-10 of the
mean variance. Annualized like the other estimators.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from regport.risk.covariance import ANNUALIZATION
from regport.risk.psd import nearest_psd

F64 = npt.NDArray[np.float64]


def factor_cov(returns: F64, k: int = 10) -> F64:
    x = np.asarray(returns, dtype=np.float64)
    s = np.cov(x, rowvar=False, ddof=1)
    k = min(k, s.shape[0] - 1)
    values, vectors = np.linalg.eigh(s)
    top = np.argsort(values)[::-1][:k]
    common = (vectors[:, top] * values[top]) @ vectors[:, top].T
    resid = np.clip(np.diag(s - common), 1e-10 * float(np.mean(np.diag(s))), None)
    return nearest_psd((common + np.diag(resid)) * ANNUALIZATION)
