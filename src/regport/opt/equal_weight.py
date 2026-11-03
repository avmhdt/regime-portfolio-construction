"""1/N benchmark (FR-21; DeMiguel, Garlappi and Uppal 2009)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


def equal_weight(n: int) -> F64:
    """Equal weights over the ``n`` eligible assets."""
    if n < 1:
        raise ValueError("need at least one asset")
    return np.full(n, 1.0 / n)
