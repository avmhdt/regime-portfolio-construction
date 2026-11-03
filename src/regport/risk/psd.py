"""Symmetrization and eigenvalue flooring of covariance matrices (FR-09, NFR-03)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


def symmetrize(cov: F64) -> F64:
    return 0.5 * (cov + cov.T)


def nearest_psd(cov: F64, floor_rel: float = 1e-10) -> F64:
    """Symmetric PSD matrix with every eigenvalue at least ``floor_rel`` times the mean eigenvalue.

    ``Σ = V diag(max(λ, floor_rel · mean(λ))) V'`` after symmetrizing; a matrix that already
    satisfies the floor is returned symmetrized but otherwise unchanged.
    """
    sym = symmetrize(np.asarray(cov, dtype=np.float64))
    values, vectors = np.linalg.eigh(sym)
    floor = floor_rel * float(np.mean(np.abs(values)))
    if values.min() >= floor:
        return sym
    clipped = np.maximum(values, floor)
    out: F64 = (vectors * clipped) @ vectors.T
    return symmetrize(out)
