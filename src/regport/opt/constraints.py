"""Portfolio constraint verification (FR-24, AC-W-02, NFR-03)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from regport.errors import ConstraintViolation

F64 = npt.NDArray[np.float64]
CLIP_TOL = 1e-9


def effective_cap(max_weight: float | None, n: int) -> float:
    """The cap actually applied: at least 1/N, so a fully invested long-only portfolio exists."""
    if max_weight is None:
        return 1.0
    return max(max_weight, 1.0 / n)


def check_constraints(w: F64, max_weight: float | None, tol: float = 1e-8) -> None:
    """Raise ``ConstraintViolation`` unless ``1'w = 1 ± tol``, ``w >= -1e-9``, ``w <= cap+tol``."""
    w = np.asarray(w, dtype=np.float64)
    if not np.all(np.isfinite(w)):
        raise ConstraintViolation("weights contain NaN or inf")
    if abs(float(w.sum()) - 1.0) > tol:
        raise ConstraintViolation(f"weights sum to {w.sum():.12f}")
    if w.min() < -CLIP_TOL:
        raise ConstraintViolation(f"negative weight {w.min():.3e}")
    cap = effective_cap(max_weight, w.size)
    if w.max() > cap + tol:
        raise ConstraintViolation(f"weight {w.max():.12f} above cap {cap}")


def clean_weights(w: F64, max_weight: float | None = None) -> F64:
    """Clip solver noise within ``1e-9`` and renormalize (NFR-03); larger violations raise."""
    w = np.asarray(w, dtype=np.float64)
    if w.min() < -CLIP_TOL:
        raise ConstraintViolation(f"negative weight {w.min():.3e} beyond clip tolerance")
    out = np.clip(w, 0.0, None)
    out = out / out.sum()
    check_constraints(out, max_weight)
    return out
