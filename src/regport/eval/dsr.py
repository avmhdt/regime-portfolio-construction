"""Deflated Sharpe ratio (AC-H-04; Bailey and López de Prado 2014).

``SR0 = sqrt(V) ((1 − γ) Φ⁻¹(1 − 1/N) + γ Φ⁻¹(1 − 1/(N e)))`` is the expected maximum Sharpe
ratio of ``N`` independent trials whose Sharpe ratios have variance ``V`` (``γ`` the
Euler–Mascheroni constant), and ``DSR = Φ((SR − SR0) sqrt(T − 1) / sqrt(1 − γ3 SR +
(γ4 − 1)/4 SR²))`` with the selected trial's per-period ``SR``, skewness ``γ3`` and non-excess
kurtosis ``γ4``. Matches backtest-overfitting-lab's trial-log contract v1.0 worked example.
"""

from __future__ import annotations

import math

from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(var_trials: float, n_trials: int) -> float:
    if n_trials < 2:
        return 0.0
    g = EULER_GAMMA
    return math.sqrt(var_trials) * (
        (1 - g) * float(norm.ppf(1 - 1 / n_trials))
        + g * float(norm.ppf(1 - 1 / (n_trials * math.e)))
    )


def deflated_sharpe(
    sr: float, n_obs: int, skew: float, kurtosis: float, var_trials: float, n_trials: int
) -> tuple[float, float]:
    """``(DSR, SR0)``; ``sr`` per period, ``kurtosis`` non-excess, ``var_trials`` per period."""
    sr0 = expected_max_sharpe(var_trials, n_trials)
    denom = math.sqrt(1 - skew * sr + (kurtosis - 1) / 4 * sr**2)
    z = (sr - sr0) * math.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z)), sr0
