"""AC-W-11: validation item (8) for the Ledoit-Wolf Sharpe test, plus Holm and the DSR."""

from __future__ import annotations

import math

import numpy as np
import pytest

from regport.eval.dsr import deflated_sharpe, expected_max_sharpe
from regport.eval.sharpe_test import (
    calibrate_block_size,
    hac_psi,
    holm,
    parzen,
    sharpe_diff_bootstrap,
    sharpe_diff_hac,
)


def _pairs(t: int, rho: float, mu: tuple[float, float], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=t)
    return z[:, 0] + mu[0], z[:, 1] + mu[1]


def test_hac_se_matches_iid_closed_form() -> None:
    """Item (8a): on T = 100,000 iid normal pairs the HAC SE is within 2% of Jobson-Korkie with
    Memmel's correction, Var = (2 − 2ρ + ½(SR1² + SR2² − 2 SR1 SR2 ρ²)) / T."""
    rho = 0.6
    r1, r2 = _pairs(100_000, rho, (0.05, 0.03), 7)
    res = sharpe_diff_hac(r1, r2, prewhiten=True)
    sr1, sr2 = 0.05, 0.03
    closed = math.sqrt((2 - 2 * rho + 0.5 * (sr1**2 + sr2**2 - 2 * sr1 * sr2 * rho**2)) / 100_000)
    assert abs(res.se / closed - 1) <= 0.02
    assert res.method == "hac" and res.block_size is None


def test_hac_size() -> None:
    """Item (8b): rejection rate at 5% under the null within [3.5%, 6.5%] over 2,000 runs."""
    rejections = 0
    for seed in range(2000):
        r1, r2 = _pairs(500, 0.5, (0.05, 0.05), seed)
        rejections += sharpe_diff_hac(r1, r2).p_value <= 0.05
    assert 0.035 <= rejections / 2000 <= 0.065


@pytest.mark.slow
def test_bootstrap_size() -> None:
    """Item (8c): bootstrap rejection rate at 5% within [3%, 7%] over 300 runs."""
    rejections = 0
    for seed in range(300):
        r1, r2 = _pairs(500, 0.5, (0.05, 0.05), 10_000 + seed)
        rejections += sharpe_diff_bootstrap(r1, r2, 999, 1, seed).p_value <= 0.05
    assert 0.03 <= rejections / 300 <= 0.07


def test_bootstrap_detects_a_large_difference() -> None:
    r1, r2 = _pairs(2000, 0.3, (0.15, -0.05), 3)
    res = sharpe_diff_bootstrap(r1, r2, 499, 2, 11)
    assert res.p_value < 0.01 and res.block_size == 2 and res.diff > 0.1


def test_block_calibration_returns_a_candidate() -> None:
    r1, r2 = _pairs(400, 0.5, (0.02, 0.02), 4)
    block = calibrate_block_size(r1, r2, seed=1, n_sim=20, n_boot=99, candidates=(1, 4))
    assert block in (1, 4)
    res = sharpe_diff_bootstrap(r1, r2, 199, None, 5, calibration_sims=10, calibration_boot=49)
    assert res.block_size in (1, 2, 4, 6, 8, 10) and 0 < res.p_value <= 1


def test_parzen_and_psi() -> None:
    assert parzen(np.array([0.0, 0.25, 0.5, 0.75, 1.0, 2.0])).tolist() == pytest.approx(
        [1.0, 1 - 6 * 0.0625 + 6 * 0.015625, 0.25, 2 * 0.25**3, 0.0, 0.0]
    )
    y = np.random.default_rng(1).normal(size=(5000, 4))
    psi = hac_psi(y, prewhiten=False)
    assert np.allclose(psi, np.eye(4), atol=0.1)


def test_input_checks() -> None:
    with pytest.raises(ValueError, match="aligned"):
        sharpe_diff_hac(np.zeros(5), np.zeros(5))
    with pytest.raises(ValueError, match="NaN"):
        sharpe_diff_hac(np.r_[np.zeros(19), np.nan], np.zeros(20))


def test_holm() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.20])
    assert holm(p).tolist() == pytest.approx([0.04, 0.09, 0.09, 0.20])
    assert holm(np.array([0.9, 0.8])).tolist() == pytest.approx([1.0, 1.0])


def test_dsr_matches_contract_worked_example() -> None:
    """backtest-overfitting-lab trial-log contract v1.0 worked example (to 1e-9)."""
    dsr, sr0 = deflated_sharpe(
        0.784662123195, 8, -0.346308523254, 2.018844051285, 0.414159210388, 4
    )
    assert sr0 == pytest.approx(0.677095775287, abs=1e-9)
    assert dsr == pytest.approx(0.594101825407, abs=1e-9)
    assert expected_max_sharpe(1.0, 1) == 0.0
