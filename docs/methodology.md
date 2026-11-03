# Methodology

Formulas behind every number in `reports/`. This file grows milestone by milestone (NFR-07): so far it covers the risk models (M3). The regime filter, BOCPD, ERC, HRP and the Ledoit–Wolf Sharpe test are added in M4 to M6.

## Covariance estimators (FR-09)

Let `r_1, …, r_T` be the daily simple returns of the N assets in the window ending at the decision date. The window is 504 trading days, or shorter if rows with missing returns are dropped (DQ-04). Every estimate is annualized by 252, symmetrized, and floored so that each eigenvalue is at least `1e-10` times the mean eigenvalue (`regport.risk.psd.nearest_psd`).

| Estimator | Formula |
|---|---|
| Sample | `S = (1/(T−1)) Σ_t (r_t − r̄)(r_t − r̄)'` |
| EWMA (RiskMetrics 1996) | Zero-mean recursion `S_t = λ S_{t−1} + (1−λ) r_t r_t'`, started at `S_0 = 0`, with `λ = 0.5^(1/h)` and half-life `h = 63` days. The result is divided by `1 − λ^T`, so the weights `(1−λ) λ^(T−t) / (1−λ^T)` sum to one over the finite window |
| Ledoit–Wolf (2004) | `δ* F + (1 − δ*) S`, shrinking toward a scaled identity `F = (tr S / N) I` with the optimal intensity `δ*` (scikit-learn `LedoitWolf`) |
| OAS (Chen et al. 2010) | The same target, with the oracle-approximating shrinkage intensity (scikit-learn `OAS`) |

## Out-of-sample risk evaluation (FR-10)

For horizon `h ∈ {1, 5, 21}` trading days, origins are every `h`-th trading date from 2005-01-03, so realized windows don't overlap. At origin `t`, a portfolio `w` fixed at `t` has:

- **Forecast variance:** `F_t = w' Σ_t w · h / 252`, with `Σ_t` estimated from returns up to and including `t`.
- **Realized h-day return:** `R_t = Σ_i w_i (Π_{s=t+1}^{t+h} (1 + r_{i,s}) − 1)`.
- **Realized variance proxy:** `RV_t = Σ_{s=t+1}^{t+h} (w' r_s)²`.

The evaluation portfolios at each origin are 1/N, 100 long-only Dirichlet(1) portfolios (one fixed draw per origin, shared by all estimators), and each estimator's own minimum-variance portfolio (long-only, cap 0.40).

- **Bias statistic.** `b = std(R_t / sqrt(F_t))` over the T origins. A correct forecast gives `b ≈ 1`, and under normality `b` lies within `1 ± 1.96 sqrt(1/(2T))` with 95% probability. `b > 1` means the model understates risk.
- **QLIKE** (Patton 2011). `L = mean_t (log F_t + RV_t / F_t)`. Among losses, it is robust to the noise in the realized-variance proxy. Lower is better.
- **Min-variance realized volatility.** `std(R_t) · sqrt(252/h)` for each estimator's own minimum-variance portfolio.

## Estimator selection (D-12)

The selection rule was fixed in `docs/decisions.md` before the first evaluation. The chosen estimator has the lowest mean QLIKE at `h = 21`, averaged over 1/N and the random portfolios, on windows that end by 2014-12-31. Its statistics are then reported on the second half and on the full OOS period. AC-W-03 requires `b ∈ [0.90, 1.10]` at `h = 21` over the full OOS, for both 1/N and the median random portfolio.

## Minimum variance (FR-17)

`min w'Σw` subject to `1'w = 1` and `0 ≤ w ≤ c`, where `c = max(0.40, 1/N)`. The problem is solved as `||L'w||²` with `Σ = LL'`, first with CLARABEL and then with OSQP if CLARABEL fails. Weights within `1e-9` of the bounds are clipped and renormalized; anything larger raises `ConstraintViolation`. Validation: when no bound binds, the solution equals the closed form `Σ⁻¹1 / (1'Σ⁻¹1)` within 1e-6.
