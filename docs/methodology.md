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

## Equal risk contribution (FR-18)

ERC uses Spinu's (2013) convex problem: `min_y ½ y'Σy − Σ_i b_i ln y_i` over `y > 0`. Its first-order condition is `y_i (Σy)_i = b_i`, so `w = y / 1'y` has relative risk contributions `RC_i = w_i (Σw)_i / (w'Σw) = b_i`, with `b_i = 1/N` by default (Maillard, Roncalli and Teïletche 2010).

- **Solver:** Newton's method with gradient `Σy − b/y`, Hessian `Σ + diag(b/y²)` and Armijo backtracking that keeps `y > 0`.
- **Fallback:** cyclical coordinate descent, where each step is the positive root of `Σ_ii y_i² + (Σ_{j≠i} Σ_ij y_j) y_i − b_i = 0`.
- **Stopping:** at `max_i |RC_i − b_i| ≤ 1e-10`. A result above 1e-6 raises.

## Hierarchical risk parity (FR-19)

HRP follows López de Prado (2016):

1. **Distance:** `d_ij = sqrt((1 − ρ_ij)/2)`, clustered by single linkage on the condensed matrix.
2. **Ordering:** the dendrogram's leaf order (no optimal reordering).
3. **Bisection:** recursive halving of the ordered list. Each half's variance is that of its inverse-variance portfolio, `v = w_ivp' Σ_half w_ivp`. The left half's weight is scaled by `α = 1 − v_L/(v_L + v_R)` and the right half's by `1 − α`.

## Mean-variance with a turnover penalty (FR-20)

The problem is `max_w μ'w − (γ/2) w'Σw − Σ_i κ_i |w_i − w_prev,i|` subject to `1'w = 1` and `0 ≤ w ≤ 0.40`, with `γ = 5`.

- **Penalty:** `κ_i = c_i · m · 12`, where `c_i` is the one-way cost, `m` the multiplier and 12 the rebalances per year, putting a one-off cost on the same annual scale as `μ`.
- **Mean:** `μ` is the James–Stein estimate `μ0 + (1 − φ)(μ̂ − μ0)`. Here `μ̂` is the mean of the last 252 daily returns, `μ0` its cross-sectional average, and `φ = min(1, (N−3)/(T (μ̂−μ0)'Σ⁻¹(μ̂−μ0)))`.

## Costs, rebalancing and timing (FR-21 to FR-24, DQ-06)

- **Decision dates:** the last trading date of each month.
- **Threshold rule:** trade to target only if `max_i |w_drift,i − w_target,i| > 0.02`.
- **Timing:** weights decided at the close of `t` execute at the close of `t+1`. Day `t+1`'s return accrues to the drifted old weights, and the cost `Σ_i c_i |trade_i|` is charged that day. New weights earn from `t+2`.
- **Drift:** between trades, `w_{t+1} = w_t ⊙ (1 + r_{t+1}) / (1 + w_t'r_{t+1})`.
- **1/N:** goes through the same eligibility, schedule, threshold, lag and cost model. Its target is `1/N_t` over the assets eligible at `t`.

## Metrics (FR-26)

On daily net returns `r` and excess returns `e = r − rf`:

| Metric | Formula |
|---|---|
| Annual return | `(Π(1+r))^(252/T) − 1` |
| Volatility | `std(r)·√252` |
| Sharpe | `mean(e)/std(e)·√252` |
| Sortino | `mean(e)/sqrt(mean(min(e,0)²))·√252` |
| Max drawdown | Minimum of `wealth/peak − 1` |
| Calmar | Annual return / abs(max drawdown) |
| CEQ | `252·(mean(e) − (γ/2)·var(e))` with `γ = 1` |
| Turnover, cost drag | `Σ|trade|` and `Σ cost` per year |
| Effective number of assets | Mean of `1/Σw²` |

The trial log's `sharpe_annual` is the contract's raw-return Sharpe `mean(r)/std(r)·√252`.
