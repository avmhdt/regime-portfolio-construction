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

## Regime features (D-05)

- **Set A:** `x_t = (log(1 + r_SPY,t), ΔBAA10Y_t)`. The BAA10Y value at `t` is the latest observation published by `t − 1` business day (DQ-02). Before 1998-01-02, Sharadar has no SPY prices, so the Kenneth French daily market return `Mkt-RF + RF` stands in for SPY (owner ruling). It is a model input only, and each row records its source.
- **Set B:** adds the 3-month change in UNRATE and the 12-month log change in INDPRO. Both ends come from the ALFRED vintage published by `t − 1` business day (DQ-09). A missing release carries the last value forward.

## Walk-forward HMM (FR-11 to FR-13)

- **Fitting:** a Gaussian HMM with full covariances, fitted by Baum–Welch (hmmlearn `fit` only) on features standardized with their training-window mean and standard deviation. Ten seeded restarts per fit; the best log-likelihood among restarts whose states each hold at least 5% of the filtered mass is kept.
- **Schedule:** refits every 21 trading days on an expanding window of at least 2,520 days. Fit `k`, with data through `f_k`, serves the dates `f_k … f_{k+1} − 1`. For each of those dates, the filter is re-run from the start of fit `k`'s training window with fit `k`'s parameters, over data up to that date.
- **Forward filter:** in log space, `log α_t = log b(x_t) + logsumexp_j(log f_{t−1,j} + log A_jk)` and `f_t = α_t / Σα_t`. The prediction is `P(S_{t+1}|x_1:t) = f_t A`. The log-likelihood `Σ_t log Σ_k α_tk` equals hmmlearn's `score` to 1e-8.
- **Label switching:** states are put in ascending order of their equity-return emission variance. They are then matched to the previous fit's states by the Hungarian algorithm on the symmetric Kullback–Leibler divergence `KL(p‖q) + KL(q‖p)` between Gaussian emissions, in raw feature units. If the matching overrules the canonical order, the matching wins and the event is logged.

## BOCPD (FR-14)

Adams and MacKay (2007), with constant hazard `H = 1/λ` and a Normal–Gamma prior. The predictive distribution of run length `r` is Student-t with `ν = 2α_r`, location `μ_r` and squared scale `β_r(κ_r + 1)/(α_r κ_r)`. The recursion, in log space, is:

- **Growth:** `log P(r_t = r + 1) ∝ log P(r_{t−1} = r) + log π_r(x_t) + log(1 − H)`.
- **Change point:** `log P(r_t = 0) ∝ logsumexp_r(log P(r_{t−1} = r) + log π_r(x_t)) + log H`.
- **Posterior update:** `μ' = (κμ + x)/(κ + 1)`, `κ' = κ + 1`, `α' = α + ½` and `β' = β + κ(x − μ)²/(2(κ + 1))`.

Run lengths beyond 1,500 are dropped and the posterior renormalized. The outputs are `P(r_t < 21)`, the MAP run length and the mean run length.

## Regime-conditioned risk inputs (FR-16)

- **Regime covariance:** with filtered weights `w_tk = p_tk / Σ_t p_tk` over the covariance window, `μ_k = Σ_t w_tk r_t` and `Σ_k = Σ_t w_tk (r_t − μ_k)(r_t − μ_k)' / (1 − Σ_t w_tk²)`. Each `Σ_k` is shrunk 50% toward its diagonal. A state with fewer than 30 effective observations falls back to the unweighted moments.
- **Forward weights:** `π = (1/21) Σ_{j=1}^{21} f_t A^j`.
- **Mixture covariance:** `Σ = Σ_k π_k (Σ_k + μ_k μ_k') − μ̄ μ̄'` with `μ̄ = Σ_k π_k μ_k`. Mean–variance uses `μ̄` shrunk 50% toward its cross-sectional mean.
- **BOCPD mode:** when `P(r_t < 21) > 0.5`, the covariance is an EWMA with half-life 10 days instead of the base estimator.

## Leakage demonstration (AC-H-09, DQ-08)

The comparison figure needs smoothed probabilities `P(S_t | x_1:T)`. They are computed by a forward–backward pass only in `regport.report.figures.smoothed_for_illustration`, with the final refit's parameters. They never feed a strategy.
