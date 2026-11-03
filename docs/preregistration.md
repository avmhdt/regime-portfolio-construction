# Preregistration (AC-H-10)

Committed on 2026-11-03, before any real-data backtest and before the first `regport grid` row in `results/trial_log.csv`. Later changes are appended under "Deviations" with a date and a reason; nothing above that heading is rewritten.

## Question

Does a regime layer built from strictly filtered probabilities improve out-of-sample, after-cost, risk-adjusted returns? Compared against the same optimizer without it, and against 1/N?

## Headline configuration (`configs/base.yaml`)

| Item | Choice | Source |
|---|---|---|
| Optimizer | Equal risk contribution | Spec default |
| Regime usage | `hmm`: mixture covariance with forward state weights over 21 days, regime covariances shrunk 50% toward their diagonal (FR-16) | Spec default |
| HMM | Gaussian, K = 4, full covariance. Features set A: SPY daily log return and daily change of BAA10Y. Before 1998-01-02, the Kenneth French daily market return stands in for SPY, as a model input only. Refit every 21 trading days on an expanding window of at least 2,520 days, 10 seeded restarts | K by BIC on 1993-02-02 to 2004-12-31 (rule fixed in `docs/decisions.md`). The proxy is an owner ruling |
| Covariance estimator | EWMA, half-life 63 days, 504-day window | D-12, chosen on 2005–2014 windows (`reports/cov_eval.md`) |
| BOCPD | Hazard 1/100, Normal–Gamma prior μ0 = 0, κ0 = 1, α0 = 2, β0 = 1.189e-4. Trigger P(run length < 21) > 0.5 switches to an EWMA with half-life 10 days | Highest pre-2005 predictive log-likelihood (rule fixed in `docs/decisions.md`) |
| Universe and entry | The 12 ETFs; staggered entry after 252 valid returns (D-02) | Spec default |
| Rebalancing | Monthly decisions on the last trading day, trading only if a weight is more than 2 points from target; execution at the next close (D-06, DQ-06) | Spec default |
| Constraints | Long-only, fully invested. Cap 0.40 for min-variance and mean–variance (D-07) | Spec default |
| Costs | One-way per-asset table, 1–5 bps (`configs/universe.yaml`, D-04) | Spec default |
| Sample | OOS from 2005-01-03 to the last complete month at the final run (D-13) | Spec default |
| Benchmark | 1/N over eligible assets, through the same eligibility, schedule, threshold, lag and costs | FR-21 |

## Tests

- **Primary:** the Ledoit–Wolf (2008) Sharpe-difference test on daily excess returns, net of costs, of the headline against 1/N. It uses a studentized circular block bootstrap (5,000 resamples, block size calibrated over {1, 2, 4, 6, 8, 10}), two-sided, 5% level. The HAC version (Parzen kernel, VAR(1) prewhitening) is reported alongside (D-08).
- **Multiple testing:** Holm across every strategy compared with 1/N in the ablation grid.
- **Ablation:** each regime arm against the same optimizer with `none`, using the paired Ledoit–Wolf test, and reporting the differences in net Sharpe, CEQ, turnover, cost drag and max drawdown.

## Ablation grid (`configs/ablation.yaml`, M6)

- **Optimizers:** min_variance, erc, hrp, mv_turnover.
- **Regime modes:** none, hmm with feature set A, hmm with feature set B (ALFRED point in time), and bocpd.
- **Benchmark:** 1/N.
- **Size:** 4 × 4 + 1 = 17 configurations, each logged as a trial.

## Robustness (M6, in this order)

1. Uniform cost grid of 0, 5, 10 and 25 bps, with the break-even cost against 1/N.
2. Common-start panel (D-02 a).
3. Kenneth French 10 and 49 industries.
4. ALFRED point-in-time vs latest-revised macro features (feature set B, AC-H-07).
5. AQR factor premia (aggregate results only, AQR cited).

## Fixed and never tuned

- The stress windows: 2007-10-09 to 2009-03-09, 2020-02-19 to 2020-03-23, and 2022-01-03 to 2022-10-12.
- All regime hyperparameters above. They were chosen on pre-2005 data only.

## Known open items at registration

- **BOCPD validation (3b).** At hazard 1/100, BOCPD raises about 6 false alarms per 1,000 stationary synthetic days, against the spec's at most 1. Detection (3a) passes. The owner rules on this before M5 merges. If the hazard changes, that is appended below as a deviation.
- **Proxy disclosure.** The regime model's pre-1998 history uses the French market proxy. The research memo says so.

## Deviations

None yet.
