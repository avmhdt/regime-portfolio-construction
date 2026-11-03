# Regime detection and portfolio construction

> Detect market regimes in real time and compare allocation methods honestly against the 1/N benchmark.

**Status:** in progress. M1 (scaffolding, CI, smoke test), M2 (point-in-time data layer), M3 (risk models) and M4 (optimizers, costs, backtest engine) are done; the research memo replaces this README at M7.

| | |
|---|---|
| Track | Quant Researcher |
| Estimated effort | 4–6 weeks |
| Stack | Python (cvxpy, hmmlearn, scikit-learn) |
| Data cost | Free |
| Strongest for | Multi-asset, risk and allocation teams |

## Requirements

- [x] A multi-asset universe of 10–20 liquid ETFs or futures proxies across equities, rates, credit, commodities and FX. (FR-01, FR-07; AC-W-05 with SPY's 1998 start as a documented exception)
- [ ] Regime models using filtered (real-time) probabilities only: Gaussian HMM and Bayesian online change-point detection.
- [x] Covariance estimators (sample, EWMA, Ledoit–Wolf, OAS) evaluated on out-of-sample risk forecasts. (FR-09, FR-10; [reports/cov_eval.md](reports/cov_eval.md))
- [x] Optimizers: minimum variance, equal risk contribution, HRP (López de Prado 2016), and constrained mean–variance with a turnover penalty. (FR-17 to FR-20, AC-W-08)
- [x] A 1/N benchmark (DeMiguel, Garlappi and Uppal 2009) and cost-aware rebalancing. (FR-21 to FR-24, AC-W-10)

## Data

Raw and licensed data live in `data/`, which is git-ignored. Only download scripts and small synthetic fixtures are committed.

| Dataset | Cost | What it gives you |
|---|---|---|
| [Stooq or Tiingo daily prices](https://stooq.com/db/h/) | Free | ETF and index histories. |
| [FRED and ALFRED](https://alfred.stlouisfed.org) | Free | Macro series. ALFRED keeps first-published vintages, which gives you point-in-time macro data. |
| [AQR datasets](https://www.aqr.com/Insights/Datasets) | Free | Long histories of momentum, value and carry across asset classes. |
| [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | Free | Industry and factor portfolios for robustness checks. |

### Credits and notices

- Prices: Sharadar Fund Prices (SFP), used under the owner's licence; only aggregates are published. Stooq daily data serve as a cross-check only.
- This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis. Series: DTB3, T10Y3M (Board of Governors); BAA10Y (Moody's, used only as a model input); VIXCLS (Chicago Board Options Exchange, CBOE Volatility Index: VIX, retrieved from FRED); ALFRED vintages of UNRATE, PAYEMS, CPIAUCSL (BLS) and INDPRO (Board of Governors).
- Kenneth R. French Data Library, Tuck School of Business at Dartmouth: Fama/French 3 factors and 10/49 industry portfolios (daily). Files are not redistributed.

## Success criteria

### Works (correct and reproducible)

- [ ] A unit test shows regime probabilities at time t never use data after t (no smoothed-state leakage).
- [x] ERC portfolios have equal risk contributions within 1e-6, and constraints always hold. (AC-W-02: property tests and a check at every rebalance; drift within D-06's 2-point band is reported, see docs/decisions.md)
- [x] The ex-ante vs ex-post volatility bias statistic is close to 1 for your chosen estimator. (AC-W-03, EWMA chosen by D-12; [reports/cov_eval.md](reports/cov_eval.md))

### Impresses hirers

- [ ] An out-of-sample comparison against 1/N with a proper Sharpe-difference test (Ledoit and Wolf 2008), reported honestly even if 1/N wins.
- [ ] An ablation showing what the regime layer adds over the optimizer alone, net of turnover.
- [ ] Stress tests through 2008, 2020 and 2022 with drawdown and recovery statistics.

## Development

```bash
uv sync
cp .env.example .env   # then add FRED_API_KEY and SHARADAR_API_KEY
uv run regport smoke
uv run pytest
```

Decisions and their reasons are in [docs/decisions.md](docs/decisions.md). Every backtest appends one row to `results/trial_log.csv` in [trial-log contract v1.0](https://github.com/avmhdt/backtest-overfitting-lab/blob/main/docs/trial-log.md).

## Questions this project must answer

- Why does 1/N so often beat mean–variance?
- How do you handle HMM label switching?
- How would this scale to 500 assets?
