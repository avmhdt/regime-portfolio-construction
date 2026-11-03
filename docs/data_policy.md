# Data policy

How the raw data are sourced, timed and checked. The results of the checks are in [reports/data_quality.md](../reports/data_quality.md), which `regport data check` regenerates.

## Sources (D-01, P-07)

| Source | Used for | Access | Published here |
|---|---|---|---|
| Sharadar SFP (`funds` table) | Daily prices of the 12 ETFs: `closeadj` (split- and distribution-adjusted), `close` (split-adjusted), `closeunadj` | `https://api.sharadar.com/v1.0/data/funds?ticker=…`, key in the `x-api-key` header | Aggregates only |
| Stooq bulk snapshot | Cross-check of the same 12 ETFs (DQ-03), fallback when no Sharadar key is set | The owner's local snapshot (through 2026-10-30), read in place, converted once to Parquet | Aggregates only; never committed |
| FRED | DTB3, BAA10Y, T10Y3M, VIXCLS (current values) | `series/observations` | Aggregates only; BAA10Y and VIXCLS are never plotted or tabulated |
| ALFRED | Every vintage of UNRATE, PAYEMS, INDPRO, CPIAUCSL | `series/observations` with `realtime_start=1776-07-04`, `realtime_end=9999-12-31` | Aggregates only |
| Kenneth French Data Library | FF3 daily factors with RF; 10 and 49 industry portfolios, value-weighted, daily | Zip downloads, capped at 8 MB/s | Aggregates only; files never redistributed |
| AQR | Century of Factor Premia, Time Series Momentum (monthly) | The owner's manual downloads (P-08), copied into `data/raw/aqr/` | Aggregates only, with AQR cited: AQR gave written permission (owner ruling, 2026-11-03). Workbooks and raw series are never redistributed |

Tiingo is implemented as an optional third source (`--source tiingo`), but it isn't used: the owner has no key, and none is needed.

Raw files live under `data/raw/<source>/` with a UTC stamp in the name. `data/raw/manifest.json` records each file's URL (keys redacted), fetch time and SHA-256. Re-downloads add files rather than overwrite them, and the build reads the latest file per key. Nothing under `data/` is committed.

## Coverage

- **Sharadar SFP starts on 1997-12-31 for SPY.** The other 11 ETFs start within 10 trading days of their inception (AC-W-05). The master calendar is SPY's Sharadar trading dates, so the panel starts on 1997-12-31, not at the spec's `data_start` of 1993-02-01.
- **Stooq starts on 2005-02-25 for SPY** and on or near inception for the later ETFs. It can't cover any pre-2005 window.

## Pre-inception policy (D-02, staggered entry)

An ETF becomes eligible on date *t* if it has at least 252 valid daily returns strictly before *t* and a price on *t*. The price may be carried forward at most 3 trading days (DQ-04, DQ-07). Ineligible assets weigh 0 in every strategy, including 1/N, which is 1/N_t over eligible assets. No proxy history is spliced in.

On the built panel:
- **At the 2005-01-03 OOS start,** 7 assets are eligible: SPY, EFA, EEM, IEF, TLT, LQD and TIP.
- **All 12 are eligible** from 2008-04-11. The D-02(a) robustness run starts there.
- **In the 2008 stress window,** 10 assets are eligible at its start.

## Returns

`ret[t] = adj_close[t] / adj_close[t-1] - 1` on consecutive calendar dates. It is null if either price is missing, so a forward-filled day has a null return, never 0.

## Availability lags (DQ-01, DQ-02, DQ-09)

- **Prices:** usable at the close of their trading date.
- **FRED daily series:** `available_date` is the first trading date after `obs_date`, a one-business-day lag. This includes DTB3.
- **Risk-free rate (D-09):** `rf[t] = (1 + y/100)^(1/252) - 1`, where *y* is the latest non-null DTB3 value with `available_date <= t`. DTB3 is a discount-basis rate, so this is an approximation.
- **ALFRED (DQ-09):** the value used at decision date *t* is the latest observation in the data as published by *t* minus 1 business day (weekdays, `numpy.busday_offset`). Concretely, the row with `realtime_start <= k <= realtime_end` and the latest `obs_date`. A release on an exchange holiday, such as the jobs report on Good Friday, counts as known on the next weekday.
- **Dates before a series' first vintage (DQ-10):** the earliest vintage is used, restricted to observations at least two months before the decision month, and the result is flagged `pit_approx`. Those dates are excluded from macro-feature evaluation. On the current panel no date is affected: every series has vintages before 1997-12-31.
- **`value_latest`** in `macro_pit` is the current revised value. It exists only for the revision-bias study (AC-H-07) and is never a model input.

## Adjustment check (DQ-03)

- **Sharadar includes distributions.** Over 2010-01-01 to 2024-12-31, the annualized return from `closeadj` exceeds the price-only return from the split-adjusted `close` by 5.9%/yr for HYG and 3.7%/yr for LQD. The threshold is 2.0%/yr.
- **Stooq is only partly adjusted.** Its HYG series returns 3.0%/yr over the same window: above the 2.0%/yr floor, but about 2 points a year below Sharadar's.
- **The two sources disagree by more than 5 bps on 2–4% of days for most ETFs**, against the spec's 99% agreement rule. Of the 1,318 disagreement days, 622 are distribution days on which Stooq's "adjusted" return equals the price-only return, meaning Stooq missed the distribution. This accounts for most of the bond ETFs' disagreements (HYG 111 of 148, LQD 134 of 193, IEF 125 of 172). The rest are close-price differences of a few bps, concentrated in 2011–2016. The per-symbol table is in the data-quality report. This failure is pending an owner ruling: see `docs/decisions.md`.

## Determinism (DQ-12)

- Processed files are written with fixed Parquet settings, from sorted frames, with no float group-by aggregations.
- `regport data check` rebuilds the panel into a temporary directory and compares the SHA-256 of every file with the main build.
- It also checks that every raw file still matches its manifest hash.

## Selection bias (DQ-13)

The universe was chosen in 2026, knowing that all 12 ETFs survived. No survivorship correction is attempted, and the README states this among its limitations.
