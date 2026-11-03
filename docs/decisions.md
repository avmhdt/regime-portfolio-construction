# Decisions

Owner decisions copied from the owner-level `DECISIONS.md` on 2026-11-03, followed by choices made while building. `DECISIONS.md` overrides the spec (`SPEC.html`, v1.0), and the spec overrides `README.md`. API keys never go in this file.

## Owner decisions

### Plan and merging

- **Scope:** project 7 of 15, for researcher and developer job applications, built a few at a time with no fixed date. Started in the third batch on 2026-11-03.
- **Merging:** agents push feature branches and open one PR per milestone. Branch protection is turned on once CI exists (M1); PRs merge with `gh pr merge --auto --squash` as `avmhdt` once CI is green.

### Identity and access

- **GitHub:** `gh` active account is `avmhdt`, with admin on the repo. Actions is enabled (P-01).
- **WRDS:** no access, and none is needed here.

### Data

- **D-01 price source:** Sharadar SFP through Sharadar's direct API.
  - Base URL `https://api.sharadar.com/v1.0`, table `funds` (SFP), key in `SHARADAR_API_KEY`, sent in the `x-api-key` header so it stays out of URLs and logs.
  - Fallback is the owner's local Stooq snapshot at `E:\trading-strategy-brainstorm\data\stooq` (through 2026-10-30), labelled as such in the `source` column. Stooq is always run as the DQ-03 cross-check.
- **Sharadar licence:** Personal Use License. Sharadar replied in writing on 2026-10-31 that public code, aggregate results, job applications, aggregate cross-checks and hosted coding tools are allowed. Publishing stays aggregate-only, with no security-level data (D-11).
- **FRED (P-04):** the owner has the FRED key; it goes in `.env` as `FRED_API_KEY`.
- **AQR (P-08):** the owner downloaded the workbooks manually from aqr.com/Insights/Datasets into `D:\trading-strategy-brainstorm\backtest-overfitting-lab\data\raw\aqr\` (manifest dated 2026-11-02). The ones this project needs ("Century of Factor Premia Monthly.xlsx", "Time Series Momentum Factors Monthly.xlsx") are copied into this repo's git-ignored `data/raw/aqr/` with their source and SHA-256 recorded; they are never committed.
- **P-07 terms review and D-11 (ruled 2026-11-03): confirmed.** Derived aggregate tables and figures may be published on these terms:
  - **FRED/ALFRED:** the README carries FRED's notice: "This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis."
  - **BAA10Y:** carries Moody's copyright. It is used only as a model input and is never plotted or tabulated.
  - **VIXCLS:** Cboe's copyright. It is cited wherever it is used.
  - **DTB3, T10Y3M and the ALFRED series:** government data.
  - **Kenneth French:** the library is cited, and its files are never redistributed. Its page states no terms beyond a copyright notice.
  - **Stooq:** never committed. No published terms were found for the bulk files.
  - **Sharadar:** already confirmed in writing; aggregates only.
- **AQR, first ruling (2026-11-03, superseded the same day):** dropped from everything public, because AQR's site terms prohibit publishing "any information obtained from this Website" without its written consent. The run was allowed locally only, until AQR gave written permission.
- **AQR, current ruling (2026-11-03):** the owner reported, through the 15-projects review session, that AQR has given written permission: "AQR has greenlighted publishing our results. Proceed." The AQR robustness run (FR-06, Could, M6) may be published as aggregate results, with AQR cited. Workbooks and raw series are never redistributed or committed. AQR is a manual download (P-08), so `regport fetch all` doesn't fetch it.

### Money

D-10: USD 0. No paid data, LLM or cloud spend for this project. The Sharadar subscription is the owner's existing one.

### Machine

- **P-02 development OS:** native Windows 11 with `uv` and Python 3.12, the spec's primary environment. WSL2 Ubuntu 22.04.5 with `uv` and Python 3.11 to 3.13 is available but not used. CI covers ubuntu-latest and windows-latest.
- **P-10 reference hardware:** Intel Core i9-14900K, 24 cores and 32 threads, 128 GB RAM on Windows (62 GB visible to WSL). `D:` is a SATA HDD.
- **Shared PC:** live recorders run on this PC (a crypto capture whose 14-day count started 2026-11-03, a Deribit cron, Tardis downloads). Downloads are capped at 10 MB/s, heavy runs use at most 8 threads, and WSL is never restarted. Data stays on `D:` (or WSL ext4), never `C:` and never in git.

### Accepted spec defaults

D-02 to D-14 stand as written in the spec:

| ID | Decision |
|---|---|
| D-02 | Staggered entry after 252 valid returns; OOS from 2005-01-03; common start as a robustness run; no proxy splicing |
| D-03 | The 12 ETFs of §7, fixed |
| D-04 | Per-asset one-way costs of 1 to 5 bps in `configs/universe.yaml`, plus a uniform grid of 0, 5, 10, 25 bps |
| D-05 | K by BIC on 1993-02 to 2004-12, then fixed; feature set A headline, B as an ablation arm |
| D-06 | Monthly decisions on the last trading day; trade only if max abs drift exceeds 2 percentage points |
| D-07 | Long-only, fully invested; cap 0.40 for min-variance and mean-variance; ERC and HRP long-only only |
| D-08 | Ledoit-Wolf 2008 circular block bootstrap (5,000 resamples) headline; HAC alongside |
| D-09 | FRED DTB3, lagged one business day |
| D-10 | No paid spend |
| D-11 | Code, configs, experiment log CSV, aggregate tables and figures; never raw or processed data |
| D-12 | Covariance estimator chosen on 2005-01-03 to 2014-12-31 only |
| D-13 | OOS from 2005-01-03 to the last complete month at the final run |
| D-14 | MIT, code only |

## Choices made while building

| Milestone | Topic | Choice | Why |
|---|---|---|---|
| M1 | FRED key transport | The client sends `Authorization: Bearer` first and falls back to the `api_key` query parameter only if FRED refuses the header. The mode in use is printed by `regport smoke`; on 2026-11-03 FRED refused the header, so the query parameter is what works. URLs are redacted (`api_key=REDACTED`) before any log line or exception, and httpx/httpcore logging is held at WARNING | FRED's v1 `series/observations` endpoint (needed for ALFRED vintages) documents the key only as a query parameter, while the v2 docs use the header. The header route keeps the key out of URLs when FRED accepts it |
| M1 | Trial-log writer | Own writer in `regport.tracking.experiment_log`, following the contract's reference writer; columns are the ten required ones plus `x_returns_basis` (always `raw`) and `x_note` | Matches the spec's `TrialRow`/`append_trial` API; the contract allows `x_` columns after the ten |
| M1 | `overfitlab` as a dev dependency | Added from git (`avmhdt/backtest-overfitting-lab`, pinned in `uv.lock`), dev group only, never imported from `src/` | Lets a unit test and the CI step run `overfitlab`'s own validator on the log, rather than skipping with a notice. It is not on PyPI |
| M1 | Leakage-guard scope | `predict`, `predict_proba` and `score_samples` are banned as calls anywhere in `src/`; hmmlearn may be imported only in `regport/regime/hmm.py`, and `decode` is banned in any module that imports it | A blanket `.decode()` ban would also reject `bytes.decode`; confining hmmlearn to one module keeps the rule exact |
| M1 | `.gitignore` | `!src/regport/data/` re-includes the package, which the existing `data/` rule had hidden | The spec layout puts ingest code in `src/regport/data/`, and the `data/` rule matches any directory of that name |
| M1 | Line endings | `.gitattributes` sets `* text=auto eol=lf` | NFR-08 and the trial-log contract need LF; Windows checkouts would otherwise convert `results/trial_log.csv` to CRLF and break the append-only check |
| M1 | Trial log append-only check | CI compares `results/trial_log.csv` on the PR with the base branch and fails unless the base version is a byte prefix | AC-W-09: no row ever edited or deleted |
| M1 | Stooq coverage | The snapshot's SPY file starts on 2005-02-25, not 1993 | Found by `regport smoke`. The Stooq fallback therefore cannot supply the 1993-2004 pre-OOS window that D-05 uses to pick K; Sharadar SFP is needed for that, and M2 records each source's first date per symbol |
| M1 | Smoke test, 2026-11-03 | `regport smoke` passed: DTB3 19,000 rows 1954-01-04 to 2026-10-30; UNRATE 2,199 rows in 800 vintages; SPY from Sharadar SFP, 7,254 rows 1997-12-31 to 2026-11-02. Output, raw files, manifest and source tree scanned for both key values: none found | M1 exit criterion |
| M2 | Sharadar SFP coverage | SFP starts on 1997-12-31 for SPY; the other 11 ETFs start within 10 trading days of inception. The panel and master calendar therefore start on 1997-12-31 | Ruled 2026-11-03: see "Owner rulings on M2" |
| M2 | Stooq adjustment | Stooq misses about half of HYG's distributions, and disagrees with Sharadar by more than 5 bps on 2-4% of days for most ETFs | DQ-03's 99% agreement rule fails; ruled a documented exception on 2026-11-03 |
| M2 | `close` columns | `prices_daily.close` is Sharadar's `closeunadj` (the schema's "unadjusted"); an extra column `close_splitadj` keeps Sharadar's split-adjusted `close`, used as the price-only series in DQ-03 and for split detection | Unadjusted closes jump at splits (EFA and EEM 2005-06-09, EEM 2008-07-24), so they can't serve as a price-only return series |
| M2 | `pit_approx` | `macro_pit` carries `pit_approx` as an extra column, so `value_asof` and the table agree; `macro_vintages.pit_approx` marks rows of each series' earliest vintage, the only rows a DQ-10 approximation can use | The schema lists `pit_approx` only on `macro_vintages` |
| M2 | ALFRED business day | `asof - 1 business day` means one weekday back (`numpy.busday_offset`), not one trading day | A release on an exchange holiday (the jobs report on Good Friday) is public on the next weekday |
| M2 | Download cap | File downloads stream at no more than 8 MB/s, below the owner's 10 MB/s cap for this PC | Live recorders share the connection |
| M2 | AQR parser | Deferred to M6 | FR-06 is Could and listed under M6. Aggregate AQR results may be published there, with AQR cited (owner ruling after AQR's permission) |
| M2 | French range check | DQ-11 fails on returns below -100% (impossible); returns above +100% are counted, not failed | Two 1930s days in the 49-industry file (Paper, MedEq) exceed +100%, which is real for thin early portfolios |
| M2 | Raw file names | Every raw file name carries a UTC stamp and, if that path is taken, a `-1`, `-2` suffix (`fresh_path`). A re-converted Stooq file once reused its name and overwrote the raw file it replaced; the integration test caught it through DQ-12 | DQ-12: raw files are immutable, and re-fetches add files |
| M2 | Documented exceptions in code | `regport.data.checks.OWNER_EXCEPTIONS` names the two ruled exceptions. A check that fails within a ruled exception still reports FAIL, marked "documented exception", and doesn't block the exit code; any other failure blocks | Thresholds stay as the spec has them |

## Owner rulings on M2 (2026-11-03)

Answered in the main session and relayed by the 15-projects review session.

1. **AC-W-05 for SPY: accepted as a documented exception.**
   - Sharadar SFP starts SPY on 1997-12-31, five years after its 1993 inception. The other 11 ETFs start within 10 trading days of inception.
   - `regport data check` still reports AC-W-05 as a failure for SPY, with the exception noted. The exception covers SPY only: any other symbol failing still blocks.
2. **Pre-2005 regime window: "French market pre-1998" (option b).**
   - Before 1998-01-02, Ken French's daily market return (Mkt-RF + RF, `ff3_daily`) stands in for SPY's return. It is an input to the HMM and BOCPD only, never an investable return.
   - The spec's window and thresholds are unchanged: K by BIC on 1993-02 to 2004-12 (D-05), and `min_train_days` 2,520 (FR-11).
   - The proxy is labelled wherever it applies: the feature column, the regime outputs and the reports. The research memo will say that the regime model's early history uses a proxy.
   - It is built in M5.
3. **DQ-03 source agreement: "Accept, explain the rest", a documented exception.**
   - The 99% threshold is unchanged, and Sharadar stays the price source. Sharadar's own adjustment tests pass.
   - `regport data check` puts every disagreement day in one category, first match wins. The categories describe the days; none says which source is wrong. Totals over 2005-02-28 to 2026-10-30, for 1,318 disagreeing days out of 64,156 common days:
     - 678: distribution days where Stooq's return is much closer to the price-only return than to the total return, so it looks unadjusted.
     - 58: other distribution days.
     - 28: the first trading day after a market holiday.
     - 250: the gap reverses the next day, so the two closing levels differ for that one day.
     - 3: one source shows a zero return.
     - 301: none of the above (unexplained).
   - None of the non-distribution disagreements falls on a split day or spans a missing day.
   - They cluster in 2006–2008 (261) and 2011–2016 (255). By asset, they are concentrated in TLT, DBC and SPY.
   - **Cause not established.** For the 250 one-day differences and the 301 unexplained days, the evidence doesn't show which source is off or why. Settling it would need a third source.
   - **Distribution days excluded,** the sources agree on 99.07% of days overall. Per asset, everything is at or above 99% except TLT (97.52%), DBC (98.35%) and SPY (98.77%).
   - The full table is in `reports/data_quality.md`.

## M3 protocol, fixed before the first real covariance evaluation (2026-11-03)

- **D-12 selection rule:**
  - The chosen estimator is the one with the lowest mean QLIKE at h = 21 trading days, over non-overlapping origins whose realized window ends on or before 2014-12-31.
  - The mean is taken over 101 evaluation portfolios: 1/N and the 100 random Dirichlet(1) portfolios.
  - QLIKE is Patton's (2011) `log F + RV / F`, where F is the forecast h-day variance and RV is the sum of squared daily portfolio returns over the h days.
  - The bias statistic and the min-variance realized volatility are reported beside it but don't decide.
  - The chosen estimator replaces `risk.estimator` in `configs/base.yaml` if it differs from the default `ledoit_wolf`.
- **Asset set at each origin:**
  - An asset must be eligible on the origin date (D-02).
  - It must have at most 5% null returns in the 504-day window, following DQ-04 literally, with pre-inception days counting as null. So a newly eligible ETF enters covariance-based portfolios only once about 480 of its last 504 returns are valid.
  - Rows with any remaining null among the selected assets are dropped from that window.
- **Origins:** every h-th trading date from the first date at or after 2005-01-03. Covariances use returns up to and including the origin date. Each realized window is the next h trading days and must end by `oos_end`, which is the last trading date of the last complete month.
- **Portfolio returns:** fixed weights over the h days. The h-day portfolio return is the weighted sum of each asset's compounded h-day return. The forecast is `sqrt(w' Σ_daily w · h)`.
- **Min-variance in the evaluation:** FR-17's long-only problem with the 0.40 cap. The cap is raised to 1/N when fewer than three assets make 0.40 infeasible.
- **EWMA:** RiskMetrics zero-mean recursion with λ = 0.5^(1/63) over the 504-day window, weights normalized to sum to 1, which is the bias correction. Sample, Ledoit–Wolf and OAS demean.
- **Reproducibility (NFR-02):** repeated runs give min-variance weights that differ at about 1e-15 (cvxpy with CLARABEL), so cov-eval tables are equal after rounding to 1e-10, which is NFR-02's rule, rather than bit for bit. `reports/tables/*.csv` are written rounded to 10 decimals.

## M4 choices (2026-11-03)

| Topic | Choice | Why |
|---|---|---|
| Investable set in backtests | An ETF enters every strategy, 1/N included, at the first decision date with 252 valid returns (D-02). The covariance window then shrinks to the assets' common history. DQ-04's 5% rule counts only gaps after an asset's first return | D-02 says the same set applies to every strategy. M3's cov-eval keeps the rule fixed before it ran, which counts pre-inception days as missing |
| Constraints with the threshold rule | `check_constraints` runs on every optimizer target, and post-trade weights satisfy the constraints whenever a trade executes. Without a trade, D-06 keeps drifted weights within 2 points of target, which can sit just above the 0.40 cap (0.414 on the synthetic fixture). The audit reports such drift | D-06 is the owner's rule. Trading whenever drift breaks the cap would change it |
| Solver fallback | Previous drifted weights, projected onto the constraints (capped, excess spread pro rata), with a logged `solver_fallback` event. A run fails if fallbacks exceed 1% of rebalances (NFR-09) | Drifted weights alone can break the cap |
| Solver tolerances | CLARABEL `tol_gap_abs`, `tol_gap_rel` and `tol_feas` at 1e-10; OSQP `eps_abs` and `eps_rel` at 1e-10 with polishing | NFR-03 allows clipping only within 1e-9; default tolerances left weights at -5e-7 |
| Turnover penalty scale | `κ_i = c_i · turnover_penalty_mult · 12` for monthly decisions (52 weekly) | μ and Σ are annual; a one-off cost amortized over one rebalance interval is that much per year |
| Mean estimator | James–Stein toward the cross-sectional grand mean, `φ = min(1, (N−3)/(T (μ̂−μ0)'Σ⁻¹(μ̂−μ0)))`, from the last 252 daily returns | FR-20's default; no shrinkage when N ≤ 3 |
| Start of each run | The first decision is on the last trading date before 2005-01-03. The portfolio starts in cash, so the first return day (2005-01-03) carries only the set-up cost, the same for 1/N and every strategy | DQ-06 timing from a cold start |
| Turnover metric | `turnover = Σ|trade|` per execution day, as the spec defines it, so it counts both legs; annual turnover is its sum per year | §7b schema |
| No real-data backtests in M4 | The engine is tested on synthetic panels only. Any real run is a trial in the log, so none happens before `docs/preregistration.md` is committed (AC-H-10) | Keeps the trial count honest |
| riskfolio-lib | Not installed locally (a large dependency tree, and `uv` can't cap download speed under the 10 MB/s rule). The xcheck job runs those tests in CI | Owner's bandwidth rule |
| ERC cross-check (AC-W-08) | Against riskfolio-lib, ERC matches within 1e-6 on 11 of the 23 matrices and differs by at most 1.7e-6 on the other 12. In every one of those 12, riskfolio's own weights miss equal risk contributions by more than 100 times ours, which stop at max \|RC − 1/N\| ≤ 1e-10. The test checks that explanation case by case and caps any difference at 1e-5. This is the spec's "documented, explained difference" | `rp_optimization` solves a convex program at its solver's default tolerance, so its weights carry about 1e-6 of solver error. HRP matches within 1e-6 on all 23 |
| Golden tolerance | Golden values are stored rounded to 1e-10 and compared within 1e-9, or 1e-8 for solver-based optimizers | CI on Ubuntu found an ERC weight one unit apart in the tenth decimal, a rounding-boundary effect |

## M5 protocol, fixed before any regime-model selection or validation run (2026-11-03)

- **HMM states (D-05):**
  - K ∈ {2, 3, 4} is chosen by the lowest BIC, `−2 log L + p log T`.
  - The data: feature set A on 1993-02-02 to 2004-12-31, standardized on that window, with the French market proxy before 1998.
  - Each K is fitted once with the walk-forward fit settings (`n_init` 10, seeded, `max_iter` 500, `tol` 1e-6).
  - That K is then fixed for every run, feature set B included.
- **BOCPD hyperparameters (FR-14):**
  - The grid: hazard λ ∈ {100, 250, 500, 1000} days and α0 ∈ {2, 5, 10, 20}, with `μ0 = 0`, `κ0 = 1` and `β0 = (α0 − 1) · s²`. Here `s²` is the variance of the SPY (proxy) log return on 1993-02-02 to 2004-12-31, so the prior mean of the variance equals the pre-2005 sample variance.
  - The chosen point has the highest one-step predictive log-likelihood `Σ_t log p(x_t | x_1:t−1)` on that same window.
  - `max_run_length` 1500, `short_run_days` 21 and `trigger_prob` 0.5 stay at the spec defaults.
- **Validation item (3), BOCPD detection:**
  - It runs with the chosen λ and α0, and with `β0 = (α0 − 1) σ²` for the synthetic base standard deviation σ.
  - **Ambiguity:** "5 variance shifts (×3)" can mean the variance or the standard deviation multiplied by 3. Both are run and both are reported. A variance tripling carries about 0.45 nats of evidence per observation against a prior change-point log-odds of about −5.5 at λ = 250, so detecting it within 10 observations is close to impossible. The standard-deviation reading (variance ×9) is the one tested against the threshold.
  - **False alarms:** the onsets per 1,000 stationary observations, i.e. days where `cp_prob_short` crosses above 0.5, after a 100-day burn-in.
- **Status:** the reading of "×3" is the agent's choice, recorded here. It goes to the owner if the result depends on it.

## M5 results and choices (2026-11-03)

| Topic | Choice or result | Why |
|---|---|---|
| K (D-05) | 4. BIC 15,657 against 15,684 for K = 3 and 15,893 for K = 2, on 3,003 days from 1993-02-02 to 2004-12-31 | Rule fixed beforehand |
| BOCPD (FR-14) | Hazard 1/100, α0 = 2, β0 = 1.189e-4. Highest pre-2005 predictive log-likelihood (9,781.0, against 9,776.3 at 1/250) | Rule fixed beforehand |
| BOCPD validation (3b) | **Fails at the chosen hazard:** about 6.8 false alarms per 1,000 stationary days, against ≤ 1. A grid over hazard × α0 (8 replications each) shows the same code at 0.4–0.8 per 1,000 at hazard 1/500, with ≥ 4 of 5 shifts detected in 88–100% of replications. Detection (3a) passes at the chosen hazard. The test is an expected failure pending the owner's ruling | Not retuned to pass: the hazard was chosen by the preregistered rule |
| Fixed during M5 | An integer prior (`mu0 = 0`) made Numba compile integer arrays for the running means, which broke BOCPD. Every argument is now cast to float. A test checks the result against a textbook implementation to 1e-12 | — |
| Label-switch count (FR-13) | A switch is a change, between consecutive refits, of the alignment permutation applied on top of the canonical order. The first real run counted every refit whose aligned labels differed from canonical order (231 of 285), which double-counts after the first real switch | A persistent relabelling is one event, not one per refit |
| HMM fitting speed | hmmlearn `implementation="scaling"` (about 2× faster than `"log"`), refits in parallel over `REGPORT_N_JOBS` processes (8 for the real run, with BLAS threads at 1). The full walk-forward (285 refits × 10 restarts) took 533 s on the reference PC (NFR-01: ≤ 30 min). Seeds are per refit, so results don't depend on the job count | Owner's 8-thread limit |
| Forward filter | Own log-space recursion compiled with Numba. It matches hmmlearn's `score` to 7.5e-14 relative and its last smoothed row to 2.4e-13 | FR-12 |
| Missing macro releases | In feature set B a missing month (e.g. October 2025, during the US government shutdown) carries the last known value forward | Uses no later data |
| BAA10Y in reports | `reports/regimes.md` tabulates state emissions for the equity-return feature only | P-07 ruling: BAA10Y is a model input, never tabulated |
| Smoothed probabilities | Own forward–backward pass in `regport.report.figures.smoothed_for_illustration`, with the final refit's parameters. Equals hmmlearn's `predict_proba` to 1e-10 in a test | DQ-08 |
