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
- **P-07 terms review:** pending, before M2.

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
| M1 | FRED key transport | The client sends `Authorization: Bearer` first and falls back to the `api_key` query parameter only if FRED refuses the header. The mode in use is printed by `regport smoke`. URLs are redacted (`api_key=REDACTED`) before any log line or exception, and httpx/httpcore logging is held at WARNING | FRED's v1 `series/observations` endpoint (needed for ALFRED vintages) documents the key only as a query parameter, while the v2 docs use the header. The header route keeps the key out of URLs when FRED accepts it |
| M1 | Trial-log writer | Own writer in `regport.tracking.experiment_log`, following the contract's reference writer; columns are the ten required ones plus `x_returns_basis` (always `raw`) and `x_note` | Matches the spec's `TrialRow`/`append_trial` API; the contract allows `x_` columns after the ten |
| M1 | `overfitlab` as a dev dependency | Added from git (`avmhdt/backtest-overfitting-lab`, pinned in `uv.lock`), dev group only, never imported from `src/` | Lets a unit test and the CI step run `overfitlab`'s own validator on the log, rather than skipping with a notice. It is not on PyPI |
| M1 | Leakage-guard scope | `predict`, `predict_proba` and `score_samples` are banned as calls anywhere in `src/`; hmmlearn may be imported only in `regport/regime/hmm.py`, and `decode` is banned in any module that imports it | A blanket `.decode()` ban would also reject `bytes.decode`; confining hmmlearn to one module keeps the rule exact |
| M1 | Line endings | `.gitattributes` sets `* text=auto eol=lf` | NFR-08 and the trial-log contract need LF; Windows checkouts would otherwise convert `results/trial_log.csv` to CRLF and break the append-only check |
| M1 | Trial log append-only check | CI compares `results/trial_log.csv` on the PR with the base branch and fails unless the base version is a byte prefix | AC-W-09: no row ever edited or deleted |
| M1 | Stooq coverage | The snapshot's SPY file starts on 2005-02-25, not 1993 | Found by `regport smoke`. The Stooq fallback therefore cannot supply the 1993-2004 pre-OOS window that D-05 uses to pick K; Sharadar SFP is needed for that, and M2 records each source's first date per symbol |
