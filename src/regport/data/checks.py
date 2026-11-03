"""``regport data check``: DQ-01 to DQ-13 and AC-W-05 on the built panel (FR-08).

Writes ``reports/data_quality.md``. Only aggregates are reported (D-11, Sharadar licence): dates,
counts, shares and annualized statistics, never price or return rows.
"""

from __future__ import annotations

import hashlib
import math
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from regport.config import ExperimentConfig, Universe
from regport.data import french as french_mod
from regport.data.ingest import resolve_price_source
from regport.data.manifest import Manifest
from regport.data.panel import PROCESSED_FILES, build_panel, next_trading_date
from regport.data.pit import value_asof
from regport.data.stooq import SNAPSHOT_DATE
from regport.errors import DataSourceError
from regport.regime.guard import scan_paths
from regport.settings import Settings

ADJ_WINDOW = (date(2010, 1, 1), date(2024, 12, 31))
ADJ_MIN_GAP = 0.02
AGREE_BPS = 5e-4
AGREE_MIN_SHARE = 0.99
DIV_DAY_GAP = 2e-4
INCEPTION_TOLERANCE_DAYS = 10
OUTLIER_ABS_RET = 0.25
SPLIT_JUMP = 0.10
PIT_SAMPLE = 50
PAYEMS_MIN_SHARE = 0.5


@dataclass
class CheckResult:
    id: str
    title: str
    passed: bool | None  # None = not applicable at the data layer
    summary: str
    table: list[dict[str, object]] = field(default_factory=list)

    def line(self) -> str:
        status = {True: "PASS", False: "FAIL", None: "N/A "}[self.passed]
        return f"[{status}] {self.id:<8} {self.title}: {self.summary}"


@dataclass
class CheckReport:
    checks: list[CheckResult]
    source: str
    calendar: tuple[date, date]

    @property
    def ok(self) -> bool:
        return all(c.passed is not False for c in self.checks)


def annualized(ret: np.ndarray) -> float:
    """Geometric annualized return of daily simple returns (252 days per year)."""
    r = ret[np.isfinite(ret)]
    return math.expm1(float(np.log1p(r).sum()) * 252 / r.size) if r.size else float("nan")


def _pct_change(frame: pl.DataFrame, col: str) -> pl.Expr:
    return pl.col(col) / pl.col(col).shift(1).over("symbol") - 1.0


class _Context:
    def __init__(self, cfg: ExperimentConfig, settings: Settings, universe: Universe) -> None:
        self.cfg, self.settings, self.universe = cfg, settings, universe
        self.processed = settings.data_dir / "processed"
        missing = [
            n for n in PROCESSED_FILES[:-1] if not (self.processed / f"{n}.parquet").exists()
        ]
        if missing:
            raise DataSourceError(f"missing processed files {missing}; run `regport data build`")
        self.prices = self._read("prices_daily")
        self.returns = self._read("returns_daily")
        self.rf = self._read("rf_daily")
        self.fred = self._read("fred_daily")
        self.vintages = self._read("macro_vintages")
        self.pit = self._read("macro_pit")
        self.calendar: list[date] = self.rf["date"].to_list()
        stooq = settings.data_dir / "interim" / "prices_stooq.parquet"
        self.stooq = pl.read_parquet(stooq) if stooq.exists() else None
        self.source = resolve_price_source(cfg, settings)

    def _read(self, name: str) -> pl.DataFrame:
        return pl.read_parquet(self.processed / f"{name}.parquet")


def check_universe(ctx: _Context) -> CheckResult:
    u = ctx.universe
    classes = sorted({a.asset_class for a in u.assets})
    return CheckResult(
        "FR-01",
        "Universe validation",
        True,
        f"{len(u.assets)} assets; classes {', '.join(classes)}",
    )


def check_coverage(ctx: _Context) -> CheckResult:
    """AC-W-05: every symbol ingested, first date <= inception + 10 trading days."""
    cal = np.array(ctx.calendar, dtype="datetime64[D]")
    rows: list[dict[str, object]] = []
    ok = True
    firsts = ctx.prices.group_by("symbol", maintain_order=True).agg(
        pl.col("date").min().alias("first"), pl.col("date").max().alias("last"), pl.len()
    )
    by_symbol = {r["symbol"]: r for r in firsts.iter_rows(named=True)}
    for asset in ctx.universe.assets:
        rec = by_symbol.get(asset.symbol)
        if rec is None:
            ok = False
            rows.append({"symbol": asset.symbol, "inception": asset.inception, "first": "missing"})
            continue
        first = rec["first"]
        lag = int(
            np.sum((cal > np.datetime64(asset.inception, "D")) & (cal <= np.datetime64(first, "D")))
        )
        if np.datetime64(asset.inception, "D") < cal[0]:
            lag_text = f"data start {(first - asset.inception).days} calendar days after inception"
            within = False
        else:
            lag_text = str(lag)
            within = lag <= INCEPTION_TOLERANCE_DAYS
        ok &= within
        rows.append(
            {
                "symbol": asset.symbol,
                "inception (approx.)": asset.inception,
                "first date": first,
                "last date": rec["last"],
                "rows": rec["len"],
                "trading days after inception": lag_text,
                "within 10": within,
            }
        )
    failed = [r["symbol"] for r in rows if not r.get("within 10")]
    summary = (
        f"{len(rows)} symbols; first date within {INCEPTION_TOLERANCE_DAYS} trading days of "
        f"inception for {len(rows) - len(failed)}" + (f"; not for {failed}" if failed else "")
    )
    return CheckResult("AC-W-05", "Coverage and inception", ok, summary, rows)


def check_availability(ctx: _Context) -> CheckResult:
    """DQ-01: macro and rf inputs carry availability dates after their observation dates."""
    fred_bad = ctx.fred.filter(pl.col("available_date") <= pl.col("obs_date")).height
    rates = ctx.fred.filter(
        (pl.col("series_id") == ctx.cfg.data.risk_free_series)
        & pl.col("value").is_not_null()
        & pl.col("available_date").is_not_null()
    )
    rng = np.random.default_rng(ctx.cfg.seed)
    picks = rng.choice(len(ctx.calendar), size=min(200, len(ctx.calendar)), replace=False)
    rf_bad = 0
    for i in sorted(picks.tolist()):
        t = ctx.calendar[i]
        known = rates.filter(pl.col("available_date") <= t)
        expected = (
            None
            if known.is_empty()
            else (1 + known.sort("available_date")["value"][-1] / 100) ** (1 / 252) - 1
        )
        got = ctx.rf["rf"][i]
        if expected is None:
            rf_bad += got is not None
        elif got is None or abs(got - expected) > 1e-15:
            rf_bad += 1
    k_bad = ctx.pit.filter(pl.col("obs_date") >= pl.col("asof_date")).height
    ok = fred_bad == 0 and rf_bad == 0 and k_bad == 0
    return CheckResult(
        "DQ-01",
        "Availability (no look-ahead)",
        ok,
        f"FRED rows available on/before their obs date: {fred_bad}; rf mismatches on 200 random "
        f"dates: {rf_bad}; macro_pit rows with obs_date >= asof_date: {k_bad}",
    )


def check_fred_lag(ctx: _Context) -> CheckResult:
    """DQ-02: available_date is the first trading date after obs_date."""
    expected = next_trading_date(ctx.fred["obs_date"].to_list(), ctx.calendar)
    got = ctx.fred["available_date"].to_list()
    bad = sum(1 for a, b in zip(expected, got, strict=True) if a != b)
    rows = [
        {"series": s, "rows": n, "first obs": f, "last obs": last, "null values": nulls}
        for s, n, f, last, nulls in ctx.fred.group_by("series_id", maintain_order=True)
        .agg(
            pl.len(),
            pl.col("obs_date").min().alias("first"),
            pl.col("obs_date").max().alias("last"),
            pl.col("value").null_count(),
        )
        .sort("series_id")
        .iter_rows()
    ]
    return CheckResult(
        "DQ-02",
        "FRED one-business-day lag",
        bad == 0,
        f"{ctx.fred.height} rows, {bad} with a wrong available_date",
        rows,
    )


def check_adjustment(ctx: _Context) -> CheckResult:
    """DQ-03: distributions are included; sources agree within 5 bps on >= 99% of days."""
    lo, hi = ADJ_WINDOW
    rows: list[dict[str, object]] = []
    ok = True
    window = ctx.prices.filter(pl.col("date").is_between(lo, hi)).sort("symbol", "date")
    window = window.with_columns(
        _pct_change(window, "adj_close").alias("ra"),
        _pct_change(window, "close_splitadj").alias("rp"),
    )
    for sym in ("HYG", "LQD"):
        d = window.filter(pl.col("symbol") == sym)
        adj = annualized(d["ra"].to_numpy().astype(float))
        price = annualized(d["rp"].to_numpy().astype(float))
        gap = adj - price
        passed = bool(gap >= ADJ_MIN_GAP)
        ok &= passed
        rows.append(
            {
                "test": f"{ctx.source} {sym} adj minus price-only, %/yr",
                "value": round(100 * gap, 2),
                "threshold": ">= 2.00",
                "pass": passed,
            }
        )
    agree_rows: list[dict[str, object]] = []
    if ctx.stooq is not None:
        s = ctx.stooq.filter(pl.col("date").is_between(lo, hi) & (pl.col("symbol") == "HYG"))
        s = s.sort("date").with_columns(_pct_change(s, "adj_close").alias("r"))
        hyg = annualized(s["r"].to_numpy().astype(float))
        passed = bool(hyg >= ADJ_MIN_GAP)
        ok &= passed
        rows.append(
            {
                "test": "stooq HYG adj_close annualized, %/yr",
                "value": round(100 * hyg, 2),
                "threshold": ">= 2.00",
                "pass": passed,
            }
        )
        both = (
            ctx.prices.select("date", "symbol", "adj_close", "close_splitadj")
            .join(
                ctx.stooq.select("date", "symbol", pl.col("adj_close").alias("s")),
                on=["date", "symbol"],
            )
            .sort("symbol", "date")
        )
        both = both.with_columns(
            _pct_change(both, "adj_close").alias("ra"),
            _pct_change(both, "s").alias("rs"),
            _pct_change(both, "close_splitadj").alias("rp"),
        ).drop_nulls(["ra", "rs", "rp"])
        both = both.with_columns(
            ((pl.col("ra") - pl.col("rs")).abs() <= AGREE_BPS).alias("agree"),
            ((pl.col("ra") - pl.col("rp")).abs() > DIV_DAY_GAP).alias("dist_day"),
            ((pl.col("rs") - pl.col("rp")).abs() <= 1e-4).alias("stooq_price_only"),
        )
        for sym in ctx.universe.symbols:
            d = both.filter(pl.col("symbol") == sym)
            if d.is_empty():
                continue
            share = float(d["agree"].mean())  # type: ignore[arg-type]
            bad = d.filter(~pl.col("agree"))
            missed = bad.filter(pl.col("dist_day") & pl.col("stooq_price_only")).height
            passed = share >= AGREE_MIN_SHARE
            ok &= passed
            agree_rows.append(
                {
                    "symbol": sym,
                    "common days": d.height,
                    "agree within 5 bps, %": round(100 * share, 2),
                    "disagreeing days": bad.height,
                    "of which distribution days Stooq left unadjusted": missed,
                    "pass": passed,
                }
            )
        pooled = float(both["agree"].mean())  # type: ignore[arg-type]
        agree_rows.append(
            {
                "symbol": "all",
                "common days": both.height,
                "agree within 5 bps, %": round(100 * pooled, 2),
                "disagreeing days": both.filter(~pl.col("agree")).height,
                "of which distribution days Stooq left unadjusted": both.filter(
                    ~pl.col("agree") & pl.col("dist_day") & pl.col("stooq_price_only")
                ).height,
                "pass": pooled >= AGREE_MIN_SHARE,
            }
        )
    failed = [str(r["symbol"]) for r in agree_rows if not r["pass"]]
    summary = "adjustment tests " + ("pass" if all(r["pass"] for r in rows) else "FAIL")
    if agree_rows:
        summary += (
            f"; {ctx.source} vs stooq agreement >= 99% for "
            f"{len(agree_rows) - len(failed)} of {len(agree_rows)} rows"
            + (f", not for {', '.join(failed)}" if failed else "")
        )
    return CheckResult("DQ-03", "Adjustment and source agreement", ok, summary, rows + agree_rows)


def check_missing(ctx: _Context) -> CheckResult:
    """DQ-04: returns on missing or forward-filled days are null, never 0."""
    joined = ctx.returns.join(
        ctx.prices.select("date", "symbol", "adj_close"), on=["date", "symbol"], how="left"
    )
    bad = joined.filter(pl.col("adj_close").is_null() & pl.col("ret").is_not_null()).height
    rows = []
    for sym in ctx.universe.symbols:
        d = joined.filter(pl.col("symbol") == sym)
        first = d.filter(pl.col("adj_close").is_not_null())["date"].min()
        after = d.filter(pl.col("date") > first) if first is not None else d.clear()
        rows.append(
            {
                "symbol": sym,
                "days after first price": after.height,
                "days without a price": after["adj_close"].null_count(),
                "null returns": after["ret"].null_count(),
            }
        )
    return CheckResult(
        "DQ-04", "Missing prices", bad == 0, f"{bad} non-null returns on days without a price", rows
    )


def check_outliers(ctx: _Context) -> CheckResult:
    """DQ-05: |ret| > 25%, and split days with an adjusted jump > 10% (listed, never edited)."""
    outliers = ctx.returns.filter(pl.col("ret").abs() > OUTLIER_ABS_RET).select("symbol", "date")
    prices = (
        ctx.prices.sort("symbol", "date")
        .with_columns((pl.col("close") / pl.col("close_splitadj")).alias("factor"))
        .with_columns(
            (pl.col("factor") / pl.col("factor").shift(1).over("symbol")).alias("factor_change"),
            _pct_change(ctx.prices, "adj_close").alias("adj_ret"),
        )
    )
    splits = prices.filter((pl.col("factor_change") - 1).abs() > 0.01)
    jumps = splits.filter(pl.col("adj_ret").abs() > SPLIT_JUMP)
    rows: list[dict[str, object]] = [
        {"type": "|return| > 25%", "symbol": s, "date": d} for s, d in outliers.iter_rows()
    ] + [
        {
            "type": "split day",
            "symbol": s,
            "date": d,
            "adjusted jump > 10%": bool((jumps["date"] == d).any()),
        }
        for s, d in splits.select("symbol", "date").iter_rows()
    ]
    return CheckResult(
        "DQ-05",
        "Outliers and splits",
        True,
        f"{outliers.height} returns above 25%; {splits.height} split days, "
        f"{jumps.height} with an adjusted jump above 10% (listed, not edited)",
        rows,
    )


def check_eligibility(ctx: _Context) -> CheckResult:
    """DQ-07: eligibility recomputed independently; D-02 consequences."""
    cfg = ctx.cfg.data
    mismatches = 0
    rows: list[dict[str, object]] = []
    has_price = ctx.returns.join(
        ctx.prices.select("date", "symbol", "adj_close"), on=["date", "symbol"], how="left"
    ).sort("symbol", "date")
    for sym in ctx.universe.symbols:
        d = has_price.filter(pl.col("symbol") == sym)
        valid = d["ret"].is_not_null().to_numpy()
        price = d["adj_close"].is_not_null().to_numpy()
        n_hist = np.concatenate([[0], np.cumsum(valid)[:-1]])
        recent = np.zeros(price.size, dtype=bool)
        for lag in range(cfg.max_ffill_days + 1):
            recent[lag:] |= price[: price.size - lag]
        expected = (n_hist >= cfg.min_history_days) & recent
        got = d["eligible"].to_numpy()
        mismatches += int(np.sum(expected != got))
        first = d.filter("eligible")["date"].min()
        rows.append(
            {
                "symbol": sym,
                "first eligible date": first,
                "eligible on OOS start": bool(
                    d.filter(pl.col("date") == ctx.cfg.sample.oos_start)["eligible"].any()
                ),
            }
        )
    counts = ctx.returns.group_by("date", maintain_order=True).agg(pl.col("eligible").sum())
    at_start = counts.filter(pl.col("date") >= ctx.cfg.sample.oos_start).sort("date")["eligible"][0]
    all_dates = counts.filter(pl.col("eligible") == len(ctx.universe.symbols)).sort("date")["date"]
    all_from = all_dates[0] if all_dates.len() else None
    summary = (
        f"{mismatches} mismatches; {at_start} assets eligible at OOS start "
        f"{ctx.cfg.sample.oos_start}; all {len(ctx.universe.symbols)} eligible from {all_from}"
    )
    return CheckResult(
        "DQ-07", "Eligibility (D-02 staggered entry)", mismatches == 0, summary, rows
    )


def check_regime_guard(ctx: _Context) -> CheckResult:
    findings = scan_paths([Path(__file__).resolve().parents[1]])
    return CheckResult(
        "DQ-08",
        "Filtered regime outputs only",
        not findings,
        f"AST guard: {len(findings)} banned hmmlearn calls in src/ "
        "(regime outputs themselves are checked from M5)",
    )


def check_pit(ctx: _Context) -> CheckResult:
    """DQ-09 and AC-W-06: vectorized macro_pit equals value_asof; vintages are really used."""
    rng = np.random.default_rng(ctx.cfg.seed + 1)
    path = ctx.processed / "macro_vintages.parquet"
    mismatches = 0
    for series_id in ctx.cfg.macro.vintage_series:
        rows = ctx.pit.filter(pl.col("series_id") == series_id)
        for i in sorted(
            rng.choice(rows.height, size=min(PIT_SAMPLE, rows.height), replace=False).tolist()
        ):
            r = rows.row(i, named=True)
            obs, val, approx = value_asof(
                series_id, r["asof_date"], path, ctx.cfg.macro.vintage_lag_bdays
            )
            mismatches += (obs, val, approx) != (r["obs_date"], r["value"], r["pit_approx"])
    pay = ctx.pit.filter(
        (pl.col("series_id") == "PAYEMS") & (pl.col("asof_date") >= date(2000, 1, 1))
    )
    monthly = (
        pay.with_columns(pl.col("asof_date").dt.truncate("1mo").alias("month"))
        .group_by("month", maintain_order=True)
        .agg(((pl.col("value") - pl.col("value_latest")).abs() > 0).any().alias("differs"))
    )
    share = float(monthly["differs"].mean()) if monthly.height else 0.0  # type: ignore[arg-type]
    ok = mismatches == 0 and share >= PAYEMS_MIN_SHARE
    table: list[dict[str, object]] = [
        {"series": s, "vintage rows": n, "first vintage": f, "vintages": v}
        for s, n, f, v in ctx.vintages.group_by("series_id", maintain_order=True)
        .agg(
            pl.len(),
            pl.col("realtime_start").min().alias("first"),
            pl.col("realtime_start").n_unique().alias("n"),
        )
        .sort("series_id")
        .iter_rows()
    ]
    return CheckResult(
        "DQ-09",
        "Point-in-time macro (AC-W-06)",
        ok,
        f"{mismatches} mismatches between macro_pit and value_asof on {PIT_SAMPLE} random dates "
        f"per series; PAYEMS PIT value differs from the latest value in {100 * share:.1f}% of "
        f"{monthly.height} months since 2000 (needs >= 50%)",
        table,
    )


def check_pit_approx(ctx: _Context) -> CheckResult:
    rows = []
    for series_id in ctx.cfg.macro.vintage_series:
        approx = ctx.pit.filter((pl.col("series_id") == series_id) & pl.col("pit_approx"))
        rows.append(
            {
                "series": series_id,
                "pit_approx days": approx.height,
                "range": (
                    f"{approx['asof_date'].min()!s} to {approx['asof_date'].max()!s}"
                    if approx.height
                    else "none"
                ),
            }
        )
    total = sum(int(str(r["pit_approx days"])) for r in rows)
    return CheckResult(
        "DQ-10",
        "Dates without a vintage",
        True,
        f"{total} series-days flagged pit_approx (excluded from macro features)",
        rows,
    )


def check_french(ctx: _Context) -> CheckResult:
    path = ctx.processed / "french_daily.parquet"
    if not path.exists():
        return CheckResult("DQ-11", "Kenneth French parsing", None, "not fetched")
    frame = pl.read_parquet(path)
    rows = []
    ok = True
    for name, ds in french_mod.DATASETS.items():
        d = frame.filter(pl.col("dataset") == name)
        n_series = d["series"].n_unique()
        bad_range = d.filter(pl.col("ret") < -1.0).height
        large = d.filter(pl.col("ret") > 1.0).height
        good = n_series == ds.n_series and bad_range == 0 and d.height > 0
        ok &= good
        rows.append(
            {
                "dataset": name,
                "series": n_series,
                "expected": ds.n_series,
                "first": d["date"].min(),
                "last": d["date"].max(),
                "missing (-99.99/-999)": d["ret"].null_count(),
                "returns below -100%": bad_range,
                "returns above +100%": large,
                "pass": good,
            }
        )
    return CheckResult(
        "DQ-11",
        "Kenneth French parsing",
        ok,
        f"{len(rows)} datasets, column counts and units checked",
        rows,
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_determinism(ctx: _Context) -> CheckResult:
    """DQ-12: raw files match the manifest; a second build is byte-identical."""
    manifest = Manifest(ctx.settings.data_dir)
    changed = [e.path for e in manifest.entries if not manifest.verify(e)]
    rows = []
    with tempfile.TemporaryDirectory(dir=ctx.settings.data_dir) as tmp:
        rebuilt = build_panel(ctx.cfg, ctx.settings, ctx.universe, out_dir=Path(tmp))
        for name, path in rebuilt.items():
            a, b = _digest(ctx.processed / f"{name}.parquet"), _digest(path)
            rows.append({"file": f"{name}.parquet", "sha256": a[:16], "identical": a == b})
    same = all(r["identical"] for r in rows)
    return CheckResult(
        "DQ-12",
        "Immutable raw files, deterministic build",
        same and not changed,
        f"{len(manifest.entries)} raw files, {len(changed)} changed since fetch; "
        f"rebuild identical for {sum(bool(r['identical']) for r in rows)} of "
        f"{len(rows)} files",
        rows,
    )


def check_disclosure(ctx: _Context) -> CheckResult:
    return CheckResult(
        "DQ-13",
        "Selection bias disclosure",
        True,
        "the 12 ETFs were chosen in 2026 knowing all survived; stated in the README "
        "limitations; no survivorship correction",
    )


CHECKS: list[Callable[[_Context], CheckResult]] = [
    check_universe,
    check_coverage,
    check_availability,
    check_fred_lag,
    check_adjustment,
    check_missing,
    check_outliers,
    check_eligibility,
    check_regime_guard,
    check_pit,
    check_pit_approx,
    check_french,
    check_determinism,
    check_disclosure,
]


def run_checks(cfg: ExperimentConfig, settings: Settings, universe: Universe) -> CheckReport:
    ctx = _Context(cfg, settings, universe)
    results = [check(ctx) for check in CHECKS]
    results.insert(
        7,
        CheckResult(
            "DQ-06",
            "Execution timing",
            None,
            "engine rule; verified by tests/integration/test_engine.py (M4)",
        ),
    )
    return CheckReport(results, ctx.source, (ctx.calendar[0], ctx.calendar[-1]))


def _md_table(rows: list[dict[str, object]]) -> list[str]:
    if not rows:
        return []
    cols: list[str] = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(_cell(r.get(c, "")) for c in cols) + " |")
    return out


def _cell(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "**no**"
    return str(value)


def write_report(report: CheckReport, path: Path) -> Path:
    lines = [
        "# Data quality",
        "",
        "Generated by `regport data check`; do not edit by hand. Aggregates only (D-11).",
        "",
        f"- Price source: `{report.source}` (D-01). Cross-check: Stooq snapshot of "
        f"{SNAPSHOT_DATE}.",
        f"- Master calendar: {report.calendar[0]} to {report.calendar[1]}.",
        f"- Overall: **{'PASS' if report.ok else 'FAIL'}**.",
        "",
        "| Check | Title | Result | Summary |",
        "|---|---|---|---|",
    ]
    for c in report.checks:
        status = {True: "pass", False: "**FAIL**", None: "n/a"}[c.passed]
        lines.append(f"| {c.id} | {c.title} | {status} | {c.summary} |")
    for c in report.checks:
        if c.table:
            lines += ["", f"## {c.id} {c.title}", "", *_md_table(c.table)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path
