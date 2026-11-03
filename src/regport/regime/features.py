"""Regime-model features (D-05), each computed from data available at its date (DQ-01).

Set A: ``spy_logret`` (SPY daily log total return) and ``baa10y_diff`` (daily change of BAA10Y as
known that day, i.e. with FRED's one-business-day lag). Set B adds ``unrate_3m_diff`` and
``indpro_12m_logdiff`` from ALFRED vintages as published by ``t − 1`` business day (DQ-09), or,
for the revision-bias study only (AC-H-07), from the latest revised values.

Before SPY's first Sharadar return (1998-01-02) the Kenneth French daily market return
``Mkt-RF + RF`` stands in for SPY (owner ruling 2026-11-03). It is a regime-model input only,
never an investable return, and every row says which source it used in ``spy_source``.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl

from regport.data.pit import VintageStates, knowledge_date

F64 = npt.NDArray[np.float64]
FEATURE_SETS: dict[str, list[str]] = {
    "A": ["spy_logret", "baa10y_diff"],
    "B": ["spy_logret", "baa10y_diff", "unrate_3m_diff", "indpro_12m_logdiff"],
}
PROXY_LABEL = "french_mkt_proxy"
_EPOCH = np.datetime64("1970-01-01", "D")


def feature_calendar(
    returns_daily: pl.DataFrame, french_daily: pl.DataFrame, start: date, symbol: str = "SPY"
) -> tuple[list[date], pl.DataFrame]:
    """Trading dates from ``start``: French dates before SPY's first return, then SPY's dates.

    Returns the calendar and the spliced ``spy_logret`` frame with its ``spy_source`` label.
    """
    spy = returns_daily.filter((pl.col("symbol") == symbol) & pl.col("ret").is_not_null())
    spy = spy.select(
        "date", pl.col("ret").log1p().alias("spy_logret"), pl.lit("sharadar").alias("spy_source")
    ).sort("date")
    first_spy = spy["date"].min()
    mkt = (
        french_daily.filter(pl.col("dataset") == "ff3_daily")
        .pivot(on="series", index="date", values="ret")
        .filter((pl.col("date") >= start) & (pl.col("date") < first_spy))
        .select(
            "date",
            (pl.col("Mkt-RF") + pl.col("RF")).log1p().alias("spy_logret"),
            pl.lit(PROXY_LABEL).alias("spy_source"),
        )
        .sort("date")
    )
    spliced = pl.concat([mkt, spy.filter(pl.col("date") >= start)])
    return spliced["date"].to_list(), spliced


def _days(dates: list[date] | pl.Series) -> npt.NDArray[np.int64]:
    return (np.array(list(dates), dtype="datetime64[D]") - _EPOCH).astype(np.int64)


def known_daily_values(fred_daily: pl.DataFrame, series_id: str, calendar: list[date]) -> F64:
    """Value of a daily FRED series as known on each calendar date (DQ-02 one-business-day lag).

    ``obs_date`` becomes usable on the first calendar date strictly after it; FRED's missing
    values are skipped, so the last published value carries over.
    """
    obs = fred_daily.filter((pl.col("series_id") == series_id) & pl.col("value").is_not_null())
    obs = obs.sort("obs_date")
    cal = _days(calendar)
    avail_idx = np.searchsorted(cal, _days(obs["obs_date"]), side="right")
    values = obs["value"].to_numpy().astype(np.float64)
    out = np.full(cal.size, np.nan)
    # Latest observation whose availability index is <= i.
    order = np.argsort(avail_idx, kind="stable")
    avail_sorted, vals_sorted = avail_idx[order], values[order]
    pos = np.searchsorted(avail_sorted, np.arange(cal.size), side="right") - 1
    ok = pos >= 0
    out[ok] = vals_sorted[pos[ok]]
    return out


def _vintage_value(
    states: VintageStates, obs: npt.NDArray[np.int64], k: npt.NDArray[np.int64]
) -> F64:
    """Value of observation ``obs[i]`` in the data published by knowledge date ``k[i]``."""
    out = np.full(obs.size, np.nan)
    for i, (o, kk) in enumerate(zip(obs, k, strict=True)):
        hit = (states.rows_obs == o) & (states.rows_start <= kk) & (states.rows_end >= kk)
        if hit.any():
            out[i] = states.rows_value[np.flatnonzero(hit)[0]]
    return out


def _months_back(day: int, months: int) -> int:
    m = np.datetime64(_EPOCH + np.timedelta64(day, "D"), "M") - np.timedelta64(months, "M")
    return int((m.astype("datetime64[D]") - _EPOCH).astype(np.int64))


def macro_change(
    vintages: pl.DataFrame,
    series_id: str,
    calendar: list[date],
    months: int,
    log: bool,
    vintage: Literal["pit", "latest"] = "pit",
    lag_bdays: int = 1,
) -> F64:
    """Change of a monthly series over ``months`` months, as known at each date.

    ``pit``: both values come from the same vintage, the one published by ``t − lag`` business
    days, and the latest observation in it (DQ-09). ``latest``: the same observation dates, with
    today's revised values (AC-H-07 contrast only; never a headline input).
    """
    states = VintageStates.from_rows(series_id, vintages.filter(pl.col("series_id") == series_id))
    cal = _days(calendar)
    k = (knowledge_date(np.array(calendar, dtype="datetime64[D]"), lag_bdays) - _EPOCH).astype(
        np.int64
    )
    latest_obs, _, _ = states.lookup(k, cal)
    out = np.full(cal.size, np.nan)
    known = latest_obs != np.iinfo(np.int64).min
    # The published set only changes at vintage breaks: evaluate once per distinct state.
    idx = np.searchsorted(states.breaks, k, side="right") - 1
    for state in np.unique(idx[known]):
        rows = np.flatnonzero(known & (idx == state))
        o = latest_obs[rows[0]]
        back = _months_back(int(o), months)
        if vintage == "pit":
            kk = np.full(2, k[rows[0]])
            now, then = _vintage_value(states, np.array([o, back]), kk)
        else:
            now, then = states.latest_values(np.array([o, back]))
        out[rows] = (np.log(now) - np.log(then)) if log else (now - then)
    # A missing release (e.g. October 2025, during the US government shutdown) leaves the
    # difference undefined; the last known value carries forward, which uses no later data.
    return pl.Series(out).fill_nan(None).forward_fill().fill_null(np.nan).to_numpy()


def build_features(
    returns_daily: pl.DataFrame,
    french_daily: pl.DataFrame,
    fred_daily: pl.DataFrame,
    vintages: pl.DataFrame | None,
    start: date,
    feature_set: Literal["A", "B"] = "A",
    vintage: Literal["pit", "latest"] = "pit",
) -> pl.DataFrame:
    """Feature frame from the first date on which every feature is finite."""
    calendar, spy = feature_calendar(returns_daily, french_daily, start)
    baa = known_daily_values(fred_daily, "BAA10Y", calendar)
    frame = spy.with_columns(pl.Series("baa10y_diff", np.diff(baa, prepend=np.nan)))
    if feature_set == "B":
        if vintages is None:
            raise ValueError("feature set B needs ALFRED vintages")
        frame = frame.with_columns(
            pl.Series(
                "unrate_3m_diff", macro_change(vintages, "UNRATE", calendar, 3, False, vintage)
            ),
            pl.Series(
                "indpro_12m_logdiff", macro_change(vintages, "INDPRO", calendar, 12, True, vintage)
            ),
        )
    cols = FEATURE_SETS[feature_set]
    finite = np.all(np.isfinite(frame.select(cols).to_numpy()), axis=1)
    first = int(np.argmax(finite))
    out = frame[first:]
    if not np.all(np.isfinite(out.select(cols).to_numpy())):
        bad = out.filter(~pl.all_horizontal(pl.col(cols).is_finite()))["date"].to_list()[:5]
        raise ValueError(f"non-finite features after the start, e.g. on {bad}")
    return out.select("date", *cols, "spy_source")
