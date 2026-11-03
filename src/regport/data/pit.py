"""Point-in-time macro values from ALFRED vintages (FR-04, DQ-09, DQ-10).

ALFRED rows (``output_type=1``) say that ``value`` for ``obs_date`` was the published figure from
``realtime_start`` to ``realtime_end`` inclusive. The figure usable at decision date ``asof`` is
the one published by the knowledge date ``k = asof - 1 business day``: among rows with
``realtime_start <= k <= realtime_end``, the one with the latest ``obs_date``.

If ``k`` precedes the series' first vintage, the earliest vintage is used, restricted to
``obs_date`` at least two months before ``asof``'s month, and the result is flagged
``pit_approx`` (DQ-10). Business days are weekdays (``numpy.busday_offset``): a release on an
exchange holiday, such as the jobs report on Good Friday, is known on the next weekday.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from regport.errors import DataSourceError

I64 = npt.NDArray[np.int64]
F64 = npt.NDArray[np.float64]
DEFAULT_VINTAGES_PATH = Path("data/processed/macro_vintages.parquet")


def knowledge_date(asof: npt.ArrayLike, lag_bdays: int = 1) -> npt.NDArray[np.datetime64]:
    """``asof - lag_bdays`` weekdays; one business day before a Saturday or Sunday is Friday."""
    days = np.asarray(asof, dtype="datetime64[D]")
    return np.busday_offset(days, -lag_bdays, roll="forward")


@dataclass(frozen=True)
class VintageStates:
    """The published data set of one series, as a step function of the knowledge date.

    ``breaks[j]`` is a date on which the published set changes; from then until ``breaks[j+1]``
    the latest observation is ``latest_obs[j]`` with value ``latest_value[j]``.
    """

    series_id: str
    breaks: I64
    latest_obs: I64
    latest_value: F64
    first_vintage: int
    rows_obs: I64
    rows_start: I64
    rows_end: I64
    rows_value: F64

    @classmethod
    def from_rows(cls, series_id: str, frame: pl.DataFrame) -> VintageStates:
        """Build from ``macro_vintages`` rows of one series."""
        if frame.is_empty():
            raise DataSourceError(f"no vintages for {series_id}")
        valid = frame.filter(pl.col("value").is_not_null()).sort("realtime_start", "obs_date")
        obs = _days(valid["obs_date"])
        start = _days(valid["realtime_start"])
        end = _days(valid["realtime_end"])
        value = valid["value"].to_numpy().astype(np.float64)
        breaks = np.unique(np.concatenate([start, end + 1]))
        latest_obs = np.full(breaks.size, np.iinfo(np.int64).min, dtype=np.int64)
        latest_value = np.full(breaks.size, np.nan)
        for j, b in enumerate(breaks):
            live = (start <= b) & (end >= b)
            if live.any():
                k = int(np.flatnonzero(live)[np.argmax(obs[live])])
                latest_obs[j], latest_value[j] = obs[k], value[k]
        return cls(
            series_id, breaks, latest_obs, latest_value, int(start.min()), obs, start, end, value
        )

    def lookup(self, k: I64, asof: I64) -> tuple[I64, F64, npt.NDArray[np.bool_]]:
        """Vectorized ``(obs_date, value, pit_approx)`` for knowledge dates ``k`` (epoch days)."""
        idx = np.searchsorted(self.breaks, k, side="right") - 1
        obs = np.where(idx >= 0, self.latest_obs[np.maximum(idx, 0)], np.iinfo(np.int64).min)
        val = np.where(idx >= 0, self.latest_value[np.maximum(idx, 0)], np.nan)
        approx = k < self.first_vintage
        if approx.any():
            first = self.rows_start == self.first_vintage
            for i in np.flatnonzero(approx):
                cutoff = _months_back(int(asof[i]), 2)
                ok = first & (self.rows_obs <= cutoff)
                if ok.any():
                    j = int(np.flatnonzero(ok)[np.argmax(self.rows_obs[ok])])
                    obs[i], val[i] = self.rows_obs[j], self.rows_value[j]
        return obs, val, approx

    def latest_values(self, obs: I64) -> F64:
        """Current (fully revised) value of each ``obs`` (``realtime_end`` = 9999-12-31)."""
        current = self.rows_end == _CURRENT_END
        lookup = dict(
            zip(self.rows_obs[current].tolist(), self.rows_value[current].tolist(), strict=True)
        )
        return np.array([lookup.get(int(o), np.nan) for o in obs], dtype=np.float64)


_EPOCH = np.datetime64("1970-01-01", "D")
_CURRENT_END = int((np.datetime64("9999-12-31", "D") - _EPOCH).astype(np.int64))


def _days(series: pl.Series) -> I64:
    return series.cast(pl.Int32).to_numpy().astype(np.int64)


def _months_back(day: int, months: int) -> int:
    """First day of the month ``months`` before the month containing epoch day ``day``."""
    month = np.datetime64(_EPOCH + np.timedelta64(day, "D"), "M") - np.timedelta64(months, "M")
    return int((month.astype("datetime64[D]") - _EPOCH).astype(np.int64))


def _to_date(day: int) -> date:
    return (_EPOCH + np.timedelta64(day, "D")).astype(date)  # type: ignore[no-any-return]


def load_vintages(path: Path = DEFAULT_VINTAGES_PATH) -> pl.DataFrame:
    if not path.exists():
        raise DataSourceError(f"{path} not found; run `regport data build`")
    return pl.read_parquet(path)


def value_asof(
    series_id: str, asof: date, vintages_path: Path | None = None, lag_bdays: int = 1
) -> tuple[date, float, bool]:
    """``(obs_date, value, pit_approx)`` of ``series_id`` as known at decision date ``asof``.

    DQ-09: from the latest vintage with ``realtime_start <= asof - 1 business day``. DQ-10: if
    no vintage existed then, the earliest vintage is used and ``pit_approx`` is True.
    """
    frame = load_vintages(vintages_path or DEFAULT_VINTAGES_PATH).filter(
        pl.col("series_id") == series_id
    )
    states = VintageStates.from_rows(series_id, frame)
    asof_day = np.array([(np.datetime64(asof, "D") - _EPOCH).astype(np.int64)])
    k = (knowledge_date(np.datetime64(asof, "D"), lag_bdays) - _EPOCH).astype(np.int64)
    obs, val, approx = states.lookup(np.atleast_1d(k), asof_day)
    if obs[0] == np.iinfo(np.int64).min:
        raise DataSourceError(f"{series_id}: no value known at {asof}")
    return _to_date(int(obs[0])), float(val[0]), bool(approx[0])


def build_macro_pit(
    vintages: pl.DataFrame, calendar: list[date], lag_bdays: int = 1
) -> pl.DataFrame:
    """``macro_pit``: one row per trading date and series (§7b), plus ``pit_approx``."""
    asof = np.array(calendar, dtype="datetime64[D]")
    asof_days = (asof - _EPOCH).astype(np.int64)
    k = (knowledge_date(asof, lag_bdays) - _EPOCH).astype(np.int64)
    frames = []
    for series_id in sorted(vintages["series_id"].unique().to_list()):
        states = VintageStates.from_rows(
            series_id, vintages.filter(pl.col("series_id") == series_id)
        )
        obs, val, approx = states.lookup(k, asof_days)
        known = obs != np.iinfo(np.int64).min
        frames.append(
            pl.DataFrame(
                {
                    "asof_date": asof_days[known].astype(np.int32),
                    "series_id": series_id,
                    "obs_date": obs[known].astype(np.int32),
                    "value": val[known],
                    "value_latest": states.latest_values(obs[known]),
                    "pit_approx": approx[known],
                }
            ).with_columns(pl.col("asof_date", "obs_date").cast(pl.Date))
        )
    schema = {
        "asof_date": pl.Date,
        "series_id": pl.String,
        "obs_date": pl.Date,
        "value": pl.Float64,
        "value_latest": pl.Float64,
        "pit_approx": pl.Boolean,
    }
    if not frames:
        return pl.DataFrame(schema=schema)
    return pl.concat(frames).select(list(schema)).sort("series_id", "asof_date")
