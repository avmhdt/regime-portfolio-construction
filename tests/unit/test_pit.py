"""AC-W-06 (fixture part), DQ-09, DQ-10: point-in-time macro values from ALFRED vintages."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from regport.data.panel import build_vintages, write_parquet
from regport.data.pit import build_macro_pit, knowledge_date, value_asof
from regport.errors import DataSourceError

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "alfred_vintages.json"


@pytest.fixture
def vintages_path(tmp_path: Path) -> Path:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    frame = build_vintages({k: v for k, v in raw.items() if not k.startswith("_")})
    path = tmp_path / "macro_vintages.parquet"
    write_parquet(frame, path)
    return path


def test_first_release_before_revision_and_revised_after(vintages_path: Path) -> None:
    # Released Friday 2020-02-07: not usable that day (k = Thursday), usable on Monday.
    assert value_asof("TESTSER", date(2020, 2, 10), vintages_path) == (
        date(2020, 1, 1),
        100.0,
        False,
    )
    # February first published 2020-03-06 at 105, revised 2020-03-20 to 106.
    assert value_asof("TESTSER", date(2020, 3, 9), vintages_path) == (
        date(2020, 2, 1),
        105.0,
        False,
    )
    assert value_asof("TESTSER", date(2020, 3, 20), vintages_path)[1] == 105.0
    assert value_asof("TESTSER", date(2020, 3, 23), vintages_path)[1] == 106.0
    assert value_asof("TESTSER", date(2020, 4, 6), vintages_path) == (
        date(2020, 3, 1),
        110.0,
        False,
    )


def test_release_day_itself_is_not_known(vintages_path: Path) -> None:
    obs, value, _ = value_asof("TESTSER", date(2020, 3, 6), vintages_path)
    assert (obs, value) == (date(2020, 1, 1), 100.0)  # January's revision lands on 03-09


def test_weekend_and_lag(vintages_path: Path) -> None:
    assert value_asof("TESTSER", date(2020, 3, 8), vintages_path)[1] == 105.0  # Sunday -> Fri
    assert value_asof("TESTSER", date(2020, 3, 23), vintages_path, lag_bdays=2)[1] == 105.0
    k = knowledge_date(np.array(["2020-03-09", "2020-03-08"], dtype="datetime64[D]"))
    assert k.tolist() == [date(2020, 3, 6), date(2020, 3, 6)]


def test_before_any_vintage(vintages_path: Path) -> None:
    with pytest.raises(DataSourceError, match="no value known"):
        value_asof("TESTSER", date(2020, 2, 7), vintages_path)


def test_pit_approx_uses_earliest_vintage(vintages_path: Path) -> None:
    # LATESER's first vintage is 2020-06-05; on 2020-04-15 only obs <= 2020-02-01 may be used.
    assert value_asof("LATESER", date(2020, 4, 15), vintages_path) == (date(2020, 2, 1), 2.5, True)
    # "." values are null and skipped: 2020-07-15 sees May, not the missing April.
    assert value_asof("LATESER", date(2020, 7, 15), vintages_path) == (date(2020, 5, 1), 5.5, False)


def test_missing_series_and_file(vintages_path: Path, tmp_path: Path) -> None:
    with pytest.raises(DataSourceError, match="no vintages"):
        value_asof("NOPE", date(2020, 4, 6), vintages_path)
    with pytest.raises(DataSourceError, match="not found"):
        value_asof("TESTSER", date(2020, 4, 6), tmp_path / "missing.parquet")


def test_vectorized_table_matches_value_asof(vintages_path: Path) -> None:
    calendar = [date(2020, 2, 3) + timedelta(days=i) for i in range(160)]
    calendar = [d for d in calendar if d.weekday() < 5]
    vintages = pl.read_parquet(vintages_path)
    pit = build_macro_pit(vintages, calendar)
    assert pit.columns == [
        "asof_date",
        "series_id",
        "obs_date",
        "value",
        "value_latest",
        "pit_approx",
    ]
    for row in pit.iter_rows(named=True):
        expected = value_asof(row["series_id"], row["asof_date"], vintages_path)
        assert (row["obs_date"], row["value"], row["pit_approx"]) == expected
    feb = pit.filter((pl.col("series_id") == "TESTSER") & (pl.col("obs_date") == date(2020, 2, 1)))
    assert set(feb["value_latest"].to_list()) == {106.0}
    early = pit.filter(
        (pl.col("series_id") == "TESTSER") & (pl.col("asof_date") <= date(2020, 2, 7))
    )
    assert early.is_empty()  # nothing known yet: no row rather than a guess
    assert pit.filter(pl.col("obs_date") >= pl.col("asof_date")).is_empty()


def test_build_macro_pit_empty() -> None:
    empty = pl.DataFrame(schema={"series_id": pl.String})
    assert build_macro_pit(empty, [date(2020, 1, 2)]).is_empty()


def test_earliest_vintage_flag(vintages_path: Path) -> None:
    frame = pl.read_parquet(vintages_path)
    flagged = frame.filter("pit_approx")
    assert set(flagged["realtime_start"].to_list()) == {date(2020, 2, 7), date(2020, 6, 5)}
