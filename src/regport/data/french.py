"""Kenneth French Data Library daily files (FR-05, DQ-11).

The zips hold one CSV with several tables (value-weighted returns first, then equal-weighted and
others). Only the first table is parsed: for the industry files that is "Average Value Weighted
Returns -- Daily"; for the factor file it is the only table. Percent becomes decimal, and the
missing markers -99.99 and -999 become null. Column counts are asserted.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from regport.errors import DataSourceError

FRENCH_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"


@dataclass(frozen=True)
class FrenchDataset:
    name: str
    file: str
    n_series: int

    @property
    def url(self) -> str:
        return f"{FRENCH_BASE_URL}/{self.file}"


DATASETS: dict[str, FrenchDataset] = {
    "ff3_daily": FrenchDataset("ff3_daily", "F-F_Research_Data_Factors_daily_CSV.zip", 4),
    "ind10_daily": FrenchDataset("ind10_daily", "10_Industry_Portfolios_daily_CSV.zip", 10),
    "ind49_daily": FrenchDataset("ind49_daily", "49_Industry_Portfolios_daily_CSV.zip", 49),
}

_DATA_ROW = re.compile(r"^\s*(\d{8})\s*,")
MISSING = (-99.99, -999.0)


def read_zip_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise DataSourceError(f"{path.name}: expected one CSV, found {names}")
        return archive.read(names[0]).decode("latin-1")


def parse_first_table(text: str, dataset: FrenchDataset) -> pl.DataFrame:
    """Long frame ``dataset, date, series, ret`` from the first daily table in ``text``."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if _DATA_ROW.match(line)), None)
    if start is None or start == 0:
        raise DataSourceError(f"{dataset.name}: no daily data rows found")
    header = [h.strip() for h in lines[start - 1].split(",")]
    series = header[1:]
    if len(series) != dataset.n_series:
        raise DataSourceError(
            f"{dataset.name}: expected {dataset.n_series} series, header has {len(series)}"
        )
    rows: list[list[str]] = []
    for line in lines[start:]:
        if not _DATA_ROW.match(line):
            break
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != dataset.n_series + 1:
            raise DataSourceError(f"{dataset.name}: bad column count in row {cells[0]}")
        rows.append(cells)
    wide = pl.DataFrame(rows, schema=["date", *series], orient="row")
    return (
        wide.unpivot(index="date", variable_name="series", value_name="pct")
        .select(
            pl.lit(dataset.name).alias("dataset"),
            pl.col("date").str.to_date("%Y%m%d"),
            pl.col("series"),
            pl.col("pct").cast(pl.Float64).alias("pct"),
        )
        .with_columns(
            pl.when(pl.col("pct").is_in(list(MISSING)))
            .then(None)
            .otherwise(pl.col("pct") / 100.0)
            .alias("ret")
        )
        .drop("pct")
        .sort("series", "date")
    )
