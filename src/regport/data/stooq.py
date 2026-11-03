"""The owner's local Stooq bulk daily snapshot (P-06, FR-02).

The snapshot stays where it is (``STOOQ_DIR``) and is never copied or committed. Files are located
by walking the tree once into a ticker -> path index, never by guessing paths, because some sit in
numbered subfolders (``nyse etfs/2/spy.us.txt``).
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import polars as pl

from regport.errors import DataSourceError

SNAPSHOT_DATE = date(2026, 10, 30)
STOOQ_HEADER = [
    "<TICKER>",
    "<PER>",
    "<DATE>",
    "<TIME>",
    "<OPEN>",
    "<HIGH>",
    "<LOW>",
    "<CLOSE>",
    "<VOL>",
    "<OPENINT>",
]


def build_index(root: Path, subdir: str = "d_us_txt") -> dict[str, Path]:
    """Map lower-case Stooq symbols (``spy.us``) to file paths under ``root/subdir``.

    Raises ``DataSourceError`` if the tree is missing or a symbol appears twice.
    """
    base = root / subdir
    if not base.is_dir():
        raise DataSourceError(f"Stooq snapshot not found at {base}")
    index: dict[str, Path] = {}
    for dirpath, _dirs, files in os.walk(base):
        for name in files:
            if not name.endswith(".txt"):
                continue
            symbol = name[:-4].lower()
            path = Path(dirpath) / name
            if symbol in index:
                raise DataSourceError(
                    f"Stooq symbol {symbol} appears twice: {index[symbol]}, {path}"
                )
            index[symbol] = path
    return index


def read_stooq_file(path: Path) -> pl.DataFrame:
    """Parse one Stooq daily file into ``date, open, high, low, close, volume``.

    Fails loudly on a header change or a non-daily period (R-01).
    """
    with path.open(encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
    if header != STOOQ_HEADER:
        raise DataSourceError(f"{path.name}: unexpected Stooq header {header}")
    raw = pl.read_csv(path, schema_overrides={"<DATE>": pl.String, "<TIME>": pl.String})
    if raw.is_empty():
        raise DataSourceError(f"{path.name}: no rows")
    periods = raw["<PER>"].unique().to_list()
    if periods != ["D"]:
        raise DataSourceError(f"{path.name}: expected daily bars, got periods {periods}")
    return raw.select(
        pl.col("<DATE>").str.to_date("%Y%m%d").alias("date"),
        pl.col("<OPEN>").cast(pl.Float64).alias("open"),
        pl.col("<HIGH>").cast(pl.Float64).alias("high"),
        pl.col("<LOW>").cast(pl.Float64).alias("low"),
        pl.col("<CLOSE>").cast(pl.Float64).alias("close"),
        pl.col("<VOL>").cast(pl.Float64).alias("volume"),
    ).sort("date")
