"""``regport data build``: the point-in-time panel (FR-07, §7b).

Outputs in ``processed/``: ``prices_daily``, ``returns_daily`` (with the D-02 eligibility mask),
``rf_daily``, ``fred_daily``, ``macro_vintages``, ``macro_pit`` and ``french_daily``. Every output
is a deterministic function of the raw files named in the manifest and the config (DQ-12): rows
are sorted, there are no float group-by aggregations, and Parquet is written with fixed settings.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from regport.config import ExperimentConfig, Universe
from regport.data import french as french_mod
from regport.data.fred import observations_frame
from regport.data.ingest import resolve_price_source, write_interim_prices
from regport.data.manifest import Manifest
from regport.data.pit import build_macro_pit
from regport.errors import DataSourceError
from regport.settings import Settings

PROCESSED_FILES = (
    "prices_daily",
    "returns_daily",
    "rf_daily",
    "fred_daily",
    "macro_vintages",
    "macro_pit",
    "french_daily",
)
TRADING_DAYS = 252


def write_parquet(frame: pl.DataFrame, path: Path) -> None:
    """Fixed writer settings, so equal frames give byte-identical files (DQ-12)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.rechunk().write_parquet(
        path, compression="zstd", compression_level=3, statistics=True, row_group_size=1 << 20
    )


def master_calendar(prices: pl.DataFrame, symbol: str, start: date) -> list[date]:
    """Trading dates of ``symbol`` with a price, from ``start`` on (§7b ``prices_daily``)."""
    dates = (
        prices.filter(
            (pl.col("symbol") == symbol)
            & pl.col("adj_close").is_not_null()
            & (pl.col("date") >= start)
        )["date"]
        .unique()
        .sort()
        .to_list()
    )
    if len(dates) < 2:
        raise DataSourceError(f"calendar symbol {symbol} has no prices after {start}")
    return dates


def build_returns(
    prices: pl.DataFrame,
    calendar: list[date],
    symbols: list[str],
    min_history: int,
    max_ffill: int,
) -> pl.DataFrame:
    """``returns_daily``: simple and log total returns, valid-history count and eligibility.

    ``ret[t] = adj[t] / adj[t-1] - 1`` on consecutive calendar dates, null if either price is
    missing (a forward-filled day has a null return, DQ-04). ``eligible[t]`` iff at least
    ``min_history`` valid returns strictly before ``t`` and a price on ``t``, where a price may be
    carried forward at most ``max_ffill`` trading days (DQ-07, D-02).
    """
    grid = pl.DataFrame({"date": calendar}).join(pl.DataFrame({"symbol": symbols}), how="cross")
    panel = grid.join(
        prices.select("date", "symbol", "adj_close"), on=["date", "symbol"], how="left"
    ).sort("symbol", "date")
    prev = pl.col("adj_close").shift(1).over("symbol")
    out = panel.with_columns(
        (pl.col("adj_close") / prev - 1.0).alias("ret"),
    ).with_columns(
        pl.col("ret").log1p().alias("logret"),
        (
            pl.col("ret").is_not_null().cast(pl.Int32).cum_sum().over("symbol")
            - pl.col("ret").is_not_null().cast(pl.Int32)
        ).alias("n_valid_history"),
        pl.col("adj_close")
        .forward_fill(limit=max_ffill)
        .over("symbol")
        .is_not_null()
        .alias("has_price"),
    )
    return out.select(
        "date",
        "symbol",
        "ret",
        "logret",
        pl.col("n_valid_history").cast(pl.Int32),
        ((pl.col("n_valid_history") >= min_history) & pl.col("has_price")).alias("eligible"),
    ).sort("symbol", "date")


def next_trading_date(obs: list[date], calendar: list[date]) -> list[date | None]:
    """First calendar date strictly after each ``obs`` (DQ-02 one-business-day lag)."""
    cal = np.array(calendar, dtype="datetime64[D]")
    idx = np.searchsorted(cal, np.array(obs, dtype="datetime64[D]"), side="right")
    return [calendar[i] if i < len(calendar) else None for i in idx]


def build_fred_daily(raw: dict[str, list[dict[str, str]]], calendar: list[date]) -> pl.DataFrame:
    frames = []
    for series_id in sorted(raw):
        frame = observations_frame(raw[series_id], series_id).select(
            "series_id", "obs_date", "value"
        )
        frames.append(
            frame.with_columns(
                pl.Series(
                    "available_date",
                    next_trading_date(frame["obs_date"].to_list(), calendar),
                    dtype=pl.Date,
                )
            )
        )
    return pl.concat(frames).sort("series_id", "obs_date")


def build_rf(fred_daily: pl.DataFrame, calendar: list[date], series_id: str) -> pl.DataFrame:
    """``rf_daily`` from DTB3 lagged one business day: ``(1 + y/100)^(1/252) - 1`` (D-09).

    The latest non-null value with ``available_date <= t`` is used (holidays carry the last
    published rate).
    """
    rates = (
        fred_daily.filter(
            (pl.col("series_id") == series_id)
            & pl.col("value").is_not_null()
            & pl.col("available_date").is_not_null()
        )
        .select("available_date", "value")
        .unique("available_date", keep="last", maintain_order=True)
        .sort("available_date")
    )
    if rates.is_empty():
        raise DataSourceError(f"no {series_id} observations for the risk-free rate")
    return (
        pl.DataFrame({"date": calendar})
        .join_asof(rates, left_on="date", right_on="available_date", strategy="backward")
        .select(
            "date",
            ((1.0 + pl.col("value") / 100.0) ** (1.0 / TRADING_DAYS) - 1.0).alias("rf"),
        )
    )


def build_vintages(raw: dict[str, list[dict[str, str]]]) -> pl.DataFrame:
    frames = []
    for series_id in sorted(raw):
        frame = observations_frame(raw[series_id], series_id)
        first = frame["realtime_start"].min()
        frames.append(frame.with_columns((pl.col("realtime_start") == first).alias("pit_approx")))
    return pl.concat(frames).sort("series_id", "obs_date", "realtime_start")


def _load_raw_series(
    manifest: Manifest, source: str, series: list[str]
) -> dict[str, list[dict[str, str]]]:
    out = {}
    for series_id in series:
        entry = manifest.latest(source, series_id)
        if entry is None:
            raise DataSourceError(
                f"no raw {source} file for {series_id}; run `regport fetch {source}`"
            )
        payload = json.loads(manifest.resolve(entry).read_text(encoding="utf-8"))
        out[series_id] = payload["observations"]
    return out


def build_french(manifest: Manifest) -> pl.DataFrame | None:
    frames = []
    for name, dataset in french_mod.DATASETS.items():
        entry = manifest.latest("french", name)
        if entry is None:
            continue
        text = french_mod.read_zip_text(manifest.resolve(entry))
        frames.append(french_mod.parse_first_table(text, dataset))
    if not frames:
        return None
    return pl.concat(frames).sort("dataset", "series", "date")


def build_panel(
    cfg: ExperimentConfig, settings: Settings, universe: Universe, out_dir: Path | None = None
) -> dict[str, Path]:
    """Build every processed file; returns name -> path for the files written."""
    manifest = Manifest(settings.data_dir)
    root = out_dir or settings.data_dir / "processed"
    source = resolve_price_source(cfg, settings)
    interim = write_interim_prices(settings, source, universe)
    if interim is None:
        raise DataSourceError(f"no raw {source} prices; run `regport fetch prices`")
    prices_all = pl.read_parquet(interim)
    missing = sorted(set(universe.symbols) - set(prices_all["symbol"].unique().to_list()))
    if missing:
        raise DataSourceError(f"no {source} prices for {missing}")
    calendar = master_calendar(prices_all, cfg.data.calendar_symbol, cfg.sample.data_start)
    prices = prices_all.join(pl.DataFrame({"date": calendar}), on="date", how="semi").sort(
        "symbol", "date"
    )
    written: dict[str, Path] = {}

    def save(name: str, frame: pl.DataFrame) -> None:
        path = root / f"{name}.parquet"
        write_parquet(frame, path)
        written[name] = path

    save("prices_daily", prices)
    save(
        "returns_daily",
        build_returns(
            prices, calendar, universe.symbols, cfg.data.min_history_days, cfg.data.max_ffill_days
        ),
    )
    fred_daily = build_fred_daily(
        _load_raw_series(manifest, "fred", cfg.macro.daily_series), calendar
    )
    save("fred_daily", fred_daily)
    save("rf_daily", build_rf(fred_daily, calendar, cfg.data.risk_free_series))
    vintages = build_vintages(_load_raw_series(manifest, "alfred", cfg.macro.vintage_series))
    save("macro_vintages", vintages)
    save("macro_pit", build_macro_pit(vintages, calendar, cfg.macro.vintage_lag_bdays))
    french = build_french(manifest)
    if french is not None:
        save("french_daily", french)
    return written
