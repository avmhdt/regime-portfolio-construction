"""``regport fetch``: download or convert raw files, record them in the manifest (§7e, FR-02-FR-06).

Ingest never transforms values. Raw files are written once under ``raw/<source>/`` with a UTC stamp
in the name; parsed per-source frames go to ``interim/``. Without ``--force``, a key that already
has a raw file is skipped, so re-running a fetch makes no network requests.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from regport.config import ExperimentConfig, Universe
from regport.data import french as french_mod
from regport.data.fred import FRED_BASE_URL, FredClient
from regport.data.http import PoliteClient
from regport.data.manifest import (
    Manifest,
    ManifestEntry,
    fresh_path,
    sha256_file,
    utc_iso,
    utc_stamp,
)
from regport.data.sharadar import SFP_TABLE, SHARADAR_BASE_URL, SharadarClient, parse_sharadar_csv
from regport.data.stooq import SNAPSHOT_DATE, build_index, read_stooq_file
from regport.data.tiingo import TIINGO_BASE_URL, TiingoClient, tiingo_frame
from regport.errors import DataSourceError
from regport.logs import get_logger
from regport.settings import Settings

log = get_logger(__name__)

PRICE_COLUMNS = {
    "date": pl.Date,
    "symbol": pl.String,
    "source": pl.String,
    "close": pl.Float64,
    "close_splitadj": pl.Float64,
    "adj_close": pl.Float64,
    "volume": pl.Float64,
    "div_cash": pl.Float64,
    "ingested_at": pl.Datetime("us", "UTC"),
}


@dataclass(frozen=True)
class FetchResult:
    source: str
    key: str
    status: str  # "fetched" | "skipped"
    path: str

    def line(self) -> str:
        return f"[{self.status:<7}] {self.source:<8} {self.key:<22} {self.path}"


def _ingested_at(entry: ManifestEntry) -> datetime:
    return datetime.strptime(entry.fetched_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def resolve_price_source(cfg: ExperimentConfig, settings: Settings) -> str:
    """D-01 ``auto``: Sharadar if ``SHARADAR_API_KEY`` is set, else the local Stooq snapshot."""
    if cfg.data.price_source == "auto":
        return "sharadar" if settings.has_key("sharadar") else "stooq"
    return cfg.data.price_source


def _skip(manifest: Manifest, source: str, key: str, force: bool) -> FetchResult | None:
    entry = manifest.latest(source, key)
    if entry is not None and not force and manifest.verify(entry):
        return FetchResult(source, key, "skipped", entry.path)
    return None


def fetch_sharadar(
    settings: Settings,
    universe: Universe,
    symbols: Iterable[str],
    force: bool = False,
    client: SharadarClient | None = None,
) -> list[FetchResult]:
    manifest = Manifest(settings.data_dir)
    results: list[FetchResult] = []
    owned = client is None
    if client is None:
        if settings.sharadar_api_key is None:
            raise DataSourceError("SHARADAR_API_KEY is not set in .env")
        client = SharadarClient(settings.sharadar_api_key)
    try:
        for symbol in symbols:
            ticker = universe.get(symbol).sharadar
            key = f"{SFP_TABLE}/{symbol}"
            if (done := _skip(manifest, "sharadar", key, force)) is not None:
                results.append(done)
                continue
            text = client.table_csv(SFP_TABLE, ticker)
            if parse_sharadar_csv(text).is_empty():
                raise DataSourceError(f"Sharadar {SFP_TABLE}/{ticker}: no rows")
            dest = fresh_path(
                settings.data_dir / "raw" / "sharadar" / SFP_TABLE / f"{symbol}_{utc_stamp()}.csv"
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8", newline="")
            url = f"{SHARADAR_BASE_URL}/data/{SFP_TABLE}?ticker={ticker}"
            entry = manifest.add("sharadar", key, dest, url, {"ticker": ticker})
            results.append(FetchResult("sharadar", key, "fetched", entry.path))
    finally:
        if owned:
            client.close()
    return results


def fetch_stooq(
    settings: Settings, universe: Universe, symbols: Iterable[str], force: bool = False
) -> list[FetchResult]:
    """Convert the snapshot's files for ``symbols`` to Parquet once (P-06); no download."""
    if settings.stooq_dir is None:
        raise DataSourceError("STOOQ_DIR is not set in .env")
    manifest = Manifest(settings.data_dir)
    results: list[FetchResult] = []
    index: dict[str, Path] | None = None
    for symbol in symbols:
        if (done := _skip(manifest, "stooq", symbol, force)) is not None:
            results.append(done)
            continue
        if index is None:
            index = build_index(Path(settings.stooq_dir))
        stooq_symbol = universe.get(symbol).stooq
        if stooq_symbol not in index:
            raise DataSourceError(f"{stooq_symbol} not in the Stooq snapshot")
        source_file = index[stooq_symbol]
        frame = read_stooq_file(source_file)
        dest = fresh_path(
            settings.data_dir
            / "raw"
            / "stooq"
            / f"{symbol}_{SNAPSHOT_DATE:%Y%m%d}_{utc_stamp()}.parquet"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(dest)
        meta = {
            "snapshot_date": SNAPSHOT_DATE.isoformat(),
            "source_file": source_file.relative_to(Path(settings.stooq_dir)).as_posix(),
            "source_sha256": sha256_file(source_file),
        }
        entry = manifest.add("stooq", symbol, dest, f"file:{stooq_symbol}.txt", meta)
        results.append(FetchResult("stooq", symbol, "fetched", entry.path))
    return results


def fetch_tiingo(
    settings: Settings,
    universe: Universe,
    symbols: Iterable[str],
    force: bool = False,
    client: TiingoClient | None = None,
) -> list[FetchResult]:
    manifest = Manifest(settings.data_dir)
    results: list[FetchResult] = []
    owned = client is None
    if client is None:
        if settings.tiingo_api_key is None:
            raise DataSourceError("TIINGO_API_KEY is not set in .env")
        client = TiingoClient(settings.tiingo_api_key)
    try:
        for symbol in symbols:
            if (done := _skip(manifest, "tiingo", symbol, force)) is not None:
                results.append(done)
                continue
            ticker = universe.get(symbol).tiingo
            rows = client.prices_json(ticker)
            tiingo_frame(rows)
            dest = fresh_path(settings.data_dir / "raw" / "tiingo" / f"{symbol}_{utc_stamp()}.json")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(rows), encoding="utf-8", newline="")
            entry = manifest.add("tiingo", symbol, dest, f"{TIINGO_BASE_URL}/{ticker}/prices")
            results.append(FetchResult("tiingo", symbol, "fetched", entry.path))
    finally:
        if owned:
            client.close()
    return results


def _fred_client(settings: Settings) -> FredClient:
    if settings.fred_api_key is None or not settings.has_key("fred"):
        raise DataSourceError("FRED_API_KEY is not set in .env")
    return FredClient(settings.fred_api_key)


def fetch_fred(
    settings: Settings,
    series: Iterable[str],
    *,
    vintages: bool,
    force: bool = False,
    client: FredClient | None = None,
) -> list[FetchResult]:
    """Daily FRED series (current values) or all ALFRED vintages, one raw JSON per series."""
    source = "alfred" if vintages else "fred"
    manifest = Manifest(settings.data_dir)
    results: list[FetchResult] = []
    owned = client is None
    client = client or _fred_client(settings)
    try:
        for series_id in series:
            if (done := _skip(manifest, source, series_id, force)) is not None:
                results.append(done)
                continue
            fetched_at = utc_iso()
            rows = client.vintages(series_id) if vintages else client.observations(series_id)
            if not rows:
                raise DataSourceError(f"{source} {series_id}: no observations")
            stamp = fetched_at.replace("-", "").replace(":", "")
            dest = fresh_path(settings.data_dir / "raw" / source / f"{series_id}_{stamp}.json")
            dest.parent.mkdir(parents=True, exist_ok=True)
            payload = {"series_id": series_id, "observations": rows}
            dest.write_text(json.dumps(payload), encoding="utf-8", newline="")
            url = f"{FRED_BASE_URL}/series/observations?series_id={series_id}"
            if vintages:
                url += "&realtime_start=1776-07-04&realtime_end=9999-12-31"
            entry = manifest.add(source, series_id, dest, url, {"rows": len(rows)}, fetched_at)
            results.append(FetchResult(source, series_id, "fetched", entry.path))
    finally:
        if owned:
            client.close()
    return results


def fetch_french(
    settings: Settings,
    names: Iterable[str] | None = None,
    force: bool = False,
    client: PoliteClient | None = None,
) -> list[FetchResult]:
    manifest = Manifest(settings.data_dir)
    results: list[FetchResult] = []
    owned = client is None
    client = client or PoliteClient(min_interval_s=2.0, timeout_s=settings.regport_http_timeout_s)
    try:
        for name in names or french_mod.DATASETS:
            dataset = french_mod.DATASETS[name]
            if (done := _skip(manifest, "french", name, force)) is not None:
                results.append(done)
                continue
            dest = fresh_path(settings.data_dir / "raw" / "french" / f"{name}_{utc_stamp()}.zip")
            client.download(dataset.url, dest)
            french_mod.parse_first_table(french_mod.read_zip_text(dest), dataset)
            entry = manifest.add("french", name, dest, dataset.url, {"file": dataset.file})
            results.append(FetchResult("french", name, "fetched", entry.path))
    finally:
        if owned:
            client.close()
    return results


def _read_raw_prices(source: str, entry: ManifestEntry, path: Path, symbol: str) -> pl.DataFrame:
    ingested = pl.lit(_ingested_at(entry)).cast(pl.Datetime("us", "UTC")).alias("ingested_at")
    if source == "sharadar":
        frame = parse_sharadar_csv(path.read_text(encoding="utf-8"))
        return frame.select(
            pl.col("date"),
            pl.lit(symbol).alias("symbol"),
            pl.lit(source).alias("source"),
            pl.col("closeunadj").cast(pl.Float64).alias("close"),
            pl.col("close").cast(pl.Float64).alias("close_splitadj"),
            pl.col("closeadj").cast(pl.Float64).alias("adj_close"),
            pl.col("volume").cast(pl.Float64),
            pl.lit(None, pl.Float64).alias("div_cash"),
            ingested,
        )
    if source == "stooq":
        frame = pl.read_parquet(path)
        return frame.select(
            pl.col("date"),
            pl.lit(symbol).alias("symbol"),
            pl.lit(source).alias("source"),
            pl.lit(None, pl.Float64).alias("close"),
            pl.lit(None, pl.Float64).alias("close_splitadj"),
            pl.col("close").alias("adj_close"),
            pl.col("volume"),
            pl.lit(None, pl.Float64).alias("div_cash"),
            ingested,
        )
    frame = tiingo_frame(json.loads(path.read_text(encoding="utf-8")))
    return frame.select(
        pl.col("date"),
        pl.lit(symbol).alias("symbol"),
        pl.lit(source).alias("source"),
        pl.col("close"),
        pl.lit(None, pl.Float64).alias("close_splitadj"),
        pl.col("adj_close"),
        pl.col("volume"),
        pl.col("div_cash"),
        ingested,
    )


def write_interim_prices(settings: Settings, source: str, universe: Universe) -> Path | None:
    """``interim/prices_<source>.parquet`` from the latest raw file of every symbol present."""
    manifest = Manifest(settings.data_dir)
    frames = []
    for symbol in universe.symbols:
        key = f"{SFP_TABLE}/{symbol}" if source == "sharadar" else symbol
        entry = manifest.latest(source, key)
        if entry is None:
            continue
        frames.append(_read_raw_prices(source, entry, manifest.resolve(entry), symbol))
    if not frames:
        return None
    out = settings.data_dir / "interim" / f"prices_{source}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.concat(frames).cast(PRICE_COLUMNS).sort("symbol", "date").write_parquet(out)  # type: ignore[arg-type]
    return out


def fetch_prices(
    cfg: ExperimentConfig,
    settings: Settings,
    universe: Universe,
    source: str,
    symbols: list[str] | None = None,
    force: bool = False,
) -> list[FetchResult]:
    """``fetch prices``: the chosen source, plus the Stooq conversion as the DQ-03 cross-check."""
    chosen = resolve_price_source(cfg, settings) if source == "auto" else source
    symbols = symbols or universe.symbols
    results: list[FetchResult] = []
    if chosen == "sharadar":
        results += fetch_sharadar(settings, universe, symbols, force)
    elif chosen == "tiingo":
        results += fetch_tiingo(settings, universe, symbols, force)
    if settings.stooq_dir is not None or chosen == "stooq":
        results += fetch_stooq(settings, universe, symbols, force)
    for src in sorted({chosen, "stooq"} if settings.stooq_dir is not None else {chosen}):
        write_interim_prices(settings, src, universe)
    return results
