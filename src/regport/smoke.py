"""``regport smoke``: M1 data-access test (§7e).

Reads one FRED series, one ALFRED series and SPY from the configured price source, prints row
counts and date ranges, and keeps nothing except Parquet copies in ``results/cache/smoke/``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from regport.config import ExperimentConfig, load_universe
from regport.data.fred import FredClient, observations_frame
from regport.data.sharadar import SharadarClient
from regport.data.stooq import SNAPSHOT_DATE, build_index, read_stooq_file
from regport.errors import DataSourceError
from regport.settings import Settings

SMOKE_FRED_SERIES = "DTB3"
SMOKE_ALFRED_SERIES = "UNRATE"


@dataclass(frozen=True)
class SmokeStep:
    name: str
    ok: bool
    source: str
    rows: int = 0
    first: date | None = None
    last: date | None = None
    detail: str = ""

    def line(self) -> str:
        status = "OK  " if self.ok else "FAIL"
        span = f"{self.first} .. {self.last}" if self.first else "-"
        text = f"[{status}] {self.name:<8} source={self.source:<9} rows={self.rows:<7} {span}"
        return f"{text}  {self.detail}".rstrip()


def resolve_price_source(cfg: ExperimentConfig, settings: Settings) -> str:
    """D-01 ``auto``: Sharadar if ``SHARADAR_API_KEY`` is set, else the local Stooq snapshot."""
    source = cfg.data.price_source
    if source == "auto":
        return "sharadar" if settings.has_key("sharadar") else "stooq"
    return source


def _span(frame: pl.DataFrame, column: str) -> tuple[date | None, date | None]:
    if frame.is_empty():
        return None, None
    first, last = frame[column].min(), frame[column].max()
    assert isinstance(first, date) and isinstance(last, date)
    return first, last


def _guard(name: str, source: str, fn: Callable[[], SmokeStep]) -> SmokeStep:
    try:
        return fn()
    except (DataSourceError, OSError, ValueError) as exc:
        return SmokeStep(name, False, source, detail=str(exc))


def run_smoke(cfg: ExperimentConfig, settings: Settings) -> list[SmokeStep]:
    """Run the three access checks; each failure is reported, not raised."""
    cache = settings.results_dir / "cache" / "smoke"
    cache.mkdir(parents=True, exist_ok=True)
    steps: list[SmokeStep] = []

    fred: FredClient | None = None
    if settings.fred_api_key is None or not settings.has_key("fred"):
        msg = "FRED_API_KEY is not set in .env"
        steps += [
            SmokeStep("fred", False, "fred", detail=msg),
            SmokeStep("alfred", False, "alfred", detail=msg),
        ]
    else:
        fred = FredClient(settings.fred_api_key)

        def fred_step() -> SmokeStep:
            assert fred is not None
            frame = observations_frame(fred.observations(SMOKE_FRED_SERIES), SMOKE_FRED_SERIES)
            frame.write_parquet(cache / f"fred_{SMOKE_FRED_SERIES}.parquet")
            first, last = _span(frame, "obs_date")
            return SmokeStep(
                "fred",
                True,
                "fred",
                frame.height,
                first,
                last,
                f"{SMOKE_FRED_SERIES}, auth={fred.auth_mode}",
            )

        def alfred_step() -> SmokeStep:
            assert fred is not None
            frame = observations_frame(fred.vintages(SMOKE_ALFRED_SERIES), SMOKE_ALFRED_SERIES)
            frame.write_parquet(cache / f"alfred_{SMOKE_ALFRED_SERIES}.parquet")
            first, last = _span(frame, "obs_date")
            n_vintages = frame["realtime_start"].n_unique()
            return SmokeStep(
                "alfred",
                True,
                "alfred",
                frame.height,
                first,
                last,
                f"{SMOKE_ALFRED_SERIES}, {n_vintages} vintages",
            )

        steps.append(_guard("fred", "fred", fred_step))
        steps.append(_guard("alfred", "alfred", alfred_step))
        fred.close()

    source = resolve_price_source(cfg, settings)
    symbol = cfg.data.calendar_symbol
    asset = load_universe(cfg.data.universe_file).get(symbol)

    def price_step() -> SmokeStep:
        if source == "sharadar":
            if settings.sharadar_api_key is None:
                raise DataSourceError("SHARADAR_API_KEY is not set in .env")
            client = SharadarClient(settings.sharadar_api_key)
            try:
                frame = client.sfp_prices(asset.sharadar)
            finally:
                client.close()
            cols = ",".join(c for c in ("close", "closeadj", "closeunadj") if c in frame.columns)
            detail = f"{symbol}, columns {cols}"
        elif source == "stooq":
            if settings.stooq_dir is None:
                raise DataSourceError("STOOQ_DIR is not set in .env")
            index = build_index(Path(settings.stooq_dir))
            if asset.stooq not in index:
                raise DataSourceError(f"{asset.stooq} not in the Stooq snapshot")
            frame = read_stooq_file(index[asset.stooq])
            detail = f"{symbol}, snapshot {SNAPSHOT_DATE}, {len(index)} files indexed"
        else:
            raise DataSourceError(f"price source {source} is not available in M1")
        frame.write_parquet(cache / f"prices_{source}_{symbol}.parquet")
        first, last = _span(frame, "date")
        return SmokeStep("prices", True, source, frame.height, first, last, detail)

    steps.append(_guard("prices", source, price_step))
    return steps
