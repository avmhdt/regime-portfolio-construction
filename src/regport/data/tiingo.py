"""Tiingo end-of-day API (optional third source, P-05). Only used when selected explicitly.

The token travels in the ``Authorization: Token`` header, never in the URL. At most one request
per second (FR-02). Untested against the live API: the owner has no Tiingo key (normally unneeded).
"""

from __future__ import annotations

from typing import Any

import polars as pl
from pydantic import SecretStr

from regport.data.http import PoliteClient
from regport.errors import DataSourceError

TIINGO_BASE_URL = "https://api.tiingo.com/tiingo/daily"


class TiingoClient:
    def __init__(self, api_key: SecretStr, client: PoliteClient | None = None) -> None:
        key = api_key.get_secret_value().strip()
        if not key:
            raise DataSourceError("TIINGO_API_KEY is not set")
        self._client = client or PoliteClient(
            min_interval_s=1.0, headers={"Authorization": f"Token {key}"}
        )

    def close(self) -> None:
        self._client.close()

    def prices_json(self, ticker: str, start_date: str = "1990-01-01") -> list[dict[str, Any]]:
        response = self._client.get(
            f"{TIINGO_BASE_URL}/{ticker}/prices", params={"startDate": start_date}
        )
        if response.status_code != 200:
            raise DataSourceError(f"Tiingo {ticker}: HTTP {response.status_code}")
        rows: list[dict[str, Any]] = response.json()
        return rows


def tiingo_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Tiingo rows to ``date, close, adj_close, volume, div_cash, split_factor``."""
    if not rows:
        raise DataSourceError("Tiingo returned no rows")
    return (
        pl.DataFrame(rows)
        .select(
            pl.col("date").str.slice(0, 10).str.to_date("%Y-%m-%d"),
            pl.col("close").cast(pl.Float64),
            pl.col("adjClose").cast(pl.Float64).alias("adj_close"),
            pl.col("volume").cast(pl.Float64),
            pl.col("divCash").cast(pl.Float64).alias("div_cash"),
            pl.col("splitFactor").cast(pl.Float64).alias("split_factor"),
        )
        .sort("date")
    )
