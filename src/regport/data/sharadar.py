"""Sharadar direct API, table ``funds`` (SFP) (FR-02, D-01).

Base URL ``https://api.sharadar.com/v1.0``; the key travels in the ``x-api-key`` header and never
in a URL or log line. Sibling repos found that AAPL and MSFT answer without a valid key (free
tier), so key checks must use a ticker outside the free tier; the 12 ETFs are such tickers.
"""

from __future__ import annotations

import io

import polars as pl
from pydantic import SecretStr

from regport.data.http import PoliteClient
from regport.errors import DataSourceError

SHARADAR_BASE_URL = "https://api.sharadar.com/v1.0"
SFP_TABLE = "funds"


class SharadarClient:
    """Per-ticker queries of a Sharadar table; responses are CSV text."""

    def __init__(self, api_key: SecretStr, client: PoliteClient | None = None) -> None:
        key = api_key.get_secret_value().strip()
        if not key:
            raise DataSourceError("SHARADAR_API_KEY is not set")
        self._client = client or PoliteClient(min_interval_s=0.5, headers={"x-api-key": key})

    def close(self) -> None:
        self._client.close()

    def table_csv(self, table: str, ticker: str) -> str:
        """Raw CSV body for one ticker of ``table``; raises on any non-200 response."""
        response = self._client.get(f"{SHARADAR_BASE_URL}/data/{table}", params={"ticker": ticker})
        if response.status_code != 200:
            detail = (
                "key refused or table not in subscription"
                if response.status_code in (401, 403)
                else "unexpected response"
            )
            raise DataSourceError(
                f"Sharadar {table}/{ticker}: HTTP {response.status_code} ({detail})"
            )
        return response.text

    def sfp_prices(self, ticker: str) -> pl.DataFrame:
        """SFP daily rows for ``ticker`` sorted by date, as returned (no transformation)."""
        frame = parse_sharadar_csv(self.table_csv(SFP_TABLE, ticker))
        if frame.is_empty():
            raise DataSourceError(f"Sharadar {SFP_TABLE}/{ticker}: no rows")
        return frame.sort("date")


def parse_sharadar_csv(text: str) -> pl.DataFrame:
    """Parse a Sharadar CSV body; ``date`` becomes a Date column."""
    if not text.strip():
        return pl.DataFrame()
    frame = pl.read_csv(io.StringIO(text), infer_schema_length=None)
    if "date" not in frame.columns:
        raise DataSourceError(f"Sharadar response has no date column: {frame.columns[:8]}")
    return frame.with_columns(pl.col("date").cast(pl.String).str.to_date("%Y-%m-%d"))
