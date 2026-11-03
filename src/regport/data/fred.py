"""FRED and ALFRED ``series/observations`` client (FR-03, FR-04).

FRED's v1 API documents the key only as the ``api_key`` query parameter; its v2 API documents an
``Authorization: Bearer`` header. The client tries the header first and falls back to the query
parameter only if FRED refuses the header. Either way the key is kept out of logs and exception
messages (``redact_url``), and httpx's own URL logging is silenced (``regport.logs``).
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

import polars as pl
from pydantic import SecretStr

from regport.data.http import PoliteClient, redact_url
from regport.errors import DataSourceError
from regport.logs import get_logger

log = get_logger(__name__)

FRED_BASE_URL = "https://api.stlouisfed.org/fred"
ALFRED_REALTIME_START = "1776-07-04"
ALFRED_REALTIME_END = "9999-12-31"
PAGE_LIMIT = 100_000

AuthMode = Literal["header", "query"]


class FredClient:
    """Paged access to ``series/observations``; at most one request per 0.5 s."""

    def __init__(
        self,
        api_key: SecretStr,
        client: PoliteClient | None = None,
        auth_mode: AuthMode | None = None,
    ) -> None:
        if not api_key.get_secret_value().strip():
            raise DataSourceError("FRED_API_KEY is not set")
        self._key = api_key
        self._client = client or PoliteClient(min_interval_s=0.5)
        self.auth_mode: AuthMode | None = auth_mode

    def close(self) -> None:
        self._client.close()

    def _request(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        url = f"{FRED_BASE_URL}/{path}"
        params = {**params, "file_type": "json"}
        modes: list[AuthMode] = [self.auth_mode] if self.auth_mode else ["header", "query"]
        last_status = 0
        for mode in modes:
            if mode == "header":
                headers = {"Authorization": f"Bearer {self._key.get_secret_value()}"}
                response = self._client.get(url, params=params, headers=headers)
            else:
                keyed = {**params, "api_key": self._key.get_secret_value()}
                response = self._client.get(url, params=keyed)
            last_status = response.status_code
            if response.status_code == 200:
                if self.auth_mode is None:
                    log.info("fred_auth_mode", mode=mode)
                self.auth_mode = mode
                payload: dict[str, Any] = response.json()
                return payload
            if response.status_code not in (400, 401, 403) or self.auth_mode is not None:
                break
        raise DataSourceError(
            f"FRED HTTP {last_status} for {redact_url(url)} series={params.get('series_id')}"
        )

    def observations(
        self,
        series_id: str,
        *,
        observation_start: date | None = None,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
    ) -> list[dict[str, str]]:
        """All observation rows (strings, as FRED returns them), following ``offset`` paging."""
        params: dict[str, str] = {"series_id": series_id, "limit": str(PAGE_LIMIT)}
        if observation_start is not None:
            params["observation_start"] = observation_start.isoformat()
        if realtime_start is not None:
            params["realtime_start"] = realtime_start
        if realtime_end is not None:
            params["realtime_end"] = realtime_end
        rows: list[dict[str, str]] = []
        offset = 0
        while True:
            page = self._request("series/observations", {**params, "offset": str(offset)})
            batch = page.get("observations", [])
            rows.extend(batch)
            count = int(page.get("count", len(rows)))
            offset += len(batch)
            if not batch or offset >= count:
                return rows

    def vintages(self, series_id: str) -> list[dict[str, str]]:
        """Every ALFRED vintage of ``series_id`` (realtime window 1776-07-04 to 9999-12-31)."""
        return self.observations(
            series_id, realtime_start=ALFRED_REALTIME_START, realtime_end=ALFRED_REALTIME_END
        )


def observations_frame(rows: list[dict[str, str]], series_id: str) -> pl.DataFrame:
    """Typed frame: FRED's ``"."`` missing marker becomes null (§7b ``fred_daily``)."""
    schema = {
        "series_id": pl.String,
        "obs_date": pl.Date,
        "realtime_start": pl.Date,
        "realtime_end": pl.Date,
        "value": pl.Float64,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows).select(
        pl.lit(series_id).alias("series_id"),
        pl.col("date").str.to_date("%Y-%m-%d").alias("obs_date"),
        pl.col("realtime_start").str.to_date("%Y-%m-%d"),
        pl.col("realtime_end").str.to_date("%Y-%m-%d"),
        pl.when(pl.col("value") == ".")
        .then(None)
        .otherwise(pl.col("value"))
        .cast(pl.Float64)
        .alias("value"),
    )
