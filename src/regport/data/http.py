"""Polite HTTP access (NFR-10): retries with exponential backoff, one request at a time per host,
a minimum interval between requests to the same host, and URLs redacted before they reach any
log line or exception message.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from types import TracebackType
from typing import Self
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from regport.errors import DataSourceError
from regport.logs import get_logger

log = get_logger(__name__)

SECRET_PARAMS = frozenset({"api_key", "apikey", "token", "key"})
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class _Retryable(Exception):
    """Transient failure; the message is already redacted."""


def redact_url(url: str | httpx.URL) -> str:
    """Replace the value of any secret-looking query parameter with ``REDACTED``."""
    parts = urlsplit(str(url))
    query = [
        (k, "REDACTED" if k.lower() in SECRET_PARAMS else v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
    ]
    return urlunsplit(parts._replace(query=urlencode(query)))


class PoliteClient:
    """Synchronous httpx client serialised per host, with retries on transient errors.

    Args:
        timeout_s: per-request timeout.
        min_interval_s: minimum seconds between two requests to the same host.
        max_attempts: attempts per request, including the first.
        backoff_s: base of the exponential backoff between attempts (capped at 30 s).
        headers: default headers (for example ``x-api-key``); never logged.
        transport: httpx transport override for tests.
    """

    def __init__(
        self,
        timeout_s: float = 30.0,
        min_interval_s: float = 0.5,
        max_attempts: int = 5,
        backoff_s: float = 1.0,
        headers: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_s),
            headers=dict(headers or {}),
            transport=transport,
            follow_redirects=False,
        )
        self._min_interval = min_interval_s
        self._max_attempts = max_attempts
        self._backoff = backoff_s
        self._locks: dict[str, threading.Lock] = {}
        self._last: dict[str, float] = {}
        self._guard = threading.Lock()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _host_lock(self, host: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(host, threading.Lock())

    def get(
        self,
        url: str,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        """GET with retries on 429/5xx and transport errors. Other non-2xx responses are
        returned to the caller, which decides what they mean."""
        host = urlsplit(url).netloc

        @retry(
            retry=retry_if_exception_type(_Retryable),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=self._backoff, min=self._backoff, max=30.0),
            reraise=True,
        )
        def attempt() -> httpx.Response:
            with self._host_lock(host):
                wait = self._last.get(host, 0.0) + self._min_interval - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                try:
                    response = self._client.get(url, params=params, headers=headers)
                except httpx.TransportError as exc:
                    raise _Retryable(f"{type(exc).__name__} for {redact_url(url)}") from None
                finally:
                    self._last[host] = time.monotonic()
            if response.status_code in RETRY_STATUS:
                log.warning("http_retry", status=response.status_code, url=redact_url(response.url))
                raise _Retryable(f"HTTP {response.status_code} for {redact_url(response.url)}")
            return response

        try:
            return attempt()
        except _Retryable as exc:
            raise DataSourceError(str(exc)) from None
