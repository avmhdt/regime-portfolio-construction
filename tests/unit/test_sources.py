"""HTTP politeness, FRED/ALFRED, Sharadar and Stooq clients, all offline (NFR-10, FR-02-FR-04)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from regport.data import fred as fred_mod
from regport.data.fred import FredClient, observations_frame
from regport.data.http import PoliteClient, redact_url
from regport.data.sharadar import SharadarClient, parse_sharadar_csv
from regport.data.stooq import build_index, read_stooq_file
from regport.errors import DataSourceError

KEY = "0123456789abcdef0123456789abcdef"
STOOQ_HEADER = "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"


def _client(handler: httpx.MockTransport, **kwargs: object) -> PoliteClient:
    return PoliteClient(min_interval_s=0.0, backoff_s=0.0, transport=handler, **kwargs)  # type: ignore[arg-type]


def test_redact_url() -> None:
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=DTB3&api_key={KEY}"
    red = redact_url(url)
    assert KEY not in red and "api_key=REDACTED" in red and "series_id=DTB3" in red


def test_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503 if calls["n"] < 3 else 200, text="ok")

    with _client(httpx.MockTransport(handler)) as client:
        assert client.get("https://example.org/x").text == "ok"
    assert calls["n"] == 3


def test_retry_exhaustion_is_redacted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    client = _client(httpx.MockTransport(handler), max_attempts=2)
    with pytest.raises(DataSourceError) as info:
        client.get("https://example.org/x", params={"api_key": KEY})
    assert KEY not in str(info.value) and "429" in str(info.value)


def test_transport_error_is_retried_and_redacted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    client = _client(httpx.MockTransport(handler), max_attempts=2)
    with pytest.raises(DataSourceError, match="ConnectError") as info:
        client.get(f"https://example.org/x?api_key={KEY}")
    assert KEY not in str(info.value)


def test_min_interval_between_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("regport.data.http.time.sleep", sleeps.append)
    client = PoliteClient(
        min_interval_s=5.0, transport=httpx.MockTransport(lambda r: httpx.Response(200))
    )
    client.get("https://example.org/a")
    client.get("https://example.org/b")
    assert len(sleeps) == 1 and 4.0 < sleeps[0] <= 5.0


def _fred_page(observations: list[dict[str, str]], count: int) -> httpx.Response:
    return httpx.Response(200, json={"count": count, "observations": observations})


def _obs(day: str, value: str, rt: str = "2026-11-03") -> dict[str, str]:
    return {"date": day, "value": value, "realtime_start": rt, "realtime_end": rt}


def test_fred_header_auth_and_paging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fred_mod, "PAGE_LIMIT", 2)
    seen: list[httpx.Request] = []
    data = [_obs("2024-01-02", "5.2"), _obs("2024-01-03", "."), _obs("2024-01-04", "5.1")]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        offset = int(request.url.params["offset"])
        return _fred_page(data[offset : offset + 2], len(data))

    fred = FredClient(SecretStr(KEY), client=_client(httpx.MockTransport(handler)))
    rows = fred.observations("DTB3", observation_start=date(2024, 1, 1))
    fred.close()
    assert len(rows) == 3 and fred.auth_mode == "header"
    assert all("api_key" not in r.url.params for r in seen)
    assert seen[0].headers["Authorization"] == f"Bearer {KEY}"
    assert seen[0].url.params["observation_start"] == "2024-01-01"
    frame = observations_frame(rows, "DTB3")
    assert frame["value"].to_list() == [5.2, None, 5.1]
    assert frame["obs_date"].to_list()[0] == date(2024, 1, 2)


def test_fred_falls_back_to_query_param() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "api_key" not in request.url.params:
            return httpx.Response(400, json={"error_message": "Bad Request. Variable api_key"})
        return _fred_page([_obs("2024-01-02", "1")], 1)

    fred = FredClient(SecretStr(KEY), client=_client(httpx.MockTransport(handler)))
    assert len(fred.observations("DTB3")) == 1
    assert fred.auth_mode == "query"
    fred.observations("DTB3")
    assert len(seen) == 3  # the second call goes straight to the query mode


def test_fred_vintage_window_and_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params["series_id"] == "BAD":
            return httpx.Response(400, json={"error_message": "bad series"})
        assert request.url.params["realtime_start"] == "1776-07-04"
        assert request.url.params["realtime_end"] == "9999-12-31"
        return _fred_page([_obs("2024-01-01", "3.7", "2024-02-02")], 1)

    fred = FredClient(SecretStr(KEY), client=_client(httpx.MockTransport(handler)))
    assert fred.vintages("UNRATE")[0]["realtime_start"] == "2024-02-02"
    with pytest.raises(DataSourceError, match="HTTP 400") as info:
        fred.observations("BAD")
    assert KEY not in str(info.value)
    with pytest.raises(DataSourceError, match="not set"):
        FredClient(SecretStr(" "))


def test_observations_frame_empty() -> None:
    assert observations_frame([], "X").columns == [
        "series_id",
        "obs_date",
        "realtime_start",
        "realtime_end",
        "value",
    ]


SFP_CSV = (
    "ticker,date,open,high,low,close,volume,closeadj,closeunadj,lastupdated\n"
    "SPY,2024-01-03,1,1,1,470.0,10,465.0,470.0,2024-01-04\n"
    "SPY,2024-01-02,1,1,1,472.0,10,467.0,472.0,2024-01-03\n"
)


def test_sharadar_sends_key_in_header_only() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=SFP_CSV)

    client = SharadarClient(
        SecretStr(KEY),
        client=_client(httpx.MockTransport(handler), headers={"x-api-key": KEY}),
    )
    frame = client.sfp_prices("SPY")
    client.close()
    assert frame["date"].to_list() == [date(2024, 1, 2), date(2024, 1, 3)]
    assert seen[0].headers["x-api-key"] == KEY
    assert KEY not in str(seen[0].url)
    assert seen[0].url.path == "/v1.0/data/funds"


def test_sharadar_default_client_sets_header() -> None:
    client = SharadarClient(SecretStr(KEY))
    assert client._client._client.headers["x-api-key"] == KEY
    client.close()


@pytest.mark.parametrize(
    ("status", "text", "match"),
    [
        (403, "Exceeds free tier", "key refused"),
        (500, "", "HTTP 500"),
        (200, "", "no rows"),
    ],
)
def test_sharadar_errors(status: int, text: str, match: str) -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(status, text=text))
    client = SharadarClient(SecretStr(KEY), client=_client(transport, max_attempts=1))
    with pytest.raises(DataSourceError, match=match):
        client.sfp_prices("SPY")
    with pytest.raises(DataSourceError, match="not set"):
        SharadarClient(SecretStr(""))


def test_parse_sharadar_csv_needs_date() -> None:
    with pytest.raises(DataSourceError, match="no date"):
        parse_sharadar_csv("ticker,close\nSPY,1\n")


def _write_stooq(path: Path, rows: list[str], header: str = STOOQ_HEADER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "".join(rows), encoding="utf-8")


def test_stooq_index_and_parse(tmp_path: Path) -> None:
    spy = tmp_path / "d_us_txt/data/daily/us/nyse etfs/2/spy.us.txt"
    _write_stooq(
        spy,
        [
            "SPY.US,D,20261030,000000,761.38,761.46,750.36,755.72,77666350.5,0\n",
            "SPY.US,D,20261029,000000,760,762,755,758,1000,0\n",
        ],
    )
    _write_stooq(tmp_path / "d_us_txt/data/daily/us/nyse etfs/tlt.us.txt", [])
    (tmp_path / "d_us_txt/readme.md").write_text("x", encoding="utf-8")
    index = build_index(tmp_path)
    assert set(index) == {"spy.us", "tlt.us"}
    frame = read_stooq_file(index["spy.us"])
    assert frame["date"].to_list() == [date(2026, 10, 29), date(2026, 10, 30)]
    assert frame["close"].to_list() == [758.0, 755.72]
    assert frame["volume"].to_list()[1] == 77666350.5
    with pytest.raises(DataSourceError, match="no rows"):
        read_stooq_file(index["tlt.us"])


def test_stooq_failures(tmp_path: Path) -> None:
    with pytest.raises(DataSourceError, match="not found"):
        build_index(tmp_path)
    _write_stooq(tmp_path / "d_us_txt/a/spy.us.txt", [])
    _write_stooq(tmp_path / "d_us_txt/b/spy.us.txt", [])
    with pytest.raises(DataSourceError, match="twice"):
        build_index(tmp_path)
    bad = tmp_path / "bad.txt"
    _write_stooq(bad, [], header="Date,Open,Close\n")
    with pytest.raises(DataSourceError, match="header"):
        read_stooq_file(bad)
    weekly = tmp_path / "weekly.txt"
    _write_stooq(weekly, ["SPY.US,W,20261030,000000,1,1,1,1,1,0\n"])
    with pytest.raises(DataSourceError, match="daily"):
        read_stooq_file(weekly)


def test_fred_error_payload_never_echoes_key() -> None:
    body = json.dumps({"error_message": "bad"})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text=body)

    fred = FredClient(SecretStr(KEY), client=_client(httpx.MockTransport(handler)))
    with pytest.raises(DataSourceError) as info:
        fred.observations("DTB3")
    assert KEY not in str(info.value)
