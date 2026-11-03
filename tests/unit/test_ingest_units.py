"""Ingest edge cases: Tiingo, French parsing errors, throttled downloads, missing keys."""

from __future__ import annotations

import zipfile
from datetime import date
from pathlib import Path

import httpx
import polars as pl
import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from regport.cli import app
from regport.config import ExperimentConfig, load_universe
from regport.data import ingest
from regport.data.french import DATASETS, FrenchDataset, parse_first_table, read_zip_text
from regport.data.http import PoliteClient
from regport.data.manifest import Manifest
from regport.data.panel import build_rf, master_calendar, next_trading_date
from regport.data.tiingo import TiingoClient, tiingo_frame
from regport.errors import DataSourceError
from regport.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
KEY = "0123456789abcdef0123456789abcdef"
TIINGO_ROWS = [
    {
        "date": "2024-01-03T00:00:00.000Z",
        "close": 470.0,
        "adjClose": 465.0,
        "volume": 10,
        "divCash": 0.0,
        "splitFactor": 1.0,
    },
    {
        "date": "2024-01-02T00:00:00.000Z",
        "close": 472.0,
        "adjClose": 467.0,
        "volume": 10,
        "divCash": 1.5,
        "splitFactor": 1.0,
    },
]


def _settings(tmp_path: Path, **kwargs: object) -> Settings:
    return Settings(_env_file=None, regport_data_dir=tmp_path / "data", **kwargs)  # type: ignore[call-arg]


def test_tiingo_client_and_fetch(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=TIINGO_ROWS)

    client = TiingoClient(
        SecretStr(KEY),
        client=PoliteClient(
            min_interval_s=0,
            transport=httpx.MockTransport(handler),
            headers={"Authorization": f"Token {KEY}"},
        ),
    )
    universe = load_universe(ROOT / "configs/universe.yaml")
    settings = _settings(tmp_path)
    results = ingest.fetch_tiingo(settings, universe, ["SPY"], client=client)
    assert results[0].status == "fetched"
    assert KEY not in str(seen[0].url) and seen[0].headers["Authorization"] == f"Token {KEY}"
    out = ingest.write_interim_prices(settings, "tiingo", universe)
    assert out is not None
    frame = pl.read_parquet(out)
    assert frame["date"].to_list() == [date(2024, 1, 2), date(2024, 1, 3)]
    assert frame["div_cash"].to_list() == [1.5, 0.0]
    assert ingest.fetch_tiingo(settings, universe, ["SPY"], client=client)[0].status == "skipped"


def test_tiingo_errors() -> None:
    with pytest.raises(DataSourceError, match="not set"):
        TiingoClient(SecretStr(""))
    bad = TiingoClient(
        SecretStr(KEY),
        client=PoliteClient(
            min_interval_s=0,
            max_attempts=1,
            transport=httpx.MockTransport(lambda r: httpx.Response(404)),
        ),
    )
    with pytest.raises(DataSourceError, match="404"):
        bad.prices_json("spy")
    with pytest.raises(DataSourceError, match="no rows"):
        tiingo_frame([])
    TiingoClient(SecretStr(KEY)).close()


def _zip(path: Path, files: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return path


def test_french_parse_errors(tmp_path: Path) -> None:
    ds = FrenchDataset("t", "t.zip", 2)
    with pytest.raises(DataSourceError, match="no daily data"):
        parse_first_table("header only\n", ds)
    with pytest.raises(DataSourceError, match="expected 2 series"):
        parse_first_table("x\n,A,B,C\n20010102,1,2,3\n", ds)
    with pytest.raises(DataSourceError, match="bad column count"):
        parse_first_table("x\n,A,B\n20010102,1,2\n20010103,1\n", ds)
    frame = parse_first_table("x\n,A,B\n20010102,1.5,-99.99\n20010103,-999,2\n\n", ds)
    assert frame["ret"].to_list() == [0.015, None, None, 0.02]
    with pytest.raises(DataSourceError, match="one CSV"):
        read_zip_text(_zip(tmp_path / "two.zip", {"a.csv": "", "b.csv": ""}))
    assert DATASETS["ind49_daily"].url.endswith("49_Industry_Portfolios_daily_CSV.zip")


def test_download_throttle_and_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("regport.data.http.time.sleep", sleeps.append)
    body = b"x" * 200_000
    client = PoliteClient(
        min_interval_s=0, transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body))
    )
    dest = tmp_path / "f.zip"
    assert client.download("https://example.org/f.zip", dest, max_bytes_per_s=100_000) == 200_000
    assert dest.read_bytes() == body and sum(sleeps) > 1.0
    missing = PoliteClient(
        min_interval_s=0, transport=httpx.MockTransport(lambda r: httpx.Response(404))
    )
    with pytest.raises(DataSourceError, match="404"):
        missing.download("https://example.org/g.zip", tmp_path / "g.zip")

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("cut", request=request)

    broken = PoliteClient(min_interval_s=0, transport=httpx.MockTransport(boom))
    with pytest.raises(DataSourceError, match="ReadError"):
        broken.download("https://example.org/h.zip", tmp_path / "h.zip")
    assert not (tmp_path / "h.zip").exists() and not (tmp_path / "h.zip.part").exists()


def test_missing_keys_and_dirs(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    universe = load_universe(ROOT / "configs/universe.yaml")
    with pytest.raises(DataSourceError, match="SHARADAR_API_KEY"):
        ingest.fetch_sharadar(settings, universe, ["SPY"])
    with pytest.raises(DataSourceError, match="TIINGO_API_KEY"):
        ingest.fetch_tiingo(settings, universe, ["SPY"])
    with pytest.raises(DataSourceError, match="STOOQ_DIR"):
        ingest.fetch_stooq(settings, universe, ["SPY"])
    with pytest.raises(DataSourceError, match="FRED_API_KEY"):
        ingest.fetch_fred(settings, ["DTB3"], vintages=False)
    assert ingest.write_interim_prices(settings, "sharadar", universe) is None


def test_stooq_symbol_missing(tmp_path: Path) -> None:
    (tmp_path / "stooq" / "d_us_txt").mkdir(parents=True)
    settings = _settings(tmp_path, stooq_dir=tmp_path / "stooq")
    with pytest.raises(DataSourceError, match=r"spy\.us not in"):
        ingest.fetch_stooq(settings, load_universe(ROOT / "configs/universe.yaml"), ["SPY"])


def test_empty_sources_raise(tmp_path: Path) -> None:
    def empty(request: httpx.Request) -> httpx.Response:
        if "fred" in request.url.host or "stlouisfed" in request.url.host:
            return httpx.Response(200, json={"count": 0, "observations": []})
        return httpx.Response(200, text="ticker,date,close\n")

    from regport.data.fred import FredClient
    from regport.data.sharadar import SharadarClient

    settings = _settings(tmp_path)
    universe = load_universe(ROOT / "configs/universe.yaml")
    poli = PoliteClient(min_interval_s=0, transport=httpx.MockTransport(empty))
    with pytest.raises(DataSourceError, match="no rows"):
        ingest.fetch_sharadar(
            settings, universe, ["SPY"], client=SharadarClient(SecretStr(KEY), client=poli)
        )
    with pytest.raises(DataSourceError, match="no observations"):
        ingest.fetch_fred(
            settings, ["DTB3"], vintages=False, client=FredClient(SecretStr(KEY), client=poli)
        )
    assert Manifest(settings.data_dir).entries == []


def test_panel_helpers() -> None:
    cal = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5)]
    assert next_trading_date([date(2024, 1, 2), date(2024, 1, 4), date(2024, 1, 5)], cal) == [
        date(2024, 1, 3),
        date(2024, 1, 5),
        None,
    ]
    prices = pl.DataFrame({"date": cal[:1], "symbol": ["SPY"], "adj_close": [1.0]})
    with pytest.raises(DataSourceError, match="no prices"):
        master_calendar(prices, "SPY", date(2024, 1, 1))
    empty = pl.DataFrame(
        schema={"series_id": pl.String, "value": pl.Float64, "available_date": pl.Date}
    )
    with pytest.raises(DataSourceError, match="risk-free"):
        build_rf(empty, cal, "DTB3")


def test_cli_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    for var in ("FRED_API_KEY", "SHARADAR_API_KEY", "STOOQ_DIR"):
        monkeypatch.setenv(var, "")
    monkeypatch.setenv("REGPORT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("REGPORT_REPORTS_DIR", str(tmp_path / "reports"))
    runner = CliRunner()
    for args in (
        ["fetch", "fred"],
        ["fetch", "alfred"],
        ["fetch", "prices", "--source", "sharadar", "--symbols", "spy"],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 1 and "not set" in result.output, args
    result = runner.invoke(app, ["data", "build"])
    assert result.exit_code == 1 and "regport fetch prices" in result.output
    result = runner.invoke(app, ["fetch", "french", "--help"])
    assert result.exit_code == 0
    assert (
        ingest.resolve_price_source(
            ExperimentConfig.model_validate({"data": {"price_source": "stooq"}}),
            Settings(_env_file=None),
        )
        == "stooq"
    )  # type: ignore[call-arg]
