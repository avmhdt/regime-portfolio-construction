"""M2 end to end on synthetic sources: fetch (mocked HTTP) -> build -> check -> report.

Prices are simulated, with monthly distributions on the credit ETFs, a 2:1 split and a three-day
price gap. ALFRED vintages revise each month's first release a month later. Nothing here is real
data.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import polars as pl
import pytest
import yaml
from typer.testing import CliRunner

from regport.cli import app
from regport.config import ExperimentConfig, load_config, load_universe
from regport.data import checks, ingest
from regport.data.fred import FredClient
from regport.data.french import DATASETS
from regport.data.http import PoliteClient
from regport.data.manifest import Manifest
from regport.data.sharadar import SharadarClient
from regport.settings import Settings

KEY = "0123456789abcdef0123456789abcdef"
START, END = date(2001, 1, 2), date(2012, 12, 31)
HOLIDAY = date(2005, 7, 4)
GAP = [date(2008, 3, 10), date(2008, 3, 11), date(2008, 3, 12)]
SPLIT = date(2006, 6, 9)
ASSETS = [
    ("SPY", "equity", START),
    ("EFA", "equity", START),
    ("VNQ", "equity", date(2004, 9, 23)),
    ("IEF", "rates", START),
    ("TLT", "rates", START),
    ("LQD", "credit", START),
    ("HYG", "credit", date(2003, 6, 2)),
    ("GLD", "commodity", date(2002, 1, 2)),
    ("DBC", "commodity", date(2004, 1, 2)),
    ("UUP", "fx", date(2004, 1, 2)),
]
DISTRIBUTING = {"HYG": 0.0035, "LQD": 0.0030}


def _weekdays(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5 and d != HOLIDAY:
            out.append(d)
        d += timedelta(days=1)
    return out


CALENDAR = _weekdays(START, END)


def _simulate() -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(7)
    out = {}
    for symbol, _cls, inception in ASSETS:
        dates = [d for d in CALENDAR if d >= inception]
        if symbol == "GLD":
            dates = [d for d in dates if d not in GAP]
        n = len(dates)
        total = rng.normal(0.07 / 252, 0.004, n)
        total[0] = 0.0
        months = np.array([d.month for d in dates])
        ex_day = np.r_[False, months[1:] != months[:-1]]
        div = np.where(ex_day, DISTRIBUTING.get(symbol, 0.0), 0.0)
        adj = 50.0 * np.cumprod(1 + total)
        price = 50.0 * np.cumprod(1 + total - div)
        factor = np.where(np.array(dates) < SPLIT, 2.0, 1.0) if symbol == "EFA" else 1.0
        out[symbol] = pl.DataFrame(
            {
                "ticker": symbol,
                "date": dates,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1e6,
                "closeadj": adj,
                "closeunadj": price * factor,
                "lastupdated": "2026-11-03",
            }
        )
    return out


PRICES = _simulate()


def _sharadar_handler(request: httpx.Request) -> httpx.Response:
    assert request.headers["x-api-key"] == KEY
    frame = PRICES[request.url.params["ticker"]].with_columns(pl.col("date").cast(pl.String))
    return httpx.Response(200, text=frame.write_csv())


def _first_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d


def _vintage_rows() -> list[dict[str, str]]:
    months = [(y, m) for y in range(1999, 2013) for m in range(1, 13)][:-1]
    release = {}
    for y, m in months:
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        release[(y, m)] = _first_friday(ny, nm)
    rows = []
    keys = list(release)
    for i, (y, m) in enumerate(keys):
        first = 100.0 + i
        if i + 1 < len(keys):
            revised_on = release[keys[i + 1]]
            rows.append(
                {
                    "date": f"{y}-{m:02d}-01",
                    "value": str(first),
                    "realtime_start": release[(y, m)].isoformat(),
                    "realtime_end": (revised_on - timedelta(days=1)).isoformat(),
                }
            )
            rows.append(
                {
                    "date": f"{y}-{m:02d}-01",
                    "value": str(first + 1),
                    "realtime_start": revised_on.isoformat(),
                    "realtime_end": "9999-12-31",
                }
            )
        else:
            rows.append(
                {
                    "date": f"{y}-{m:02d}-01",
                    "value": str(first),
                    "realtime_start": release[(y, m)].isoformat(),
                    "realtime_end": "9999-12-31",
                }
            )
    return rows


def _daily_rows(series_id: str) -> list[dict[str, str]]:
    rng = np.random.default_rng(len(series_id))
    rows = []
    for d in _weekdays(date(2000, 6, 1), END):
        value = "." if d.month == 12 and d.day == 25 else f"{2.0 + rng.normal(0, 0.1):.2f}"
        rows.append(
            {
                "date": d.isoformat(),
                "value": value,
                "realtime_start": "2026-11-03",
                "realtime_end": "2026-11-03",
            }
        )
    return rows


def _fred_handler(request: httpx.Request) -> httpx.Response:
    params = request.url.params
    if "api_key" not in params:
        return httpx.Response(400, json={"error_message": "api_key is not set"})
    rows = _vintage_rows() if "realtime_start" in params else _daily_rows(params["series_id"])
    return httpx.Response(200, json={"count": len(rows), "observations": rows})


def _french_zip(n_series: int) -> bytes:
    names = ["Mkt-RF", "SMB", "HML", "RF"] if n_series == 4 else [f"I{i}" for i in range(n_series)]
    lines = [
        "This file was created by a synthetic generator.",
        "",
        "  Average Value Weighted Returns -- Daily",
        "," + ",".join(names),
    ]
    rng = np.random.default_rng(n_series)
    for d in CALENDAR[:300]:
        vals = rng.normal(0.05, 1.0, n_series)
        cells = [
            "-99.99" if (d == CALENDAR[5] and j == 0) else f"{v:.2f}" for j, v in enumerate(vals)
        ]
        lines.append(d.strftime("%Y%m%d") + "," + ",".join(cells))
    lines += [
        "",
        "  Average Equal Weighted Returns -- Daily",
        "," + ",".join(names),
        "20010102," + ",".join(["999"] * n_series),
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("data.csv", "\r\n".join(lines) + "\r\n")
    return buf.getvalue()


def _french_handler(request: httpx.Request) -> httpx.Response:
    for ds in DATASETS.values():
        if request.url.path.endswith(ds.file):
            return httpx.Response(200, content=_french_zip(ds.n_series))
    return httpx.Response(404)


def _write_stooq(root: Path, price_only: set[str] = frozenset()) -> Path:  # type: ignore[assignment]
    header = "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"
    for symbol, frame in PRICES.items():
        col = "close" if symbol in price_only else "closeadj"
        body = "".join(
            f"{symbol}.US,D,{d:%Y%m%d},000000,{c},{c},{c},{c},1000,0\n"
            for d, c in zip(frame["date"], frame[col], strict=True)
        )
        sub = "2" if symbol == "SPY" else ""
        path = (
            root
            / "d_us_txt"
            / "data"
            / "daily"
            / "us"
            / "nyse etfs"
            / sub
            / f"{symbol.lower()}.us.txt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + body, encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("pipeline")
    universe = {
        "assets": [
            {
                "symbol": s,
                "asset_class": c,
                "exposure": "synthetic",
                "inception": i,
                "sharadar": s,
                "tiingo": s.lower(),
                "stooq": f"{s.lower()}.us",
                "cost_bps": 2,
            }
            for s, c, i in ASSETS
        ]
    }
    (root / "universe.yaml").write_text(yaml.safe_dump(universe), encoding="utf-8")
    cfg = ExperimentConfig().to_params()
    cfg["data"]["universe_file"] = str(root / "universe.yaml")
    cfg["sample"]["data_start"] = START.isoformat()
    (root / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    _write_stooq(root / "stooq")
    return {"root": root, "config": root / "config.yaml"}


@pytest.fixture
def mocked(monkeypatch: pytest.MonkeyPatch, workspace: dict[str, Path]) -> dict[str, Path]:
    root = workspace["root"]

    def sharadar(key: Any) -> SharadarClient:
        return SharadarClient(
            key,
            client=PoliteClient(
                min_interval_s=0,
                transport=httpx.MockTransport(_sharadar_handler),
                headers={"x-api-key": KEY},
            ),
        )

    def fred(key: Any) -> FredClient:
        return FredClient(
            key, client=PoliteClient(min_interval_s=0, transport=httpx.MockTransport(_fred_handler))
        )

    def polite(**_: Any) -> PoliteClient:
        return PoliteClient(min_interval_s=0, transport=httpx.MockTransport(_french_handler))

    monkeypatch.setattr(ingest, "SharadarClient", sharadar)
    monkeypatch.setattr(ingest, "FredClient", fred)
    monkeypatch.setattr(ingest, "PoliteClient", polite)
    monkeypatch.chdir(root)
    for var, value in {
        "FRED_API_KEY": KEY,
        "SHARADAR_API_KEY": KEY,
        "TIINGO_API_KEY": "",
        "STOOQ_DIR": str(root / "stooq"),
        "DATA_ROOT": "",
        "REGPORT_DATA_DIR": str(root / "data"),
        "REGPORT_RESULTS_DIR": str(root / "results"),
        "REGPORT_REPORTS_DIR": str(root / "reports"),
    }.items():
        monkeypatch.setenv(var, value)
    return workspace


def test_fetch_build_check(mocked: dict[str, Path]) -> None:
    runner = CliRunner()
    config = str(mocked["config"])
    result = runner.invoke(app, ["fetch", "all", "--config", config])
    assert result.exit_code == 0, result.output
    assert result.output.count("[fetched]") == 10 + 10 + 4 + 4 + 3
    again = runner.invoke(app, ["fetch", "all", "--config", config])
    assert again.output.count("[skipped]") == 31 and "[fetched]" not in again.output

    result = runner.invoke(app, ["data", "build", "--config", config])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["data", "check", "--config", config])
    assert result.exit_code == 0, result.output
    report = (mocked["root"] / "reports" / "data_quality.md").read_text(encoding="utf-8")
    assert "Overall: **PASS**" in report
    assert KEY not in report

    data = mocked["root"] / "data"
    manifest = json.loads((data / "raw" / "manifest.json").read_text(encoding="utf-8"))
    assert all(KEY not in json.dumps(e) for e in manifest["entries"])
    assert {e["source"] for e in manifest["entries"]} == {
        "sharadar",
        "stooq",
        "fred",
        "alfred",
        "french",
    }

    returns = pl.read_parquet(data / "processed" / "returns_daily.parquet")
    gld = returns.filter(pl.col("symbol") == "GLD").sort("date")
    gap_rows = gld.filter(pl.col("date").is_in([*GAP, date(2008, 3, 13)]))
    assert gap_rows["ret"].null_count() == 4  # three missing days and the day after
    assert gap_rows["eligible"].to_list() == [True, True, True, True]  # ffill <= 3 days
    hyg = returns.filter(pl.col("symbol") == "HYG")
    first_eligible = hyg.filter("eligible")["date"].min()
    assert hyg.filter(pl.col("date") < first_eligible)["n_valid_history"].max() == 251
    rf = pl.read_parquet(data / "processed" / "rf_daily.parquet")
    assert rf["date"].to_list() == CALENDAR
    assert rf["rf"][0] is None or rf["rf"][0] > 0

    prices = pl.read_parquet(data / "processed" / "prices_daily.parquet")
    assert set(prices["source"].unique()) == {"sharadar"}
    french = pl.read_parquet(data / "processed" / "french_daily.parquet")
    assert french["ret"].null_count() == 3  # one -99.99 per dataset; the second table is ignored


def test_check_fails_when_stooq_misses_distributions(mocked: dict[str, Path]) -> None:
    cfg = load_config(mocked["config"])
    settings = Settings()
    universe = load_universe(cfg.data.universe_file)
    if not (settings.data_dir / "processed" / "prices_daily.parquet").exists():
        CliRunner().invoke(app, ["fetch", "all", "--config", str(mocked["config"])])
        CliRunner().invoke(app, ["data", "build", "--config", str(mocked["config"])])
    _write_stooq(mocked["root"] / "stooq", price_only={"HYG"})
    ingest.fetch_stooq(settings, universe, ["HYG"], force=True)
    ingest.write_interim_prices(settings, "stooq", universe)
    report = checks.run_checks(cfg, settings, universe)
    dq03 = next(c for c in report.checks if c.id == "DQ-03")
    assert dq03.passed is False and not report.ok
    hyg = next(r for r in dq03.table if r.get("symbol") == "HYG")
    assert hyg["of which distribution days Stooq left unadjusted"] == hyg["disagreeing days"]
    assert len(Manifest(settings.data_dir).entries) == 32  # the old Stooq file is kept
    _write_stooq(mocked["root"] / "stooq")


def test_missing_processed_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(_env_file=None, regport_data_dir=tmp_path / "data")  # type: ignore[call-arg]
    cfg = ExperimentConfig()
    with pytest.raises(Exception, match="regport data build"):
        checks.run_checks(
            cfg,
            settings,
            load_universe(Path(__file__).resolve().parents[2] / "configs" / "universe.yaml"),
        )
