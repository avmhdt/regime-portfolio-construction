"""``regport smoke`` end to end with fake sources (the real run is marked ``data``)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from regport import smoke
from regport.cli import app
from regport.config import ExperimentConfig
from regport.errors import DataSourceError
from regport.logs import configure_logging, get_logger
from regport.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
KEY = "0123456789abcdef0123456789abcdef"
STOOQ_HEADER = "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"


class FakeFred:
    auth_mode = "header"

    def __init__(self, api_key: SecretStr) -> None:
        assert api_key.get_secret_value() == KEY

    def observations(self, series_id: str, **_: Any) -> list[dict[str, str]]:
        return [
            {
                "date": "2024-01-02",
                "value": "5.2",
                "realtime_start": "2026-11-03",
                "realtime_end": "2026-11-03",
            }
        ]

    def vintages(self, series_id: str) -> list[dict[str, str]]:
        return [
            {
                "date": "2024-01-01",
                "value": "3.7",
                "realtime_start": "2024-02-02",
                "realtime_end": "2024-03-07",
            },
            {
                "date": "2024-01-01",
                "value": "3.8",
                "realtime_start": "2024-03-08",
                "realtime_end": "9999-12-31",
            },
        ]

    def close(self) -> None:
        pass


class FakeSharadar:
    def __init__(self, api_key: SecretStr) -> None:
        pass

    def sfp_prices(self, ticker: str) -> pl.DataFrame:
        return pl.DataFrame({"date": [date(2024, 1, 2)], "close": [1.0], "closeadj": [1.0]})

    def close(self) -> None:
        pass


@pytest.fixture
def stooq_dir(tmp_path: Path) -> Path:
    path = tmp_path / "stooq/d_us_txt/data/daily/us/nyse etfs/2/spy.us.txt"
    path.parent.mkdir(parents=True)
    path.write_text(STOOQ_HEADER + "SPY.US,D,20261030,000000,1,2,0.5,1.5,100,0\n", encoding="utf-8")
    return tmp_path / "stooq"


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(smoke, "FredClient", FakeFred)
    monkeypatch.setattr(smoke, "SharadarClient", FakeSharadar)
    monkeypatch.chdir(ROOT)


def _settings(tmp_path: Path, **kwargs: Any) -> Settings:
    return Settings(_env_file=None, regport_results_dir=tmp_path / "results", **kwargs)  # type: ignore[call-arg]


def test_smoke_with_stooq(fakes: None, tmp_path: Path, stooq_dir: Path) -> None:
    settings = _settings(tmp_path, fred_api_key=KEY, stooq_dir=stooq_dir)
    steps = smoke.run_smoke(ExperimentConfig(), settings)
    assert [s.name for s in steps] == ["fred", "alfred", "prices"]
    assert all(s.ok for s in steps), [s.line() for s in steps]
    assert steps[1].detail.endswith("2 vintages")
    assert steps[2].source == "stooq" and steps[2].rows == 1
    assert (tmp_path / "results/cache/smoke/prices_stooq_SPY.parquet").exists()
    assert "OK" in steps[2].line()


def test_smoke_auto_prefers_sharadar(fakes: None, tmp_path: Path) -> None:
    settings = _settings(tmp_path, fred_api_key=KEY, sharadar_api_key=KEY)
    steps = smoke.run_smoke(ExperimentConfig(), settings)
    assert steps[2].ok and steps[2].source == "sharadar"
    assert "closeadj" in steps[2].detail


def test_smoke_reports_missing_keys(fakes: None, tmp_path: Path) -> None:
    steps = smoke.run_smoke(ExperimentConfig(), _settings(tmp_path))
    assert not any(s.ok for s in steps)
    assert "FRED_API_KEY" in steps[0].detail
    assert "STOOQ_DIR" in steps[2].detail
    assert "FAIL" in steps[0].line()


def test_smoke_explicit_sources(fakes: None, tmp_path: Path, stooq_dir: Path) -> None:
    settings = _settings(tmp_path, fred_api_key=KEY, stooq_dir=tmp_path / "nowhere")
    for source, match in [
        ("sharadar", "SHARADAR_API_KEY"),
        ("tiingo", "not available"),
        ("stooq", "not found"),
    ]:
        cfg = ExperimentConfig.model_validate({"data": {"price_source": source}})
        step = smoke.run_smoke(cfg, settings)[2]
        assert not step.ok and match in step.detail


def test_smoke_missing_symbol(fakes: None, tmp_path: Path, stooq_dir: Path) -> None:
    (stooq_dir / "d_us_txt/data/daily/us/nyse etfs/2/spy.us.txt").rename(
        stooq_dir / "d_us_txt/qqq.us.txt"
    )
    step = smoke.run_smoke(ExperimentConfig(), _settings(tmp_path, stooq_dir=stooq_dir))[2]
    assert not step.ok and "spy.us not in" in step.detail


def test_smoke_source_failure_is_reported(
    fakes: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken(FakeFred):
        def observations(self, series_id: str, **_: Any) -> list[dict[str, str]]:
            raise DataSourceError("FRED HTTP 500")

    monkeypatch.setattr(smoke, "FredClient", Broken)
    steps = smoke.run_smoke(ExperimentConfig(), _settings(tmp_path, fred_api_key=KEY))
    assert not steps[0].ok and "HTTP 500" in steps[0].detail
    assert steps[1].ok


def test_cli_smoke_exit_codes(
    fakes: None, tmp_path: Path, stooq_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in ("FRED_API_KEY", "SHARADAR_API_KEY"):
        monkeypatch.setenv(var, "")
    monkeypatch.setenv("REGPORT_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("STOOQ_DIR", str(stooq_dir))
    runner = CliRunner()
    result = runner.invoke(app, ["smoke", "--config", "configs/base.yaml"])
    assert result.exit_code == 1 and "FRED_API_KEY" in result.output
    monkeypatch.setenv("FRED_API_KEY", KEY)
    result = runner.invoke(app, ["smoke"])
    assert result.exit_code == 0, result.output
    assert result.output.count("[OK  ]") == 3
    assert KEY not in result.output


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0 and "smoke" in result.output


def test_jsonl_logging(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "logs/run.jsonl"
    configure_logging("INFO", path)
    get_logger("t").info("refit", refit_id=3)
    get_logger("t").debug("hidden")
    configure_logging("INFO")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event"] == "refit" and event["refit_id"] == 3 and event["level"] == "info"
    assert "refit" in capsys.readouterr().err


@pytest.mark.data
def test_real_smoke() -> None:
    """Local only: needs FRED_API_KEY and a price path (M1 exit criterion)."""
    from regport.config import load_config

    settings = Settings()
    if not settings.has_key("fred"):
        pytest.skip("FRED_API_KEY not set")
    steps = smoke.run_smoke(load_config(ROOT / "configs/base.yaml"), settings)
    assert all(s.ok for s in steps), [s.line() for s in steps]
