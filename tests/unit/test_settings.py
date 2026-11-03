"""Settings: path resolution and secret masking (NFR-05)."""

from __future__ import annotations

from pathlib import Path

import pytest

from regport.settings import Settings

ENV_VARS = (
    "FRED_API_KEY",
    "SHARADAR_API_KEY",
    "TIINGO_API_KEY",
    "STOOQ_DIR",
    "DATA_ROOT",
    "REGPORT_DATA_DIR",
    "REGPORT_RESULTS_DIR",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


def test_defaults_without_env() -> None:
    s = Settings()
    assert s.data_dir == Path("data")
    assert s.results_dir == Path("results")
    assert s.reports_dir == Path("reports")
    assert not s.has_key("fred")


def test_dotenv_with_inline_comments(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "FRED_API_KEY=                 # required: comment\n"
        "SHARADAR_API_KEY=abc123secret   # set by the owner\n"
        "DATA_ROOT=\n"
        "REGPORT_LOG_LEVEL=DEBUG  # optional\n",
        encoding="utf-8",
    )
    s = Settings()
    assert s.fred_api_key is None
    assert s.has_key("sharadar")
    assert s.regport_log_level == "DEBUG"
    assert s.data_dir == Path("data")


def test_secrets_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "0123456789abcdef0123456789abcdef")
    s = Settings()
    assert "0123456789abcdef" not in repr(s)
    assert "0123456789abcdef" not in str(s.model_dump())
    assert s.has_key("fred")


def test_data_dir_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ROOT", "~/data/rpc")
    assert Settings().data_dir == Path("~/data/rpc").expanduser()
    monkeypatch.setenv("REGPORT_DATA_DIR", "elsewhere")
    assert Settings().data_dir == Path("elsewhere")


def test_whitespace_key_is_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIINGO_API_KEY", "   ")
    assert not Settings().has_key("tiingo")
