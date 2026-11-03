"""Environment settings loaded from ``.env`` (pydantic-settings).

API keys are ``SecretStr``: they never appear in ``repr``, logs or error messages (NFR-05).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide settings. Paths are resolved relative to the current working directory."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    fred_api_key: SecretStr | None = None
    sharadar_api_key: SecretStr | None = None
    tiingo_api_key: SecretStr | None = None
    stooq_dir: Path | None = None
    data_root: str | None = None
    regport_data_dir: Path | None = None
    regport_results_dir: Path = Path("results")
    regport_reports_dir: Path = Path("reports")
    regport_log_level: Literal["DEBUG", "INFO", "WARNING"] = "INFO"
    regport_http_timeout_s: float = Field(default=30.0, gt=0)
    regport_n_jobs: int = Field(default=1, ge=1)

    @field_validator(
        "fred_api_key",
        "sharadar_api_key",
        "tiingo_api_key",
        "stooq_dir",
        "data_root",
        "regport_data_dir",
        mode="before",
    )
    @classmethod
    def _empty_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def data_dir(self) -> Path:
        """Root of ``raw/``, ``interim/`` and ``processed/`` (§7d).

        ``REGPORT_DATA_DIR`` wins; else ``DATA_ROOT`` with ``~`` expanded; else ``./data``.
        """
        if self.regport_data_dir is not None:
            return self.regport_data_dir
        if self.data_root is not None:
            return Path(self.data_root).expanduser()
        return Path("data")

    @property
    def results_dir(self) -> Path:
        return self.regport_results_dir

    @property
    def reports_dir(self) -> Path:
        return self.regport_reports_dir

    def has_key(self, name: Literal["fred", "sharadar", "tiingo"]) -> bool:
        """True if the key is set and non-empty (the value itself is never exposed)."""
        secret: SecretStr | None = getattr(self, f"{name}_api_key")
        return secret is not None and bool(secret.get_secret_value().strip())


def get_settings() -> Settings:
    """Load settings from the environment and ``.env``."""
    return Settings()
