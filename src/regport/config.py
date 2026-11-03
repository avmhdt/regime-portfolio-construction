"""Pydantic models for the YAML experiment and universe configs (spec §10).

Every model forbids unknown keys, so a typo in a config is an error rather than a silent default.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REQUIRED_ASSET_CLASSES = frozenset({"equity", "rates", "credit", "commodity", "fx"})


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Strict):
    price_source: Literal["auto", "sharadar", "tiingo", "stooq"] = "auto"
    universe_file: Path = Path("configs/universe.yaml")
    calendar_symbol: str = "SPY"
    min_history_days: int = Field(default=252, ge=1)
    max_ffill_days: int = Field(default=3, ge=0)
    risk_free_series: str = "DTB3"


class MacroConfig(_Strict):
    daily_series: list[str] = ["DTB3", "BAA10Y", "T10Y3M", "VIXCLS"]
    daily_lag_bdays: int = Field(default=1, ge=0)
    vintage_series: list[str] = ["UNRATE", "PAYEMS", "INDPRO", "CPIAUCSL"]
    vintage_lag_bdays: int = Field(default=1, ge=0)


class SampleConfig(_Strict):
    data_start: date = date(1993, 2, 1)
    oos_start: date = date(2005, 1, 3)
    oos_end: date | None = None
    estimator_selection_end: date = date(2014, 12, 31)

    @model_validator(mode="after")
    def _ordered(self) -> SampleConfig:
        if not self.data_start < self.oos_start < self.estimator_selection_end:
            raise ValueError("need data_start < oos_start < estimator_selection_end")
        if self.oos_end is not None and self.oos_end <= self.oos_start:
            raise ValueError("oos_end must be after oos_start")
        return self


class RiskConfig(_Strict):
    estimator: Literal["sample", "ewma", "ledoit_wolf", "oas"] = "ledoit_wolf"
    window_days: int = Field(default=504, ge=2)
    ewma_half_life_days: float = Field(default=63, gt=0)
    eval_horizons_days: list[int] = [1, 5, 21]
    n_random_portfolios: int = Field(default=100, ge=0)


class HMMConfig(_Strict):
    feature_set: Literal["A", "B"] = "A"
    n_states: int | Literal["auto"] = "auto"
    candidate_states: list[int] = [2, 3, 4]
    covariance_type: Literal["full"] = "full"
    window: Literal["expanding", "rolling"] = "expanding"
    rolling_days: int = Field(default=2520, ge=1)
    min_train_days: int = Field(default=2520, ge=1)
    refit_every_days: int = Field(default=21, ge=1)
    n_init: int = Field(default=10, ge=1)
    max_iter: int = Field(default=500, ge=1)
    tol: float = Field(default=1.0e-6, gt=0)
    state_cov_shrinkage: float = Field(default=0.5, ge=0, le=1)
    # ALFRED values as published (headline) or as revised today (AC-H-07 contrast only).
    macro_vintage: Literal["pit", "latest"] = "pit"


class NormalGammaPriorConfig(_Strict):
    mu0: float = 0.0
    kappa0: float = Field(default=1.0, gt=0)
    alpha0: float = Field(default=2.0, gt=0)
    beta0: float = Field(default=1.0e-4, gt=0)


class BOCPDConfig(_Strict):
    series: str = "spy_logret"
    hazard_lambda_days: float = Field(default=250, gt=1)
    prior: NormalGammaPriorConfig = NormalGammaPriorConfig()
    max_run_length: int = Field(default=1500, ge=2)
    short_run_days: int = Field(default=21, ge=1)
    trigger_prob: float = Field(default=0.5, gt=0, lt=1)
    fast_ewma_half_life_days: float = Field(default=10, gt=0)


class RegimeConfig(_Strict):
    usage: Literal["none", "hmm", "bocpd", "hmm_bocpd"] = "hmm"
    horizon_days: int = Field(default=21, ge=1)
    hmm: HMMConfig = HMMConfig()
    bocpd: BOCPDConfig = BOCPDConfig()


class MVConfig(_Strict):
    risk_aversion: float = Field(default=5.0, gt=0)
    turnover_penalty_mult: float = Field(default=1.0, ge=0)
    mean_estimator: Literal["james_stein", "sample"] = "james_stein"
    mean_window_days: int = Field(default=252, ge=2)


class HRPConfig(_Strict):
    linkage: Literal["single", "complete", "average", "ward"] = "single"


class SolverConfig(_Strict):
    primary: str = "CLARABEL"
    fallback: str = "OSQP"


class OptimizerConfig(_Strict):
    name: Literal["min_variance", "erc", "hrp", "mv_turnover", "equal_weight"] = "erc"
    long_only: Literal[True] = True
    max_weight: float = Field(default=0.40, gt=0, le=1)
    mv: MVConfig = MVConfig()
    hrp: HRPConfig = HRPConfig()
    solver: SolverConfig = SolverConfig()


class ExecutionConfig(_Strict):
    frequency: Literal["monthly", "weekly"] = "monthly"
    policy: Literal["calendar", "threshold"] = "threshold"
    threshold_abs: float = Field(default=0.02, ge=0)
    lag_days: int = Field(default=1, ge=1)


class CostsConfig(_Strict):
    model: Literal["proportional"] = "proportional"
    override_bps: float | None = Field(default=None, ge=0)


class SharpeTestConfig(_Strict):
    methods: list[Literal["bootstrap", "hac"]] = ["bootstrap", "hac"]
    n_boot: int = Field(default=5000, ge=1)
    block_size: int | Literal["auto"] = "auto"
    kernel: Literal["parzen"] = "parzen"
    prewhiten: bool = True


class EvaluationConfig(_Strict):
    benchmark: Literal["equal_weight"] = "equal_weight"
    sharpe_test: SharpeTestConfig = SharpeTestConfig()
    multiple_testing: Literal["holm"] = "holm"
    ceq_risk_aversion: float = Field(default=1.0, gt=0)
    stress_windows: dict[str, tuple[date, date]] = {
        "gfc_2008": (date(2007, 10, 9), date(2009, 3, 9)),
        "covid_2020": (date(2020, 2, 19), date(2020, 3, 23)),
        "rates_2022": (date(2022, 1, 3), date(2022, 10, 12)),
    }


class LoggingConfig(_Strict):
    experiment_log: Path = Path("results/trial_log.csv")


class ExperimentConfig(_Strict):
    """Fully resolved experiment configuration; its hash identifies a run's settings."""

    project: str = "regime-portfolio-construction"
    seed: int = 20261030
    data: DataConfig = DataConfig()
    macro: MacroConfig = MacroConfig()
    sample: SampleConfig = SampleConfig()
    risk: RiskConfig = RiskConfig()
    regime: RegimeConfig = RegimeConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    execution: ExecutionConfig = ExecutionConfig()
    costs: CostsConfig = CostsConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    logging: LoggingConfig = LoggingConfig()

    def to_params(self) -> dict[str, Any]:
        """JSON-compatible dict (dates as ISO strings, paths as POSIX strings)."""
        params: dict[str, Any] = json.loads(self.model_dump_json())
        return params

    def config_hash(self) -> str:
        """SHA-256 (first 16 hex chars) of the canonical JSON of the resolved config."""
        canonical = json.dumps(
            self.to_params(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("ascii")).hexdigest()[:16]


class Asset(_Strict):
    symbol: str = Field(pattern=r"^[A-Z][A-Z0-9.]{0,9}$")
    asset_class: Literal["equity", "rates", "credit", "commodity", "fx"]
    exposure: str
    inception: date
    sharadar: str
    tiingo: str
    stooq: str
    cost_bps: float = Field(ge=0)


class Universe(_Strict):
    """FR-01: 10 to 20 assets covering equity, rates, credit, commodity and FX."""

    assets: list[Asset]

    @model_validator(mode="after")
    def _valid(self) -> Universe:
        n = len(self.assets)
        if not 10 <= n <= 20:
            raise ValueError(f"universe must have 10 to 20 assets, got {n}")
        symbols = [a.symbol for a in self.assets]
        if len(set(symbols)) != n:
            raise ValueError("duplicate symbols in universe")
        missing = REQUIRED_ASSET_CLASSES - {a.asset_class for a in self.assets}
        if missing:
            raise ValueError(f"universe is missing asset classes: {sorted(missing)}")
        return self

    @property
    def symbols(self) -> list[str]:
        return [a.symbol for a in self.assets]

    def get(self, symbol: str) -> Asset:
        for asset in self.assets:
            if asset.symbol == symbol:
                return asset
        raise KeyError(symbol)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return loaded


def load_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment config; unknown keys raise ``ValidationError``."""
    return ExperimentConfig.model_validate(_read_yaml(path))


def load_universe(path: Path) -> Universe:
    """Load and validate ``configs/universe.yaml`` (FR-01)."""
    return Universe.model_validate(_read_yaml(path))
