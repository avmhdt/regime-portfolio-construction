"""FR-01 universe validation and strict experiment configs (§10)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from regport.config import ExperimentConfig, Universe, load_config, load_universe

ROOT = Path(__file__).resolve().parents[2]


def test_base_config_matches_spec_defaults() -> None:
    cfg = load_config(ROOT / "configs/base.yaml")
    assert cfg == ExperimentConfig()
    assert cfg.sample.oos_start == date(2005, 1, 3)
    assert cfg.optimizer.max_weight == 0.40
    assert cfg.execution.threshold_abs == 0.02
    assert cfg.evaluation.stress_windows["gfc_2008"] == (date(2007, 10, 9), date(2009, 3, 9))


def test_unknown_key_is_error(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "configs/base.yaml").read_text(encoding="utf-8"))
    raw["risk"]["windw_days"] = 10
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError, match="windw_days"):
        load_config(path)


def test_non_mapping_config(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_sample_order_checked() -> None:
    with pytest.raises(ValidationError, match="data_start"):
        ExperimentConfig.model_validate({"sample": {"oos_start": "1990-01-02"}})
    with pytest.raises(ValidationError, match="oos_end"):
        ExperimentConfig.model_validate({"sample": {"oos_end": "2004-01-02"}})


def test_config_hash_stable_and_sensitive() -> None:
    a = ExperimentConfig()
    assert a.config_hash() == ExperimentConfig().config_hash()
    b = ExperimentConfig.model_validate({"costs": {"override_bps": 10}})
    assert a.config_hash() != b.config_hash()
    assert a.to_params()["sample"]["oos_start"] == "2005-01-03"


def test_universe_is_the_twelve_etfs() -> None:
    universe = load_universe(ROOT / "configs/universe.yaml")
    assert universe.symbols == [
        "SPY",
        "EFA",
        "EEM",
        "VNQ",
        "IEF",
        "TLT",
        "TIP",
        "LQD",
        "HYG",
        "GLD",
        "DBC",
        "UUP",
    ]
    assert universe.get("HYG").cost_bps == 3
    assert universe.get("UUP").stooq == "uup.us"
    assert all(1 <= a.cost_bps <= 5 for a in universe.assets)
    with pytest.raises(KeyError):
        universe.get("QQQ")


def _assets() -> list[dict[str, object]]:
    raw = yaml.safe_load((ROOT / "configs/universe.yaml").read_text(encoding="utf-8"))
    assets: list[dict[str, object]] = raw["assets"]
    return assets


def test_universe_size_limits() -> None:
    with pytest.raises(ValidationError, match="10 to 20"):
        Universe.model_validate({"assets": _assets()[:9]})
    many = _assets() + [dict(a, symbol=f"X{i}") for i, a in enumerate(_assets())]
    with pytest.raises(ValidationError, match="10 to 20"):
        Universe.model_validate({"assets": many[:21]})


def test_universe_requires_all_classes() -> None:
    no_fx = [a for a in _assets() if a["asset_class"] != "fx"]
    no_fx.append(dict(no_fx[0], symbol="IWM"))
    with pytest.raises(ValidationError, match="fx"):
        Universe.model_validate({"assets": no_fx})


def test_universe_rejects_duplicates() -> None:
    assets = [*_assets()[:11], _assets()[0]]
    with pytest.raises(ValidationError, match="duplicate"):
        Universe.model_validate({"assets": assets})
