"""AC-W-05, AC-W-06 on the owner's real data (local only, marker ``data``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from regport.config import load_config, load_universe
from regport.data.checks import run_checks
from regport.settings import Settings

pytestmark = pytest.mark.data
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def report():  # type: ignore[no-untyped-def]
    settings = Settings()
    if not (settings.data_dir / "processed" / "macro_pit.parquet").exists():
        pytest.skip("run `regport fetch all` and `regport data build` first")
    cfg = load_config(ROOT / "configs/base.yaml")
    return run_checks(cfg, settings, load_universe(ROOT / cfg.data.universe_file))


def _check(report, check_id: str):  # type: ignore[no-untyped-def]
    return next(c for c in report.checks if c.id == check_id)


def test_payems_vintages_used(report) -> None:  # type: ignore[no-untyped-def]
    """AC-W-06: PAYEMS PIT differs from the latest value in >= 50% of months since 2000."""
    assert _check(report, "DQ-09").passed


def test_pit_and_lags(report) -> None:  # type: ignore[no-untyped-def]
    for check_id in ("DQ-01", "DQ-02", "DQ-04", "DQ-07", "DQ-12"):
        assert _check(report, check_id).passed, check_id


def test_adjustment_includes_distributions(report) -> None:  # type: ignore[no-untyped-def]
    dq03 = _check(report, "DQ-03")
    adjustment = [r for r in dq03.table if "test" in r]
    assert adjustment and all(r["pass"] for r in adjustment)
