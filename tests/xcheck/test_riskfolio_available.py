"""R-09: riskfolio-lib installs next to the locked cvxpy and NumPy (FR-32 comparisons from M4)."""

from __future__ import annotations

from importlib.metadata import version

import pytest

pytestmark = pytest.mark.xcheck


def test_riskfolio_imports() -> None:
    rp = pytest.importorskip("riskfolio")
    assert hasattr(rp, "Portfolio") and hasattr(rp, "HCPortfolio")
    major, minor = (int(p) for p in version("riskfolio-lib").split(".")[:2])
    assert (major, minor) >= (6, 3)
