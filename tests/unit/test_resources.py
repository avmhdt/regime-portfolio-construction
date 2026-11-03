"""The memory guard waits while free memory is below the floor, then proceeds or gives up."""

from __future__ import annotations

import pytest

from regport.resources import MemoryGuardError, free_memory_gb, wait_for_memory


def test_free_memory_is_positive() -> None:
    assert free_memory_gb() > 0


def test_waits_then_proceeds() -> None:
    readings = iter([10.0, 20.0, 45.0])
    sleeps: list[float] = []
    free = wait_for_memory("t", 30.0, 5.0, probe=lambda: next(readings), sleep=sleeps.append)
    assert free == 45.0 and sleeps == [60.0, 60.0]


def test_gives_up_after_the_limit() -> None:
    sleeps: list[float] = []
    with pytest.raises(MemoryGuardError, match="below 30 GB for 2 min"):
        wait_for_memory("t", 30.0, 2.0, probe=lambda: 5.0, sleep=sleeps.append)
    assert len(sleeps) == 2


def test_environment_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REGPORT_MIN_FREE_GB", "1")
    assert wait_for_memory("t", probe=lambda: 2.0) == 2.0
