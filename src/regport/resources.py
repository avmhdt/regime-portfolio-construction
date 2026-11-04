"""Memory guard for heavy stages (owner rule, 2026-11-03).

Live recorders and other projects share this PC. Before a heavy parallel stage starts, free
physical memory must be at least ``REGPORT_MIN_FREE_GB`` (default 30 GB); otherwise the stage
waits, checking every minute, and gives up with an error after ``REGPORT_MEMORY_WAIT_MIN``
minutes (default 30). Worker counts come from ``REGPORT_N_JOBS``; the owner's rule is at most 4
while another session runs heavy jobs.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from collections.abc import Callable

from regport.logs import get_logger

log = get_logger(__name__)


class MemoryGuardError(RuntimeError):
    """Free memory stayed below the floor for longer than the allowed wait."""


def free_memory_gb() -> float:
    """Available physical memory in GB (Windows: GlobalMemoryStatusEx; Linux: /proc/meminfo)."""
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return float(status.ullAvailPhys) / 2**30
    with open("/proc/meminfo", encoding="ascii") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 2**20
    return float("inf")  # pragma: no cover - unknown platform


def wait_for_memory(
    stage: str,
    min_gb: float | None = None,
    max_wait_min: float | None = None,
    probe: Callable[[], float] = free_memory_gb,
    sleep: Callable[[float], None] = time.sleep,
) -> float:
    """Block until free memory >= ``min_gb``; returns the free GB seen. Raises after the wait."""
    floor = min_gb if min_gb is not None else float(os.environ.get("REGPORT_MIN_FREE_GB", "30"))
    limit = (
        max_wait_min
        if max_wait_min is not None
        else float(os.environ.get("REGPORT_MEMORY_WAIT_MIN", "30"))
    )
    waited = 0.0
    while True:
        free = probe()
        if free >= floor:
            return free
        if waited >= limit:
            raise MemoryGuardError(
                f"{stage}: free memory {free:.1f} GB stayed below {floor:g} GB for {limit:g} min"
            )
        log.warning("memory_guard_wait", stage=stage, free_gb=round(free, 1), floor_gb=floor)
        sleep(60.0)
        waited += 1.0
