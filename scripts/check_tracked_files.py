"""Data guard (NFR-05, NFR-11): fail if git tracks data, secrets or any file over 5 MB.

Usage: ``python scripts/check_tracked_files.py`` (all tracked files) or with explicit paths, as the
pre-commit hook passes staged files.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MAX_BYTES = 5 * 1024 * 1024
BANNED_PREFIXES = (
    "data/",
    "results/runs/",
    "results/regime/",
    "results/cov_eval/",
    "results/logs/",
    "results/cache/",
)
BANNED_SUFFIXES = (".parquet", ".feather", ".h5", ".xlsx", ".zip", ".gz")
BANNED_NAMES = (".env",)
ALLOWED_PREFIXES = ("tests/fixtures/",)


def problems(paths: list[str]) -> list[str]:
    found = []
    for name in paths:
        posix = name.replace("\\", "/")
        if posix.startswith(ALLOWED_PREFIXES):
            continue
        if posix.startswith(BANNED_PREFIXES):
            found.append(f"{posix}: data and run outputs never go into git")
        elif posix.endswith(BANNED_SUFFIXES):
            found.append(f"{posix}: data file type outside tests/fixtures/")
        elif Path(posix).name in BANNED_NAMES:
            found.append(f"{posix}: secrets file")
        path = Path(name)
        if path.is_file() and path.stat().st_size > MAX_BYTES:
            found.append(f"{posix}: larger than 5 MB")
    return found


def main(argv: list[str]) -> int:
    paths = (
        argv
        or subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, check=True
        ).stdout.splitlines()
    )
    found = problems(paths)
    for line in found:
        print(line)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
