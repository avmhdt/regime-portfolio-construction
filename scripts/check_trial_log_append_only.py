"""AC-W-09: the committed trial log only ever grows.

Usage: ``python scripts/check_trial_log_append_only.py <base-ref>``. Fails if the version of
``results/trial_log.csv`` at ``<base-ref>`` is not a byte prefix of the working-tree version.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

LOG = "results/trial_log.csv"


def main(argv: list[str]) -> int:
    base = argv[0] if argv else "origin/main"
    old = subprocess.run(["git", "show", f"{base}:{LOG}"], capture_output=True, check=False)
    if old.returncode != 0:
        print(f"{LOG} does not exist at {base}; nothing to compare")
        return 0
    new = Path(LOG).read_bytes() if Path(LOG).exists() else b""
    if not new.startswith(old.stdout):
        print(f"{LOG}: rows at {base} were edited or deleted; the log is append-only")
        return 1
    added = new[len(old.stdout) :].count(b"\n")
    print(f"{LOG}: append-only check passed ({added} new lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
