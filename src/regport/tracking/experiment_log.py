"""Append-only trial log in backtest-overfitting-lab's trial-log contract v1.0 (FR-25, AC-W-09).

Contract: https://github.com/avmhdt/backtest-overfitting-lab/blob/main/docs/trial-log.md.
The ten required columns come first, in order; this project adds ``x_returns_basis`` (always
``raw``: ``ret`` is the daily simple net return, not an excess return) and ``x_note``.
The CSV is UTF-8 without BOM, comma separated, RFC 4180 quoted, LF line endings.
"""

from __future__ import annotations

import csv
import io
import itertools
import json
import math
import re
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

REQUIRED_COLUMNS: tuple[str, ...] = (
    "run_id",
    "timestamp",
    "git_sha",
    "strategy_family",
    "params_json",
    "start",
    "end",
    "n_obs",
    "sharpe_annual",
    "returns_path",
)
EXTRA_COLUMNS: tuple[str, ...] = ("x_returns_basis", "x_note")
COLUMNS: tuple[str, ...] = REQUIRED_COLUMNS + EXTRA_COLUMNS

PERIODS_PER_YEAR = 252
DEFAULT_LOG_PATH = Path("results/trial_log.csv")

_RUN_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}(-dirty)?$")
_FAMILY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class TrialLogError(ValueError):
    """A row or file violates trial-log contract v1.0."""


@dataclass(frozen=True)
class TrialRow:
    """One trial: a backtest invocation, successful or failed."""

    run_id: str
    timestamp: str
    git_sha: str
    strategy_family: str
    params_json: str
    start: str
    end: str
    n_obs: int
    sharpe_annual: float | None
    returns_path: str
    x_returns_basis: str = "raw"
    x_note: str = ""

    def validate(self) -> None:
        """Raise ``TrialLogError`` if any field breaks the contract's format rules."""
        if not _RUN_ID.match(self.run_id):
            raise TrialLogError(f"run_id is not a lower-case UUID v4: {self.run_id!r}")
        if not _TIMESTAMP.match(self.timestamp):
            raise TrialLogError(f"bad timestamp: {self.timestamp!r}")
        if not _GIT_SHA.match(self.git_sha):
            raise TrialLogError(f"bad git_sha: {self.git_sha!r}")
        if not _FAMILY.match(self.strategy_family):
            raise TrialLogError(f"bad strategy_family: {self.strategy_family!r}")
        try:
            parsed = json.loads(self.params_json)
        except json.JSONDecodeError as exc:
            raise TrialLogError("params_json is not valid JSON") from exc
        if not isinstance(parsed, dict) or canonical_json(parsed) != self.params_json:
            raise TrialLogError("params_json is not a canonical JSON object")
        if self.n_obs < 0:
            raise TrialLogError("n_obs must be >= 0")
        if self.n_obs == 0:
            if self.sharpe_annual is not None or self.returns_path:
                raise TrialLogError("failed rows have empty sharpe_annual and returns_path")
            return
        for value in (self.start, self.end):
            if not _DATE.match(value):
                raise TrialLogError(f"bad date: {value!r}")
        if self.start > self.end:
            raise TrialLogError("start is after end")
        if self.sharpe_annual is None or not math.isfinite(self.sharpe_annual):
            raise TrialLogError("sharpe_annual must be finite on a row with a series")
        if self.returns_path != f"runs/{self.run_id}/returns.parquet":
            raise TrialLogError("returns_path must be runs/<run_id>/returns.parquet")
        if self.x_returns_basis not in ("raw", "excess"):
            raise TrialLogError("x_returns_basis must be raw or excess")

    def as_csv_dict(self) -> dict[str, str]:
        out = {k: ("" if v is None else str(v)) for k, v in asdict(self).items()}
        if self.sharpe_annual is not None:
            out["sharpe_annual"] = repr(float(self.sharpe_annual))
        return out


def canonical_json(params: Mapping[str, Any]) -> str:
    """The contract's canonical JSON: sorted keys, compact, ASCII, no NaN."""
    return json.dumps(
        params, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def new_run_id() -> str:
    return str(uuid.uuid4())


def utc_timestamp(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_sha(log_path: Path = DEFAULT_LOG_PATH) -> str:
    """HEAD commit, with ``-dirty`` if tracked files other than the log itself changed.

    Outside a git checkout the all-zero hash with ``-dirty`` is returned, so a run is still logged.
    """

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout

    try:
        sha = git("rev-parse", "HEAD").strip()
        root = Path(git("rev-parse", "--show-toplevel").strip())
        try:
            log_rel = log_path.resolve().relative_to(root.resolve()).as_posix()
            pathspec = [".", f":(exclude){log_rel}"]
        except ValueError:
            pathspec = ["."]
        status = git("status", "--porcelain", "--untracked-files=no", "--", *pathspec)
    except (OSError, subprocess.CalledProcessError):
        return "0" * 40 + "-dirty"
    return f"{sha}-dirty" if status.strip() else sha


def sharpe_annual(ret: npt.ArrayLike, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Contract Sharpe: ``mean(ret) / std(ret, ddof=1) * sqrt(q)``; 0.0 if the std is 0."""
    arr = np.asarray(ret, dtype=np.float64)
    if arr.size < 2:
        raise TrialLogError("need at least two returns for a Sharpe ratio")
    std = float(np.std(arr, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(arr)) / std * math.sqrt(periods_per_year)


def _header_line() -> str:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerow(COLUMNS)
    return buf.getvalue()


def append_trial(row: TrialRow, path: Path = DEFAULT_LOG_PATH) -> None:
    """Validate ``row`` and append it; write the header first if the file is new.

    The file is opened in append mode only, so earlier rows are never rewritten. An existing file
    whose header differs from ``COLUMNS`` is refused rather than silently extended.
    """
    row.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    if not is_new:
        with path.open(encoding="utf-8", newline="") as fh:
            first = fh.readline()
        if first != _header_line():
            raise TrialLogError(f"{path}: header does not match trial-log contract v1.0")
        if row.run_id in {r["run_id"] for r in read_trial_log(path)}:
            raise TrialLogError(f"duplicate run_id {row.run_id}")
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerow(row.as_csv_dict())


def read_trial_log(path: Path = DEFAULT_LOG_PATH) -> list[dict[str, str]]:
    """All rows as strings, in file order."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_returns(
    path: Path,
    dates: Sequence[date],
    ret: npt.ArrayLike,
    extra: Mapping[str, npt.ArrayLike] | None = None,
) -> None:
    """Write ``returns.parquet`` with ``date`` (date32), ``ret`` (float64) and extra f64 columns."""
    values = np.asarray(ret, dtype=np.float64)
    if len(dates) != values.size:
        raise TrialLogError("dates and ret differ in length")
    if not np.all(np.isfinite(values)):
        raise TrialLogError("ret contains NaN or inf")
    if any(b <= a for a, b in itertools.pairwise(dates)):
        raise TrialLogError("dates must be strictly increasing")
    columns: dict[str, pa.Array] = {
        "date": pa.array(list(dates), pa.date32()),
        "ret": pa.array(values, pa.float64()),
    }
    for name, col in (extra or {}).items():
        columns[name] = pa.array(np.asarray(col, dtype=np.float64), pa.float64())
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)


def log_trial(
    family: str,
    params: Mapping[str, Any],
    dates: Sequence[date],
    ret: npt.ArrayLike,
    *,
    path: Path = DEFAULT_LOG_PATH,
    run_id: str | None = None,
    extra: Mapping[str, npt.ArrayLike] | None = None,
    note: str = "",
) -> TrialRow:
    """Write ``runs/<run_id>/returns.parquet`` next to the log, then append the row."""
    run_id = run_id or new_run_id()
    values = np.asarray(ret, dtype=np.float64)
    relative = f"runs/{run_id}/returns.parquet"
    write_returns(path.parent / relative, dates, values, extra)
    row = TrialRow(
        run_id=run_id,
        timestamp=utc_timestamp(),
        git_sha=git_sha(path),
        strategy_family=family,
        params_json=canonical_json(params),
        start=dates[0].isoformat(),
        end=dates[-1].isoformat(),
        n_obs=int(values.size),
        sharpe_annual=sharpe_annual(values),
        returns_path=relative,
        x_note=note,
    )
    append_trial(row, path)
    return row


def log_failed(
    family: str,
    params: Mapping[str, Any],
    note: str,
    *,
    path: Path = DEFAULT_LOG_PATH,
    run_id: str | None = None,
) -> TrialRow:
    """Log a failed or aborted run: ``n_obs = 0``, empty Sharpe and path. It still counts in N."""
    row = TrialRow(
        run_id=run_id or new_run_id(),
        timestamp=utc_timestamp(),
        git_sha=git_sha(path),
        strategy_family=family,
        params_json=canonical_json(params),
        start="",
        end="",
        n_obs=0,
        sharpe_annual=None,
        returns_path="",
        x_note=note,
    )
    append_trial(row, path)
    return row
