"""AC-W-09: the trial log follows contract v1.0 and is append-only."""

from __future__ import annotations

import csv
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from regport.tracking import experiment_log as el
from regport.tracking.experiment_log import (
    COLUMNS,
    REQUIRED_COLUMNS,
    TrialLogError,
    TrialRow,
    append_trial,
    canonical_json,
    log_failed,
    log_trial,
    read_trial_log,
    sharpe_annual,
)

PARAMS = {"optimizer": "erc", "regime": {"usage": "none"}, "note": 'a "quoted", text'}


def _dates(n: int) -> list[date]:
    start = date(2020, 1, 2)
    return [start + timedelta(days=i) for i in range(n)]


def _row(**overrides: object) -> TrialRow:
    run_id = "3f2b8c1e-9a4d-4c2e-8f7a-1b2c3d4e5f60"
    fields: dict[str, object] = {
        "run_id": run_id,
        "timestamp": "2026-11-03T12:00:00Z",
        "git_sha": "a" * 40,
        "strategy_family": "regport_erc",
        "params_json": canonical_json(PARAMS),
        "start": "2020-01-02",
        "end": "2020-12-31",
        "n_obs": 250,
        "sharpe_annual": 0.5,
        "returns_path": f"runs/{run_id}/returns.parquet",
    }
    fields.update(overrides)
    return TrialRow(**fields)  # type: ignore[arg-type]


def test_columns_are_contract_order() -> None:
    assert COLUMNS[:10] == (
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
    assert COLUMNS[:10] == REQUIRED_COLUMNS
    assert all(c.startswith("x_") for c in COLUMNS[10:])


def test_log_trial_writes_series_and_row(tmp_path: Path) -> None:
    log = tmp_path / "results" / "trial_log.csv"
    rng = np.random.default_rng(1)
    ret = rng.normal(0.0004, 0.01, 300)
    dates = _dates(300)
    row = log_trial("regport_erc", PARAMS, dates, ret, path=log, extra={"ret_gross": ret + 1e-5})

    raw = log.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw
    assert raw.split(b"\n")[0].decode() == ",".join(COLUMNS)

    rows = read_trial_log(log)
    assert len(rows) == 1
    got = rows[0]
    assert got["run_id"] == row.run_id
    assert json.loads(got["params_json"]) == PARAMS
    assert got["params_json"] == canonical_json(PARAMS)
    assert got["start"] == "2020-01-02" and got["end"] == dates[-1].isoformat()
    assert got["n_obs"] == "300"
    assert got["x_returns_basis"] == "raw"
    assert got["returns_path"] == f"runs/{row.run_id}/returns.parquet"
    expected = float(np.mean(ret) / np.std(ret, ddof=1) * math.sqrt(252))
    assert float(got["sharpe_annual"]) == expected

    table = pq.read_table(log.parent / got["returns_path"])
    assert table.schema.field("date").type == "date32[day]"
    assert table.schema.field("ret").type == "double"
    assert table.num_rows == 300
    assert "ret_gross" in table.column_names


def test_params_json_is_rfc4180_quoted(tmp_path: Path) -> None:
    log = tmp_path / "trial_log.csv"
    append_trial(_row(), log)
    line = log.read_text(encoding="utf-8").splitlines()[1]
    assert '"{""note"":""a \\""quoted\\"", text""' in line
    with log.open(encoding="utf-8", newline="") as fh:
        parsed = list(csv.reader(fh))
    assert len(parsed[1]) == len(COLUMNS)


def test_append_only_keeps_earlier_bytes(tmp_path: Path) -> None:
    log = tmp_path / "trial_log.csv"
    log_trial("regport_hrp", PARAMS, _dates(10), np.linspace(-0.01, 0.01, 10), path=log)
    before = log.read_bytes()
    log_failed("regport_hrp", PARAMS, "solver failed", path=log)
    after = log.read_bytes()
    assert after.startswith(before)
    rows = read_trial_log(log)
    assert [r["n_obs"] for r in rows] == ["10", "0"]
    failed = rows[1]
    assert failed["sharpe_annual"] == "" and failed["returns_path"] == ""
    assert failed["start"] == "" and failed["x_note"] == "solver failed"


def test_header_written_once(tmp_path: Path) -> None:
    log = tmp_path / "trial_log.csv"
    for _ in range(3):
        log_failed("regport_erc", PARAMS, "x", path=log)
    text = log.read_text(encoding="utf-8")
    assert text.count("run_id,timestamp") == 1
    assert len(read_trial_log(log)) == 3


def test_refuses_foreign_header(tmp_path: Path) -> None:
    log = tmp_path / "trial_log.csv"
    log.write_text("run_id,timestamp\n", encoding="utf-8")
    with pytest.raises(TrialLogError, match="header"):
        append_trial(_row(), log)


def test_refuses_duplicate_run_id(tmp_path: Path) -> None:
    log = tmp_path / "trial_log.csv"
    append_trial(_row(), log)
    with pytest.raises(TrialLogError, match="duplicate"):
        append_trial(_row(), log)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"run_id": "not-a-uuid"}, "UUID"),
        ({"run_id": "3F2B8C1E-9A4D-4C2E-8F7A-1B2C3D4E5F60"}, "UUID"),
        ({"timestamp": "2026-11-03 12:00:00"}, "timestamp"),
        ({"git_sha": "abc"}, "git_sha"),
        ({"strategy_family": "Regport-ERC"}, "strategy_family"),
        ({"params_json": "{not json"}, "JSON"),
        ({"params_json": '{"b": 1, "a": 2}'}, "canonical"),
        ({"params_json": "[1,2]"}, "canonical"),
        ({"n_obs": -1}, "n_obs"),
        ({"start": "2020/01/02"}, "date"),
        ({"start": "2021-01-01"}, "after"),
        ({"sharpe_annual": None}, "finite"),
        ({"sharpe_annual": float("nan")}, "finite"),
        ({"returns_path": "elsewhere.parquet"}, "returns_path"),
        ({"x_returns_basis": "log"}, "x_returns_basis"),
        ({"n_obs": 0, "sharpe_annual": 1.0, "returns_path": ""}, "failed"),
    ],
)
def test_row_validation(overrides: dict[str, object], match: str) -> None:
    with pytest.raises(TrialLogError, match=match):
        _row(**overrides).validate()


def test_dirty_suffix_is_valid() -> None:
    _row(git_sha="b" * 40 + "-dirty").validate()


def test_sharpe_annual() -> None:
    assert sharpe_annual([0.01, 0.01, 0.01]) == 0.0
    assert sharpe_annual([0.01, -0.01]) == 0.0
    assert sharpe_annual([0.02, 0.0]) == pytest.approx(
        0.01 / np.std([0.02, 0.0], ddof=1) * math.sqrt(252)
    )
    with pytest.raises(TrialLogError):
        sharpe_annual([0.01])


def test_write_returns_rejects_bad_series(tmp_path: Path) -> None:
    target = tmp_path / "r.parquet"
    with pytest.raises(TrialLogError, match="length"):
        el.write_returns(target, _dates(3), [0.0, 0.1])
    with pytest.raises(TrialLogError, match="NaN"):
        el.write_returns(target, _dates(2), [0.0, float("nan")])
    with pytest.raises(TrialLogError, match="increasing"):
        el.write_returns(target, [date(2020, 1, 2)] * 2, [0.0, 0.1])


def test_git_sha_format(tmp_path: Path) -> None:
    sha = el.git_sha(Path("results/trial_log.csv"))
    assert el._GIT_SHA.match(sha)


def test_git_sha_outside_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise OSError("no git")

    monkeypatch.setattr(el.subprocess, "run", boom)
    assert el.git_sha(tmp_path / "log.csv") == "0" * 40 + "-dirty"


def test_git_sha_ignores_generated_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def git(*args: str) -> None:
        el.subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    for name in ("code.py", "reports/table.csv", "results/grids/g.json", "log.csv"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("a", encoding="utf-8")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "init")
    monkeypatch.chdir(tmp_path)
    for name in ("reports/table.csv", "results/grids/g.json", "log.csv"):
        (tmp_path / name).write_text("b", encoding="utf-8")
    assert not el.git_sha(tmp_path / "log.csv").endswith("-dirty")
    (tmp_path / "code.py").write_text("b", encoding="utf-8")
    assert el.git_sha(tmp_path / "log.csv").endswith("-dirty")


def test_git_sha_log_outside_repo(tmp_path: Path) -> None:
    assert el._GIT_SHA.match(el.git_sha(tmp_path / "elsewhere.csv"))


def test_utc_timestamp_format() -> None:
    assert el._TIMESTAMP.match(el.utc_timestamp())


def test_read_missing_log(tmp_path: Path) -> None:
    assert read_trial_log(tmp_path / "missing.csv") == []


def test_overfitlab_validates_our_log(tmp_path: Path) -> None:
    validate = pytest.importorskip("overfitlab.triallog").validate_trial_log
    log = tmp_path / "trial_log.csv"
    log_trial(
        "regport_erc", PARAMS, _dates(100), np.random.default_rng(2).normal(0, 0.01, 100), path=log
    )
    log_failed("regport_erc", PARAMS, "aborted", path=log)
    issues = validate(log)
    assert [i for i in issues if i.severity == "error"] == []
