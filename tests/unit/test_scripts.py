"""Repository guards in ``scripts/`` (NFR-05, AC-W-09)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tracked_files_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    guard = _load("check_tracked_files")
    monkeypatch.chdir(tmp_path)
    big = tmp_path / "big.bin"
    big.write_bytes(b"0" * (5 * 1024 * 1024 + 1))
    found = guard.problems(
        [
            "data/raw/x.csv",
            "results/runs/a/returns.parquet",
            "x.parquet",
            ".env",
            "tests/fixtures/ok.parquet",
            "src/ok.py",
            "big.bin",
        ]
    )
    assert len(found) == 5
    assert any("big.bin" in f for f in found)
    assert not any(f.startswith(("tests/fixtures/", "src/ok.py")) for f in found)
    assert guard.main(["src/ok.py"]) == 0
    assert guard.main([".env"]) == 1


def test_repository_is_clean() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_tracked_files.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout


def test_append_only_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = _load("check_trial_log_append_only")

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.org")
    git("config", "user.name", "t")
    git("config", "core.autocrlf", "false")
    log = tmp_path / "results/trial_log.csv"
    log.parent.mkdir()
    monkeypatch.chdir(tmp_path)
    log.write_text("h\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-qm", "base")
    assert script.main(["HEAD~0"]) == 0
    log.write_text("h\nrow1\n", encoding="utf-8")
    assert script.main(["HEAD"]) == 0
    log.write_text("x\n", encoding="utf-8")
    assert script.main(["HEAD"]) == 1
    assert script.main(["no-such-ref"]) == 0
