"""Leakage guard (FR-15, DQ-08): an AST scan for hmmlearn calls that use the full sequence.

``predict``, ``predict_proba``, ``decode`` and ``score_samples`` return smoothed or Viterbi
posteriors that condition on data after t. The rules:

1. ``predict``, ``predict_proba`` and ``score_samples`` may not be called anywhere in ``src/``,
   whatever the receiver, nor reached through ``getattr`` with a literal name.
2. hmmlearn may be imported only in ``regport/regime/hmm.py``, and ``decode`` may not be called
   in any module that imports hmmlearn. (``bytes.decode`` elsewhere stays legal.)

Run as ``python -m regport.regime.guard [paths...]``; exits 1 and lists every finding.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

BANNED_EVERYWHERE = frozenset({"predict", "predict_proba", "score_samples"})
BANNED_WITH_HMMLEARN = frozenset({"decode"})
HMMLEARN_ALLOWED_MODULE = ("regport", "regime", "hmm.py")


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


def _imports_hmmlearn(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "hmmlearn" for alias in node.names):
                lines.append(node.lineno)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] == "hmmlearn"
        ):
            lines.append(node.lineno)
    return lines


def scan_source(source: str, path: Path) -> list[Finding]:
    """All guard findings for one module's source text."""
    tree = ast.parse(source, filename=str(path))
    findings: list[Finding] = []
    hmm_lines = _imports_hmmlearn(tree)
    if hmm_lines and path.parts[-3:] != HMMLEARN_ALLOWED_MODULE:
        findings += [
            Finding(path, line, "hmmlearn may only be imported in regport/regime/hmm.py")
            for line in hmm_lines
        ]
    banned = BANNED_EVERYWHERE | (BANNED_WITH_HMMLEARN if hmm_lines else frozenset())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in banned:
            findings.append(Finding(path, node.lineno, f"banned call .{func.attr}()"))
        elif (
            isinstance(func, ast.Name)
            and func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in banned
        ):
            findings.append(Finding(path, node.lineno, f"banned getattr {node.args[1].value!r}"))
    return findings


def iter_python_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(path.rglob("*.py"))
        elif path.suffix == ".py":
            yield path


def scan_paths(paths: Iterable[Path]) -> list[Finding]:
    """Scan files and directories (recursively); sorted findings."""
    findings: list[Finding] = []
    for file in iter_python_files(paths):
        findings += scan_source(file.read_text(encoding="utf-8"), file)
    return findings


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv) or ["src"]
    findings = scan_paths(Path(a) for a in args)
    for finding in findings:
        print(finding)
    return 1 if findings else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
