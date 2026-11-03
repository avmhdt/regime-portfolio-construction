"""Markdown and CSV table helpers for generated reports."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl


def cell(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "**no**"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def md_table(
    frame: pl.DataFrame, columns: Sequence[str] | None = None, digits: int = 3
) -> list[str]:
    """A GitHub Markdown table of ``frame`` (selected ``columns``, floats rounded)."""
    cols = list(columns or frame.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in frame.select(cols).iter_rows():
        out.append("| " + " | ".join(cell(v, digits) for v in row) + " |")
    return out


def write_csv(frame: pl.DataFrame, path: Path, digits: int = 10) -> Path:
    """CSV with floats rounded to ``digits`` (NFR-02 compares tables after rounding to 1e-10)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rounded = frame.with_columns(pl.col(pl.Float64).round(digits))
    path.write_text(rounded.write_csv(line_terminator="\n"), encoding="utf-8", newline="")
    return path
