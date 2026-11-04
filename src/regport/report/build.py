"""``regport report`` and ``regport reproduce`` (FR-33, AC-W-04).

``reports/manifest.json`` records the SHA-256 of every generated table in ``reports/tables/``.
``--check`` compares a fresh run with the committed manifest: a table passes if its hash matches,
or if every float differs by at most 1e-9 and every other cell is equal (NFR-02 allows rounding
noise at 1e-10). Timing tables and the trial summary are recorded but not compared.

The README's results live between ``<!-- regport:NAME:start -->`` and ``<!-- regport:NAME:end -->``
markers and are regenerated from the tables, so no number in them is typed by hand.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl

# Recorded but not compared: wall-clock timings vary between runs, and the trial summary grows
# with every rerun because reproduce itself logs its backtests as trials.
UNCOMPARED = {
    "scaling.csv": "wall-clock times are reported, not compared",
    "trials.csv": "the trial count grows with every run, including this one",
}
FLOAT_TOL = 1e-9


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(reports_dir: Path) -> dict[str, object]:
    tables = sorted((reports_dir / "tables").glob("*.csv"))
    return {
        "tables": {t.name: sha256(t) for t in tables},
        "uncompared": UNCOMPARED,
        "rule": "hash match, or floats within 1e-9 and other cells equal",
    }


def write_manifest(reports_dir: Path) -> Path:
    path = reports_dir / "manifest.json"
    path.write_text(
        json.dumps(build_manifest(reports_dir), indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


@dataclass(frozen=True)
class TableCheck:
    table: str
    status: str  # "hash" | "tolerance" | "uncompared" | "missing" | "new" | "differs"
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("hash", "tolerance", "uncompared")


def _close(a: pl.DataFrame, b: pl.DataFrame) -> tuple[bool, str]:
    if a.columns != b.columns or a.height != b.height:
        return False, f"shape or columns differ ({a.shape} vs {b.shape})"
    for col in a.columns:
        x, y = a[col], b[col]
        if x.dtype.is_float() or y.dtype.is_float():
            xs = x.cast(pl.Float64).to_numpy()
            ys = y.cast(pl.Float64).to_numpy()
            both_nan = (xs != xs) & (ys != ys)
            diff = abs(xs - ys)
            bad = ~both_nan & ~(diff <= FLOAT_TOL)
            if bad.any():
                return False, f"column {col}: max difference {float(diff[~both_nan].max()):.3g}"
        elif not x.equals(y, null_equal=True):
            return False, f"column {col} differs"
    return True, ""


def check_tables(
    reference: dict[str, str], reference_dir: Path | None, reports_dir: Path
) -> list[TableCheck]:
    """Compare fresh tables with the reference manifest (and, if given, the reference CSVs)."""
    out = []
    fresh = {t.name: t for t in (reports_dir / "tables").glob("*.csv")}
    for name, digest in sorted(reference.items()):
        if name in UNCOMPARED:
            out.append(TableCheck(name, "uncompared", UNCOMPARED[name]))
            continue
        if name not in fresh:
            out.append(TableCheck(name, "missing"))
            continue
        if sha256(fresh[name]) == digest:
            out.append(TableCheck(name, "hash"))
            continue
        ref_csv = reference_dir / name if reference_dir else None
        if ref_csv is not None and ref_csv.exists():
            ok, detail = _close(pl.read_csv(ref_csv), pl.read_csv(fresh[name]))
            out.append(TableCheck(name, "tolerance" if ok else "differs", detail))
        else:
            out.append(TableCheck(name, "differs", "hash differs and no reference CSV kept"))
    out += [TableCheck(n, "new") for n in sorted(set(fresh) - set(reference))]
    return out


_BLOCK = re.compile(
    r"(<!-- regport:(?P<name>[a-z_]+):start -->\n)(.*?)(<!-- regport:(?P=name):end -->)", re.S
)


def fill_readme(readme: Path, blocks: dict[str, str]) -> list[str]:
    """Replace every marked block whose name is in ``blocks``; returns the names filled."""
    text = readme.read_text(encoding="utf-8")
    filled: list[str] = []

    def repl(m: re.Match[str]) -> str:
        name = m.group("name")
        if name not in blocks:
            return m.group(0)
        filled.append(name)
        return m.group(1) + blocks[name].rstrip("\n") + "\n" + m.group(4)

    readme.write_text(_BLOCK.sub(repl, text), encoding="utf-8", newline="\n")
    return filled
