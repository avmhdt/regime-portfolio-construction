"""``regport grid``: run every cell of an ablation grid (FR-28) and record which run is which.

Each cell is one ``run_backtest`` call, so each appends one trial-log row (FR-25). A grid manifest
in ``results/grids/<grid_id>.json`` maps cells to run ids for ``regport evaluate --latest-grid``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

from regport.backtest.engine import Panel, run_backtest
from regport.config import ExperimentConfig, load_config
from regport.logs import get_logger
from regport.settings import Settings
from regport.tracking.experiment_log import new_run_id, utc_timestamp

log = get_logger(__name__)


class RegimeArm(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    usage: Literal["none", "hmm", "bocpd", "hmm_bocpd"]
    feature_set: Literal["A", "B"] | None = None
    macro_vintage: Literal["pit", "latest"] | None = None


class Headline(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    optimizer: str
    regime: str


class GridConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    base: Path
    optimizers: list[Literal["min_variance", "erc", "hrp", "mv_turnover"]]
    regimes: list[RegimeArm]
    benchmark: Literal["equal_weight"] = "equal_weight"
    headline: Headline | None = None


def load_grid(path: Path) -> GridConfig:
    return GridConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Cell:
    optimizer: str
    regime: str
    cfg: ExperimentConfig

    @property
    def key(self) -> str:
        return f"{self.optimizer}|{self.regime}"


def cell_config(
    base: ExperimentConfig, optimizer: str, arm: RegimeArm | None, **overrides: Any
) -> ExperimentConfig:
    data = base.model_dump()
    data["optimizer"]["name"] = optimizer
    data["regime"]["usage"] = arm.usage if arm else "none"
    if arm and arm.feature_set:
        data["regime"]["hmm"]["feature_set"] = arm.feature_set
    if arm and arm.macro_vintage:
        data["regime"]["hmm"]["macro_vintage"] = arm.macro_vintage
    for dotted, value in overrides.items():
        section, field = dotted.split("__")
        data[section][field] = value
    return ExperimentConfig.model_validate(data)


def grid_cells(grid: GridConfig, base: ExperimentConfig) -> list[Cell]:
    cells = [
        Cell(o, arm.name, cell_config(base, o, arm))
        for o in grid.optimizers
        for arm in grid.regimes
    ]
    cells.append(Cell(grid.benchmark, "none", cell_config(base, grid.benchmark, None)))
    return cells


def run_cells(
    cells: list[Cell], settings: Settings, tag: str, kind: str, extra: dict[str, Any] | None = None
) -> Path:
    """Run each cell (one trial row each) and write the manifest; returns its path."""
    grid_id = new_run_id()
    panel = Panel.load(settings.data_dir / "processed")
    runs = []
    for cell in cells:
        result = run_backtest(cell.cfg, settings, panel=panel, tag=f"{tag}:{cell.key}")
        runs.append(
            {
                "optimizer": cell.optimizer,
                "regime": cell.regime,
                "run_id": result.run_id,
                "config_hash": cell.cfg.config_hash(),
            }
        )
        log.info("grid_cell_done", cell=cell.key, run_id=result.run_id)
    out = Path(cells[0].cfg.logging.experiment_log).parent / "grids" / f"{kind}_{grid_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "grid_id": grid_id,
        "kind": kind,
        "finished": utc_timestamp(),
        "tag": tag,
        "runs": runs,
        **(extra or {}),
    }
    out.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    return out


def run_grid(path: Path, settings: Settings) -> Path:
    grid = load_grid(path)
    base = load_config(grid.base)
    headline = grid.headline.model_dump() if grid.headline else None
    return run_cells(
        grid_cells(grid, base),
        settings,
        "grid",
        "ablation",
        {"config": str(path), "headline": headline},
    )


def latest_manifest(results_dir: Path, kind: str = "ablation") -> Path:
    found = sorted(
        (results_dir / "grids").glob(f"{kind}_*.json"),
        key=lambda p: json.loads(p.read_text(encoding="utf-8"))["finished"],
    )
    if not found:
        raise FileNotFoundError(f"no {kind} grid in {results_dir / 'grids'}; run `regport grid`")
    return found[-1]
