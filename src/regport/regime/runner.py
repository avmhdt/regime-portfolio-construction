"""``regport regime run``: compute or load cached walk-forward regime outputs.

Outputs go to ``results/regime/`` (git-ignored), keyed by a hash of the regime settings, the seed
and the feature values, so any change in data or config yields a new file rather than a stale one:
``hmm_<key>.parquet`` (long probabilities), ``hmm_<key>_fits.json`` (per-refit parameters) and
``bocpd_<key>.parquet``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from regport.config import ExperimentConfig
from regport.logs import get_logger
from regport.regime.bocpd import NormalGammaPrior, bocpd
from regport.regime.features import FEATURE_SETS, build_features
from regport.regime.walkforward import WalkForwardHMM

log = get_logger(__name__)


@dataclass
class RegimeOutputs:
    hmm: pl.DataFrame | None
    transmats: dict[int, np.ndarray] | None
    bocpd: pl.DataFrame | None
    features: pl.DataFrame
    label_switches: list[date]


def load_features(
    cfg: ExperimentConfig,
    processed: Path,
    feature_set: Literal["A", "B"] | None = None,
    vintage: Literal["pit", "latest"] | None = None,
) -> pl.DataFrame:
    fs = feature_set or cfg.regime.hmm.feature_set
    vt = vintage or cfg.regime.hmm.macro_vintage
    vintages = pl.read_parquet(processed / "macro_vintages.parquet") if fs == "B" else None
    return build_features(
        pl.read_parquet(processed / "returns_daily.parquet"),
        pl.read_parquet(processed / "french_daily.parquet"),
        pl.read_parquet(processed / "fred_daily.parquet"),
        vintages,
        cfg.sample.data_start,
        fs,
        vt,
    )


def _key(kind: str, cfg: ExperimentConfig, features: pl.DataFrame, cols: list[str]) -> str:
    settings = cfg.regime.hmm.model_dump() if kind == "hmm" else cfg.regime.bocpd.model_dump()
    blob = json.dumps(
        {"kind": kind, "settings": settings, "seed": cfg.seed, "start": str(cfg.sample.data_start)},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(blob.encode())
    digest.update(features.select("date", *cols).write_ipc(None).getvalue())
    return digest.hexdigest()[:16]


def _n_states(cfg: ExperimentConfig) -> int:
    k = cfg.regime.hmm.n_states
    if k == "auto":
        raise ValueError("regime.hmm.n_states is 'auto'; fix it from the pre-OOS BIC (D-05)")
    return int(k)


def hmm_paths(cfg: ExperimentConfig, features: pl.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Cache paths of the walk-forward HMM outputs for these settings and features."""
    key = _key("hmm", cfg, features, FEATURE_SETS[cfg.regime.hmm.feature_set])
    return out_dir / f"hmm_{key}.parquet", out_dir / f"hmm_{key}_fits.json"


def run_hmm(
    cfg: ExperimentConfig,
    features: pl.DataFrame,
    out_dir: Path,
    n_jobs: int = 1,
    force: bool = False,
) -> tuple[pl.DataFrame, dict[int, np.ndarray], list[date]]:
    cols = FEATURE_SETS[cfg.regime.hmm.feature_set]
    probs_path, fits_path = hmm_paths(cfg, features, out_dir)
    if probs_path.exists() and fits_path.exists() and not force:
        fits = json.loads(fits_path.read_text(encoding="utf-8"))
        return (
            pl.read_parquet(probs_path),
            {int(k): np.array(v["transmat"]) for k, v in fits["fits"].items()},
            [date.fromisoformat(d) for d in fits["label_switches"]],
        )
    model = WalkForwardHMM(cfg.regime.hmm, _n_states(cfg), cfg.seed, n_jobs)
    frame = model.run(features.select(cols).to_numpy(), features["date"].to_list())
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.probs.write_parquet(probs_path)
    record = {
        "feature_set": cfg.regime.hmm.feature_set,
        "features": cols,
        "label_switches": [str(d) for d in frame.label_switches],
        "failed_refits": [str(d) for d in frame.failed_refits],
        "fits": {
            str(i): {
                "fit_end": str(f.fit_end),
                "n_obs": f.n_obs,
                "loglik": f.loglik,
                "transmat": f.transmat.tolist(),
                "means": f.raw_means().tolist(),
                "covars": f.raw_covars().tolist(),
                "seed": f.seed,
            }
            for i, f in enumerate(frame.fits)
        },
    }
    fits_path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
    log.info("hmm_done", refits=len(frame.fits), label_switches=len(frame.label_switches))
    return frame.probs, {i: f.transmat for i, f in enumerate(frame.fits)}, frame.label_switches


def run_bocpd(
    cfg: ExperimentConfig, features: pl.DataFrame, out_dir: Path, force: bool = False
) -> pl.DataFrame:
    b = cfg.regime.bocpd
    key = _key("bocpd", cfg, features, [b.series])
    path = out_dir / f"bocpd_{key}.parquet"
    if path.exists() and not force:
        return pl.read_parquet(path)
    prior = NormalGammaPrior(b.prior.mu0, b.prior.kappa0, b.prior.alpha0, b.prior.beta0)
    res = bocpd(
        features[b.series].to_numpy(),
        b.hazard_lambda_days,
        prior,
        b.max_run_length,
        b.short_run_days,
    )
    frame = pl.DataFrame(
        {
            "date": features["date"],
            "model": "bocpd",
            "cp_prob_short": res.cp_prob_short,
            "rl_map": res.rl_map,
            "rl_mean": res.rl_mean,
            "spy_source": features["spy_source"],
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return frame


def load_or_run(
    cfg: ExperimentConfig,
    processed: Path,
    results_dir: Path,
    n_jobs: int = 1,
    models: tuple[str, ...] = ("hmm", "bocpd"),
) -> RegimeOutputs:
    if "hmm" in models:
        _n_states(cfg)  # fail fast on n_states "auto", before reading any data
    features = load_features(cfg, processed)
    out_dir = results_dir / "regime"
    hmm = transmats = None
    switches: list[date] = []
    if "hmm" in models:
        hmm, transmats, switches = run_hmm(cfg, features, out_dir, n_jobs)
    bocpd_frame = run_bocpd(cfg, features, out_dir) if "bocpd" in models else None
    return RegimeOutputs(hmm, transmats, bocpd_frame, features, switches)
