"""Walk-forward Gaussian HMM (FR-11 to FR-13).

Refit every ``refit_every_days`` rows on an expanding (or rolling) window of at least
``min_train_days`` rows. Fit ``k`` uses rows up to ``f_k`` and serves the dates ``f_k`` to
``f_{k+1} − 1``. For those dates the forward filter is re-run from the start of fit ``k``'s training
window with fit ``k``'s parameters, over data up to each date only. States are relabelled after
every fit (canonical order, then Hungarian alignment with the previous fit).

The fits are independent, so they run in parallel (``n_jobs`` processes); each fit's restart
seeds come from ``SeedSequence([seed, k])``, so results don't depend on ``n_jobs`` (NFR-02).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import numpy.typing as npt
import polars as pl

from regport.config import HMMConfig
from regport.errors import LookaheadError
from regport.logs import get_logger
from regport.regime.filter import forward_filter
from regport.regime.hmm import HMMParams, fit_hmm
from regport.regime.labels import relabel
from regport.resources import wait_for_memory

log = get_logger(__name__)
F64 = npt.NDArray[np.float64]


@dataclass
class RegimeFrame:
    """Filtered and predicted state probabilities per date (long format), plus fit records."""

    probs: pl.DataFrame
    fits: list[HMMParams]
    label_switches: list[date] = field(default_factory=list)
    failed_refits: list[date] = field(default_factory=list)

    def at(self, asof: date) -> pl.DataFrame:
        """Rows for decision date ``asof``; raises if any row used a later fit (DQ-08)."""
        rows = self.probs.filter(pl.col("date") == asof)
        if rows.filter(pl.col("params_fit_end") > asof).height:
            raise LookaheadError(f"regime row at {asof} uses parameters fitted after it")
        return rows


def _train_start(end: int, cfg: HMMConfig) -> int:
    return 0 if cfg.window == "expanding" else max(0, end - cfg.rolling_days + 1)


def _fit_job(args: tuple[F64, int, int, int, date, int, float]) -> HMMParams | None:
    x, n_states, n_init, seed, fit_end, max_iter, tol = args
    try:
        return fit_hmm(x, n_states, n_init, seed, fit_end, max_iter, tol)
    except ValueError:
        return None  # every restart degenerate: the caller keeps the previous fit


class WalkForwardHMM:
    def __init__(self, cfg: HMMConfig, n_states: int, seed: int, n_jobs: int = 1) -> None:
        if cfg.min_train_days < 2 * n_states:
            raise ValueError("min_train_days is too small for the number of states")
        self.cfg = cfg
        self.n_states = n_states
        self.seed = seed
        self.n_jobs = n_jobs

    def fit_points(self, n_rows: int) -> list[int]:
        first = self.cfg.min_train_days - 1
        return list(range(first, n_rows, self.cfg.refit_every_days))

    def run(self, x: F64, dates: list[date]) -> RegimeFrame:
        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] != len(dates):
            raise ValueError("features and dates differ in length")
        points = self.fit_points(len(dates))
        if not points:
            raise ValueError("not enough rows for a first fit")
        jobs = [
            (
                x[_train_start(f, self.cfg) : f + 1],
                self.n_states,
                self.cfg.n_init,
                int(np.random.SeedSequence([self.seed, k]).generate_state(1)[0]),
                dates[f],
                self.cfg.max_iter,
                self.cfg.tol,
            )
            for k, f in enumerate(points)
        ]
        if self.n_jobs > 1:
            wait_for_memory("walk-forward HMM")
            with ProcessPoolExecutor(max_workers=self.n_jobs) as pool:
                raw = list(pool.map(_fit_job, jobs))
        else:
            raw = [_fit_job(j) for j in jobs]
        fits: list[HMMParams] = []
        switches: list[date] = []
        frames = []
        prev: HMMParams | None = None
        failed_refits: list[date] = []
        prev_perm: npt.NDArray[np.int64] | None = None
        for k, (f, fitted) in enumerate(zip(points, raw, strict=True)):
            if fitted is None:
                if prev is None:
                    raise ValueError(f"the first HMM fit (ending {dates[f]}) failed")
                failed_refits.append(dates[f])
                log.warning("refit_failed", refit_id=k, fit_end=str(dates[f]))
                params = prev  # keep serving the previous fit; its fit_end stays earlier
                fits.append(params)
                stop = points[k + 1] if k + 1 < len(points) else len(dates)
                start = _train_start(points[k - 1], self.cfg) if k else 0
                filtered, predicted, _ = forward_filter(x[start:stop], params)
                frames.append(
                    self._frame(dates, f, stop, start, filtered, predicted, params.fit_end, k)
                )
                continue
            params, perm = relabel(prev, fitted)
            switched = prev_perm is not None and not np.array_equal(perm, prev_perm)
            prev_perm = perm
            if switched:
                switches.append(dates[f])
                log.info("label_switch", refit_id=k, fit_end=str(dates[f]))
            fits.append(params)
            prev = params
            stop = points[k + 1] if k + 1 < len(points) else len(dates)
            start = _train_start(f, self.cfg)
            filtered, predicted, _ = forward_filter(x[start:stop], params)
            frames.append(self._frame(dates, f, stop, start, filtered, predicted, dates[f], k))
        probs = pl.concat(frames).with_columns(pl.col("date").cast(pl.Date))
        return RegimeFrame(probs, fits, switches, failed_refits)

    def _frame(
        self,
        dates: list[date],
        f: int,
        stop: int,
        start: int,
        filtered: F64,
        predicted: F64,
        fit_end: date,
        refit_id: int,
    ) -> pl.DataFrame:
        rows = slice(f - start, stop - start)
        out_dates = dates[f:stop]
        kk = self.n_states
        return pl.DataFrame(
            {
                "date": np.repeat(np.array(out_dates, dtype="datetime64[D]"), kk),
                "model": "hmm",
                "state": np.tile(np.arange(kk, dtype=np.int8), len(out_dates)),
                "p_filtered": filtered[rows].ravel(),
                "p_predicted": predicted[rows].ravel(),
                "params_fit_end": fit_end,
                "refit_id": np.int32(refit_id),
            }
        )
