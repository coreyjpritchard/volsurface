"""Calibrate SABR with ``beta`` held fixed, one parameter set per expiry.

``alpha`` and ``beta`` trade off in the leading factor ``alpha / F^(1 - beta)``,
so a joint search does not identify both. The fitter takes ``beta`` as given
and minimises a vega-weighted price residual in ``(alpha, rho, nu)``. Bounds
are the corresponding rows of :meth:`~volsurface.models.sabr.SABR.param_spec`.

Each expiry is its own search. ``slice_report`` has one row per expiry, indexed
by ``T``. ``surface`` is the fit at the expiry nearest one year, for a caller
that wants a single model. The parameters of the other expiries are only in
the report.

Several starts are tried on each expiry. The smallest sum of squares wins.
Rows marked with an exercise other than European are refused. The pricer is
European. ``allow_american=True`` drops those rows and records the count.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from volsurface.core.params import from_unconstrained, to_unconstrained
from volsurface.core.types import FitResult, MarketData
from volsurface.fit._price_fit import (
    black_vega,
    gaussian_starts,
    in_spread,
    model_prices,
    outside_spread,
    quote_frame,
    rectangle,
    unconstrained_bounds,
)
from volsurface.models.sabr import SABR, free_spec


class SABRFitter:
    """Multi-start least squares for ``alpha``, ``rho`` and ``nu`` on each expiry."""

    def __init__(
        self,
        beta: float = 1.0,
        n_starts: int = 4,
        loss: str = "vega",
        seed: int = 0,
        allow_american: bool = False,
    ):
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must lie in [0, 1], got {beta}")
        if n_starts < 1:
            raise ValueError(f"n_starts must be at least 1, got {n_starts}")
        if loss not in ("vega", "in_spread"):
            raise ValueError(f"loss must be 'vega' or 'in_spread', got {loss!r}")
        self.beta = float(beta)
        self.n_starts = int(n_starts)
        self.loss = loss
        self.seed = int(seed)
        self.allow_american = bool(allow_american)

    def fit(self, data: MarketData) -> FitResult:
        started = time.perf_counter()
        frame, n_american = quote_frame(
            data,
            fitter="SABRFitter",
            allow_american=self.allow_american,
        )
        residuals = np.empty(len(frame), dtype=float)
        rows = []
        worst = 0.0
        representative = None
        distance = np.inf
        for maturity, index in frame.groupby("T").groups.items():
            part = frame.loc[index].reset_index(drop=True)
            fitted, part_residual, cost = _fit_slice(
                part,
                self.beta,
                self.n_starts,
                self.loss,
                self.seed,
            )
            residuals[frame.index.get_indexer(index)] = part_residual
            rows.append(
                {
                    "T": float(maturity),
                    "alpha": fitted.alpha,
                    "beta": fitted.beta,
                    "rho": fitted.rho,
                    "nu": fitted.nu,
                    "rmse_price": float(np.sqrt(np.mean(part_residual**2))),
                    "max_abs_price": float(np.max(np.abs(part_residual))),
                    "n_american": n_american,
                }
            )
            worst = max(worst, cost)
            gap = abs(float(maturity) - 1.0)
            if gap < distance:
                distance = gap
                representative = fitted
        assert representative is not None
        report = pd.DataFrame(rows).sort_values("T").reset_index(drop=True)
        return FitResult(
            surface=rectangle(representative, data, frame),
            residuals=residuals,
            in_spread=_book_in_spread(frame, report, self.beta),
            wall_time=time.perf_counter() - started,
            solver_status=f"{len(rows)} slices cost<={worst:.3e} starts={self.n_starts}",
            slice_report=report,
        )


def _book_in_spread(frame: pd.DataFrame, report: pd.DataFrame, beta: float) -> float:
    """Fraction of two-sided quotes inside the spread, each expiry its own model."""
    if "bid" not in frame.columns or "ask" not in frame.columns:
        return float("nan")
    priced = np.empty(len(frame), dtype=float)
    maturity = frame["T"].to_numpy(dtype=float)
    for row in report.itertuples(index=False):
        mask = np.isclose(maturity, row.T)
        model = SABR(alpha=row.alpha, beta=beta, rho=row.rho, nu=row.nu)
        part = frame.loc[mask].reset_index(drop=True)
        priced[mask] = model_prices(model, part)
    return in_spread(frame, priced)


def _fit_slice(frame, beta, n_starts, loss, seed):
    market = frame["price"].to_numpy(dtype=float)
    vega = black_vega(frame)
    specs = free_spec()
    bounds = unconstrained_bounds(specs)
    base = _origin(beta)
    origin = to_unconstrained(
        {"alpha": base.alpha, "rho": base.rho, "nu": base.nu},
        specs,
    )
    starts = gaussian_starts(origin, specs, n_starts, seed)
    best = None
    best_cost = np.inf
    for start in starts:
        result = least_squares(
            lambda x: _residual(x, specs, beta, frame, market, vega, loss),
            start,
            bounds=bounds,
            method="trf",
        )
        if result.cost < best_cost:
            best = result
            best_cost = float(result.cost)
    assert best is not None
    fitted = _model_from(best.x, specs, beta)
    priced = model_prices(fitted, frame)
    return fitted, priced - market, best_cost


def _origin(beta: float) -> SABR:
    values = {spec.name: spec.default for spec in SABR.param_spec()}
    values["beta"] = beta
    return SABR(**values)


def _model_from(x: np.ndarray, specs, beta: float) -> SABR:
    values = from_unconstrained(np.asarray(x, dtype=float), specs)
    values["beta"] = beta
    return SABR(**values)


def _residual(x, specs, beta, frame, market, vega, loss: str) -> np.ndarray:
    model = _model_from(x, specs, beta)
    priced = model_prices(model, frame)
    gap = priced - market
    if loss == "in_spread":
        gap = outside_spread(priced, frame, gap)
    return gap / vega
