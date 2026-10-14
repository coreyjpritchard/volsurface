"""Calibrate Heston to a book of option prices.

The search runs in the unconstrained coordinates of
:meth:`~volsurface.models.heston.Heston.param_spec`. Bounds are that spec.
The Feller ratio ``2 kappa theta / xi^2`` is written onto the result and is
not a constraint: a fitted model may be one whose variance reaches zero.

``loss="vega"`` minimises ``(model - market) / vega``. That is an implied-vol
error to first order, and it is the loss used to recover a known parameter
set. ``loss="in_spread"`` minimises the vega-weighted distance outside the
bid and ask, and is zero for every model that sits inside the spread, so it
does not identify a unique parameter. Quotes with no two-sided market fall
back to the vega residual against the working price.

Several starts are tried. The smallest sum of squares wins. The returned
surface is a :class:`~volsurface.surface.model.ModelSurface` of the fitted
model on the rectangle of the quotes.

Rows marked with an exercise other than European are refused. The pricer
is European. ``allow_american=True`` drops those rows and records the count.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

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
from volsurface.models.heston import Heston


class HestonFitter:
    """Multi-start least squares for one Heston parameter set. See the module note."""

    def __init__(
        self,
        n_starts: int = 4,
        loss: str = "vega",
        seed: int = 0,
        allow_american: bool = False,
    ):
        if n_starts < 1:
            raise ValueError(f"n_starts must be at least 1, got {n_starts}")
        if loss not in ("vega", "in_spread"):
            raise ValueError(f"loss must be 'vega' or 'in_spread', got {loss!r}")
        self.n_starts = int(n_starts)
        self.loss = loss
        self.seed = int(seed)
        self.allow_american = bool(allow_american)

    def fit(self, data: MarketData) -> FitResult:
        started = time.perf_counter()
        frame, n_american = quote_frame(
            data,
            fitter="HestonFitter",
            allow_american=self.allow_american,
        )
        market = frame["price"].to_numpy(dtype=float)
        vega = black_vega(frame)
        specs = Heston.param_spec()
        bounds = unconstrained_bounds(specs)
        base = Heston(**{spec.name: spec.default for spec in specs})
        starts = gaussian_starts(base.to_unconstrained(), specs, self.n_starts, self.seed)
        best = None
        best_cost = np.inf
        for start in starts:
            result = least_squares(
                lambda x: _residual(x, frame, market, vega, self.loss),
                start,
                bounds=bounds,
                method="trf",
            )
            if result.cost < best_cost:
                best = result
                best_cost = float(result.cost)
        assert best is not None
        fitted = Heston.from_unconstrained(best.x)
        model_price = model_prices(fitted, frame)
        residuals = model_price - market
        surface = rectangle(fitted, data, frame)
        report = pd.DataFrame(
            [
                {
                    "v0": fitted.v0,
                    "kappa": fitted.kappa,
                    "theta": fitted.theta,
                    "xi": fitted.xi,
                    "rho": fitted.rho,
                    "feller": fitted.feller_ratio(),
                    "rmse_price": float(np.sqrt(np.mean(residuals**2))),
                    "max_abs_price": float(np.max(np.abs(residuals))),
                    "n_american": n_american,
                }
            ]
        )
        return FitResult(
            surface=surface,
            residuals=residuals,
            in_spread=in_spread(frame, model_price),
            wall_time=time.perf_counter() - started,
            solver_status=f"{best.message} cost={best.cost:.3e} starts={self.n_starts}",
            slice_report=report,
        )


def _residual(x, frame, market, vega, loss: str) -> np.ndarray:
    model = Heston.from_unconstrained(x)
    priced = model_prices(model, frame)
    gap = priced - market
    if loss == "in_spread":
        gap = outside_spread(priced, frame, gap)
    return gap / vega
