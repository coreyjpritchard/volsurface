"""Calibrate SABR with ``beta`` held fixed.

``alpha`` and ``beta`` trade off in the leading factor ``alpha / F^(1 - beta)``,
so a joint search does not identify both. The fitter takes ``beta`` as given
and minimises a vega-weighted price residual in ``(alpha, rho, nu)``. Bounds
are the corresponding rows of :meth:`~volsurface.models.sabr.SABR.param_spec`.

Several starts are tried. The smallest sum of squares wins. The returned
surface is a :class:`~volsurface.surface.model.ModelSurface` of the fitted
model on the rectangle of the quotes.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from volsurface import bs
from volsurface.core.params import from_unconstrained, to_unconstrained, to_unconstrained_value
from volsurface.core.types import Curves, Domain, FitResult, MarketData
from volsurface.models.sabr import SABR, free_spec
from volsurface.surface.model import ModelSurface


class SABRFitter:
    """Multi-start least squares for ``alpha``, ``rho`` and ``nu`` at one ``beta``."""

    def __init__(self, beta: float = 1.0, n_starts: int = 4, loss: str = "vega", seed: int = 0):
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

    def fit(self, data: MarketData) -> FitResult:
        started = time.perf_counter()
        frame = _frame(data)
        market = frame["price"].to_numpy(dtype=float)
        vega = _vega(frame)
        specs = free_spec()
        bounds = _bounds(specs)
        base = _origin(self.beta)
        starts = _starts(base, specs, self.n_starts, self.seed)
        best = None
        best_cost = np.inf
        for start in starts:
            result = least_squares(
                lambda x: _residual(x, specs, self.beta, frame, market, vega, self.loss),
                start,
                bounds=bounds,
                method="trf",
            )
            if result.cost < best_cost:
                best = result
                best_cost = float(result.cost)
        assert best is not None
        fitted = _model_from(best.x, specs, self.beta)
        model_price = _prices(fitted, frame)
        residuals = model_price - market
        surface = _surface(fitted, data, frame)
        report = pd.DataFrame(
            [
                {
                    "alpha": fitted.alpha,
                    "beta": fitted.beta,
                    "rho": fitted.rho,
                    "nu": fitted.nu,
                    "rmse_price": float(np.sqrt(np.mean(residuals**2))),
                    "max_abs_price": float(np.max(np.abs(residuals))),
                }
            ]
        )
        return FitResult(
            surface=surface,
            residuals=residuals,
            in_spread=_in_spread(frame, model_price),
            wall_time=time.perf_counter() - started,
            solver_status=f"{best.message} cost={best.cost:.3e} starts={self.n_starts}",
            slice_report=report,
        )


def _origin(beta: float) -> SABR:
    values = {spec.name: spec.default for spec in SABR.param_spec()}
    values["beta"] = beta
    return SABR(**values)


def _model_from(x: np.ndarray, specs, beta: float) -> SABR:
    values = from_unconstrained(np.asarray(x, dtype=float), specs)
    values["beta"] = beta
    return SABR(**values)


def _frame(data: MarketData) -> pd.DataFrame:
    quotes = data.quotes
    if "price" not in quotes.columns:
        raise ValueError("SABRFitter needs a working price column")
    frame = quotes.dropna(subset=["price", "K", "T", "F", "df"]).copy()
    frame = frame[np.isfinite(frame["price"].to_numpy(dtype=float))]
    if frame.empty:
        raise ValueError("no quote had a finite price")
    if "k" not in frame.columns:
        frame["k"] = np.log(frame["K"].to_numpy(dtype=float) / frame["F"].to_numpy(dtype=float))
    return frame.reset_index(drop=True)


def _vega(frame: pd.DataFrame) -> np.ndarray:
    """Black vega at the quote's implied vol, floored so a wing still has a weight."""
    iv = frame["iv"].to_numpy(dtype=float) if "iv" in frame.columns else np.full(len(frame), np.nan)
    missing = ~np.isfinite(iv) | (iv <= 0.0)
    if missing.any():
        filled = bs.implied_vol(
            frame.loc[missing, "price"].to_numpy(dtype=float),
            frame.loc[missing, "F"].to_numpy(dtype=float),
            frame.loc[missing, "K"].to_numpy(dtype=float),
            frame.loc[missing, "T"].to_numpy(dtype=float),
            frame.loc[missing, "df"].to_numpy(dtype=float),
            frame.loc[missing, "kind"].to_numpy() if "kind" in frame.columns else "call",
        )
        iv = iv.copy()
        iv[missing] = filled
    iv = np.where(np.isfinite(iv) & (iv > 0.0), iv, 0.2)
    raw = np.asarray(
        bs.vega(
            frame["F"].to_numpy(dtype=float),
            frame["K"].to_numpy(dtype=float),
            frame["T"].to_numpy(dtype=float),
            iv,
            frame["df"].to_numpy(dtype=float),
        ),
        dtype=float,
    )
    floor = 1e-4 * frame["df"].to_numpy(dtype=float) * frame["F"].to_numpy(dtype=float)
    return np.maximum(raw, floor)


def _prices(model: SABR, frame: pd.DataFrame) -> np.ndarray:
    out = np.empty(len(frame), dtype=float)
    for maturity, index in frame.groupby("T").groups.items():
        rows = frame.loc[index]
        kind = rows["kind"].to_numpy() if "kind" in rows.columns else "call"
        priced = model.price(
            float(rows["F"].iloc[0]),
            rows["K"].to_numpy(dtype=float),
            float(maturity),
            float(rows["df"].iloc[0]),
            kind,
        )
        out[frame.index.get_indexer(index)] = np.asarray(priced, dtype=float)
    return out


def _residual(x, specs, beta, frame, market, vega, loss: str) -> np.ndarray:
    model = _model_from(x, specs, beta)
    priced = _prices(model, frame)
    gap = priced - market
    if loss == "in_spread" and "bid" in frame.columns and "ask" in frame.columns:
        bid = frame["bid"].to_numpy(dtype=float)
        ask = frame["ask"].to_numpy(dtype=float)
        two_sided = np.isfinite(bid) & np.isfinite(ask) & (ask > bid)
        outside = np.maximum(bid - priced, 0.0) + np.maximum(priced - ask, 0.0)
        gap = np.where(two_sided, outside, gap)
    return gap / vega


def _bounds(specs) -> tuple[list[float], list[float]]:
    lower: list[float] = []
    upper: list[float] = []
    for spec in specs:
        lo, hi = spec.lower, spec.upper
        if spec.transform == "logit":
            pad = 1e-9 * (hi - lo)
            lo, hi = lo + pad, hi - pad
        lower.append(to_unconstrained_value(lo, spec))
        upper.append(to_unconstrained_value(hi, spec))
    return lower, upper


def _starts(base: SABR, specs, n_starts: int, seed: int) -> list[np.ndarray]:
    """The spec default at this beta, then Gaussian kicks in unconstrained space."""
    rng = np.random.default_rng(seed)
    values = {"alpha": base.alpha, "rho": base.rho, "nu": base.nu}
    origin = to_unconstrained(values, specs)
    lower, upper = _bounds(specs)
    starts = [np.clip(origin, lower, upper)]
    for _ in range(n_starts - 1):
        kicked = origin + rng.normal(scale=0.4, size=origin.size)
        starts.append(np.clip(kicked, lower, upper))
    return starts


def _surface(model: SABR, data: MarketData, frame: pd.DataFrame) -> ModelSurface:
    k = frame["k"].to_numpy(dtype=float)
    t = frame["T"].to_numpy(dtype=float)
    domain = Domain(
        t_min=float(np.min(t)),
        t_max=float(np.max(t)),
        k_min=float(np.min(k)),
        k_max=float(np.max(k)),
    )
    return ModelSurface(model, Curves(data.F, data.df), domain)


def _in_spread(frame: pd.DataFrame, model_price: np.ndarray) -> float:
    if "bid" not in frame.columns or "ask" not in frame.columns:
        return float("nan")
    bid = frame["bid"].to_numpy(dtype=float)
    ask = frame["ask"].to_numpy(dtype=float)
    two_sided = np.isfinite(bid) & np.isfinite(ask) & (ask > bid)
    if not two_sided.any():
        return float("nan")
    slack = 1e-8
    inside = (model_price[two_sided] >= bid[two_sided] - slack) & (
        model_price[two_sided] <= ask[two_sided] + slack
    )
    return float(np.mean(inside))
