"""Least-squares fit of raw SVI to total implied variance.

The parameters are stepped in unconstrained coordinates: ``a`` and ``m`` free,
``b = exp(x)``, ``sigma = exp(x)``, ``rho = tanh(x)``. A negative minimum
variance is penalised. The objective does not contain Durrleman's ``g``, so a
fitted slice can still have a butterfly arbitrage. Two starting points are
tried; the smaller residual wins.

This is a fit, not the SANOS linear program. It will happily match an
arbitrageable smile.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from volsurface.core.types import Curves, FitResult, MarketData
from volsurface.surface.svi import RawSVI, svi_surface


def fit_raw_svi(k: np.ndarray, total_var: np.ndarray) -> tuple[RawSVI, np.ndarray]:
    """Fit one slice. Returns the slice and the total-variance residuals."""
    k = np.asarray(k, dtype=float)
    total_var = np.asarray(total_var, dtype=float)
    starts = [_initial(k, total_var), _initial(k, total_var, rho_sign=-1.0)]
    best = None
    best_cost = np.inf
    for start in starts:
        result = least_squares(lambda x: _residual(x, k, total_var), start, method="trf")
        if result.cost < best_cost:
            best = result
            best_cost = float(result.cost)
    assert best is not None
    slice_ = _unpack(best.x)
    return slice_, slice_.w(k) - total_var


class SVIFitter:
    """Fit every expiry in a ``MarketData`` and stack the slices into a surface."""

    def fit(self, data: MarketData) -> FitResult:
        started = time.perf_counter()
        quotes = data.quotes.dropna(subset=["iv"])
        pillars = []
        residual_blocks = []
        k_min = np.inf
        k_max = -np.inf
        for _, group in quotes.groupby("expiry", sort=True):
            group = group.drop_duplicates("k").sort_values("k")
            if len(group) < 5:
                continue
            k = group["k"].to_numpy(dtype=float)
            T = float(group["T"].iloc[0])
            total_var = group["iv"].to_numpy(dtype=float) ** 2 * T
            slice_, residual = fit_raw_svi(k, total_var)
            pillars.append((T, slice_))
            residual_blocks.append(residual)
            k_min = min(k_min, float(k[0]))
            k_max = max(k_max, float(k[-1]))
        if not pillars:
            raise ValueError("no expiry had five implied vols to fit")
        surface = svi_surface(pillars, Curves(data.F, data.df), (k_min, k_max))
        residuals = np.concatenate(residual_blocks)
        in_spread = _in_spread(data.quotes, surface)
        return FitResult(
            surface=surface,
            residuals=residuals,
            in_spread=in_spread,
            wall_time=time.perf_counter() - started,
            solver_status="least_squares",
        )


def _initial(k: np.ndarray, total_var: np.ndarray, rho_sign: float = 1.0) -> np.ndarray:
    """A cheap start: ATM level, a modest wing slope, curvature 0.2."""
    atm = float(total_var[np.argmin(np.abs(k))])
    slope = float(np.polyfit(k, total_var, 1)[0]) if k.size >= 2 else 0.0
    b = 0.1
    rho = float(np.clip(rho_sign * slope / b, -0.8, 0.8))
    m = 0.0
    sigma = 0.2
    a = atm - b * (-rho * m + np.sqrt(m * m + sigma * sigma))
    return np.array([a, np.log(b), np.arctanh(rho), m, np.log(sigma)], dtype=float)


def _unpack(x: np.ndarray) -> RawSVI:
    a, log_b, artanh_rho, m, log_sigma = (float(v) for v in x)
    return RawSVI(
        a=a,
        b=float(np.exp(log_b)),
        rho=float(np.tanh(artanh_rho)),
        m=m,
        sigma=float(np.exp(log_sigma)),
    )


def _residual(x: np.ndarray, k: np.ndarray, total_var: np.ndarray) -> np.ndarray:
    slice_ = _unpack(x)
    gap = slice_.w(k) - total_var
    # Minimum of w is a + b * sigma * sqrt(1 - rho^2). Keep it non-negative.
    floor = slice_.a + slice_.b * slice_.sigma * np.sqrt(1.0 - slice_.rho**2)
    if floor < 0.0:
        gap = np.append(gap, 10.0 * floor * np.sqrt(k.size))
    return gap


def _in_spread(quotes: pd.DataFrame, surface) -> float:
    """Fraction of two-sided quotes whose model call lies between bid and ask.

    NaN when the snapshot has no two-sided market, which is the committed SPY file.
    """
    usable = quotes.dropna(subset=["bid", "ask", "F", "df", "k", "T"])
    usable = usable[(usable["bid"] > 0.0) & (usable["ask"] > usable["bid"])]
    if usable.empty:
        return float("nan")
    inside = 0
    for _, row in usable.iterrows():
        model = float(surface.call([row["T"]], [row["k"]])[0, 0])
        cash = model * float(row["df"]) * float(row["F"])
        # Compare on the call. A quoted put is converted by parity.
        if row["kind"] == "put":
            bid = float(row["bid"]) + float(row["df"]) * (float(row["F"]) - float(row["K"]))
            ask = float(row["ask"]) + float(row["df"]) * (float(row["F"]) - float(row["K"]))
        else:
            bid, ask = float(row["bid"]), float(row["ask"])
        lo, hi = min(bid, ask), max(bid, ask)
        inside += int(lo <= cash <= hi)
    return inside / len(usable)
