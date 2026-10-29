"""Least-squares fit of raw SVI to total implied variance.

The parameters are stepped in unconstrained coordinates: ``a`` and ``m`` free,
``b = exp(x)``, ``sigma = exp(x)``, ``rho = tanh(x)``. A negative minimum
variance is penalised, and that penalty row is always present so the Jacobian
does not change length when the floor crosses zero. The objective does not
contain Durrleman's ``g``, so a fitted slice can still have a butterfly
arbitrage. The fitter reports ``min_g`` instead of hiding it in a penalty.
Two starting points are tried; the smaller residual wins.

When a quote has a bid and an ask, the total-variance residual is weighted by
the inverse variance of the half-spread mapped through vega. A quote with no
spread takes the median of those two-sided weights, or weight 1 when the
slice has no two-sided quote. There is no SSVI or eSSVI fit: the
parameterisation stays raw SVI.

This is a fit, not the SANOS linear program. It will happily match an
arbitrageable smile.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from volsurface import bs
from volsurface.core.types import Curves, FitResult, MarketData
from volsurface.surface.svi import RawSVI, svi_surface

# Lee's moment bound on the slope of total variance: b * (1 + |rho|) <= 2.
_LEE_SLOPE = 2.0
_DIAGNOSTIC_POINTS = 401


def fit_raw_svi(
    k: np.ndarray, total_var: np.ndarray, weights: np.ndarray | None = None
) -> tuple[RawSVI, np.ndarray]:
    """Fit one slice. Returns the slice and the unweighted total-variance residuals.

    ``weights`` multiply the squared residuals. They default to 1. The returned
    residual is ``w(k) - total_var``, without the weights and without the floor
    penalty, so a caller can report an in-sample gap in total variance.
    """
    k = np.asarray(k, dtype=float)
    total_var = np.asarray(total_var, dtype=float)
    if weights is None:
        weights = np.ones_like(total_var)
    else:
        weights = np.asarray(weights, dtype=float)
    starts = [_initial(k, total_var), _initial(k, total_var, rho_sign=-1.0)]
    best = None
    best_cost = np.inf
    for start in starts:
        result = least_squares(lambda x: _residual(x, k, total_var, weights), start, method="trf")
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
        groups = []
        k_min = np.inf
        k_max = -np.inf
        for _, group in quotes.groupby("expiry", sort=True):
            group = group.drop_duplicates("k").sort_values("k")
            if len(group) < 5:
                continue
            k = group["k"].to_numpy(dtype=float)
            T = float(group["T"].iloc[0])
            total_var = group["iv"].to_numpy(dtype=float) ** 2 * T
            weights = _spread_weights(group)
            slice_, residual = fit_raw_svi(k, total_var, weights)
            pillars.append((T, slice_))
            residual_blocks.append(residual)
            groups.append(group)
            k_min = min(k_min, float(k[0]))
            k_max = max(k_max, float(k[-1]))
        if not pillars:
            raise ValueError("no expiry had five implied vols to fit")
        # Groupby sorts the expiry label, not the year fraction. Calendar order is by T.
        order = np.argsort([t for t, _ in pillars])
        pillars = [pillars[i] for i in order]
        groups = [groups[i] for i in order]
        residual_blocks = [residual_blocks[i] for i in order]
        surface = svi_surface(pillars, Curves(data.F, data.df), (k_min, k_max))
        residuals = np.concatenate(residual_blocks)
        in_spread = _in_spread(data.quotes, surface)
        report = _slice_report(pillars, groups, surface, k_min, k_max)
        return FitResult(
            surface=surface,
            residuals=residuals,
            in_spread=in_spread,
            wall_time=time.perf_counter() - started,
            solver_status="least_squares",
            slice_report=report,
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


def _residual(
    x: np.ndarray, k: np.ndarray, total_var: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    slice_ = _unpack(x)
    gap = (slice_.w(k) - total_var) * np.sqrt(weights)
    # Minimum of w is a + b * sigma * sqrt(1 - rho^2). The penalty row is
    # always appended. Dropping it when the floor is positive changes the
    # length of the residual, and least_squares then rejects the Jacobian.
    floor = slice_.a + slice_.b * slice_.sigma * np.sqrt(1.0 - slice_.rho**2)
    penalty = 10.0 * min(floor, 0.0) * np.sqrt(k.size)
    return np.append(gap, penalty)


def _spread_weights(group: pd.DataFrame) -> np.ndarray:
    """Inverse-variance weights from the cash half-spread, mean normalised to 1.

    The map from a cash price to total variance is ``dw/dP = 2 * iv * T / vega``,
    with ``vega = dCash/dsigma``. ``sigma_w`` is that slope times the half-spread.
    A tiny vega is floored so the slope stays finite. A quote with no two-sided
    market takes the median two-sided weight on the slice, or 1 when the slice
    has none. Weights are then scaled to average 1, so the floor penalty stays
    on the same scale as an unweighted fit.
    """
    k = group["k"].to_numpy(dtype=float)
    T = float(group["T"].iloc[0])
    iv = group["iv"].to_numpy(dtype=float)
    forward = group["F"].to_numpy(dtype=float)
    discount = group["df"].to_numpy(dtype=float)
    bid = group["bid"].to_numpy(dtype=float) if "bid" in group else np.full(k.size, np.nan)
    ask = group["ask"].to_numpy(dtype=float) if "ask" in group else np.full(k.size, np.nan)
    strike = group["K"].to_numpy(dtype=float)
    weights = np.ones(k.size)
    two_sided_weights: list[float] = []
    one_sided = np.zeros(k.size, dtype=bool)
    for i in range(k.size):
        has_market = np.isfinite(bid[i]) and np.isfinite(ask[i]) and ask[i] > bid[i]
        if not has_market:
            one_sided[i] = True
            continue
        if not (np.isfinite(iv[i]) and iv[i] > 0.0):
            continue
        half = 0.5 * (ask[i] - bid[i])
        vega = float(bs.vega(forward[i], strike[i], T, iv[i], discount[i]))
        vega_floor = 1e-4 * discount[i] * forward[i] * np.sqrt(max(T, 1e-12))
        dw_dp = 2.0 * iv[i] * T / max(vega, vega_floor)
        sigma_w = max(abs(dw_dp) * half, 1e-8)
        weights[i] = 1.0 / sigma_w**2
        two_sided_weights.append(weights[i])
    # A one-sided quote has no half-spread, so it cannot form an inverse-variance
    # weight. Giving it 1 while its neighbours sit at 1/sigma_w^2 lets a last
    # trade dominate a tight market. Use the median two-sided weight instead.
    # With no two-sided quote on the slice every weight stays 1.
    if two_sided_weights and one_sided.any():
        weights[one_sided] = float(np.median(two_sided_weights))
    weights *= weights.size / np.sum(weights)
    return weights


def _diagnostic_grid(k_min: float, k_max: float) -> np.ndarray:
    """Strikes covering the quoted window and half a unit of log-moneyness each side."""
    return np.linspace(k_min - 0.5, k_max + 0.5, _DIAGNOSTIC_POINTS)


def _slice_report(pillars, groups, surface, k_min: float, k_max: float) -> pd.DataFrame:
    """One row per fitted expiry: ``min_g``, Lee's bound, calendar order, in-spread.

    ``min_g`` is the smallest Durrleman ``g`` on :func:`_diagnostic_grid`.
    ``lee_ok`` is ``b (1 + |rho|) <= 2``, the slope bound on total variance.
    ``calendar_ok`` is true when this slice's total variance sits above the
    previous expiry's on that same grid. The first expiry has nothing to clear.
    ``in_spread`` is the fraction of two-sided quotes at this expiry, or NaN.
    ``n_one_sided`` counts quotes on the slice with no two-sided bid and ask.
    Those rows are weighted by the median two-sided weight. See
    :func:`_spread_weights`.
    """
    grid = _diagnostic_grid(k_min, k_max)
    rows = []
    previous = None
    for (T, slice_), group in zip(pillars, groups, strict=True):
        g_values = np.asarray(slice_.g(grid), dtype=float)
        finite = g_values[np.isfinite(g_values)]
        min_g = float(np.min(finite)) if finite.size else float("nan")
        lee_ok = bool(slice_.b * (1.0 + abs(slice_.rho)) <= _LEE_SLOPE)
        if previous is None:
            calendar_ok = True
        else:
            gap = slice_.w(grid) - previous.w(grid)
            calendar_ok = bool(np.nanmin(gap) >= -1e-8)
        rows.append(
            {
                "T": T,
                "min_g": min_g,
                "lee_ok": lee_ok,
                "calendar_ok": calendar_ok,
                "in_spread": _in_spread(group, surface),
                "n_one_sided": _n_one_sided(group),
            }
        )
        previous = slice_
    return pd.DataFrame.from_records(rows)


def _n_one_sided(group: pd.DataFrame) -> int:
    """Quotes on this slice that do not have a strictly positive bid-ask spread."""
    n = len(group)
    if "bid" not in group or "ask" not in group:
        return int(n)
    bid = group["bid"].to_numpy(dtype=float)
    ask = group["ask"].to_numpy(dtype=float)
    two_sided = np.isfinite(bid) & np.isfinite(ask) & (ask > bid)
    return int(n - int(two_sided.sum()))


def _in_spread(quotes: pd.DataFrame, surface) -> float:
    """Fraction of two-sided quotes whose model call lies between bid and ask.

    NaN when the quotes have no two-sided market.
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
