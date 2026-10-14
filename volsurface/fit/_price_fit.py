"""Shared pieces of a European price fit.

Heston and SABR both minimise a vega-weighted price residual. The quote
frame, the Black vega, the price map, the unconstrained bounds and starts,
the spread score and the rectangle of the surface are the same. The model
class and the parameter report stay with the fitter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volsurface import bs
from volsurface.core.params import to_unconstrained_value
from volsurface.core.types import Curves, Domain, MarketData
from volsurface.surface.model import ModelSurface


def quote_frame(
    data: MarketData,
    *,
    fitter: str,
    allow_american: bool = False,
) -> tuple[pd.DataFrame, int]:
    """Working rows, and how many American rows were dropped.

    A row with ``exercise`` other than ``european`` is a European-model
    error unless ``allow_american`` is set, in which case those rows are
    dropped. Dropping every row is still an error. A book with no
    ``exercise`` column is left as it is.
    """
    quotes = data.quotes
    if "price" not in quotes.columns:
        raise ValueError(f"{fitter} needs a working price column")
    frame = quotes.dropna(subset=["price", "K", "T", "F", "df"]).copy()
    frame = frame[np.isfinite(frame["price"].to_numpy(dtype=float))]
    if frame.empty:
        raise ValueError("no quote had a finite price")
    dropped = 0
    if "exercise" in frame.columns:
        american = frame["exercise"].astype(str).str.casefold() != "european"
        dropped = int(american.sum())
        if dropped and not allow_american:
            raise ValueError(
                f"{fitter} prices a European model. {dropped} quotes have "
                "exercise other than european."
            )
        if dropped:
            frame = frame.loc[~american]
            if frame.empty:
                raise ValueError("every quote was American and was dropped")
    if "k" not in frame.columns:
        frame["k"] = np.log(frame["K"].to_numpy(dtype=float) / frame["F"].to_numpy(dtype=float))
    return frame.reset_index(drop=True), dropped


def black_vega(frame: pd.DataFrame) -> np.ndarray:
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


def model_prices(model, frame: pd.DataFrame) -> np.ndarray:
    """``model.price`` on each expiry, in the row order of ``frame``."""
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


def outside_spread(priced: np.ndarray, frame: pd.DataFrame, gap: np.ndarray) -> np.ndarray:
    """Vega residual's price gap, replaced by the distance outside a two-sided market."""
    if "bid" not in frame.columns or "ask" not in frame.columns:
        return gap
    bid = frame["bid"].to_numpy(dtype=float)
    ask = frame["ask"].to_numpy(dtype=float)
    two_sided = np.isfinite(bid) & np.isfinite(ask) & (ask > bid)
    outside = np.maximum(bid - priced, 0.0) + np.maximum(priced - ask, 0.0)
    return np.where(two_sided, outside, gap)


def unconstrained_bounds(specs) -> tuple[list[float], list[float]]:
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


def gaussian_starts(origin: np.ndarray, specs, n_starts: int, seed: int) -> list[np.ndarray]:
    """``origin``, then Gaussian kicks of scale 0.4, clipped to the spec bounds."""
    rng = np.random.default_rng(seed)
    lower, upper = unconstrained_bounds(specs)
    starts = [np.clip(np.asarray(origin, dtype=float), lower, upper)]
    for _ in range(n_starts - 1):
        kicked = origin + rng.normal(scale=0.4, size=origin.size)
        starts.append(np.clip(kicked, lower, upper))
    return starts


def rectangle(model, data: MarketData, frame: pd.DataFrame) -> ModelSurface:
    k = frame["k"].to_numpy(dtype=float)
    t = frame["T"].to_numpy(dtype=float)
    domain = Domain(
        t_min=float(np.min(t)),
        t_max=float(np.max(t)),
        k_min=float(np.min(k)),
        k_max=float(np.max(k)),
    )
    return ModelSurface(model, Curves(data.F, data.df), domain)


def in_spread(frame: pd.DataFrame, model_price: np.ndarray) -> float:
    if "bid" not in frame.columns or "ask" not in frame.columns:
        return float("nan")
    bid = frame["bid"].to_numpy(dtype=float)
    ask = frame["ask"].to_numpy(dtype=float)
    two_sided = np.isfinite(bid) & np.isfinite(ask) & (ask > bid)
    if not two_sided.any():
        return float("nan")
    # A residual of a few ulps sits on the boundary when the loss is exactly zero.
    slack = 1e-8
    inside = (model_price[two_sided] >= bid[two_sided] - slack) & (
        model_price[two_sided] <= ask[two_sided] + slack
    )
    return float(np.mean(inside))
