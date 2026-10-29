"""Build a :class:`~volsurface.core.types.MarketData` from a tidy chain.

Forwards and discount factors are those already inferred by
:func:`volsurface.chains.market_implied_vols`. This module does not refit them.
It records an uncertainty on the forward and fills exercise style, which the
parquet snapshot does not store.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from volsurface import chains
from volsurface.core.types import MarketData

# The SPX snapshot (and the synthetic test chain) is SPX: European and cash-settled. A ticker in
# _EQUITY_AMERICAN is still marked American and physical if a chain with that
# underlying is passed. Index options SPX and XSP are European and cash settled.
# Anything else must be labelled by the caller. This is a loader assumption,
# not a field we read.
_EQUITY_AMERICAN = {"SPY"}
_INDEX_EUROPEAN = {"SPX", "XSP"}

QUOTE_COLUMNS = [
    "T",
    "K",
    "kind",
    "bid",
    "ask",
    "exercise",
    "settlement",
    "volume",
    "price",
    "iv",
    "k",
    "F",
    "df",
    "expiry",
    "price_source",
    "underlying",
]


def market_data_from_snapshot(
    chain: pd.DataFrame,
    *,
    exercise: str | None = None,
    settlement: str | None = None,
    n_atm: int = 8,
) -> MarketData:
    """``MarketData`` for one tidy chain (see :mod:`volsurface.chains`).

    ``exercise`` and ``settlement`` default from the underlying. The committed
    chain is SPX, marked European and cash. SPY, if a chain with that ticker is
    passed, is marked American and physical. XSP is European and cash. Pass
    both explicitly for any other name.

    The working price is the chain's ``price`` column (mid when the loader had
    a two-sided quote, otherwise the last trade). Bid and ask are copied
    through and may be missing. ``F_err(T)`` is the standard deviation of the
    near-the-money forwards implied by put-call parity at that expiry, held
    flat outside the pillars. ``F`` and ``df`` follow :func:`_forward_curve`
    and :func:`_discount_curve`.
    """
    if chain.empty:
        raise ValueError("chain is empty")
    underlying = str(chain["underlying"].iloc[0])
    exercise, settlement = _style(underlying, exercise, settlement)

    market = chains.market_implied_vols(chain, n_atm=n_atm)
    if market.empty:
        raise ValueError("no expiry had a usable put-call pair")

    quotes = _join_quotes(chain, market, exercise, settlement)
    pillars = (
        market.groupby("expiry", sort=True)
        .agg(T=("T", "first"), F=("F", "first"), df=("df", "first"))
        .reset_index()
    )
    t_pillars = pillars["T"].to_numpy(dtype=float)
    f_of = _forward_curve(t_pillars, pillars["F"].to_numpy(dtype=float))
    # Log-linear discount factor: a constant forward rate on each gap, df(0) = 1.
    df_of = _discount_curve(t_pillars, pillars["df"].to_numpy(dtype=float))
    err_values = np.array(
        [
            _forward_scatter(chain, row.expiry, float(row.F), float(row.df), n_atm)
            for row in pillars.itertuples(index=False)
        ]
    )
    err_of = _interpolant(t_pillars, err_values)
    asof = pd.Timestamp(chain["quote_time"].iloc[0]).to_pydatetime()
    return MarketData(
        asof=asof,
        quotes=quotes,
        F=f_of,
        df=df_of,
        F_err=err_of,
        day_count="ACT/365",
    )


def _style(underlying: str, exercise: str | None, settlement: str | None) -> tuple[str, str]:
    if exercise is None or settlement is None:
        if underlying in _EQUITY_AMERICAN:
            exercise = exercise or "american"
            settlement = settlement or "physical"
        elif underlying in _INDEX_EUROPEAN:
            exercise = exercise or "european"
            settlement = settlement or "cash"
        else:
            raise ValueError(
                f"pass exercise and settlement for underlying {underlying!r}; "
                "the snapshot does not record them"
            )
    return exercise, settlement


def _join_quotes(
    chain: pd.DataFrame, market: pd.DataFrame, exercise: str, settlement: str
) -> pd.DataFrame:
    """One row per out-of-the-money quote that received an implied vol."""
    extra = chain[["expiry", "kind", "strike", "bid", "ask", "volume", "underlying"]].copy()
    extra["expiry"] = pd.to_datetime(extra["expiry"])
    merged = market.merge(extra, on=["expiry", "kind", "strike"], how="left")
    out = pd.DataFrame(
        {
            "T": merged["T"].to_numpy(dtype=float),
            "K": merged["strike"].to_numpy(dtype=float),
            "kind": merged["kind"].to_numpy(),
            "bid": merged["bid"].to_numpy(dtype=float),
            "ask": merged["ask"].to_numpy(dtype=float),
            "exercise": exercise,
            "settlement": settlement,
            "volume": merged["volume"].to_numpy(dtype=float),
            "price": merged["price"].to_numpy(dtype=float),
            "iv": merged["iv"].to_numpy(dtype=float),
            "k": merged["k"].to_numpy(dtype=float),
            "F": merged["F"].to_numpy(dtype=float),
            "df": merged["df"].to_numpy(dtype=float),
            "expiry": merged["expiry"].to_numpy(),
            "price_source": merged["price_source"].to_numpy(),
            "underlying": merged["underlying"].to_numpy(),
        }
    )
    return out.loc[:, QUOTE_COLUMNS].reset_index(drop=True)


def _forward_scatter(chain: pd.DataFrame, expiry, F: float, df: float, n_atm: int) -> float:
    """Standard deviation of parity-implied forwards near the money.

    Each strike with both a call and a put gives ``F_hat = K + (C - P) / df``.
    The scatter of those values is a rough error bar on the forward. It is not
    a standard error from a sampling model.
    """
    group = chain[pd.to_datetime(chain["expiry"]) == pd.to_datetime(expiry)]
    wide = group.pivot_table(index="strike", columns="kind", values="price", aggfunc="mean")
    if not {"call", "put"} <= set(wide.columns):
        return float("nan")
    both = wide.dropna(subset=["call", "put"])
    if len(both) < 2:
        return float("nan")
    K = both.index.to_numpy(dtype=float)
    implied = K + (both["call"].to_numpy(dtype=float) - both["put"].to_numpy(dtype=float)) / df
    idx = np.argsort(np.abs(K - F))[:n_atm]
    if idx.size < 2:
        return float("nan")
    return float(np.std(implied[idx], ddof=1))


def _interpolant(xp: np.ndarray, fp: np.ndarray) -> Callable:
    """Linear interpolation in expiry, flat outside the first and last pillar.

    Used for ``F_err``, which has no carry to extend. Forwards and discount
    factors use :func:`_forward_curve` and :func:`_discount_curve`.
    """
    order = np.argsort(xp)
    xp = np.asarray(xp, dtype=float)[order]
    fp = np.asarray(fp, dtype=float)[order]

    def evaluate(t):
        t_arr = np.asarray(t, dtype=float)
        y = np.interp(np.atleast_1d(t_arr), xp, fp)
        if t_arr.ndim == 0:
            return y.reshape(())
        return y.reshape(t_arr.shape)

    return evaluate


def _forward_curve(times: np.ndarray, forwards: np.ndarray) -> Callable:
    """Forward curve, linear in calendar time between pillars.

    Left of the first pillar the first carry is extended, and right of the
    last pillar the last carry is extended. The carry on a gap is
    ``log(F_right / F_left) / (T_right - T_left)``, so

        F(T) = F_edge * exp(carry * (T - T_edge)).

    A single pillar has no second point, so the level is held outside it.
    A non-positive forward has no logarithm, and that gap's carry is 0.
    """
    order = np.argsort(times)
    times = np.asarray(times, dtype=float)[order]
    forwards = np.asarray(forwards, dtype=float)[order]

    def carry(i_left: int, i_right: int) -> float:
        dt = float(times[i_right] - times[i_left])
        left = float(forwards[i_left])
        right = float(forwards[i_right])
        if dt <= 0.0 or left <= 0.0 or right <= 0.0:
            return 0.0
        return float(np.log(right / left) / dt)

    if times.size >= 2:
        left_carry = carry(0, 1)
        right_carry = carry(-2, -1)
    else:
        left_carry = 0.0
        right_carry = 0.0

    def evaluate(t):
        t_arr = np.asarray(t, dtype=float)
        flat = np.atleast_1d(t_arr)
        y = np.interp(flat, times, forwards).astype(float)
        before = flat < times[0]
        after = flat > times[-1]
        if np.any(before) or np.any(after):
            y = y.copy()
            y[before] = forwards[0] * np.exp(left_carry * (flat[before] - times[0]))
            y[after] = forwards[-1] * np.exp(right_carry * (flat[after] - times[-1]))
        if t_arr.ndim == 0:
            return y.reshape(())
        return y.reshape(t_arr.shape)

    return evaluate


def _discount_curve(times: np.ndarray, dfs: np.ndarray) -> Callable:
    """Discount factor, log-linear between pillars (a constant forward rate on each gap).

    If the first pillar is strictly after 0, ``log df`` is anchored at
    ``(0, 0)``, so ``df(0) = 1`` and times before the first pillar are not
    stuck at ``df(T_1)``. Past the last pillar the slope of ``log df`` on the
    last gap between market pillars is extended. One market pillar has no
    second point to define that rate, so ``df`` is held flat past the pillar.
    The anchor, when present, still fixes the segment from 0 to that pillar.
    Times before 0 stay at ``df(0)``.
    """
    order = np.argsort(times)
    times = np.asarray(times, dtype=float)[order]
    log_df = np.log(np.asarray(dfs, dtype=float)[order])
    if times.size >= 2 and times[-1] > times[-2]:
        right_slope = float((log_df[-1] - log_df[-2]) / (times[-1] - times[-2]))
    else:
        right_slope = 0.0
    if times[0] > 0.0:
        knots_t = np.concatenate(([0.0], times))
        knots_y = np.concatenate(([0.0], log_df))
    else:
        knots_t = times
        knots_y = log_df

    def evaluate(t):
        t_arr = np.asarray(t, dtype=float)
        flat = np.atleast_1d(t_arr)
        y = np.interp(flat, knots_t, knots_y).astype(float)
        after = flat > times[-1]
        if np.any(after):
            y = y.copy()
            y[after] = log_df[-1] + right_slope * (flat[after] - times[-1])
        y = np.exp(y)
        if t_arr.ndim == 0:
            return y.reshape(())
        return y.reshape(t_arr.shape)

    return evaluate
