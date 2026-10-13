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

# Equity options in this repo's snapshots are American and physically settled.
# Index options SPX and XSP are European and cash settled. Anything else must
# be labelled by the caller. This is a loader assumption, not a field we read.
_EQUITY_AMERICAN = {"SPY"}
_INDEX_EUROPEAN = {"SPX", "XSP"}

QUOTE_COLUMNS = [
    "T", "K", "kind", "bid", "ask", "exercise", "settlement", "volume",
    "price", "iv", "k", "F", "df", "expiry", "price_source", "underlying",
]


def market_data_from_snapshot(
    chain: pd.DataFrame,
    *,
    exercise: str | None = None,
    settlement: str | None = None,
    n_atm: int = 8,
) -> MarketData:
    """``MarketData`` for one tidy chain (see :mod:`volsurface.chains`).

    ``exercise`` and ``settlement`` default from the underlying: SPY is marked
    American and physical, SPX and XSP European and cash. Pass them explicitly
    for any other name.

    The working price is the chain's ``price`` column (mid when the loader had
    a two-sided quote, otherwise the last trade). Bid and ask are copied
    through and may be missing. ``F_err(T)`` is the standard deviation of the
    near-the-money forwards implied by put-call parity at that expiry.
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
    f_of = _interpolant(t_pillars, pillars["F"].to_numpy(dtype=float))
    # Discount factors are log-linear between pillars: a constant forward rate on each gap.
    df_of = _interpolant(t_pillars, np.log(pillars["df"].to_numpy(dtype=float)), exp=True)
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


def _interpolant(xp: np.ndarray, fp: np.ndarray, *, exp: bool = False) -> Callable:
    """Linear interpolation in expiry, flat outside the first and last pillar."""
    order = np.argsort(xp)
    xp = np.asarray(xp, dtype=float)[order]
    fp = np.asarray(fp, dtype=float)[order]

    def evaluate(t):
        t_arr = np.asarray(t, dtype=float)
        y = np.interp(np.atleast_1d(t_arr), xp, fp)
        if exp:
            y = np.exp(y)
        if t_arr.ndim == 0:
            return y.reshape(())
        return y.reshape(t_arr.shape)

    return evaluate
