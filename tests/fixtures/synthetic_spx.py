"""A small synthetic SPX-like option chain for the tests. It is not market data.

The real Cboe snapshot is not redistributed with this repository (see
``data/snapshots/README.md``), so the tests that need a chain read this file instead:
``tests/fixtures/synthetic_spx_chain.parquet``, written by

    python -m tests.fixtures.synthetic_spx

Every price comes from one Heston model (Lewis Fourier prices, so the chain is free
of static arbitrage before any noise), on ten SPXW-style expiries from one day to a
year. Each quote is a bid and an ask on a 0.05 tick that bracket the model price at a
random point inside the spread, so the mids are noisy but the model price is always
inside the quote. ``source`` is ``"synthetic"`` on every row. The draws come from one
``numpy.random.default_rng(SEED)``, so the file is the same on every run.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from volsurface import chains, heston

PATH = Path(__file__).resolve().parent / "synthetic_spx_chain.parquet"
SEED = 20261014
QUOTE_TIME = pd.Timestamp("2026-10-14 14:30:00", tz="UTC")
SPOT = 7800.0
RATE = 0.04
DIVIDEND_YIELD = 0.013
# v0, kappa, theta, xi, rho: a steep, SPX-like skew.
HESTON = (0.012, 2.5, 0.032, 0.7, -0.8)
EXPIRIES = [
    dt.date(2026, 10, 15),
    dt.date(2026, 10, 16),
    dt.date(2026, 10, 23),
    dt.date(2026, 10, 30),
    dt.date(2026, 11, 20),
    dt.date(2026, 12, 18),
    dt.date(2027, 1, 15),
    dt.date(2027, 3, 19),
    dt.date(2027, 6, 17),
    dt.date(2027, 10, 15),
]
TICK = 0.05


def _strikes(F: float, T: float) -> np.ndarray:
    """At most 40 strikes from 6 to 3 standard deviations around the forward."""
    width = 0.18 * np.sqrt(T)
    step = 5.0 if T < 0.05 else 25.0 if T < 0.3 else 50.0
    lo, hi = F * np.exp(-6.0 * width), F * np.exp(3.0 * width)
    grid = np.arange(np.floor(lo / step) * step, hi + step, step)
    if len(grid) > 40:
        grid = grid[np.unique(np.linspace(0, len(grid) - 1, 40).round().astype(int))]
    return grid


def build(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for expiry in EXPIRIES:
        close = chains.expiry_close_utc(expiry)
        T = (close - QUOTE_TIME).total_seconds() / (365.0 * 86400.0)
        F = SPOT * np.exp((RATE - DIVIDEND_YIELD) * T)
        df = np.exp(-RATE * T)
        K = _strikes(F, T)
        for kind in ("call", "put"):
            model = heston.price(F, K, T, *HESTON, df=df, kind=kind)
            spread = np.clip(0.05 + 0.04 * model, 0.10, 20.0)
            u = rng.uniform(0.15, 0.85, size=K.shape)
            bid = np.floor((model - u * spread) / TICK) * TICK
            ask = np.ceil((model + (1.0 - u) * spread) / TICK) * TICK
            for k, b, a in zip(K, bid, ask, strict=True):
                rows.append(
                    {
                        "expiry": expiry,
                        "kind": kind,
                        "strike": float(k),
                        "bid": round(float(b), 2),
                        "ask": round(float(a), 2),
                        "last": np.nan,
                        "volume": 0.0,
                        "open_interest": 0.0,
                    }
                )
    return chains.tidy(
        pd.DataFrame(rows),
        underlying="SPX",
        quote_time=QUOTE_TIME,
        spot=SPOT,
        source="synthetic",
    )


def load() -> pd.DataFrame:
    """The committed synthetic chain."""
    return chains.load_snapshot(PATH)


if __name__ == "__main__":
    chain = build()
    chain.to_parquet(PATH, index=False, compression="zstd")
    print(f"wrote {PATH} ({len(chain)} rows, {chain['expiry'].nunique()} expiries)")
