"""In-memory synthetic option chain for committed figures and the session app.

Nothing here is market data. The chain is European, cash-settled, with spot 100,
and every row has ``source="synthetic"``. Figure scripts write their committed
outputs from this chain by default. Pass ``--snapshot path/to/chain.parquet`` to
run the same scripts on a chain you supply; those outputs go to ``data/cache/``
and are not committed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from volsurface import chains, heston
from volsurface.core.market import market_data_from_snapshot

SOURCE_SYNTHETIC = "Synthetic option chain (Heston mid quotes with bid-ask noise). Not market data."
SPOT = 100.0
RATE = 0.03
DIVIDEND_YIELD = 0.01
# Steep short skew, Feller-friendly enough for stable prices.
HESTON = (0.04, 1.5, 0.04, 0.8, -0.7)
SEED = 20200102
QUOTE_TIME = pd.Timestamp("2020-01-02 21:00:00", tz="UTC")
# Ten expiries from one day to a year, generic calendar dates.
EXPIRIES = [
    dt.date(2020, 1, 3),
    dt.date(2020, 1, 9),
    dt.date(2020, 1, 16),
    dt.date(2020, 1, 31),
    dt.date(2020, 2, 21),
    dt.date(2020, 3, 20),
    dt.date(2020, 4, 17),
    dt.date(2020, 7, 17),
    dt.date(2020, 10, 16),
    dt.date(2021, 1, 15),
]
TICK = 0.01
UNDERLYING = "DEMO"
REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_FIG = REPO_ROOT / "docs" / "figures"
CACHE_FIG = REPO_ROOT / "data" / "cache" / "figures"


def _strikes(F: float, T: float) -> np.ndarray:
    """At most 40 strikes from 6 to 3 standard deviations around the forward."""
    width = 0.18 * np.sqrt(max(T, 1.0 / 365.0))
    step = 0.5 if T < 0.05 else 1.0 if T < 0.3 else 2.0
    lo, hi = F * np.exp(-6.0 * width), F * np.exp(3.0 * width)
    grid = np.arange(np.floor(lo / step) * step, hi + step, step)
    grid = grid[grid > 0]
    if len(grid) > 40:
        grid = grid[np.unique(np.linspace(0, len(grid) - 1, 40).round().astype(int))]
    return grid


def build(seed: int = SEED) -> pd.DataFrame:
    """A small, reproducible synthetic chain. Same file on every run."""
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
            spread = np.clip(0.01 + 0.04 * model, 0.02, 2.0)
            u = rng.uniform(0.15, 0.85, size=K.shape)
            bid = np.floor((model - u * spread) / TICK) * TICK
            ask = np.ceil((model + (1.0 - u) * spread) / TICK) * TICK
            for k, b, a in zip(K, bid, ask, strict=True):
                rows.append(
                    {
                        "expiry": expiry,
                        "kind": kind,
                        "strike": float(k),
                        "bid": round(float(max(b, 0.0)), 2),
                        "ask": round(float(max(a, TICK)), 2),
                        "last": np.nan,
                        "volume": 0.0,
                        "open_interest": 0.0,
                    }
                )
    return chains.tidy(
        pd.DataFrame(rows),
        underlying=UNDERLYING,
        quote_time=QUOTE_TIME,
        spot=SPOT,
        source="synthetic",
    )


def market_data(chain: pd.DataFrame):
    """``MarketData`` for a demo or user-supplied chain, labelled European and cash."""
    return market_data_from_snapshot(chain, exercise="european", settlement="cash")


def parse_snapshot_arg(argv: list[str] | None = None) -> Path | None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="local option-chain parquet. Outputs go to data/cache/figures/ and are not committed.",
    )
    args, _unknown = parser.parse_known_args(argv if argv is not None else sys.argv[1:])
    return args.snapshot


def load_figure_market(argv: list[str] | None = None):
    """Load the synthetic demo chain, or a user snapshot if ``--snapshot`` is given.

    Returns ``(chain, data, source_label, out_dir, is_synthetic)``.
    """
    snapshot = parse_snapshot_arg(argv)
    if snapshot is None:
        chain = build()
        data = market_data(chain)
        return chain, data, SOURCE_SYNTHETIC, DOCS_FIG, True
    if not snapshot.is_file():
        raise SystemExit(f"snapshot not found: {snapshot}")
    chain = chains.load_snapshot(snapshot)
    data = market_data(chain)
    label = f"User-supplied chain from {snapshot.name}. Not redistributed."
    CACHE_FIG.mkdir(parents=True, exist_ok=True)
    return chain, data, label, CACHE_FIG, False
