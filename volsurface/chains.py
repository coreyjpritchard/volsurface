"""Option chains: fetch, store and convert to market implied volatilities.

Every backend returns the same tidy table, one row per option, with these columns:

=================  ==========================================================================
underlying         ticker, e.g. ``"SPY"``
quote_time         UTC timestamp at which the prices are valid
spot               underlying price at ``quote_time``
expiry             expiry date (options expire at 16:00 New York time on this date)
T                  time to expiry in years, ACT/365, from ``quote_time`` to 16:00 New York on expiry
kind               ``"call"`` or ``"put"``
strike             strike price
bid, ask           quotes (NaN if absent)
mid                ``(bid + ask) / 2`` when both quotes are positive, otherwise NaN
last               last trade price (NaN if absent)
price              the price used downstream: ``mid`` if available, else ``last`` (see below)
price_source       ``"mid"`` or ``"last"``
volume             contracts traded in the session (NaN if absent)
open_interest      open interest (NaN if absent)
source             ``"yfinance"`` or ``"ibkr"``
rate               continuously compounded short rate at ``quote_time`` (NaN if unknown)
=================  ==========================================================================

Outside US market hours yfinance reports zero bids and asks. In that case the yfinance
backend falls back to last trade prices from the most recent session, sets ``quote_time`` to that
session's close and records ``price_source = "last"``. Last prices are not synchronous, so
such a snapshot is noisier than one taken from live quotes.

``market_implied_vols`` infers the forward and discount factor per expiry from put-call
parity, ``C - P = df * (F - K)``, by regressing ``C - P`` on ``K`` near the money, and then
inverts out-of-the-money prices with :func:`volsurface.bs.implied_vol`. SPY options are
American; near the money and away from dividend dates the early exercise premium is small,
and using out-of-the-money options keeps it out of the implied vols.

The IBKR backend is the only code in the package that talks to Interactive Brokers. It
reads ``IB_HOST``, ``IB_PORT`` and ``IB_CLIENT_ID`` from the environment or from a ``.env``
file in the repository root (see ``.env.example``).
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Iterable
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from volsurface import bs

NEW_YORK = ZoneInfo("America/New_York")
REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = REPO_ROOT / "data" / "snapshots"

COLUMNS = [
    "underlying", "quote_time", "spot", "expiry", "T", "kind", "strike", "bid", "ask", "mid",
    "last", "price", "price_source", "volume", "open_interest", "source", "rate",
]


# %% Shared tidy step


def expiry_close_utc(expiry: dt.date) -> pd.Timestamp:
    """16:00 New York time on the expiry date, as a UTC timestamp."""
    close = dt.datetime.combine(expiry, dt.time(16, 0), tzinfo=NEW_YORK)
    return pd.Timestamp(close).tz_convert("UTC")


def tidy(
    raw: pd.DataFrame,
    *,
    underlying: str,
    quote_time: pd.Timestamp,
    spot: float,
    source: str,
    rate: float = float("nan"),
    allow_last: bool = False,
    moneyness: tuple[float, float] = (0.5, 1.3),
    min_price: float = 0.03,
) -> pd.DataFrame:
    """Turn raw quotes into the tidy schema and drop unusable rows.

    ``raw`` needs columns ``expiry`` (date), ``kind``, ``strike``, ``bid``, ``ask`` and may
    have ``last``, ``volume``, ``open_interest``. Rows are kept when the strike lies within
    ``moneyness`` times spot and the price is at least ``min_price``. When ``allow_last`` is
    true, rows without a two-sided quote use the last trade price.
    """
    df = raw.copy()
    for col in ("bid", "ask", "last", "volume", "open_interest"):
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df.loc[df["bid"] <= 0, "bid"] = np.nan
    df.loc[df["ask"] <= 0, "ask"] = np.nan
    df.loc[df["last"] <= 0, "last"] = np.nan

    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["price"] = df["mid"]
    df["price_source"] = "mid"
    if allow_last:
        use_last = df["mid"].isna()
        df.loc[use_last, "price"] = df.loc[use_last, "last"]
        df.loc[use_last, "price_source"] = "last"

    quote_time = pd.Timestamp(quote_time).tz_convert("UTC")
    expiry = pd.to_datetime(df["expiry"]).dt.date
    close = expiry.map(expiry_close_utc)
    df["T"] = (close - quote_time).dt.total_seconds().to_numpy() / (365.0 * 86400.0)
    df["expiry"] = pd.to_datetime(expiry)
    df["underlying"] = underlying
    df["quote_time"] = quote_time
    df["spot"] = float(spot)
    df["source"] = source
    df["rate"] = float(rate)

    keep = (
        (df["T"] > 0)
        & df["price"].notna()
        & (df["price"] >= min_price)
        & (df["strike"] >= moneyness[0] * spot)
        & (df["strike"] <= moneyness[1] * spot)
    )
    out = df.loc[keep, COLUMNS].sort_values(["expiry", "kind", "strike"])
    return out.reset_index(drop=True)


# %% Backend 1: yfinance


def fetch_yfinance(symbol: str = "SPY", max_years: float = 1.0) -> pd.DataFrame:
    """Fetch every expiry up to ``max_years`` from Yahoo Finance in the tidy schema."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    now = pd.Timestamp.now(tz="UTC")
    expiries = [
        d for d in (dt.date.fromisoformat(s) for s in ticker.options)
        if (expiry_close_utc(d) - now).days <= 365.0 * max_years + 1
    ]

    frames = []
    underlying_info: dict = {}
    for expiry in expiries:
        chain = ticker.option_chain(expiry.isoformat())
        underlying_info = chain.underlying or underlying_info
        for kind, table in (("call", chain.calls), ("put", chain.puts)):
            frames.append(
                pd.DataFrame(
                    {
                        "expiry": expiry,
                        "kind": kind,
                        "strike": table["strike"],
                        "bid": table["bid"],
                        "ask": table["ask"],
                        "last": table["lastPrice"],
                        "last_trade": pd.to_datetime(table["lastTradeDate"], utc=True),
                        "volume": table["volume"],
                        "open_interest": table["openInterest"],
                    }
                )
            )
    raw = pd.concat(frames, ignore_index=True)

    quoted = float(((raw["bid"] > 0) & (raw["ask"] > 0)).mean())
    state = underlying_info.get("marketState")
    live = state == "REGULAR" if state else quoted > 0.5
    if live:
        quote_time = now
        spot = float(underlying_info.get("regularMarketPrice") or _last_close(ticker))
    else:
        # Market closed: use the last session's trades, priced against its close.
        market_time = underlying_info.get("regularMarketTime")
        quote_time = (
            pd.Timestamp(market_time, unit="s", tz="UTC") if market_time
            else raw["last_trade"].max()
        )
        spot = float(underlying_info.get("regularMarketPrice") or _last_close(ticker))
        session_start = quote_time.tz_convert(NEW_YORK).normalize().tz_convert("UTC")
        raw = raw[raw["last_trade"] >= session_start]
    return tidy(raw, underlying=symbol, quote_time=quote_time, spot=spot, source="yfinance",
                rate=_tbill_rate(), allow_last=not live)


def _last_close(ticker) -> float:
    return float(ticker.history(period="5d")["Close"].iloc[-1])


def _tbill_rate() -> float:
    """13-week US Treasury bill rate (Yahoo ``^IRX``), continuously compounded.

    ``^IRX`` quotes the discount yield ``d`` in percent: price = 1 - d * 91 / 360.
    """
    import yfinance as yf

    try:
        d = float(yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1]) / 100.0
    except Exception:  # network or schema failure: fall back to the parity estimate
        return float("nan")
    return float(-np.log(1.0 - d * 91.0 / 360.0) / (91.0 / 365.0))


# %% Backend 2: Interactive Brokers (ib_async)


def read_env(path: Path | None = None) -> dict[str, str]:
    """Read ``KEY=VALUE`` lines from ``.env``; process environment variables take precedence."""
    path = path or REPO_ROOT / ".env"
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("'\"")
    for key in ("IB_HOST", "IB_PORT", "IB_CLIENT_ID", "IB_MARKET_DATA_TYPE"):
        if key in os.environ:
            values[key] = os.environ[key]
    return values


def ib_settings() -> tuple[str, int, int]:
    env = read_env()
    return (
        env.get("IB_HOST", "127.0.0.1"),
        int(env.get("IB_PORT", "4002")),
        int(env.get("IB_CLIENT_ID", "17")),
    )


def ib_reachable(timeout: float = 1.0) -> bool:
    """True if something is listening on the configured gateway host and port."""
    import socket

    host, port, _ = ib_settings()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _chunks(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_ibkr(
    symbol: str = "SPY",
    max_years: float = 1.0,
    moneyness: tuple[float, float] = (0.5, 1.3),
    batch: int = 90,
) -> pd.DataFrame:
    """Fetch the option chain from a running TWS or IB Gateway in the tidy schema.

    Steps: qualify the stock, read its price, list expiries and strikes with
    ``reqSecDefOptParams``, qualify every option contract (strikes do not exist for every
    expiry, so unqualified ones are dropped), then request snapshot tickers in batches below
    the 100 simultaneous market data lines limit. ``IB_MARKET_DATA_TYPE`` selects live (1),
    frozen (2), delayed (3) or delayed frozen (4) data; the default is 1.
    """
    from ib_async import IB, Option, Stock

    host, port, client_id = ib_settings()
    data_type = int(read_env().get("IB_MARKET_DATA_TYPE", "1"))

    ib = IB()
    ib.connect(host, port, clientId=client_id, readonly=True, timeout=10)
    try:
        ib.reqMarketDataType(data_type)
        stock = Stock(symbol, "SMART", "USD")
        if not ib.qualifyContracts(stock) or not stock.conId:
            raise ValueError(f"IB could not qualify stock {symbol}")
        [stock_ticker] = ib.reqTickers(stock)
        spot = stock_ticker.marketPrice()
        if not np.isfinite(spot):
            spot = stock_ticker.close
        quote_time = pd.Timestamp.now(tz="UTC")

        chains = ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)
        chain = next(c for c in chains if c.exchange == "SMART" and c.tradingClass == symbol)
        expiries = [
            d for d in (dt.datetime.strptime(e, "%Y%m%d").date() for e in chain.expirations)
            if 0 < (expiry_close_utc(d) - quote_time).days <= 365.0 * max_years + 1
        ]
        strikes = [k for k in chain.strikes if moneyness[0] * spot <= k <= moneyness[1] * spot]
        contracts = [
            Option(symbol, d.strftime("%Y%m%d"), k, right, "SMART", currency="USD",
                   tradingClass=symbol)
            for d in expiries
            for k in strikes
            for right in ("C", "P")
        ]
        qualified = []
        for group in _chunks(contracts, 200):
            # ib_async returns None in place of contracts it cannot qualify.
            qualified += [c for c in ib.qualifyContracts(*group) if c is not None and c.conId]

        rows = []
        for group in _chunks(qualified, batch):
            for t in ib.reqTickers(*group):
                c = t.contract
                rows.append(
                    {
                        "expiry": dt.datetime.strptime(
                            c.lastTradeDateOrContractMonth[:8], "%Y%m%d"
                        ).date(),
                        "kind": "call" if c.right == "C" else "put",
                        "strike": c.strike,
                        "bid": t.bid,
                        "ask": t.ask,
                        "last": t.last,
                        "volume": t.volume,
                    }
                )
    finally:
        ib.disconnect()
    raw = pd.DataFrame(rows)
    return tidy(raw, underlying=symbol, quote_time=quote_time, spot=spot, source="ibkr",
                moneyness=moneyness)


# %% Storage


def snapshot_path(chain: pd.DataFrame, directory: Path = SNAPSHOT_DIR) -> Path:
    underlying = chain["underlying"].iloc[0]
    date = pd.Timestamp(chain["quote_time"].iloc[0]).tz_convert(NEW_YORK).date()
    return Path(directory) / f"{underlying}_{date.isoformat()}.parquet"


def save_snapshot(chain: pd.DataFrame, path: Path | None = None) -> Path:
    path = Path(path) if path else snapshot_path(chain)
    path.parent.mkdir(parents=True, exist_ok=True)
    chain.to_parquet(path, index=False, compression="zstd")
    return path


def load_snapshot(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def latest_snapshot(underlying: str = "SPY", directory: Path = SNAPSHOT_DIR) -> Path:
    paths = sorted(Path(directory).glob(f"{underlying}_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no {underlying} snapshot in {directory}")
    return paths[-1]


# %% Forward inference and implied vols


def infer_forward(
    strikes: np.ndarray, calls: np.ndarray, puts: np.ndarray, spot: float, n_atm: int = 8
) -> tuple[float, float]:
    """Forward and discount factor from put-call parity near the money.

    Regress ``C - P`` on ``K`` over the ``n_atm`` strikes closest to the point where
    ``C - P`` changes sign: ``C - P = df * F - df * K``, so ``df = -slope`` and
    ``F = intercept / df``. Returns ``(F, df)``.
    """
    strikes, calls, puts = (np.asarray(a, dtype=float) for a in (strikes, calls, puts))
    diff = calls - puts
    # Rough forward: strike where |C - P| is smallest, then the n_atm strikes nearest to it.
    centre = strikes[np.argmin(np.abs(diff))] if len(strikes) else spot
    idx = np.argsort(np.abs(strikes - centre))[:n_atm]
    if len(idx) < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(strikes[idx], diff[idx], 1)
    df = -slope
    return float(intercept / df), float(df)


def market_implied_vols(snapshot: pd.DataFrame, n_atm: int = 8,
                        rate: float | None = None) -> pd.DataFrame:
    """Implied vols of out-of-the-money options, one row per (expiry, strike).

    Adds columns ``F`` and ``df`` (inferred per expiry), ``k = log(K / F)`` and ``iv``.
    Puts are used for ``K < F`` and calls for ``K >= F``. Expiries where parity cannot be
    fitted (fewer than two strikes with both a call and a put) are dropped.

    Each expiry gets a parity regression of ``C - P`` on ``K`` (:func:`infer_forward`). The
    regression pins down where ``C - P`` crosses zero, so ``F`` is well determined, but the
    slope ``-df`` is not: SPY puts are American and last trade prices are not synchronous,
    and both tilt the line. So the discount factors come from one continuously compounded
    rate, ``df_i = exp(-r T_i)``, taken from (in order) the ``rate`` argument, the
    snapshot's ``rate`` column, or a least squares fit of ``-log(df_i) = r T_i`` to the
    regression slopes. Each ``F_i`` is then re-estimated as the median of
    ``K + (C - P) / df_i`` over the near-the-money strikes.
    """
    fits = {}
    for expiry, group in snapshot.groupby("expiry", sort=True):
        wide = group.pivot_table(index="strike", columns="kind", values="price")
        if not {"call", "put"} <= set(wide.columns):
            continue
        both = wide.dropna(subset=["call", "put"])
        if len(both) < 2:
            continue
        K, C, P = both.index.to_numpy(), both["call"].to_numpy(), both["put"].to_numpy()
        F, df = infer_forward(K, C, P, float(group["spot"].iloc[0]), n_atm=n_atm)
        if np.isfinite(F) and 0.5 < df < 1.5:
            fits[expiry] = (float(group["T"].iloc[0]), F, df, K, C - P)

    if rate is None and "rate" in snapshot and np.isfinite(snapshot["rate"].iloc[0]):
        rate = float(snapshot["rate"].iloc[0])
    if fits:
        T = np.array([f[0] for f in fits.values()])
        y = -np.log([f[2] for f in fits.values()])
        r = float(T @ y / (T @ T)) if rate is None else rate
        for expiry, (t, F, _, K, diff) in fits.items():
            df = float(np.exp(-r * t))
            idx = np.argsort(np.abs(K - F))[:n_atm]
            fits[expiry] = (t, float(np.median(K[idx] + diff[idx] / df)), df, K, diff)

    out = []
    for expiry, (_, F, df, _, _) in fits.items():
        group = snapshot[snapshot["expiry"] == expiry]
        otm = group[
            ((group["kind"] == "put") & (group["strike"] < F))
            | ((group["kind"] == "call") & (group["strike"] >= F))
        ].copy()
        otm["F"] = F
        otm["df"] = df
        otm["k"] = np.log(otm["strike"].to_numpy() / F)
        otm["iv"] = bs.implied_vol(
            otm["price"].to_numpy(), F, otm["strike"].to_numpy(), otm["T"].to_numpy(), df,
            otm["kind"].to_numpy(),
        )
        out.append(otm)
    cols = ["expiry", "T", "kind", "strike", "price", "price_source", "F", "df", "k", "iv"]
    if not out:
        return pd.DataFrame(columns=cols)
    result = pd.concat(out, ignore_index=True)[cols]
    return result.dropna(subset=["iv"]).reset_index(drop=True)


# %% Command line: python -m volsurface.chains [yfinance|ibkr] [SYMBOL]

if __name__ == "__main__":
    import sys

    backend = sys.argv[1] if len(sys.argv) > 1 else "yfinance"
    symbol = sys.argv[2] if len(sys.argv) > 2 else "SPY"
    fetch = {"yfinance": fetch_yfinance, "ibkr": fetch_ibkr}[backend]
    chain = fetch(symbol)
    path = save_snapshot(chain)
    print(f"{len(chain)} rows, {chain['expiry'].nunique()} expiries -> {path} "
          f"({path.stat().st_size / 1e3:.0f} kB, prices from {chain['price_source'].unique()})")
