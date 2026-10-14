"""Build a tidy SPX snapshot from CBOE's public delayed quotes JSON.

The committed file is one fetch, not whatever is live when this script is
re-run. Pass a saved JSON path. With no argument the script downloads

    https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json

and writes ``data/snapshots/SPX_<New York date>.parquet``. A later download
overwrites that path when the New York date of the file timestamp is the same.

Only the SPXW root is kept. SPXW options are PM-settled at 16:00 New York,
which is the expiry close :func:`volsurface.chains.expiry_close_utc` uses.
The SPX root is the standard monthly series and is AM-settled against the
opening print. Dates where both roots are listed keep SPXW only, so one
expiry is never a mix of the two settlements. Every SPXW expiry with positive
time to expiry inside the tidy window is kept, including dailies and weeklies.

``quote_time`` is the JSON ``timestamp`` read as UTC. That field has no zone.
The fetch that produced the committed file matched the clock in UTC. ``rate``
is left NaN. :func:`volsurface.chains.market_implied_vols` then keeps the
put-call parity regression at each expiry (``df = -slope``, ``F = intercept / df``).
No dividend yield is fitted.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

from volsurface.chains import save_snapshot, tidy

CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"
_OCC = re.compile(r"^([A-Z0-9]+?)(\d{6})([CP])(\d{8})$")
# Tidy defaults. Stated here so the fixture does not depend on a later change
# to the function signature's defaults.
_MONEYNESS = (0.5, 1.3)
_MIN_PRICE = 0.03


def load_json(path: Path | None) -> dict:
    if path is None:
        with urllib.request.urlopen(CBOE_URL, timeout=60) as response:
            return json.load(response)
    return json.loads(path.read_text())


def spxw_rows(payload: dict) -> tuple[pd.DataFrame, pd.Timestamp, float]:
    """SPXW quotes, the file timestamp as UTC, and the index print."""
    stamp = payload["timestamp"]
    quote_time = pd.Timestamp(stamp)
    if quote_time.tzinfo is None:
        quote_time = quote_time.tz_localize("UTC")
    else:
        quote_time = quote_time.tz_convert("UTC")
    spot = float(payload["data"]["current_price"])
    rows = []
    for opt in payload["data"]["options"]:
        match = _OCC.match(opt["option"])
        if match is None or match.group(1) != "SPXW":
            continue
        expiry = dt.datetime.strptime(match.group(2), "%y%m%d").date()
        rows.append(
            {
                "expiry": expiry,
                "kind": "call" if match.group(3) == "C" else "put",
                "strike": int(match.group(4)) / 1000.0,
                "bid": opt.get("bid"),
                "ask": opt.get("ask"),
                "last": opt.get("last_trade_price"),
                "volume": opt.get("volume"),
                "open_interest": opt.get("open_interest"),
            }
        )
    raw = pd.DataFrame(rows)
    # A crossed market has no usable mid. tidy() would still average the two quotes.
    crossed = raw["bid"].notna() & raw["ask"].notna() & (raw["ask"] < raw["bid"])
    raw = raw.loc[~crossed].copy()
    return raw, quote_time, spot


def build(payload: dict) -> pd.DataFrame:
    raw, quote_time, spot = spxw_rows(payload)
    return tidy(
        raw,
        underlying="SPX",
        quote_time=quote_time,
        spot=spot,
        source="cboe",
        rate=float("nan"),
        allow_last=False,
        moneyness=_MONEYNESS,
        min_price=_MIN_PRICE,
    )


def main(argv: list[str]) -> None:
    path = Path(argv[1]) if len(argv) > 1 else None
    payload = load_json(path)
    chain = build(payload)
    out = save_snapshot(chain)
    print(
        f"{len(chain)} rows, {chain['expiry'].nunique()} expiries, "
        f"quote_time {chain['quote_time'].iloc[0]}, spot {chain['spot'].iloc[0]} -> {out}"
    )


if __name__ == "__main__":
    main(sys.argv)
