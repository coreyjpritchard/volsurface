# Snapshots

**The Cboe snapshot is not redistributed.** The SPX results in this repository (figures,
notes, fitted parameters) were computed on `SPX_2026-10-14.parquet`, a 14 October 2026
snapshot of the SPXW chain from Cboe's public delayed quotes. The file was removed from
the repository on 29 October 2026: Cboe's terms of use (<https://www.cboe.com/en/terms/>)
restrict republishing their data.
Everything in this folder but this README is ignored by git, so a local copy stays local.

To work with real data, build your own snapshot with `scripts/build_cboe_spx_snapshot.py`,
from a saved copy of the JSON below (`python scripts/build_cboe_spx_snapshot.py _SPX.json`).
Cboe's data is subject to Cboe's terms, and complying with them is up to you. A snapshot
taken on another day is a different chain, so its numbers will not match the committed
figures. Without a snapshot the tests run on the synthetic chain in `tests/fixtures/`, and
the figure scripts stop with a message pointing here.

The rest of this file describes the 14 October 2026 snapshot the results were computed on.

## `SPX_2026-10-14.parquet`

Source: CBOE public delayed quotes,

<https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json>

The JSON field `timestamp` is `2026-10-14 10:01:12` and has no timezone. The
fetch that produced this file matched that clock in UTC, so `quote_time` is
that instant as UTC. That is 06:01 in New York, inside the CBOE global
trading hours session and before the regular cash open. On the same index
object, `last_trade_time` is `2026-10-13T16:14:59` (no zone), `current_price`
is 7805.4502, `prev_day_close` is 7805.4502, `close` is 7805.45, and
`price_change` is -1.7898. The index print is the previous close, not a
cash-session trade. Parity forwards do not use it. The option rows still
carry a bid and an ask. Neither the 13 MB JSON nor the parquet is committed.
A later snapshot should be taken in the regular session.

`scripts/build_cboe_spx_snapshot.py` builds the parquet from a saved copy of
the JSON, or downloads it from the URL when no path is given. Cboe's delayed
quote pages prohibit downloading their data with automated programs, so check
Cboe's terms before using the download.

### Settlement

Only the root SPXW is kept. SPXW options are PM-settled at 16:00 New York.
The root SPX is the standard monthly series and is AM-settled against the
opening print. Dates on which both roots are listed keep SPXW only, so one
expiry is never a mix of the two settlements. Every SPXW expiry with positive
time to expiry is kept, including dailies and weeklies, not only third
Fridays. The short end of the skew is the point of the series. This is the
reading of "standard expiries" used here: one PM settlement, applied to the
whole SPXW book. A later cut to Friday weeklies, or to third-Friday PM
contracts only, would be a different file.

### Filters

`volsurface.chains.tidy` with moneyness 0.5 to 1.3 times the index print,
minimum price 0.03, and `allow_last` false, so a row needs a two-sided quote.
Rows with ask below bid are dropped before that step. On this fetch none of
the kept rows were crossed. `rate` is stored as NaN. `source` is `cboe`.
The tidy table has 18,543 rows and 40 expiries. Time to expiry runs from
0.001139 year (14 October 2026, about 10 hours) to 0.962783 year
(30 September 2027).

### Forward and discount factor

`market_implied_vols` does not read a dividend yield and does not read a
Treasury bill. At each expiry it regresses `C - P` on `K` over the eight
strikes nearest the point where `C - P` changes sign:

```
C - P = df * F - df * K
```

so `df = -slope` and `F = intercept / df`. That is put-call parity for a
European option. The forward already contains whatever dividends the index
implies. An explicit `rate` argument, or a finite `rate` column, still
replaces `df` with `exp(-r T)` and sets `F` to the median of
`K + (C - P) / df` near the money. This file leaves the column blank, so the
parity pair stands.

Measured pillars (not a smoothed curve):

| expiry | T | F | df |
| --- | --- | --- | --- |
| 2026-10-14 | 0.001139 | 7823.9057 | 1.000357 |
| 2026-10-15 | 0.003879 | 7824.2157 | 1.001786 |
| 2026-10-16 | 0.006619 | 7826.7561 | 0.998571 |
| 2027-09-30 | 0.962783 | 8133.2863 | 0.953810 |

The two front discount factors sit above 1. `-log(df) / T` is `-0.313` on
the front expiry (about 10 hours) and `-0.460` on the next (about 1.4
days). The last expiry gives `-log(df) / T = 0.0491`. The loader does not
cap `df` at 1 and does not replace the parity pair with a fitted short
rate. Doing either would change every implied vol downstream of this file.

Re-running the script against a later download overwrites this path when the
New York date of the timestamp is still 14 October 2026, and writes a new
dated file when it is not. The numbers above are for the 14 October 2026 fetch.
