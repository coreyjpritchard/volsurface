# Snapshots

## `SPX_2026-10-14.parquet`

Source: CBOE public delayed quotes,

<https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json>

The JSON field `timestamp` is `2026-10-14 10:01:12` and has no timezone. The
fetch that produced this file matched that clock in UTC, so `quote_time` is
that instant as UTC. On the same index object, `last_trade_time` is
`2026-10-13T16:14:59` (no zone), `current_price` is 7805.4502, `prev_day_close`
is 7805.4502, `close` is 7805.45, and `price_change` is -1.7898. At 10:01 UTC
the US cash session is not open, so this is a pre-open delayed book. The
option rows still carry a bid and an ask. The 13 MB JSON is not committed.
`scripts/build_cboe_spx_snapshot.py` rebuilds the parquet from a saved copy
or from the URL.

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

The two front discount factors sit above 1. The last one gives
`-log(df) / T = 0.0491`. The loader does not replace those values.

Re-running the script against a later download overwrites this path when the
New York date of the timestamp is still 14 October 2026, and writes a new
dated file when it is not. The committed parquet is this fetch.
