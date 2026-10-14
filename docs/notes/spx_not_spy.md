# Why SPX, not SPY

SPY options are American. The holder can exercise before expiry, so the
European identity `C - P = df * (F - K)` does not hold, and a Black implied
volatility inverted from the price is not the volatility of a European claim
on the forward. SPY also pays dividends, so the forward is not the spot rolled
at the interest rate.

SPX options are European and cash-settled. Put-call parity then gives both the
forward and the discount factor from the quotes at each expiry. No dividend
yield is estimated. The dividends of the stocks in the index are already inside
that forward.

The committed chain keeps the SPXW root only. SPXW settles at the 16:00 New
York close, which is the expiry time `chains.expiry_close_utc` uses. The root
SPX is the standard monthly series and settles against the opening print. A
calendar day that lists both roots is two settlements, so the snapshot drops
SPX even when the date coincides. Dailies and weeklies in the file are SPXW.
They are not a third-Friday AM contract.

The file, the timestamp, and the parity regression are described in
`data/snapshots/README.md`.
