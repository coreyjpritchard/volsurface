# Snapshots

This folder is for a **local** option-chain snapshot you build yourself. Everything
in it except this README is gitignored. The repository does not redistribute
Cboe data, or figures and fitted parameters derived from it.

To work with real index-option data, build a snapshot with
`scripts/build_cboe_spx_snapshot.py` from a saved copy of Cboe's delayed-quotes
JSON (`python scripts/build_cboe_spx_snapshot.py _SPX.json`). Cboe's data is
subject to [Cboe's terms of use](https://www.cboe.com/en/terms/), and complying
with them is up to you. You can also fetch an equity chain with
`python -m volsurface.chains yfinance SPY`, which writes
`data/snapshots/SPY_<date>.parquet`.

Without a snapshot the tests and the committed figures run on synthetic chains
(`tests/fixtures/` and `volsurface.demo_market`). Figure scripts accept
`--snapshot path/to/chain.parquet` and write those outputs under
`data/cache/figures/` (also gitignored).

`scripts/build_cboe_spx_snapshot.py` keeps the SPXW root (PM-settled), tidies
quotes, and leaves `rate` as NaN so forwards come from the put-call parity
regression at each expiry. Check Cboe's terms before using the optional
download.
