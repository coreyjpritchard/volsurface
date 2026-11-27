# Session 00 Heston fit

Earlier versions of the session 00 figure compared two Heston fits on a
market option chain. Those fitted parameters and the market numbers have
been removed from this public repository, and
`scripts/heston_fit_old_vs_new.py` no longer runs that comparison.

The committed session 00 figure is drawn from the synthetic demo chain
(`docs/figure_00.py`, `volsurface.demo_market`). To fit Heston on a chain
you supply, run:

    python docs/figure_00.py --snapshot path/to/chain.parquet

Outputs go to `data/cache/figures/` and are not committed.
