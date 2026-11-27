# Exports

Calibration of a SANOS-calibrated market: a parametric fit to the SANOS
surface from one historical snapshot of index options, in a normalised
market with spot 100. The target is the SANOS surface, not raw quotes.

- `site/snapshot_markets.json` is the site pack: the normalised contract,
  premiums and labels. It does not contain fitted Heston parameters, an
  implied-vol grid, dates or expiry labels.
- `local/` is gitignored. It holds the training document and, when a local
  snapshot or grid is supplied, the normalised implied-vol grid used to
  build it. Those files are for training a hedger, not for the site.

Rebuild with `make calibration` after passing a grid or a local snapshot
path, for example `GRID=path/to/grid.json make calibration`.
