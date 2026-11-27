# Exports

Calibration of a SANOS-calibrated market: a parametric fit to the SANOS
surface from one historical snapshot of index options, in a normalised
market with spot 100. The target is the SANOS surface, not raw quotes.

- `examples/synthetic_snapshot_markets.json` is a committed schema example.
  Its premiums are from a labelled synthetic Heston surface, not from a
  historical snapshot. Rebuild it with `make synthetic-example`.
- `local/` is gitignored. It holds the training document (fitted parameters
  for a hedger, including a full-surface Heston fit and a contract-weighted
  fit) and, when a local snapshot or grid is supplied, the normalised
  implied-vol grid and a site pack built from that grid. Those files are
  for local training and a private website handoff, not for the public
  repository.

Rebuild a local calibration with `make calibration` after passing a grid
or a local snapshot path, for example `GRID=path/to/grid.json make
calibration`. Heston is fitted twice: equally across the surface with the
default numerical box, and with residual weights peaked at the 30-day
contract under a wider kappa box. Heston cannot match the short-dated
skew of a typical equity-index surface; both fits remain parametric
Heston models, not a simulation of the surface.
