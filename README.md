# volsurface

Pull an option chain, turn it into an implied volatility surface, and price it under
Black-Scholes, Heston, SABR and rough Bergomi through one interface.

![SPX smiles against one Heston fit, and the at-the-money skew term structure](docs/00_the_smile.png)

The left panel shows SPX implied vols on 14 October 2026 (points) and the best of three
Heston starts fitted to their at-the-money levels and skews (lines). The right panel is
the reason this repository exists: the market's at-the-money skew grows like a power of
$1/T$ as expiry shrinks, while Heston's flattens to a constant. Rough volatility models
predict the power law. The sessions below work towards pricing them. The chain is SPX,
not SPY: see [docs/notes/spx_not_spy.md](docs/notes/spx_not_spy.md).

## Why

A personal research project in mathematical finance. Each session asks one question,
answers it with an interactive figure, and builds on the last. See [ROADMAP.md](ROADMAP.md)
for the ladder and [LOG.md](LOG.md) for what each session found.

## Run it

Python 3.11. Dependencies are pinned in `uv.lock` ([uv](https://docs.astral.sh/uv/)); torch
comes from the CPU wheel index. One command rebuilds the environment, runs the fast tests
and redraws the README figure:

```bash
make reproduce
# the same by hand:
uv sync --locked --extra dev --extra autodiff
uv run pytest -m "not slow"
uv run python docs/figure_00.py
```

The interactive app:

```bash
uv run marimo run sessions/00_the_smile.py   # sliders, surface, smile
uv run marimo edit sessions/00_the_smile.py  # the same, with the code
```

Plain pip also works (`pip install -e ".[dev]"`, then `pytest -m "not slow"`). Without the
`autodiff` extra the torch tests are skipped rather than failing at collection.

Without Python, open the static export: [docs/00_the_smile.html](docs/00_the_smile.html).

Fetch a new snapshot (during US market hours for mid quotes; outside them the yfinance
backend falls back to the last session's trade prices):

```bash
python -m volsurface.chains yfinance SPX     # writes data/snapshots/SPX_<date>.parquet
python scripts/build_cboe_spx_snapshot.py    # CBOE delayed SPXW quotes; the committed chain
```

## Layout

| Path | Contents |
|---|---|
| `volsurface/bs.py` | Black-Scholes in forward form, Greeks, vectorised implied vol |
| `volsurface/heston.py` | Heston characteristic function (trap-free form), Fourier prices, Euler simulation |
| `volsurface/models/` | The `Model` interface, Black-Scholes, Heston, and `implied_surface` |
| `volsurface/core/` | Shared types and protocols (`docs/DESIGN.md`) |
| `docs/DESIGN.md` | Source of truth for interfaces, coordinates, and the pull-request stack |
| `volsurface/chains.py` | Option chains from yfinance or IBKR, snapshots, forwards, market implied vols |
| `volsurface/smile.py` | At-the-money level and skew, power-law fits |
| `sessions/` | marimo apps, one per session |
| `tests/` | pytest; mathematical tests run against `volsurface/` or `exercises/` |
| `data/snapshots/` | committed option chain snapshots (parquet) |
| `docs/` | static exports and the README figure (`python docs/figure_00.py`) |

## Found so far

- **Session 00.** On the 14 October 2026 SPXW mids, the at-the-money skew follows
  $|\psi(T)| \approx 0.305\, T^{-0.333}$ from about 10 hours to 351 days. The best of three
  Heston starts in the slider box has $\rho = -0.533$, inside $[-0.99, 0.99]$. The front
  skew is 1.87 in the fit against the market's 2.96. At 351 days the fit is 0.18 against
  the market's 0.28.

## Licence

MIT.
