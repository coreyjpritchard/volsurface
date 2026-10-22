# volsurface

Pull an option chain, turn it into an implied volatility surface, and price it under
Black-Scholes, Heston, SABR and rough Bergomi through one interface.

![SPX smiles against one Heston fit, and the at-the-money skew term structure](docs/00_the_smile.png)

The left panel shows SPXW mid quotes on 14 October 2026 (dots) and one calibrated Heston
model (lines), fitted to the at-the-money level and skew of all 40 expiries. The right panel
is the reason this repository exists: the market's at-the-money skew grows like
$T^{-1/3}$ as expiry shrinks. Rough volatility models predict the power law. Heston follows
it only with its vol-of-vol on the upper bound of the fit ($\xi = 5$), and then its
short-dated at-the-money vol is far too low (5.6% against 15.1% at 10 hours). The chain is SPX, not SPY: see
[docs/notes/spx_not_spy.md](docs/notes/spx_not_spy.md).

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
  $|\psi(T)| \approx 0.305\, T^{-0.333}$ from about 10 hours to 351 days. The Heston fit
  in `docs/figure_00.py` (35 seeded starts, $v_0 \ge 10^{-4}$, $\xi \le 5$) has
  $\rho = -0.622$ with $\xi$ on its bound and a Feller ratio of 0.05. Its skew follows the
  market (2.78 against 2.96 at the front, 0.29 against 0.28 at 351 days), but its
  at-the-money vol at 10 hours is 5.6% against 15.1%. With $\xi \le 10$ the fit moves to
  that bound too, so the result is bound-limited: on this loss Heston trades short-dated
  level for short-dated skew. Earlier versions, in a box with $\xi \le 2$, reported
  $\rho = -0.533$ and then $-0.551$ and a skew that went flat below about 2 days;
  [docs/notes/heston_fit_old_vs_new.md](docs/notes/heston_fit_old_vs_new.md) reproduces
  each version and explains the differences.

## Licence

MIT.
