# volsurface

**On the 14 October 2026 SPX option chain (SPXW mid quotes, 40 expiries from 10 hours
to 351 days), the at-the-money skew follows a power law, $|\psi(T)| \approx 0.305\,
T^{-0.333}$. Rough volatility predicts this shape, with Hurst index
$H = 0.5 - 0.333 \approx 0.17$. Fitted to the at-the-money level and skew, Heston can follow
the power law only with its vol-of-vol on whatever upper bound it is given
($\xi = 5$, Feller ratio 0.05), and then its short-dated at-the-money vol is far too low
(5.6% against the market's 15.1% at 10 hours).** A rough Bergomi model with three
parameters, on a forward variance curve read from the same chain, matches the skew at all
40 expiries to within 5.2%.

This is a personal research project in mathematical finance: a Python library that turns
an option chain into an implied volatility surface and prices it under Black-Scholes,
Heston, SABR, local volatility and rough Bergomi through one interface. Every number on
this page is printed by a script in this repository; `make reproduce` rebuilds the
environment, runs the tests and redraws the first figure. [LOG.md](LOG.md) records what
each session found, dated, including the results that did not survive.

![SPX smiles against one Heston fit, and the at-the-money skew term structure](docs/00_the_smile.png)

**How to read this.** The left panel shows SPXW smiles at five expiries (dots) against one
Heston model (lines); the right panel shows the at-the-money skew, the slope of implied
vol in log-moneyness, for all 40 expiries on log-log axes, where a power law is a straight
line. The market points lie on a line of slope $-1/3$. Rough volatility models predict a
short-maturity skew of order $T^{H-1/2}$ (Alòs, León and Vives 2007; Fukasawa 2011), so
the slope gives $H \approx 0.17$, and rough Bergomi, built on this idea, fits SPX skews well
(Bayer, Friz and Gatheral 2016). In Heston (1993) the variance is a diffusion, so as
$T \to 0$ the at-the-money skew tends to a finite limit, $\rho\xi / (4\sqrt{v_0})$, and the
at-the-money vol tends to $\sqrt{v_0}$. To keep the skew rising down to 10 hours the fit has
to push that limit far above the data (14.4 here), and the cheapest way is a large $\xi$
and a small $v_0$: $\xi$ ends on its bound and $\sqrt{v_0} = 5.4\%$, which is why the
34-hour smile in the left panel dips far below the quotes at the money. The caveat: this
is one snapshot, and the Heston optimum is bound-limited. With $\xi \le 10$ the fit moves
again ($\xi$ on the new bound, Feller ratio 0.03, cost 0.020 against 0.066) and its
10-hour at-the-money vol is still only 7.0%, so the failure is the level-skew trade-off,
not a missing local minimum.

![SPX skew against Heston and three rough Bergomi calibrations](docs/figures/p3e_skew.png)

**How to read this.** Black dots are the market skews again; the open markers are a
Heston fit to option prices (not to the skew) and rough Bergomi calibrated to the
at-the-money level and skew with $H$ free (fitted $H = 0.207$), and with $H$ fixed at
0.167 and at 0.149. Rough Bergomi tracks the market from 10 hours to a year: the free-$H$
fit is within 5.2% of the market skew at every expiry (median 1.8%), with three
parameters plus a forward variance curve read off the market. The price-fitted Heston
needs $\xi = 2.70$ (Feller ratio 0.14) to reach the 10-hour skew and then overshoots by more
than 5% from 1.4 to 128 days (by up to 52%, at 5 days). In Heston the skew is roughly
constant for maturities well below $1/\kappa$ and decays like $1/T$ well above it (Gatheral
2006); with one mean-reversion speed ($1/\kappa \approx 27$ days here) it cannot decay like
$T^{-1/3}$ over three orders of magnitude in maturity. Caveats: the rough Bergomi skews
are Monte Carlo estimates (4,000 paths, 64 steps, seed 0) with no error bars drawn,
the fitted at-the-money level is up to 2.4 vol points above the market at the front, and
$H = 0.21$ from one day's option prices is not the $H \approx 0.1$ that
Gatheral, Jaisson and Rosenbaum (2018) estimate from realised volatility time series.

Earlier versions of the first figure reported a Heston fit with $\rho = -0.533$ and a
10-hour skew of 1.87, and then $\rho = -0.551$ and 2.12, both in the narrower session box
($v_0 \ge 0.01$, $\xi \le 2$). [docs/notes/heston_fit_old_vs_new.md](docs/notes/heston_fit_old_vs_new.md)
reproduces each version and explains why they differ.

## What is in the library

| Path | Contents |
|---|---|
| `volsurface/bs.py`, `volsurface/models/bs.py`, `models/gbm.py` | Black-Scholes in forward form, Greeks, vectorised implied vol; GBM Monte Carlo |
| `volsurface/models/heston.py`, `volsurface/heston.py` | Heston: Lewis (2001) Fourier prices, the COS method of Fang and Oosterlee (2008), Euler simulation |
| `volsurface/models/sabr.py` | SABR with the Hagan et al. (2002) expansion |
| `volsurface/models/localvol.py` | Dupire (1994) local volatility |
| `volsurface/surface/` | Implied surfaces: linear, spline, raw SVI (Gatheral 2004), SANOS, discrete local vol |
| `volsurface/fit/` | Calibration: Heston, SABR, SVI, SANOS (arXiv:2601.11209) as a sparse linear programme, rough Bergomi |
| `volsurface/arbitrage/` | Static arbitrage checks: bounds, call spreads, butterflies, calendar order |
| `volsurface/greeks/` | Greeks five ways: analytic, finite difference, pathwise, likelihood ratio, autodiff (torch) |
| `volsurface/dynamic/` | A minimal dynamic SANOS (arXiv:2608.12587) |
| `volsurface/markets/` | Simulated markets with regime switching, a Hamilton filter, and pricers for the regime experiment |
| `volsurface/mc/` | Random number streams and time grids for Monte Carlo |
| `volsurface/chains.py` | Option chains from yfinance or IBKR (CBOE through `scripts/build_cboe_spx_snapshot.py`), snapshots, forwards, market implied vols |
| `volsurface/smile.py` | At-the-money level and skew, power-law fits |
| `volsurface/core/`, `docs/DESIGN.md` | Shared types and protocols; the design document |
| `scripts/`, `docs/figures/` | One script per figure, with the CSV or JSON it printed |
| `sessions/` | marimo apps, one per session |
| `dashboard/`, `volsurface/environments.py` | The volatility environment lab and the library code behind it |
| `data/snapshots/` | The committed SPX chain (parquet) |

Rough Bergomi itself (the hybrid scheme of Bennedsen, Lunde and Pakkanen 2017) lives in
a separate package, [rough-volatility](https://github.com/coreyjpritchard/rough-volatility),
pinned to its v0.1.0 tag in `pyproject.toml` (commit 65f3f99 in `uv.lock`) and loaded
through a plugin entry point.

## How to reproduce

Python 3.11. Dependencies are pinned in `uv.lock` ([uv](https://docs.astral.sh/uv/)); torch
comes from the CPU wheel index. One command rebuilds the environment, runs the fast tests
and redraws the first figure (about five minutes, most of it the Heston multi-start):

```bash
make reproduce
# the same by hand:
uv sync --locked --extra dev --extra autodiff
uv run pytest -m "not slow"
uv run python docs/figure_00.py
```

The other figures: `uv run python scripts/figures_p3e.py` (rough Bergomi, Monte
Carlo), `scripts/figures_p2c.py` (Heston price fit and COS checks), and so on, one script
per figure. `XI_MAX=10 uv run python docs/figure_00.py` reruns the Heston fit with the wider
$\xi$ bound quoted above.

The interactive app:

```bash
uv run marimo run sessions/00_the_smile.py   # sliders, surface, smile
uv run marimo edit sessions/00_the_smile.py  # the same, with the code
```

Plain pip also works (`pip install -e ".[dev]"`, then `pytest -m "not slow"`). Without the
`autodiff` extra the torch tests are skipped rather than failing at collection. Without
Python, download the static export [docs/00_the_smile.html](docs/00_the_smile.html) and
open it in a browser (GitHub shows the file as source rather than rendering it).

GitHub Actions runs `ruff check` and the fast tests on every push to `main` and on pull
requests (`.github/workflows/ci.yml`); the slow tests are run by hand.

Fetch a new snapshot:

```bash
python scripts/build_cboe_spx_snapshot.py    # CBOE delayed SPXW quotes; the committed chain
python -m volsurface.chains yfinance SPX     # writes data/snapshots/SPX_<date>.parquet
```

## Volatility environment lab

```bash
make dashboard        # = uv run --locked --extra dev marimo run dashboard/vol_lab.py
```

A marimo app for playing with volatility environments. Pick one of seven: Black-Scholes
(one flat vol), Heston, SABR, a raw SVI surface, rough Bergomi (Monte Carlo, through the
plug-in), and the raw SVI and SANOS fits to the 14 October 2026 SPXW chain. Each has
sliders and a few presets, including ones built to break: SVI with calendar arbitrage,
SVI and SABR with butterfly arbitrage, and the README's bound-limited Heston fit. Every
panel is computed live by the library; the app holds no pricing code
(`volsurface/environments.py` does the work). The SPX fits are read from
`data/presets/spx_fits.npz`, which `scripts/build_dashboard_presets.py` rebuilds in about
three minutes.

![The dashboard: Heston controls, implied volatility surface and one smile](docs/figures/dashboard_surface.png)

**How to read this.** Left: the environment, its parameters and the arbitrage status.
Middle: Black implied vol over strike and expiry (log-moneyness $k = \log(K/F)$, ten
expiries from 7 days to 2 years, each spanning the same number of standard deviations).
Right: the smile at one expiry, with the other expiries in grey. Below these (not shown)
are the at-the-money vol and the at-the-money skew against expiry on log-log axes, with
the fitted slope. The takeaway is that each parameter has a fingerprint: in Heston,
$\rho$ tilts the smile, $\xi$ bends it, and $\kappa$ sets where the skew stops being
flat and starts decaying like $1/T$, so the log-log skew line bends; in rough Bergomi it
stays straight with slope $H - 1/2$. Red markers appear where `check_surface` finds a
call-spread, butterfly or calendar violation. What it does not show: the checks are
pointwise on the grid, so a pass says nothing between nodes, and the rough Bergomi
surface is a Monte Carlo estimate (4,000 paths) with no error bars.

![The dashboard: a vanna surface and all six Greeks at one expiry](docs/figures/dashboard_greeks.png)

**How to read this.** Left: one Greek as a surface over strike and expiry (here vanna,
the change in delta per vol point). Right: delta, gamma, vega, vanna, volga and theta per
day at the chosen expiry. All are Black Greeks at each option's own implied vol, per 100
of forward, with zero rates. Gamma and theta are the same surface with opposite sign
($\Theta = -\tfrac12\sigma^2F^2\Gamma$, checked live in the app), gamma piles up at the
money and at short expiries, vega grows like $\sqrt T$, vanna changes sign across the
money and volga is positive in the wings. The delta panel has two lines: the
sticky-strike Black delta, and the delta when the smile moves with the forward, which is
the model's own delta for Heston and the alpha-fixed Hagan delta for SABR. What it does
not show: these are not minimum-variance hedge ratios, and for the Monte Carlo environment they are Black Greeks
of the Monte Carlo smile, not pathwise model Greeks. `tests/test_environments.py` checks
the Greeks against finite differences and closed forms.

### Static web version

```bash
make public-vollab    # = python scripts/build_public_vollab.py build/public-vollab
python -m http.server -d build/public-vollab
```

The same app as one static folder that runs entirely in the visitor's browser (Pyodide,
through `marimo export html-wasm`), with no server behind it. It ships the `volsurface`
wheel, the plotly wheel and `data/presets/spx_fits.npz`, and by default it also
self-hosts the Pyodide runtime, its packages and MathJax, so the page requests nothing
from any other host (`--cdn` loads those from public CDNs instead). Every environment
works except rough Bergomi, which needs the separate rough-volatility package and shows
a note to clone this repository instead. The folder is about 80 MB; a first visit
downloads about 67 MB and shows the first plots after roughly 15 to 30 seconds,
depending on the connection, and later moves take one to three seconds.

## Next: the regime experiment (draft pre-registration, no results)

[experiments/regime/PREREG.md](experiments/regime/PREREG.md) pins, before any run, a test
of whether a hedging policy trained in one volatility regime transfers to another. The
market is rough Bergomi whose vol-of-vol switches between two values on a Markov chain;
the hedgers (Buehler et al. 2019) live in a separate repository. The draft fixes the
liabilities, instruments, risk measure, the three hypotheses and their multiple-testing
correction. It is a draft and is not frozen, and there are no results yet.

## Honest limits

- One snapshot. Every SPX number here comes from one chain, taken at 06:01 New York time on
  14 October 2026, in the overnight (global trading hours) session before the cash open,
  not in the cash session.
- The skew is a quadratic fit within $\max(0.15\sqrt{T}, 0.01)$ of the money; another
  window would move the exponent slightly.
- Calibrations are local least squares from seeded starts, not global optimisation. The
  headline Heston fit ends on its $\xi$ bound, and $\kappa$ is poorly identified (8.0 to
  10.8 across the starts that end within 1% of the best cost).
- Because that fit sits in a flat valley in $\kappa$, the last digits of the first figure's
  numbers (the Feller ratio, the short-skew limit) can differ slightly from machine to
  machine with the linear algebra library and thread count (Feller 0.045 against 0.046,
  limit 14.4 against 14.6, on two Linux machines); the conclusions do not.
- Raw SVI fitted slice by slice is not free of static arbitrage on this chain: 2,246
  violations on the check grid, against 0 for SANOS (`docs/figures/p3c_shootout.csv`).
- The yfinance fetcher has no test, and the IBKR test needs a live gateway.

## References

- Alòs, E., León, J. A. and Vives, J. (2007). On the short-time behavior of the implied volatility for jump-diffusion models with stochastic volatility. *Finance and Stochastics* 11(4), 571-589.
- Bayer, C., Friz, P. and Gatheral, J. (2016). Pricing under rough volatility. *Quantitative Finance* 16(6), 887-904.
- Bennedsen, M., Lunde, A. and Pakkanen, M. S. (2017). Hybrid scheme for Brownian semistationary processes. *Finance and Stochastics* 21(4), 931-965.
- Buehler, H., Gonon, L., Teichmann, J. and Wood, B. (2019). Deep hedging. *Quantitative Finance* 19(8), 1271-1291.
- Dupire, B. (1994). Pricing with a smile. *Risk* 7(1), 18-20.
- Fang, F. and Oosterlee, C. W. (2008). A novel pricing method for European options based on Fourier-cosine series expansions. *SIAM Journal on Scientific Computing* 31(2), 826-848.
- Fukasawa, M. (2011). Asymptotic analysis for stochastic volatility: martingale expansion. *Finance and Stochastics* 15(4), 635-654.
- Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility parameterization with application to the valuation of volatility derivatives. Global Derivatives and Risk Management, Madrid.
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley.
- Gatheral, J. and Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance* 14(1), 59-71.
- Gatheral, J., Jaisson, T. and Rosenbaum, M. (2018). Volatility is rough. *Quantitative Finance* 18(6), 933-949.
- Hagan, P. S., Kumar, D., Lesniewski, A. S. and Woodward, D. E. (2002). Managing smile risk. *Wilmott Magazine*, September, 84-108.
- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. *Review of Financial Studies* 6(2), 327-343.
- Lewis, A. L. (2001). A simple option formula for general jump-diffusion and other exponential Lévy processes. Working paper, OptionCity.net.
- SANOS: arXiv:2601.11209. Dynamic SANOS: arXiv:2608.12587.

## Licence

MIT.

---

<sub>Built with the help of AI coding tools; all results are reproducible from the scripts in this repo.</sub>
