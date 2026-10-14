# Log

Dated entries, newest first. What was learned, what surprised, open questions.

**Current state:** Session 00 built. **Next:** Session 01, SVI on the SPY snapshot.

Library work below is not a session. Roadmap boxes stay unticked until Corey
works through the material.

## 2026-10-14 Library P1c review: z-rule, calibration, out-of-the-money wings

The z-rule bounds come from `volsurface/mc/zrule.py` and scale with the node
count. A shared-path grid checks the maximum on one draw and the mean on 32
replications, using the observed standard deviation of those means. The
standard-error test is a calibration over 200 seeds, not a slope that holds
because `stderr = sd / sqrt(n)`. The implied-vol coverage grid is
standardised moneyness on `[-4, 4]`.

The figure prices puts for `k < 0` and calls otherwise. Two of 63 nodes have
no implied vol, both the one-week wings at `|k| = 0.10`, where the
out-of-the-money price on this seed is zero. The one-year curve sits about
0.45 volatility points below Black because the strikes share the paths.

## 2026-10-14 Library P1b review: checker, SVI diagnostics, two-sided quotes

`SVISurface` accepts list inputs, which is what `in_spread` passes. The floor
penalty in the SVI residual is always one row, so the Jacobian length stays
put. Each fitted slice reports `min_g`, Lee's bound, calendar order, and its
own `in_spread`. Spreads weight the fit when they exist. On the saved SPY
chain they do not, and 14 of 24 slices still have `g < 0` on `k` in
`[-0.5, 0.5]`. The figure says so, and shows the short-dated slice where `g`
reaches about -0.18.

The quote linear program carries a tail mass and moment instead of a strike
at 1.5 times the last quote. A normalised call of 0.8 at the money is
feasible. A call of 1 is not, up to a cap of `1e6` on the tail strike. An
empty bid-ask file returns `ok is None` and `n_checked == 0`. A flat Black
surface passes the default butterfly tolerance because its derivatives are
analytic. The mixin second-difference step is `1e-3`, with a `1e-8` floor
when the checker is still differencing.

No g-penalty and no SSVI fit. Both would change the 14 of 24 count instead
of reporting it.

## 2026-10-14 Library review: curves, seeds, and the z-rule

`ArbReport.ok` is None when the checker evaluated nothing, and `n_checked`
counts what it did evaluate. `as_seed_sequence` copies a `SeedSequence` so a
second spawn with common random numbers still draws child 0. The discount
curve is anchored at `df(0) = 1` and both curves extend the last carry past
the final pillar. The z-rule bounds scale with the node count; the helper is
`volsurface/mc/zrule.py`. Heston `c2` and `c4` are checked against a second
stencil, not against a copied closed form.

`Spot` stays a one-for-one forward bump with `df` fixed. That is the reading
in DESIGN section 8, and the greek tests lock `Spot == Forward`.

## 2026-10-13 Library P1c: geometric Brownian motion by Monte Carlo, in review

Exact and Euler stepping of the unit-mean martingale `Z`. One seed covers a
strike by expiry grid. The standard error is kept on `PriceResult`, off the
array `price` returns. Antithetic pairs are the second half of the batch. The
control is `Z` at expiry, mean 1.

Against Black, 200 nodes with independent seeds and 8,000 paths pass the
z-rule (max |z| below 4.5, the fraction inside 2 between 0.90 and 0.99, mean
z inside 0.5). From 10^3 to 10^6 paths the log standard error slopes at -0.5
within 0.05. One-step Euler at volatility 1 matches the normal formula and
misses Black, on 100,000 paths. Implied vol round trip: 85 of 124 nodes on the
closed-form grid have normalised vega at or below 1e-8 and are excluded. The
figure, 8,000 shared paths, has no implied vol at 6 of 63 nodes, all of them
one-week wings.

Hypotheses, not findings: the smooth shift of the one-year curve is the shared
seed; empty wings fill from more paths, not from a smoother interpolant; Euler
bias at volatility 0.2 hides inside a few thousand paths; the control helps
most at the money.

## 2026-10-13 Library P1b: surfaces and the checker, in review

Three interpolators of the normalised call: linear in the strike, a natural
cubic of implied volatility, and raw SVI with total variance linear in
expiry. The surface checker is pointwise (bounds, call spread, butterfly,
calendar). The quote checker is the feasibility linear program, not a fit.

On the saved SPY chain the figure uses the 30 September 2027 expiry. The
spline passes through the last-trade vols and still goes negative in density
between those strikes; past the lowest strike the same cubic crosses through
zero volatility. Raw SVI on that slice stays positive in density. Across all
24 expiries, 14 fitted slices have Durrleman's `g` negative somewhere on
`k` in `[-0.5, 0.5]`. The file has no bid or ask, so the quote checker drops
every row. `in_spread` on the SVI fit is NaN for that reason.

Hypotheses, not findings: flat extrapolation of the linear call is the SANOS
linear model; the SPY marks mix stale lasts, American premium, and real
convexity breaks; short-dated negative `g` is the least squares, not a
tradable wing; equal slack weights can book a calendar gap as a quote slack.

## 2026-10-13 Library P1a: interfaces, in review

`docs/DESIGN.md` is now the source of truth for coordinates (normalised call,
`k = log(K/F)`), the `Model` / `Surface` / `GreeksEngine` protocols, and the
pull-request stack. Black-Scholes and Heston implement it. `price` still
returns an array; a standard error lives on `PriceResult`. `state=None` means
the initial state, which is empty for Black-Scholes and the variance for Heston.

The Heston first cumulant matches `-1/2` times the integrated expected variance.
The second and fourth come from a fixed stencil on the log characteristic
function. The committed SPY snapshot still has no bid or ask.

Hypotheses, not findings: the SANOS HTML's normalised call matches the PDF;
marking SPY American and physical is the right contract label; Heston
cumulants from the stencil are accurate enough to place a cosine grid, which
is all Phase 1 asks of them.

## 2026-10-08 Session 00: the smile is the model's fingerprint

Built the package core (Black-Scholes, Heston, the `Model` interface, yfinance and IBKR
chain loaders) and the first app: a Heston implied vol surface with sliders, a Black-Scholes
switch and SPY market points on the same $(k, T)$ grid.

- Heston pricing matches Fang and Oosterlee (2008) to $2 \times 10^{-8}$ (5.785155434
  against 5.785155450 at $T = 1$) and converges to Black-Scholes as $\xi \to 0$.
- Snapshot `data/snapshots/SPY_2026-10-07.parquet`: 3,695 options, 24 expiries from 1 to 358
  days. Fetched before the US open, so prices are last trades of the 7 October session,
  priced against the close of 777.22. Rate 4.11% from the 13-week T-bill.
- SPY at-the-money skew: 1.56 at 1 day, 0.74 at 1 week, 0.28 at 1 year. A power law fits it
  well: $|\psi(T)| \approx 0.30\, T^{-0.26}$, so $H \approx 0.24$ if read through rough
  volatility.
- One Heston fit to all expiries' ATM vols and skews (scipy `least_squares`, three starts,
  box as in the app sliders with $\kappa \le 20$): $v_0 = 0.0098$, $\kappa = 4.26$,
  $\theta = 0.0373$, $\xi = 0.388$, $\rho = -0.99$ (at the bound). Skew 0.97 at 1 day
  against 1.56; 0.19 at 1 year against 0.28. Heston's skew is flat for $T \ll 1/\kappa$ and
  decays like $1/T$ beyond it; the market's does neither.
- Correction to the session brief: Heston can make a one-day skew steeper than the market's
  (6.3 at $\xi = 2$, $\rho = -0.99$). What it cannot do is match the short and long ends with
  one parameter set.

Open questions: why is the exponent 0.26 here and about 0.4 in the literature (window
choice, SPY against SPX, last trades against mids)? Does a mid-quote snapshot move it?
