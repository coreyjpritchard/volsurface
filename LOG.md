# Log

Dated entries, newest first. What was learned, what surprised, open questions.

**Current state:** Session 00 built. **Next:** Session 01, SVI on the SPY snapshot.

Library work below is not a session. Roadmap boxes stay unticked until Corey
works through the material.

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
