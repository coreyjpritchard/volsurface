# Log

Dated entries, newest first. What was learned, what surprised, open questions.

**Current state:** Session 00 built. **Next:** Session 01, SVI on the SPY snapshot.

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
