# P1c: one set of paths, a whole surface

## How to read this

The top panel is the implied vol recovered from Monte Carlo prices of
out-of-the-money options under a model whose true implied vol is a flat 20%;
the bottom panel is the same error in vol points. The takeaway is that one
set of 8,000 paths prices a whole surface to within about 0.6 vol points at 3
months and a year. The 1-week put wing is the exception. There a strike 3 or
more standard deviations out is reached by only a handful of paths, so the
price estimate is noisy, and a tiny vega turns a small price error into a
large vol error. At the two outermost 1-week strikes no path arrives at all
and there is nothing to invert. The caveat: this is one seed, and the shift
of the one-year curve is sampling noise of that seed, not a bias.

## What to watch

The figure `docs/figures/p1c_gbm.png` is geometric Brownian motion with
volatility 20%, simulated exactly, 8,000 paths, one seed. Three expiries share
those paths. Strikes with `k < 0` are priced as puts and the rest as calls,
so every point is the out-of-the-money contract. Black's formula is the
horizontal line.

The one-year curve sits about 0.45 volatility points below the line, and it
is smooth. Every strike saw the same terminal draws, so a low draw moves the
whole curve. The three-month curve is almost on the line (the mean gap on
this seed is a few thousandths of a volatility point).

Two of the 63 nodes have no implied volatility. Both are the one-week wings
at `k = -0.10` and `k = +0.10`. The out-of-the-money price there is exactly
zero: on this seed no path crossed the strike. About one path was expected.
The price is the intrinsic value of an out-of-the-money option, which is
zero, so there is nothing to invert. At `|k| = 0.06` on the same expiry about
120 paths are expected through the strike, and those nodes do invert.

The bottom panel is the same gap, in volatility points. A large gap there can
still be an ordinary price error, because a short-dated wing has almost no
vega. The price test is the z-rule, not the picture.

## The definition

`Z` is the forward divided by today's forward, so it starts at 1 and its mean
stays 1. Exact stepping multiplies by a lognormal. Euler stepping adds
`sigma Z dW` and is a martingale with the wrong shape: at volatility 1 and one
step, 100,000 paths match the normal formula and do not match Black.

The standard error is the sample standard deviation of the batch payoffs
divided by the square root of the number of batches. That identity makes a
slope of log(standard error) on log(paths) hold by construction. The check
is a calibration: over 200 seeds, with antithetic pairs and the `Z` control
on, the scatter of the prices divided by the mean reported standard error
lies in `[0.9, 1.1]`.

Put minus call, on each path, is the forward payoff. Averaged, the two prices
differ by the discounted forward times the average of `Z` on those paths, which
is not exactly the strike gap.

Inversion: where a volatility is returned, price to volatility to price comes
back to a relative error of 1e-12. Once a failed inversion is NaN, that check
holds by construction on the nodes that invert. Volatility to price to
volatility comes back to 1e-8 where the normalised vega `n(d1) sqrt(T)`
exceeds 1e-8. The grid for that check is standardised moneyness
`k / (sigma sqrt(T))` on `[-4, 4]`. The fraction of nodes above the vega
threshold is what is asserted. A volatility the solver cannot stand behind
is returned as NaN.

## Hypotheses

These are not results.

- The 0.45 volatility-point shift of the one-year curve is that one seed, not
  a bias in the exact scheme. A second seed moves it the other way.
- The two missing week-long wings are strikes no path crossed. More paths fill
  them in from the inside. They do not fill in by changing the interpolant.
- At a volatility near 0.2 the Euler bias is smaller than the standard error
  of a few thousand paths, which is why the bias test uses volatility 1.
- Antithetic pairs and the `Z` control cut variance most at the money. The
  z-rule check with both switched on does not measure by how much.

## Explain it back

1. Why does the one-year curve move as a whole, while the one-week curve is missing only at the two strikes no path crossed?
2. The averaged call minus the averaged put is not exactly `df (F - K)`. What is it equal to?
3. Why does a price that is only one standard error off Black turn into a large implied-volatility error at short expiry?
