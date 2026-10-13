# P1c: one set of paths, a whole surface

## What to watch

The figure `docs/figures/p1c_gbm.png` is geometric Brownian motion with
volatility 20%, simulated exactly, 8,000 paths, one seed. Three expiries share
those paths. Black's formula is the horizontal line.

The one-year curve sits about half a volatility point below the line, and it
is smooth. Every strike saw the same terminal draws, so a low draw moves the
whole curve. The three-month curve is the same kind of shift, smaller. The
one-week curve is not a shift. Past about 8% of spot on the put side, and at
the far call, no path finished in the money, the price is intrinsic, and there
is no implied volatility. Six of the 63 plotted nodes are missing for that
reason. Where the week does invert, the wing is the noisy part.

The bottom panel is the same gap, in volatility points. A large gap there can
still be an ordinary price error, because a short-dated wing has almost no
vega. The price test is the z-rule, not the picture.

## The definition

`Z` is the forward divided by today's forward, so it starts at 1 and its mean
stays 1. Exact stepping multiplies by a lognormal. Euler stepping adds
`sigma Z dW` and is a martingale with the wrong shape: at volatility 1 and one
step, 100,000 paths match the normal formula and do not match Black.

The standard error is the sample standard deviation of the batch payoffs
divided by the square root of the number of batches. From 1,000 paths to
1,000,000, independent seeds, the slope of log(standard error) on log(paths)
is -0.5 within 0.05.

Put minus call, on each path, is the forward payoff. Averaged, the two prices
differ by the discounted forward times the average of `Z` on those paths, which
is not exactly the strike gap.

Inversion: where a volatility is returned, price to volatility to price comes
back to a relative error of 1e-12. Volatility to price to volatility comes back
to 1e-8 where the normalised vega `n(d1) sqrt(T)` exceeds 1e-8. On a grid of
31 strikes and four expiries, 85 of 124 nodes are below that threshold. They
are counted. A volatility the solver cannot stand behind is returned as NaN,
not as a number that fails the round trip.

## Hypotheses

These are not results.

- The half-point shift of the one-year curve is that one seed, not a bias in
  the exact scheme. A second seed moves it the other way.
- The six missing week-long wings are empty of paths. More paths fill them in
  from the inside. They do not fill in by changing the interpolant.
- At a volatility near 0.2 the Euler bias is smaller than the standard error
  of a few thousand paths, which is why the bias test uses volatility 1.
- Antithetic pairs and the `Z` control cut variance most at the money. The
  z-rule check with both switched on does not measure by how much.

## Explain it back

1. Why does the one-year curve move as a whole, while the one-week curve breaks in the wing?
2. The averaged call minus the averaged put is not exactly `df (F - K)`. What is it equal to?
3. Why does a price that is only one standard error off Black turn into a large implied-volatility error at short expiry?
