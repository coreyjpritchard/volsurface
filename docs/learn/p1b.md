# P1b: three ways to join the dots, and two ways to say "arbitrage"

## What to watch

The figure `docs/figures/p1b_surfaces.png` uses the saved SPY chain. The dots
are last trades. There is no bid or ask in that file, and the loader marks the
options American.

The top two panels are the last expiry in the file, 30 September 2027, about
0.98 year, thirty strikes. It is chosen because the natural cubic, continued
past the lowest quoted strike, crosses through zero implied volatility. It is
not chosen as a slice where SVI is well behaved.

Inside the quoted strikes the cubic spline passes through every dot. Raw SVI
is a smooth curve fitted by least squares in total variance. The gap on these
knots is an in-sample root-mean-square of about 0.13 volatility points. The
linear surface joins normalised call prices with straight lines in the strike
and is then inverted, so it is not a straight line in volatility.

The grey bands are past the last quote. There the cubic is still a cubic. On
this slice it falls through zero on the low-strike side (the minimum on the
plotted wing is about -100%). The crosses are nodes where the surface checker
says the spline's call is not convex. Many of those nodes sit between the
quoted strikes, not only in the wing: the spline is following wiggles in the
last trades.

The middle panel is the density of log(F_T / F). On `k` in `[-0.5, 0.5]`,
Durrleman's `g` is negative for 14 of the 24 fitted slices. The September
2027 slice is one of the 10 that pass, and its density stays positive. The
spline's density is negative on 84 of 257 plotted points inside the quotes.
The blue line is clipped at -2 and 8; the spikes are larger than that, and
the clip is why the line sits on the frame.

The bottom panel is `g` itself. The red curve is the worst slice on that same
grid, about 0.0027 year, with minimum `g` about -0.18. The fit does not
penalise `g`. A penalty would hide the count.

## The definition

The normalised call is the cash call divided by `df * F`. Linear interpolation
joins that number in the normalised strike `x = exp(k)`. A cubic spline joins
implied volatility in `k`. Raw SVI joins total variance with five parameters,

```
w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2)).
```

The surface checker looks at one grid. It asks whether the call stays between
intrinsic and 1, whether its slope in `x` stays in `[-1, 0]`, whether it is
convex in `x`, and whether it rises in expiry at fixed `k`. A pass on the
nodes says nothing about the gaps between them.

The quote checker is a different question. It asks whether any discrete
martingale measure can sit inside the bid and ask. The tail beyond the last
quoted strike is a mass and a moment, not a strike planted at 1.5 times the
last quote. The saved SPY file has no bid or ask, so that check evaluates
nothing: `ok` is None and `n_checked` is 0. A synthetic two-sided book is
what exercises the linear program and `in_spread`.

The slice used as a known failure is the one Axel Vogt posted, printed as
equation (3.8) in Gatheral and Jacquier (2014). The printed order is
`(a, b, m, rho, sigma)`, which is not the field order of the dataclass.
Durrleman's `g` is negative somewhere on it. The fitter does not include `g`
in its objective. It records `min_g`, Lee's slope bound, and whether total
variance rises in expiry. On this snapshot 14 of the 24 fitted slices have
`g < 0` somewhere on `k` in `[-0.5, 0.5]`. When a quote has a bid and an ask,
the squared error in total variance is weighted by the half-spread mapped
through vega. This file has neither, so those weights are 1.

## Hypotheses

These are not results.

- Flat extrapolation of the linear call past the last strike is what the
  SANOS linear model does. The PDF has not been checked equation by equation.
- Violations on these last trades mix stale prints, the American early-exercise
  premium, and any genuine butterfly in the marks. The figure is not a
  tradable arbitrage.
- Negative `g` on the short-dated SVI fits is the least squares chasing noisy
  lasts, rather than a property that would survive a bid-ask fit.
- With equal weights on quote slacks and calendar slacks, a calendar gap can
  be booked as a quote violation. The size of the gap is identified. The kind
  is not unique.

## Explain it back

1. Why can a cubic that passes through every quoted volatility still have a negative density between the quotes?
2. What does the quote checker do with the saved SPY file, and why?
3. The Vogt slice is arbitrageable. Does fitting SVI by least squares prevent that?
