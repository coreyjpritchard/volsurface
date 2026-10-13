# P1b: three ways to join the dots, and two ways to say "arbitrage"

## What to watch

The figure `docs/figures/p1b_surfaces.png` is one expiry from the saved SPY
chain: 30 September 2027, about 0.98 year, thirty strikes. The dots are last
trades. There is no bid or ask in that file, and the loader marks the options
American.

The top panel is implied volatility against log-moneyness. Inside the quoted
strikes the cubic spline passes through every dot. Raw SVI is a smooth curve
fitted by least squares in total variance; on these knots the root-mean-square
gap in volatility is about 0.13 percentage points. The linear surface joins
normalised call prices with straight lines in the strike and is then inverted,
so it is not a straight line in volatility.

The grey bands are past the last quote. There the cubic is still a cubic. On
this slice it falls through zero on the low-strike side (the minimum on the
plotted wing is about -100%). The crosses are nodes where the surface checker
says the spline's call is not convex. Many of those nodes sit between the
quoted strikes, not only in the wing: the spline is following wiggles in the
last trades.

The bottom panel is the density of log(F_T / F). SVI's density stays positive
on this slice. The spline's density is negative on 84 of 257 plotted points
inside the quotes. The blue line is clipped at -2 and 8; the spikes are larger
than that, and the clip is why the line sits on the frame.

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
martingale measure can sit inside the bid and ask. The saved SPY file has no
bid or ask, so that check drops every quote and reports nothing to constrain.

The slice used as a known failure is the one Axel Vogt posted, printed as
equation (3.8) in Gatheral and Jacquier (2014). Durrleman's `g` is negative
somewhere on it. The fitter does not include `g` in its objective, and on this
snapshot 14 of the 24 fitted slices have `g < 0` somewhere on `k` in
`[-0.5, 0.5]`. The September 2027 slice in the figure is one of the ones that
does not.

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
