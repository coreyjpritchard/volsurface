# P1b: three ways to join the dots, and two ways to say "arbitrage"

## How to read this

The top panel is one expiry of the synthetic demo chain, drawn three ways;
the middle panel is the risk-neutral density each curve implies; the bottom
panel is Durrleman's `g`, which is negative exactly where an SVI slice
implies a negative density. The takeaway is that passing through every quote
is not the same as being arbitrage-free. A cubic spline in implied vol bends
freely between noisy mids, and since the density is the second strike
derivative of the call price, every wiggle turns into a spike or a negative
patch. Past the last strike nothing anchors it at all. Raw SVI is smooth, but
a least-squares fit does not impose `g >= 0`. The caveat: the expiry was
picked by the script as the worst spline wing in the file, so it shows the
spline at its worst, not a typical slice.

## What to watch

The figure `docs/figures/p1b_surfaces.png` uses the synthetic demo chain
(not market data). The dots are mid quotes. The loader marks the options
European and cash-settled. There is a bid and an ask on every row.

The top panels show one expiry chosen by the script as the worst spline wing
in the file, so they show the spline at its worst, not a typical slice. A
cubic spline through the quoted mids can go negative in the wings and imply a
negative risk-neutral density; raw SVI and a better interpolant are compared
on the same slice.

Inside the quoted strikes the cubic spline passes through every dot. Raw SVI
is the weighted least-squares slice from `SVIFitter` (half-spread through
vega). The linear surface joins normalised call prices with straight lines in
the strike and is then inverted, so it is not a straight line in volatility.

The grey bands are past the last quote. There the cubic is still a cubic. The
crosses are nodes where the surface checker says the spline's call is not
convex.

The middle panel is the density of log(F_T / F). The bottom panel is `g`
itself. The red curve is the worst slice on the window `k` in `[-0.5, 0.5]`.
The fit does not penalise `g`. A penalty would hide the count.

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
last quote. A book with no bid and no ask still evaluates nothing: `ok` is
None and `n_checked` is 0. A synthetic book is what locks that path. Scale
measurements on a market snapshot are not recorded in this public note; run
the checker on a chain you supply.

The slice used as a known failure is the one Axel Vogt posted, printed as
equation (3.8) in Gatheral and Jacquier (2014). The printed order is
`(a, b, m, rho, sigma)`, which is not the field order of the dataclass.
Durrleman's `g` is negative somewhere on it. The fitter does not include `g`
in its objective. It records `min_g`, Lee's slope bound, and whether total
variance rises in expiry. When a quote has a bid and an ask, the squared
error in total variance is weighted by the half-spread mapped through vega.
The synthetic demo chain has both, so those weights are not 1.

## Hypotheses

These are not results.

- Flat extrapolation of the linear call past the last strike is what the
  SANOS linear model does. The PDF has not been checked equation by equation.
- A slack on these European mids would be a failure of one martingale measure
  to sit inside the quotes, or a stale print in a pre-open book. It would not
  be an American early-exercise premium. The figure's negative spline density
  is the interpolant, not a quote-checker violation.
- Negative `g` on a least-squares SVI slice is the least squares, not a
  penalty hiding the count. Whether it would shrink under a different weight
  is open.
- With equal weights on quote slacks and calendar slacks, a calendar gap can
  be booked as a quote violation. The size of the gap is identified. The kind
  is not unique.

## Explain it back

1. Why can a cubic that passes through every quoted volatility still have a negative density between the quotes?
2. What does the quote checker return when a book has no bid and no ask, and why is that different from a recorded violation?
3. The Vogt slice is arbitrageable. Does fitting SVI by least squares prevent that?
