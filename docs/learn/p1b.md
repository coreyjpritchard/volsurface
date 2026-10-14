# P1b: three ways to join the dots, and two ways to say "arbitrage"

## What to watch

The figure `docs/figures/p1b_surfaces.png` uses the saved SPX chain. The dots
are mid quotes from the CBOE delayed book. The loader marks the options
European and cash-settled. There is a bid and an ask on every row.

The top two panels are 30 October 2026, T = 0.044975, 466 strikes, log-moneyness
from -0.555 to 0.099. The script picks this expiry because the natural cubic,
continued 0.12 in log-moneyness past each end of the quotes, reaches the lowest
implied vol in the file. On this slice that minimum is -24.95. The same grid's
maximum is 1.53. The panel's axis runs from -100% to 80%, so both ends leave
the frame. The last expiry in the file, 30 September 2027, does not do this:
its extrapolated minimum is about 0.040.

Inside the quoted strikes the cubic spline passes through every dot. Raw SVI
is the weighted least-squares slice from `SVIFitter` (half-spread through vega).
The gap on these knots is an in-sample root-mean-square of 0.00213 in
volatility, 0.21 volatility points. The linear surface joins normalised call
prices with straight lines in the strike and is then inverted, so it is not a
straight line in volatility.

The grey bands are past the last quote. There the cubic is still a cubic. The
crosses are nodes where the surface checker says the spline's call is not
convex: 129 of 361 nodes on the plotted grid.

The middle panel is the density of log(F_T / F). On `k` in `[-0.5, 0.5]`,
Durrleman's `g` is negative for 24 of the 40 fitted slices. This October slice
is one of the 16 that pass on that window (`g` minimum 0.0578), and its SVI
density stays positive (minimum 0.0003 on the plotted grid). The spline's
density is negative on 125 of 263 plotted points inside the quotes. The blue
line is clipped at -2 and 8. The spikes are larger than that, and the clip is
why the line sits on the frame.

The bottom panel is `g` itself. The red curve is the worst slice on that same
window, T = 0.1218 (27 November 2026), minimum `g` -0.0923. The fit does not
penalise `g`. A penalty would hide the count. On the fitter's wider diagnostic
grid, which runs half a unit of log-moneyness past the quoted strikes, the
minimum is much smaller: -5509.8 at T = 0.711 and -13054 at T = 0.963. That
grid is not the one drawn here.

Across the whole chain, `in_spread` is 0.558: that fraction of the two-sided
quotes has the SVI call between the bid and the ask.

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
last quote. The two shortest expiries of this file, checked together, are
feasible: 278 quotes, total slack 0, `ok` true. The first 24 expiries together
(4,398 quotes) are feasible as well. The program on all 40 expiries was killed
at a 15.6 GB memory limit before it returned a status. A book with no bid and
no ask still evaluates nothing: `ok` is None and `n_checked` is 0. A synthetic
book is what locks that path.

The slice used as a known failure is the one Axel Vogt posted, printed as
equation (3.8) in Gatheral and Jacquier (2014). The printed order is
`(a, b, m, rho, sigma)`, which is not the field order of the dataclass.
Durrleman's `g` is negative somewhere on it. The fitter does not include `g`
in its objective. It records `min_g`, Lee's slope bound, and whether total
variance rises in expiry. On this snapshot 24 of the 40 fitted slices have
`g < 0` somewhere on `k` in `[-0.5, 0.5]`. When a quote has a bid and an ask,
the squared error in total variance is weighted by the half-spread mapped
through vega. This file has both, so those weights are not 1.

## Hypotheses

These are not results.

- Flat extrapolation of the linear call past the last strike is what the
  SANOS linear model does. The PDF has not been checked equation by equation.
- A slack on these European mids would be a failure of one martingale measure
  to sit inside the quotes, or a stale print in a pre-open book. It would not
  be an American early-exercise premium. The figure's negative spline density
  is the interpolant, not a quote-checker violation. The sampled quote programs
  above were feasible.
- Negative `g` on 24 of 40 slices is the least squares, not a penalty hiding
  the count. Whether it would shrink under a different weight is open.
- With equal weights on quote slacks and calendar slacks, a calendar gap can
  be booked as a quote violation. The size of the gap is identified. The kind
  is not unique.

## Explain it back

1. Why can a cubic that passes through every quoted volatility still have a negative density between the quotes?
2. What does the quote checker do with the two shortest expiries of the saved SPX file, and what happened when all 40 were passed in together?
3. The Vogt slice is arbitrageable. Does fitting SVI by least squares prevent that?
