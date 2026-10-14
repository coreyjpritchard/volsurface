# Dupire local variance, and why the interpolant shows up in it

The normalised call is a function of log-moneyness `k` and total variance
`w(k, T)`. Dupire's local variance, written so that the only derivatives
are of `w`, is `(dw/dT) / g(k)`. The denominator `g` is the same function
that multiplies the lognormal density in Durrleman's formula:

```
g = 1 - (k/w) dw/dk
    + (1/4) (-1/4 - 1/w + k^2/w^2) (dw/dk)^2
    + (1/2) d²w/dk².
```

`g` is the risk-neutral density of `log(F_T / F)`, up to the positive
lognormal factor. Where `g` is zero the local variance is undefined.
Where `g` is negative the density is negative, which is a butterfly
arbitrage, and the local variance comes out negative. Clipping it to a
small positive number would hide the arbitrage and would feed a Monte
Carlo a variance the surface does not have. The implementation leaves
the number and counts the cell.

Two derivatives in `g` are strike derivatives of `w`. A cubic spline of
implied volatility, extrapolated as a cubic, curves away from the quotes.
`d²w/dk²` then grows like the second derivative of that cubic, and `g`
crosses zero or becomes tiny while `dw/dT` stays the size of a market
variance. The ratio explodes. That explosion is the interpolant, not a
property of the option prices between the strikes. Raw SVI has an
explicit `w''` that decays as `1/k^3`, so the same quotes do not produce
that wing. Linear interpolation of the call in the strike has no second
derivative between nodes, so Dupire is not defined there: a finite
difference of the inverted total variance chatters instead of settling.

`dw/dT` is the other half. Between two expiries this library takes `w`
linear in calendar time, which is a constant `dw/dT` on that interval
and a different constant on the next. Local variance therefore jumps at
each expiry even when the smile is smooth in `k`. The jump is the
calendar interpolant. It is visible on the SVI and spline surfaces of
the SPX snapshot, at fixed `k`, as a step from one expiry to the next.

The Monte Carlo reads that grid by bilinear interpolation and steps
`log Z` with the sampled variance. A query outside the grid is clipped
to the edge cell, and that sample is counted as clipped. A cell that is
not strictly positive is counted as invalid and adds no shock. Adding
paths shrinks the standard error and leaves the time-step bias where it
is. Flat variance is the exception: the log-Euler step is the exact
lognormal transition, and the only error is sampling. A variance that
depends on `k` is not exact.

On the Heston parameters `v0 = theta = 0.04`, `kappa = 2`, `xi = 0.15`,
`rho = -0.3`, four standard deviations of `log Z` at `T = 1` is about
`0.8`. The finite-difference grid on `k` in `[-0.8, 0.8]` has 3 negative
cells and 1 exploding cell, so a path clipped onto that edge would stop.
The widest grid that stays positive is `k` in `[-0.5, 0.5]` (minimum
local variance `0.014`). At 200,000 paths, 48 steps, seed 4, strikes 95,
100 and 105, the implied-vol gaps are `-4.9`, `-2.2` and `+0.1` bp.
Standard errors are `8.5`, `7.1` and `6.2` bp. The budget is 15 bp plus
three standard errors. 4,898 of the 200,000 paths touch the edge at least
once, and none hit a non-positive cell.

The same `k` grid in `[-0.3, 0.3]`, with `Domain` `±0.4` and with
`Domain` `±0.8`, produces identical local-variance arrays (maximum
absolute difference 0) and the same gaps, `-5.0`, `-2.2` and `+0.1` bp,
on that 200,000-path draw. Once the grid sits inside the domain, the
domain width is not a second extrapolation.
