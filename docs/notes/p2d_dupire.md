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
the SPY snapshot, at fixed `k`, as a step from one expiry to the next.

The Monte Carlo reads that grid by bilinear interpolation and steps
`log Z` with the sampled variance. Adding paths shrinks the standard
error and leaves the time-step bias where it is. Flat variance is the
exception: the log-Euler step is the exact lognormal transition, and
the only error is sampling. A variance that depends on `k` is not
exact. The Heston test allows 40 bp of implied vol for that bias and
for the finite-difference estimate of `g`, plus one standard error.
