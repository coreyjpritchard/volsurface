# Where Hagan's SABR formula stops being a density

The forward and its volatility are

```
dF = alpha F^beta dW,    d(alpha) = nu alpha dZ,    corr = rho dt.
```

Hagan, Kumar, Lesniewski and Woodward (2002) expand the Black implied
volatility in powers of `nu^2 T`. The leading factor is `alpha` times a
function of log-moneyness. That function is `z / x(z)`, with

```
z = (nu / alpha) (F K)^((1 - beta) / 2) log(F / K)
x(z) = log( (sqrt(1 - 2 rho z + z^2) + z - rho) / (1 - rho) ).
```

At the money, `z = 0` and `x = 0`. The ratio tends to 1. The code uses that
limit, written out, instead of dividing the two zeros. The maturity
correction that multiplies the whole expression is

```
1 + [ (1 - beta)^2 alpha^2 / (24 (F K)^(1 - beta))
      + rho beta nu alpha / (4 (F K)^((1 - beta) / 2))
      + (2 - 3 rho^2) nu^2 / 24 ] T.
```

`beta` stays a parameter. `alpha / F^(1 - beta)` is the at-the-money level
when `T` and `nu` are small, so `alpha` and `beta` are not separately
visible in a smile. Calibration holds `beta` fixed and fits `alpha`, `rho`
and `nu` separately on each expiry. SABR does not share one parameter set
across the book.

The expansion does not enforce a positive density. `z / x(z)` and the
bracket in `T` can bend the smile so hard that the call, as a function of
strike, stops being convex. The risk-neutral density of log-moneyness is
that convexity, up to a positive factor. Where the second derivative of the
call in the strike is negative, the density is negative. The implied
volatility itself can still be positive everywhere on the same strikes: the
formula is quoting a Black vol that does not come from a probability. On the committed example (`alpha = 0.3`, `beta = 1`, `rho = -0.3`,
`nu = 1.5`, `T = 10`, `nu^2 T = 22.5`) the density is negative from
`k = -1.5` to `k = -0.15`, positive from `k = -0.125` to `k = 0.30`, and
negative again from `k = 0.325` to `k = 0.50`. The most negative value on
that grid is `-1.099` at `k = -0.325`. Implied vol on the same strikes
stays between `0.66` and `2.07`. Short maturity and small `nu` keep
`nu^2 T` small, which is the regime the expansion was built for, and there
the log-Euler price sits inside the Monte Carlo standard error.

When `beta = 1` and `rho = 0`, `alpha` does not see the forward, and the
price is the average of Black prices on `(1/T) int alpha_t^2 dt`. On
`alpha = 0.25`, `nu = 1`, `T = 2`, a mixture of 100,000 paths and 1,000
steps has implied vol `0.2733` at the money and `0.2913` at `k = ±0.2`,
with a standard error of about `8` bp. Hagan is `184` bp high at the money
and `271` bp high at those wings. A 20,000-path log-Euler with 64 steps is
`187` to `290` bp low. The gap between that sample and Hagan, about `380`
to `561` bp, adds the two and is not the formula on its own.

For `beta < 1` the forward can reach 0. It has to stay there. A log-Euler
step of a positive number never hits 0, so the discretisation also forms
the plain Euler predictor `Z + a Z^beta sqrt(dt) Z_F`. A predictor that is
not strictly positive sets `Z` to 0 for the rest of the path. So does a
level below `1e-12`, and a power that overflowed. With `nu = 0`,
`alpha = 0.8` and a unit forward, `P(Z_T = 0) = exp(-2 / 0.64) = 0.04394`.
At 64 steps and 20,000 paths the estimate is `0.0433`.
