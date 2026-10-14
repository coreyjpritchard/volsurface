# Heston: the characteristic function, the cosine price, the Euler scheme

The forward `F` and the variance `v` are an affine pair. The joint transform
`E[exp(i u log(F_T / F) + i w v_T)]` is exponential in the initial variance,
with coefficients that solve a Riccati equation. Setting `w = 0` leaves the
characteristic function of `X_T = log(F_T / F)`. The Riccati has two roots.
The form that stays continuous in `u` is the one Albrecher, Mayer, Schoutens
and Tistaert (2007) isolate, Gatheral's `r-` written as
`2 alpha / (beta + d)`. The other root crosses the branch cut of the complex
logarithm for ordinary parameters. That is the little Heston trap: a price
that is smooth in the strike at one parameter set and nonsense at the next,
with no change in the algebra that a reader can see. `volsurface.heston.char_func`
is the continuous branch. Calls in `price` are the Lewis integral of that
function. It is the reference for everything else in this note.

Fang and Oosterlee (2008) replace the integral by a cosine series of the
density on a finite interval. The interval is
`c1 ± L sqrt(c2 + sqrt(c4))`, with `L = 10` when `c4` is usable. The
cumulants are the ones `Heston.cumulants` returns: `c1` from the integrated
expected variance, `c2` and `c4` from a degree-8 stencil on `log phi`. The
step is `f / sqrt(c2)` with `f` in `0.001`, `0.002` and `0.005`, halved
while `phi(-i t)` is not finite. A first pass uses `sqrt(-2 c1)`, and a
second pass uses the `c2` that pass kept. `c4` is used only when the finest
pair of neighbouring steps agrees. On the Fang and Oosterlee parameters at
`T = 1` the kept values sit on the mpmath derivatives
`c2 = 0.031571152013` and `c4 = 0.007486782`. The same parameters at one
day and at one week now return a positive `c4` (one-week kurtosis `0.572`).
Steps of `0.02 / sqrt(c2)` and wider do not: on a kurtosis near 1000 they
return a negative `c4` while `phi` is still finite. When the three factors
disagree, the fit is repeated at `0.0005`, `0.001` and `0.002`. A large
positive fourth cumulant is kept. At `v0 = theta = 0.04`, `kappa = 0.5`,
`xi = 1.5`, `rho = -0.9` and `T = 1` the kurtosis is about 103. At
`kappa = 1`, `xi = 5`, `rho = -0.7` it is about 281 at `T = 0.25` and
about 798 at `T = 1`.

`L` then doubles, and `N` is chosen again, until two successive prices
agree to `1e-7` or the next width would pass 160 or 65536 terms. The price
kept is the narrower of the two that agreed. If they never agree, the
widest width whose last doubling of `N` moved the price by at most
`2.5e-4` is kept. Kurtosis near 1000 still leaves a tail outside `L = 10`,
so this widening runs when `c4` is usable. A fixed multiple of `sqrt(c2)`
is not the width.

The series is applied to the put. The put payoff is bounded by the strike,
so its cosine coefficients decay. The call payoff grows like the forward,
and the same truncation leaves a visible error. The call is then
`put + df (F - K)`. That step uses `E[F_T] = F`, which the characteristic
function satisfies, and it does not expand the call. `N` starts at 256, or
higher when `32 (b - a) / sqrt(c2)` says so, and doubles until successive
prices agree to `1e-7`. On the Fang and Oosterlee at-the-money one-year
call the published nine-digit value is `5.785155450`. A 40-digit Lewis
integral is `5.785155434376`. The library Lewis is `5.785155434375`
(`-1e-12` against that integral, and `1.56e-8` under the published
digits). The cosine price with the chosen `N` is `5.785155412`, which is
`2.2e-8` under the integral. Fixed `N = 256` is `5.785155373`, `6.1e-8`
under the integral. The published comparison of `-7.7e-8` was mostly the
rounding of `5.785155450`. The same rule stays within `1e-6` of Lewis
from one week to two years on `k` in `[-1, 1]`. Passing `n_terms` fixes
`N` and is how the error-against-`N` figure is drawn.

Puts at high kurtosis, `F = 100` and `k` in `{-1, -0.4, 0, 0.3}`, stay
within `1e-5` of Lewis. The largest gap is `2.5e-7` at `xi = 1.5`,
`kappa = 0.5`, `rho = -0.9`, `T = 0.25`, and `9.0e-8` at `T = 1`
(kurtosis about 103). At `xi = 5`, `kappa = 1`, `rho = -0.7` the largest
gap is `6.2e-6` at `T = 0.25` (kurtosis about 281) and `5.6e-6` at
`T = 1` (kurtosis about 798). The corner `v0 = 6.582e-4`, `kappa = 2.1`,
`theta = 0.00999`, `xi = 3.56`, `rho = -0.54`, `T = 0.25` has kurtosis
1049 and a put gap of `1.0e-8` on `k` in `{-1, -0.5, -0.086, 0, 0.3}`.
A one-day slice with `v0 = 1e-4`, `xi = 5`, `rho = -0.8` has kurtosis
1095 and a gap of `1.1e-5`. A `T = 1` draw with kurtosis about 24,000
has a gap of `7.3e-6`.

The simulation is Euler full truncation. The variance that multiplies the
Brownian increment and the mean-reversion gap is `max(v, 0)`. The value
stored for the next step is not floored. Each step draws the forward normal
and then the variance normal, from one child of the seed. Adding paths
shrinks the standard error and does not shrink the discretisation bias.
On the z-rule parameter set (`v0 = theta = 0.04`, `kappa = 1.5`, `xi = 0.25`,
`rho = -0.4`, Feller ratio 4.8), 64 steps and 240,000 paths give a bias of
`-0.035`, `-0.023` and `-0.013` at strikes 90, 100 and 110. An 8,000-path
standard error on those strikes is about `0.17`, `0.13` and `0.10`. The
bias is a fraction of that standard error, which is why a 24-node z-rule
test can pass. It is not a statement that the scheme is unbiased.

The Fang and Oosterlee parameters have Feller ratio 0.38, so the variance
does hit zero. At 64 steps the bias on strikes 90, 100 and 110, averaged
over seeds 0 to 3 at 1,000,000 paths each, is `+0.0159`, `+0.0241` and
`+0.0205`. The sample standard deviation across those four runs is
`0.0067`, `0.0068` and `0.0037`. An 8,000-path z-rule test subtracts that
budget before scoring. The budget is smaller than the 8,000-path standard
error, and the z-rule at 24 nodes allows a mean z of 0.71, so the test
does not have power to detect that budget. It fails if the scheme is
grossly wrong.
