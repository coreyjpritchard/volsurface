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
`c1 ± L sqrt(c2 + sqrt(c4))`, with `L = 10`. The cumulants are the ones
`Heston.cumulants` returns: `c1` from the integrated expected variance,
`c2` and `c4` from a degree-8 stencil on `log phi`. The step is a fraction
of `sqrt(2 |c1|)`, and it is halved while `phi(-i t)` is not finite. `c4`
is used only when two neighbouring steps agree. On the Fang and Oosterlee
parameters at `T = 1` the kept values sit on the mpmath derivatives
`c2 = 0.031571152013` and `c4 = 0.007486782`. When the steps disagree,
`c4` is discarded and the half-width becomes `2 L sqrt(c2)`. That is wider
than the Gaussian interval. It is what a nearly Gaussian law (`xi` near 0)
and a very short maturity produce in float64. A large positive fourth
cumulant is not discarded. At `v0 = theta = 0.04`, `kappa = 0.5`,
`xi = 1.5`, `rho = -0.9` and `T = 1` the kurtosis is about 103. At
`kappa = 1`, `xi = 5`, `rho = -0.7` it is about 281 at `T = 0.25` and
about 798 at `T = 1`. Zeroing those would shrink the interval.

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
`T = 1` (kurtosis about 798). On that second set the library Lewis put
at `k = -1`, `T = 1` is `0.166864108`. The 40-digit integral is
`0.166861460`. The gap is `2.65e-6`. The at-the-money put on the same
set agrees to `4e-10`.

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
budget before scoring. The scheme is still biased. The test checks that
the Monte Carlo noise, once the budget is removed, looks like the standard
errors it reports.
