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
`Heston.cumulants` already returns: `c1` from the integrated expected
variance, `c2` and `c4` from a degree-8 stencil on `log phi`. On the Fang
and Oosterlee parameters at `T = 1` that stencil gives
`c2 = 0.031571152011` against the mpmath value `0.031571152013`, and
`c4 = 0.007486836` against `0.007486782`. A negative `c4`, or one larger
than `80 c2²`, is replaced by zero before the square root. That replacement
is the stencil in round-off, which is what happens as the volatility of
variance goes to zero and the law becomes Gaussian. It is not a claim that
the fourth cumulant is zero.

The series is applied to the put. The put payoff is bounded by the strike,
so its cosine coefficients decay. The call payoff grows like the forward,
and the same truncation leaves a visible error. The call is then
`put + df (F - K)`. That step uses `E[F_T] = F`, which the characteristic
function satisfies, and it does not expand the call. With `N = 256` terms
the at-the-money one-year call on the Fang and Oosterlee parameters is
within `1e-7` of their reference `5.785155450`, and the same `N` stays
within `1e-6` of Lewis from one week to two years on `k` in `[-1, 1]`,
including the parameter set whose Feller ratio is `0.38`. A harsher
violation, ratio near `0.05`, still needs more terms: the characteristic
function decays slowly and `N = 256` is no longer enough. The error against
`N` is the figure, not a second tolerance.

The simulation is Euler full truncation. The variance that multiplies the
Brownian increment and the mean-reversion gap is `max(v, 0)`. The value
stored for the next step is not floored. Each step draws the forward normal
and then the variance normal, from one child of the seed. Adding paths
shrinks the standard error and does not shrink the discretisation bias.
On the z-rule parameter set (`v0 = theta = 0.04`, `kappa = 1.5`, `xi = 0.25`,
`rho = -0.4`), 64 steps and 240,000 paths give a bias of `-0.035`, `-0.023`
and `-0.013` at strikes 90, 100 and 110. An 8,000-path standard error on
those strikes is about `0.17`, `0.13` and `0.10`. The bias is a fraction of
that standard error, which is why a 24-node z-rule test can pass. It is not
a statement that the scheme is unbiased.
