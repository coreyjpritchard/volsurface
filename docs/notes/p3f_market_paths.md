# Market paths and the regime chain

The chain is continuous-time Markov and independent of the Brownian
drivers. This phase switches `nu` only. `xi_level` and `both` raise.
The variance is

`v_t = xi0(t) exp(nu_{r_t} sqrt(2H) Y_t - nu_{r_t}^2 t^{2H} / 2)`.

`Y` has variance `t^{2H} / (2H)`, so the conditional expectation of `v_t`
given the chain is `xi0(t)`. At `H = 0.5` the driver `Y` is Brownian and
is drawn here. Any other `H` asks `create("rbergomi")` for `driver_y`.
A missing plugin raises. The hybrid scheme is not in this repository, and
a stand-in driver is not labelled rBergomi.

The spot step freezes `v` on the left of the interval and multiplies `Z`
by `exp(sqrt(v) dW - v dt / 2)`, with `dW = rho dY + sqrt(1 - rho^2) dW_perp`.
Each step has conditional mean 1 given the past. Whether that survives a
switch in `nu` is the z-rule on the terminal mean, not a theorem in this
note.

`P(t) = expm(Q t)` is the transition with rows summing to 1.
`P(t)[i, j]` is the probability of `j` given `i`. A row law updates on
the right, `p0 @ P(t)`. The chi-square test compares the simulated
occupation at one time with that law.

The forward curve stored on the path, at `H = 0.5`, is

`xi_t(u) = xi0(u) sum_r P(r_u = r | r_t) exp(nu_r Y_t - nu_r^2 t / 2)`.

That equals `E[v_u | Y_t, r_t]`. The variance-swap pricer integrates it
with a left Riemann sum. The same sum of the left-point variance is the
discrete realised variance, so the two agree in expectation.

The filter observation is Gaussian with a mean that depends on the
regime. The Hamilton probability at `t` uses the observation up to `t`.
Reliability bins that probability and checks the hit count against the
0.1 and 99.9 percent binomial quantiles.

Seeds are `SeedSequence(entropy=E, spawn_key=(split,))` with split 0, 1,
2 for train, validation, and test. Each run spawns four children: regime,
vol, spot, surrogate. The surrogate child is spawned and not drawn.

`ClosedFormPricer` uses the model's formula before expiry and holds the
payoff after expiry. `SurrogatePricer` raises when its validation error,
in basis points, is above the bar it was constructed with. The pilot that
will call these pricers is a CPU run.
