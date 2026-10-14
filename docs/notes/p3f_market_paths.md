# Market paths and the regime chain

The chain is continuous-time Markov and independent of the Brownian
drivers. This phase switches `nu` only. `xi_level` and `both` raise.

A switch acts on increments only. The new `nu` multiplies the Volterra
increment on the step where it is in force. It does not rescale variance
already accumulated, so `v` has no jump at a switch. On a grid the step
`(t_{i-1}, t_i]` uses the regime at `t_{i-1}`. At `H = 1/2` that step is

`v_i = v_{i-1} exp(nu_{i-1} dW_i - nu_{i-1}^2 dt_i / 2)`,

with `v_0 = xi0`. Constant `nu` reduces to the usual compensator
`nu^2 t^{2H} / 2`, and that diagonal matches the closed form.

`H = 1/2` is drawn here. Any other `H` calls `volterra_components` on
`create("rbergomi")`. The method must return `(dW, I_1, ..., I_kappa)`
of shape `(n_paths, n_steps, kappa + 1)`, drawn from the volatility
child with no further spawn. `kappa` is the plugin's, 1 or 2. This
module multiplies each step's vector by the `nu` in force on that step.
`I_k` lands at lag `k`. The tail uses `power_kernel_weights`, whose
first `kappa` entries are 0. `dW` alone is the regression of the kernel
onto the Wiener increment. At `H = 0.1` that regression keeps `5/9` of
`Var(I_1)`, and `E[v]` falls short of `xi0`. The compensator stays the
exact one. The Cholesky of the hybrid scheme is not in this repository.
A missing method raises, and a stand-in is not labelled rBergomi.
`H = 1/2` does not call the plugin. The dev extra pins rough-volatility
`2aca346571ccfaaab0b6af8730e90d43a57a35ab`, the commit that returns the near field.

The spot step freezes `v` on the left of the interval and multiplies `Z`
by `exp(sqrt(v) dW_spot - v dt / 2)`, with
`dW_spot = rho dW + sqrt(1 - rho^2) dW_perp`. Each step has conditional
mean 1 given the past. Whether that survives a switch in `nu` is the
z-rule on the terminal mean, not a theorem in this note.

`P(t) = expm(Q t)` is the transition with rows summing to 1. A row law
updates on the right, `p0 @ P(t)`. The chi-square test compares the
simulated occupation at one time with that law.

Every price uses the trader's information. `state["xi"]` is the mixture
of the regime-conditional forward curves, weighted by `regime_prob`. It
is not conditioned on the true label `r_t`. At `H = 1/2` each
conditional expectation equals `v_t` when `xi0` is flat, so the mixture
is that common curve. The variance-swap pricer integrates it. Behind
the clock the forward-variance entry is NaN. On the clock it is `v_t`.
At `H` other than `1/2` a later date is the closed-form constant-`nu`
curve for that regime. The closed-form diagonal drops the hybrid near
field, so it is not used.

`surface_at` is a separate Monte Carlo of the call. Each regime holds
`nu` fixed. The variance level is that regime's forward curve `xi_t(u)`,
and the future kernel is restarted from the clock. The past Volterra
integral sits in the curve and is not drawn again. The prices are
weighted by `regime_prob` at the clock. The seed is the path entropy
and `spawn_key + (4, regime)`. The clock is not in the key, so every
hedge date draws the same shocks. `n_mc` defaults to 1024. The unit
forward is 1 and the strike is `exp(k)`. At `H = 1/2` the curve is flat
at `v_t` and the variance step is the exponential martingale. A Black
call on the expected variance is flat in strike there, and independent
of `nu`, so the surface simulates the payoff instead. The plugin is not
called. Any other `H` reads `state["dW"]` and restarts
`volterra_components` from the clock. The same shocks drive a Black
path at the deterministic left forward variance. That Black price is
the control. The copies are split in half, and the coefficient on each
half is the other half's `cov / var`, so the residual variance is not
the in-sample minimum. `stderr` is the residual standard deviation
divided by `sqrt(n_mc)`. The absolute price still carries that Monte
Carlo error. Two mixtures on the same paths share a seed per regime, so
the gap between them is the gap in the weights.
`total_variance` is the controlled log contract `-2 E[log Z]`.

The rough restart has one limit that stays open. A later date is the
closed-form constant-`nu` curve, and the clock entry is `v_t`. It is
not the conditional curve of a path whose past `nu` switched, because
that curve would read the latent label. The grid from the clock to the
last expiry has to be equally spaced.

Prices read the spot filter. `price_filter` defaults to `"spot"`, and
`regime_prob` is that filter. The windowed realised-variance filter is
observable, and the default does not price it. `price_filter="rv"` does.
The latent-variance benchmark is not a price.

The spot observation is one-step log realised variance,
`log((d log Z)^2 / dt)`. Time 0 has no return, and the probability there
is `p0`. The emission is Gaussian with mean `log(xi0) - gamma - log(2)`
in every regime, because the return's variance is the spot variance and
not `nu`. The variance in regime `j` is `pi^2 / 2 + nu_j^2`. `pi^2 / 2`
is about 4.93. For `nu` of 0.4 and 1.1 the extra term is 0.16 and 1.21,
so the two Gaussians are close.

The windowed level refines each clock step into `n_sub` sub-steps,
default 21, with a Brownian bridge that sums to the coarse shock. `Z`
does not move. Variance is frozen at the start of the step, so the
level is `log v` there plus the log of a chi-squared on 21 degrees,
scaled by 21, up to the drift and the bridge. It is not the log of
integrated variance. The filter reads the change in that level. The
change at date `i` is emitted by the regime at date `i - 2`. The
emission table is the sample mean and variance of the change on pilot
paths that stay in one regime. The pilot does not read `v`. Its seed is
entropy 101 and spawn key `(7, regime)`.

The latent-variance benchmark reads `log v_i - log v_{i-1}` through a
pilot table of the same kind. That table does read `v`. It is used at
every `H`. The exact half-Brownian likelihood is not the benchmark, and
the market does not call it. At `H` other than `1/2` that Gaussian is
not the rough likelihood.

The bar is the Brier skill `1 - Brier / Brier_base`, scored at each
setting. The price bins are a separate claim. Their lift over the base
rate sits between 0.05 and 0.35.

The constant-`nu` forward curve, used for dates after the clock, remains

`xi_t(u) = xi0(u) exp(nu sqrt(2H) I_t(u) - nu^2 (u^{2H} - (u - t)^{2H}) / 2)`,

`I_t(u) = int_0^t (u - s)^{H - 1/2} dW_s`.

At `H = 1/2` the kernel is 1, so `I_t(u) = W_t` and the exponent does
not depend on `u`. On a 5-path, 8-step draw the curve matches `roughvol`
to `1e-10`.

Seeds are `SeedSequence(entropy=E, spawn_key=(split,))` with split 0, 1,
2 for train, validation, and test. Each run spawns four children: regime,
vol, spot, surrogate. The surrogate child is spawned and not drawn. The
regime child draws the chain and does not draw the filter observation.

`ClosedFormPricer` uses the model's formula before expiry and holds the
payoff after expiry. `SurrogatePricer` raises when its validation error,
in basis points, is above the bar it was constructed with. The pilot that
will call these pricers is a CPU run.

Chain, 4,000 paths, seed entropy 1, test split, time 1. Generator rates
1.5 and 0.8, initial law `(0.7, 0.3)`. Counts 1,474 and 2,526. The law
`p0 @ expm(Q t)` expects 1,532.5 and 2,467.5. Chi-square statistic
`3.625`, p-value `0.0569`. The bar in the test is a p-value above `0.001`.

Terminal time 1, eight runs, 2,000 paths each, `nu` in `{0.4, 1.1}`,
`rho = -0.4`, `xi0 = 0.04`, `H = 0.5`. The z-score of `E[Z] - 1` has
maximum absolute value `1.256` and mean `-0.152`. The z-score of
`E[v] - xi0` has maximum absolute value `2.414` and mean `-0.498`. Both
sets pass the z-rule.

The variance swap at inception against realised left-point variance,
eight runs, 1,500 paths. The z-score of the gap has maximum absolute
value `1.602` and mean `-0.194`. It passes the z-rule.

Six hundred paths, 21 times, clock index 10, seed entropy 12, test split.
The generator, initial law, `rho`, and `xi0` are the pre-registered pins.
The hedge cost `1e-4` does not enter the filters. The chain does not
depend on `H` or on `nu`, so both settings have 277 paths in the low
regime and 323 in the high regime. The base rate is 0.5383. Quoting it
scores Brier 0.4971 and log score -0.6902. Eight quantile bins of the
at-the-money call, and the same bins of the variance swap, each contain
both regimes. With 1,024 Monte Carlo copies the call's modal prediction
is right on 0.6600 of the paths, a lift of 0.1217, under the spot
weights and under the windowed weights. The variance swap is right on
0.6600 of the paths, a lift of 0.1217. Both lifts sit inside the band
from 0.05 to 0.35. That 0.6600 is the price bin, not a filter hit rate.

Market-test settings, `H = 1/2`, `nu` of 0.4 and 1.1, 600 paths. Spot
filter: 287 of 600, confusion 129, 148, 165, 158, Brier 0.5222, log
score -0.7185, Brier skill -0.0505, calibration error 0.0713. Windowed
filter: 345 of 600, confusion 109, 168, 87, 236, Brier 0.4953, log score
-0.6893, Brier skill 0.0035, calibration error 0.0295. Latent benchmark:
487 of 600, confusion 223, 54, 59, 264, Brier 0.2806, log score -0.4412,
Brier skill 0.4355, calibration error 0.0221. Mean absolute call gaps
against the true-regime mixture are 0.0002898, 0.0002834, and 0.0001549.
Implied-vol root-mean-square error on 35,032 finite nodes is 0.005348,
0.005232, and 0.003915. At `H = 1/2` the stored forward variance does
not depend on the weights, so the variance swap does not move.

The same market-test settings, 4,000 paths. Spot hit rate 0.5023, skill
-0.0292, calibration error 0.0591. Windowed hit rate 0.5543, skill
0.0094, calibration error 0.0106. Latent hit rate 0.8295, skill 0.4785,
calibration error 0.0066.

Pre-registered settings, `H = 0.1`, `nu` of 1.5 and 3, 600 paths. Spot
filter: 410 of 600, hit rate 0.6833, confusion 176, 101, 89, 234, Brier
0.4193, log score -0.6157, Brier skill 0.1565, calibration error 0.0632.
Windowed filter: 452 of 600, confusion 207, 70, 78, 245, Brier 0.3560,
log score -0.5406, Brier skill 0.2838, calibration error 0.0255. Latent
benchmark: 464 of 600, confusion 207, 70, 66, 257, Brier 0.3316, log
score -0.5099, Brier skill 0.3330, calibration error 0.0487. Mean
absolute call gaps are 0.001650, 0.001452, and 0.001310. Implied-vol
root-mean-square error on 31,684 finite nodes is 0.02069, 0.01906, and
0.01805. The figure is `docs/figures/p3f_paths.png`.

The same pre-registered settings, 4,000 paths. Spot hit rate 0.6608,
skill 0.1178, calibration error 0.0527. Windowed hit rate 0.7048, skill
0.2071, calibration error 0.0319. Latent hit rate 0.7438, skill 0.2849,
calibration error 0.0338. The windowed skill at clocks 5, 10 and 20 is
0.2079, 0.2071 and 0.2089.

Constant `nu`, eight runs of 500 paths, clock index 8. The mean forward
curve after the clock, divided by `xi0`, is 1.019 at `nu = 1.5` and
1.031 at `nu = 3`. Both pass the z-rule.

One path, `v = 0.04`, `rho = -0.4`, `H = 1/2`, 32 steps, expiry 1,
8,192 copies, seed entropy 7, test split. The implied-vol span at
`nu = 0.01` is 0.000383. At `nu = 1.5` the span is 0.04358. The implied
vol at that `nu` is 0.2107 at log-moneyness -0.2 and 0.1676 at 0.2.
With `nu = 1.1` and 1,024 copies the at-the-money relative standard
error at expiry 1 is 0.0207.

At `H = 0.1`, eight paths, 17 times, clock index 8, maturity 1, 4,096
copies, seed entropy 21, test split. The flat anchor `v_t (T - t)`
misses the variance swap by a median relative gap of 0.281. The surface
log contract stays inside 4 standard errors of that swap on every
expiry. The largest absolute z-score on a later expiry is 1.80. The
out-of-the-money strip on log-moneyness from -0.8 to 0.8 stays inside
4 standard errors on the last expiry. Its largest absolute z-score is
1.56.

`H = 0.1`, the same two regimes, eight runs, 2,000 paths, terminal time
1, 11 times, seed entropies 70 to 77, test split. The driver is the
plugin's near field at `kappa = 1`. The z-score of `E[Z] - 1` has
maximum absolute value 2.493 and mean -0.137. The z-score of
`E[v] - xi0` has maximum absolute value 2.305 and mean -0.388. Both sets
pass the z-rule. The bound on the maximum absolute z-score at 8 runs is
3.836, and the bound on the mean is 1.237.
