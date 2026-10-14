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
`37eb89ba`, the commit that returns the near field.

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
is that common curve. The variance-swap pricer integrates it. `surface_at`
prices each regime's Black call and mixes the prices with the same
weights. At `H` other than `1/2` the tenor slope is the constant-`nu`
rough ratio for each regime, anchored at `v_t`, then mixed. Behind the
clock the entry is NaN. On the clock it is `v_t`.

The filter observation is one-step log realised variance of the simulated
spot, `log((d log Z)^2 / dt)`. Time 0 has no return, and the probability
there is `p0`. The emission is Gaussian with mean
`log(xi0) - gamma - log(2)` in every regime, because the return's
variance is the spot variance and not `nu`. The variance in regime `j`
is `pi^2 / 2 + nu_j^2`. `pi^2 / 2` is about 4.93. For `nu` of 0.4 and
1.1 the extra term is 0.16 and 1.21, so the two Gaussians are close. A
trailing window of the same series was scored on a separate draw. On
2,000 paths and 21 times, a 5-step change in log windowed realised
variance, and the product of the return with the one-step change in log
realised variance, each stayed within about 0.02 of the base rate. The
chi-squared noise in a window that still sits inside one regime visit
dominates `nu^2` times the window length, and a longer window mixes the
regimes. The observation in the market is therefore still the one-step
series. A later return does not enter an earlier probability.

Prices from this information do not determine the latent regime.
Quantile bins of the call and of the variance swap each contain both
regimes. The bar is no longer that the modal accuracy is below 1. It is
a two-sided band on the accuracy minus the base rate, the larger share
of the latent regime at the clock. The price lift sits between 0.05 and
0.35. The log-RV filter's lift sits between -0.10 and 0.10.

The constant-`nu` forward curve, used as the rough ratio inside that
mixture, remains

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
The base rate, the larger share of the latent regime, is 0.5383. Eight
quantile bins of the at-the-money call, and the same bins of the variance
swap, each contain both regimes. The modal prediction is right on 0.6600
of the paths, for both prices. The lift over the base rate is 0.1217,
inside the band from 0.05 to 0.35. The log-RV filter's modal hit rate on
the same paths is 0.4783, a lift of -0.0600, inside the band from -0.10
to 0.10. The figure is `docs/figures/p3f_paths.png`.

`H = 0.1`, the same two regimes, eight runs, 2,000 paths, terminal time
1, 11 times, seed entropies 70 to 77, test split. The driver is the
plugin's near field at `kappa = 1`. The z-score of `E[Z] - 1` has
maximum absolute value 2.493 and mean -0.137. The z-score of
`E[v] - xi0` has maximum absolute value 2.305 and mean -0.388. Both sets
pass the z-rule. The bound on the maximum absolute z-score at 8 runs is
3.836, and the bound on the mean is 1.237.
