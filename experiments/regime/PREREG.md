# Regime experiment, pre-registration

Status: draft. To freeze before the pilot ends. This file is not frozen.

The pilot is a CPU run. No GPU is assumed. The regime variable is `nu`
only. `xi_level` and `both` are not in the pilot. The hedger and the
experiment runner are not in this repository. They live in
`deep-hedging`.

`Gap(X -> Y)` is the risk of the policy trained on `X` and evaluated on
`Y`, minus the risk of the policy trained on `Y` and evaluated on `Y`.

## Liabilities and instruments

L1 is the 30-day at-the-money call. L2 is the forward-start at-the-money
straddle that starts in 15 days and expires in 45 days.

I1 is the spot. I2 adds the 30-day variance swap to the spot. I3 adds the
30-day at-the-money call to the spot, priced by `SurrogatePricer`. I3
enters a comparison only when that pricer's validation error is under the
bar it was given.

## Policies

`pi_L` is trained in the low-`nu` regime. `pi_H` is trained in the
high-`nu` regime. `pi_mix` is trained on a mixture of the two. `pi_filt`
sees the Hamilton filter probability and not the latent regime. `pi_oracle`
sees the latent regime. The oracle is a bound. It is not a deployable
policy.

Two benchmarks sit beside those policies. One is the Black-Scholes delta
at `sqrt(v_t)`. The other is the rBergomi model delta, used where the
pilot can afford it.

## Metrics

The primary risk is entropic risk at the deep-hedging default lambda.
That default is the number the hedger ships. This draft does not write a
different lambda. The same paths also report CVaR at 95 percent, the mean,
the standard deviation, turnover, and cost.

Test paths are paired across policies. Uncertainty is a hierarchical
bootstrap over paths and over network seeds.

## H1. Transfer cost

`Gap(L -> H) > 0` for L1 hedged with I1. The bar is a lower 95 percent
bound above 0.

## H2. Vol-of-vol content

The premium-normalised `Gap(L -> H)` is larger for L2 than for L1. The
bar is a lower 95 percent bound on that difference above 0.

## H3. Hedgeability

I2 is the variance swap priced under the trader's information set. The
price is the mixture of the regime-conditional variance swaps, weighted
by the filtered regime probabilities. It is not the variance swap
conditioned on the true regime. `surface_at` uses the same mixture.

For L1, adding I2 reduces `Gap(L -> H)`. The variance swap hedges the
level of variance. For L2, the reduction from I2 is smaller than the
reduction from I3, if the surrogate validation error is under its bar.
The directions are fixed.

## Sample size

The pilot uses at least 5 network seeds and 20,000 test paths. Before the
main run it reports the minimum detectable effect at 80 percent power. If
that effect is larger than 10 percent of `pi_H`'s risk on the high-`nu`
regime, the pilot raises the path count or the seed count before the main
run.

## Controls

Negative control: `nu_H = nu_L`. Every gap interval covers 0.

Positive control, same regime: train and test in one regime. The gap is
about 0.

Positive control, wrong correlation: the sign of `rho` used in training
is the opposite of the sign used in testing. The gap is clearly above 0.

## Design knobs, not results

Regime durations of 5 times maturity and of 0.5 times maturity are a
design axis. The ratio `nu_H / nu_L` is taken from the grid `{1, 1.5, 2}`.
Those values are settings. They are not measurements.

## Not the gate

H4 is exploratory. The Hamilton filter is run on log realised variance of
the simulated paths, `log((d log Z)^2 / dt)`. It is not run on a synthetic
observation drawn from the latent regime. The fraction of variance that
filter explains is the exploratory summary. Prices, including the I2
variance swap and `surface_at`, do not determine the latent regime: a
price bin contains more than one regime. H5, a link between the filter
and the toy dynamic surface, is exploratory. Neither is a hypothesis this
draft freezes. Statements about DYSANOS that were the motivation for H5
remain UNVERIFIED, and the dynamic surface in this repository is a toy
rather than a v3 baseline.
