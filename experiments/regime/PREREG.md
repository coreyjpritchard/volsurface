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

## Fixed settings

These are settings, not measurements.

The generative Hurst index is `H = 0.1`. That is the default of the rough
Bergomi plugin. It is not the SPX skew-route value 0.167.

The low-regime vol of vol is `nu_L = 1.5`, the same plugin's default
`nu`. The primary ratio `nu_H / nu_L` is 2, so `nu_H = 3`. H1, H2 and H3
use this one specification.

The primary risk is entropic risk at `lambda = 1`. For a sample of P&L
the risk is `(1/lambda) log E[exp(-lambda X)]` with `lambda = 1`. That is
the deep-hedging default. CVaR at 95 percent is a report, not the gate.

The regime chain is the one the market tests run. The generator is

```
[[-1.5, 1.5],
 [ 0.8, -0.8]]
```

Mean holding times are 2/3 of a year and 1.25 years. The initial law is
`p0 = (0.7, 0.3)`. That is not the stationary law, which is about
`(0.348, 0.652)`. Duration multiples of 5 times and of 0.5 times
maturity are a sensitivity that replaces this matrix. They are not a
second gate.

The spot-variance correlation is `rho = -0.4`. The forward-variance
level `xi0` is flat at `0.04`.

The spot hedge costs `1e-4` of notional per unit traded, one basis
point, proportional to the absolute change in delta. The variance swap
and the calls are transacted at the model mid. They carry no extra
spread.

## Metrics

The primary risk is the entropic risk in the fixed settings, at
`lambda = 1`. The same paths also report CVaR at 95 percent, the mean,
the standard deviation, turnover, and cost. Cost uses the spot rate
in the fixed settings.

Test paths are paired across policies. Uncertainty is a hierarchical
bootstrap over paths and over network seeds.

## Multiple testing

H1, H2 and H3 are one family of three contrasts. A contrast counts as met
only when its lower confidence bound at level `1 - 0.05/3` is above 0.
That is a Bonferroni correction at family-wise level 0.05, one-sided.
The ratio grid, the two durations, and the conditional L2 sentence in H3
are outside the family. They are not extra gates.

## H1. Transfer cost

`Gap(L -> H) > 0` for L1 hedged with I1. The bar is the family lower
bound above 0.

## H2. Vol-of-vol content

The premium of a liability is its time-0 model price, one number for that
liability, not a number per path. The premium-normalised gap is
`Gap(L -> H)` divided by that premium. H2 is

`Gap(L2 -> H) / premium(L2) - Gap(L1 -> H) / premium(L1) > 0`.

L1 is the 30-day at-the-money call. L2 is the forward-start at-the-money
straddle that starts in 15 days and expires in 45 days. The bar is the
family lower bound on that difference, above 0.

## H3. Hedgeability

I2 is the variance swap priced under the trader's information set. The
price is the mixture of the regime-conditional variance swaps, weighted
by the filtered regime probabilities. It is not the variance swap
conditioned on the true regime. `surface_at` uses the same mixture.
At `H` other than `1/2` each regime's Monte Carlo is conditioned on
that regime's forward curve `xi_t(u)`.

The gate contrast is the L1 reduction: `Gap` with I1 minus `Gap` with I2,
for L1, above 0. The variance swap hedges the level of variance. The bar
is the family lower bound above 0. For L2, the reduction from I2 is
smaller than the reduction from I3, if the surrogate validation error is
under its bar. That L2 sentence is reported when I3 is eligible. It is
not a fourth member of the family.

## Sample size

The pilot uses 5 network seeds and 20,000 test paths. Before the main run
it reports the minimum detectable effect at 80 percent power. If that
effect is larger than 10 percent of `pi_H`'s risk on the high-`nu`
regime, the pilot raises the path count or the seed count and writes the
new counts down before the main run.

## Controls

Negative control: `nu_H = nu_L`. Every gap interval covers 0.

Positive control, same regime: train and test in one regime. The gap is
about 0.

Positive control, wrong correlation: the sign of `rho` used in training
is the opposite of the sign used in testing. The gap is clearly above 0.

## Design knobs, not results

Regime durations of 5 times maturity and of 0.5 times maturity are a
design axis. They replace the pinned transition matrix. The ratio grid
`{1, 1.5, 2}` is a sensitivity around the primary ratio 2. Those values
are settings. They are not measurements, and they are outside the family
of three.

## Not the gate

H4 is exploratory. Two Hamilton filters are defined. `pi_filt` sees the
spot filter and not the latent regime. The spot filter reads one-step
log realised variance, `log((d log Z)^2 / dt)`. It is not run on a
synthetic observation drawn from the latent regime. The emission mean is
the same in every regime, because the return's variance is the spot
variance and not `nu`. The variance of `log(chi^2_1)` is `pi^2 / 2`,
which dwarfs the gap in `nu^2` between the two regimes, so this
observation does not separate them. A trailing window that still sits
inside one regime visit has the same problem, and a longer window mixes
the regimes. Prices, including the I2 variance swap and `surface_at`,
read this filter. They do not read the latent label.

The log-variance filter is an oracle upper bound, in the same sense as
`pi_oracle`. It reads `log v_i - log v_{i-1}`, the true latent log
variance. At `H = 1/2` that increment is
`Normal(-nu^2 dt / 2, nu^2 dt)` for the regime on the left of the step.
It is not a deployable policy and it is not `pi_filt`. At any other `H`
that Gaussian is not the rough likelihood.

On the market-test draw, 600 paths, clock index 10, seed entropy 12,
test split, `nu` is 0.4 and 1.1. That pair is not the pilot pair 1.5
and 3. The generator, initial law `(0.7, 0.3)`, `rho = -0.4`, and flat
`xi0 = 0.04` are the pins above. The spot cost `1e-4` is not an input
to either filter. The base rate is 0.5383. The spot filter's modal hit
rate is 0.4783, lift -0.0600, confusion `[[129, 148], [165, 158]]`,
Brier score 0.5222, log score -0.7185. Quoting the base rate scores
Brier 0.4971 and log score -0.6902. The log-variance filter's modal
hit rate is 0.8067, lift 0.2683, confusion `[[219, 58], [58, 265]]`,
Brier score 0.2832, log score -0.4444. The call-bin accuracy 0.6600 on
this draw is the price bin, not the spot filter. With 1,024 copies the
mean absolute call gap against the true-regime mixture is 0.0002966
for the spot filter and 0.0001595 for the log-variance filter.
Implied-vol root-mean-square error on 35,060 finite nodes is 0.005305
and 0.003815. The variance swap does not move at `H = 1/2`.

H5, a link between the filter and minimal DYSANOS, is exploratory.
Neither H4 nor H5 is a hypothesis this draft freezes. Statements about
DYSANOS that were the motivation for H5 remain UNVERIFIED. The dynamic
surface in this repository is minimal DYSANOS. The full v3 baseline is
future work.
