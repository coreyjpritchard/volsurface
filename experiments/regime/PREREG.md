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

H4 is exploratory. The Hamilton filter is run on one-step log realised
variance of the simulated paths, `log((d log Z)^2 / dt)`. It is not run
on a synthetic observation drawn from the latent regime. The emission
mean is the same in every regime, because the return's variance is the
spot variance and not `nu`. The variance of `log(chi^2_1)` is `pi^2 / 2`,
which dwarfs the gap in `nu^2` between the two regimes, so this
observation does not separate them. A trailing window that still sits
inside one regime visit has the same problem, and a longer window mixes
the regimes. The exploratory summary is the filter's hit rate minus the
unconditional base rate, the larger marginal frequency of the latent
regime. Prices, including the I2 variance swap and `surface_at`, do not
determine the latent regime: a price bin contains more than one regime.
H5, a link between the filter and minimal DYSANOS, is exploratory.
Neither is a hypothesis this draft freezes. Statements about DYSANOS that
were the motivation for H5 remain UNVERIFIED. The dynamic surface in
this repository is minimal DYSANOS. The full v3 baseline is future work.
