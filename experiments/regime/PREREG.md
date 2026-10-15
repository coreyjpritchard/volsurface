# Regime experiment, pre-registration

Status: draft. Pinned on 15 October 2026. This file is not frozen.

The pilot is a CPU run. No GPU is assumed. The regime variable is `nu`
only. `xi_level` and `both` are not in the pilot. The hedger and the
experiment runner are not in this repository. They live in
`deep-hedging`.

`Gap(X -> Y)` is the risk of the policy trained on `X` and evaluated on
`Y`, minus the risk of the policy trained on `Y` and evaluated on `Y`.

## Liabilities and instruments

A day is `1/252` of a trading year. Every maturity below uses that day.

L1 is the at-the-money call with maturity 21 trading days, one month.
L2 is the forward-start at-the-money straddle that starts 15 trading days
after the episode and expires 45 trading days after the episode. Its life
is 30 trading days.

I1 is the spot. I2 adds the 21-trading-day variance swap to the spot. I3
adds the 21-trading-day at-the-money call to the spot, priced by
`SurrogatePricer`. I3 enters a comparison only when that pricer's
validation error is under the bar it was given. The martingale pre-gate
does not include I3. That instrument stays untested until the surrogate
exists.

The hedging episode starts at `t0 = 0.5`, which is 126 trading days. Paths
are drawn from time 0, so the filter at every episode date sees the
burn-in.

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
use this one specification. The proposal's symbol for this vol of vol is
`eta`. In this repository the parameter is `nu`.

The primary risk is entropic risk of P&L per unit of the liability's
time-0 premium, at `lambda = 0.1`. For a sample of that P&L the risk is
`(1/lambda) log E[exp(-lambda X)]`. CVaR at 95 percent is a report, not
the gate.

The tail-weight check is part of the pin. On the evaluation paths, form
the weights `exp(-lambda X)`. The paths with the largest 1 percent of
those weights, the count rounded up, must carry under 10 percent of the
total weight. If the share is 10 percent or more, replace `lambda` by
`lambda / 2`, recompute the share on the same paths, and freeze the new
value. Do not halve a second time. The measured check is in the pre-gate
section. It passes at `lambda = 0.1`, so the value stays 0.1.

The time between hedge trades is one day, `1/252` of a trading year.

The regime chain is the one the market tests run. The generator is

```
[[-1.5, 1.5],
 [ 0.8, -0.8]]
```

Mean holding times are 2/3 of a year and 1.25 years. The initial law is
`p0 = (0.7, 0.3)`. That is not the stationary law, which is about
`(0.348, 0.652)`. The same initial law is kept when the generator is
scaled.

Regime duration is the unweighted mean of the two mean holding times,
`(2/3 + 5/4) / 2 = 23/24` of a year. Scaling the generator by `c` divides
that duration by `c`. L1's maturity is `21/252 = 1/12` of a year. The
scale that makes the duration 5 times that maturity is

`c5 = (23/24) / (5/12) = 2.3`,

so `Q5 = 2.3 Q`. The scale that makes the duration 0.5 times that
maturity is

`c0.5 = (23/24) / (1/24) = 23`,

so `Q0.5 = 23 Q`. The scaled generators are

```
Q5   = [[-3.45,  3.45],
        [ 1.84, -1.84]]

Q0.5 = [[-34.5,  34.5],
        [ 18.4, -18.4]]
```

These two replace the primary generator for that sensitivity. They are
not a second gate. L2's life is 30 trading days. The same generators then
have duration `5 * 21/30 = 3.5` times that life, and `0.5 * 21/30 = 0.35`
times that life. Those two multiples are what falls out of the L1 scales.
They are not a separate pin.

The ratio grid is a dose-response report. It is outside the family of
three. `nu_L` stays 1.5. `nu_H` is in `{1.5, 2.25, 3}`, so the ratios are
1, 1.5 and 2. Ratio 1 is the negative control: `nu_H = nu_L`, and every
gap interval covers 0. The primary gate remains ratio 2.

The spot-variance correlation is `rho = -0.4`. The forward-variance
level `xi0` is flat at `0.04`.

The spot hedge costs `1e-4` of notional per unit traded, one basis
point, proportional to the absolute change in delta. The variance swap
and the calls are transacted at the model mid. They carry no extra
spread.

## Metrics

The primary risk is the entropic risk in the fixed settings, at
`lambda = 0.1`, on P&L per unit of premium. The same paths also report
CVaR at 95 percent, the mean, the standard deviation, turnover, and
cost. Cost uses the spot rate in the fixed settings.

Test paths are paired across policies. Uncertainty is a hierarchical
paired percentile bootstrap. The resample draws network seeds first, then
paths within the drawn seeds, and the same draw is applied to every
policy. The resample count is `B = 10,000`. The gate is the one-sided
lower percentile at level `1 - 0.05/3`. BCa intervals, computed on the
same replicates, are a sensitivity report. They are not the gate.

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
liability, not a number per path. Each premium is the Monte Carlo mean
at `n_mc = 65536` copies, which is `2^16`. That count is for these
premiums. The H3 surface stays at 1024 copies. The premium is held fixed
in the bootstrap. Its Monte Carlo standard error is reported beside the
premium. The premium-normalised gap is `Gap(L -> H)` divided by that
premium. H2 is

`Gap(L2 -> H) / premium(L2) - Gap(L1 -> H) / premium(L1) > 0`.

L1 is the 21-trading-day at-the-money call. L2 is the forward-start
at-the-money straddle with a life of 30 trading days. The bar is the
family lower bound on that difference, above 0.

## H3. Hedgeability

I2 is the variance swap priced under the trader's information set. The
price is the mixture of the regime-conditional variance swaps, weighted
by the filtered regime probabilities. It is not the variance swap
conditioned on the true regime. `surface_at` uses the same mixture.
At `H` other than `1/2` each regime's Monte Carlo is conditioned on
that regime's forward curve `xi_t(u)`. A later date on that curve is
the closed-form constant-`nu` curve. The clock entry is `v_t`.

The gate uses `price_filter="rv"`. That filter reads the change in
windowed log realised variance, timed to the regime two steps back.
The library default stays `"spot"`. This draft does not use the default
for H3 or H4. `n_mc` is 1024. The error bar on a price is
`SurfaceBatch.stderr`, the residual standard deviation divided by
`sqrt(n_mc)`. A price that enters the gate is the Monte Carlo mean, and
the standard error is reported beside it. The surface seed is
`spawn_key + (4, regime)`. The clock is not in the key, so hedge dates
share shocks. A gap across hedge dates is the gap in the weights.

The gate contrast is the L1 reduction: `Gap` with I1 minus `Gap` with I2,
for L1, above 0. The variance swap hedges the level of variance. The bar
is the family lower bound above 0. For L2, the reduction from I2 is
smaller than the reduction from I3, if the surrogate validation error is
under its bar. That L2 sentence is reported when I3 is eligible. It is
not a fourth member of the family.

## Sample size

Training uses 100,000 paths. Validation uses 20,000 paths. The test uses
20,000 paths. These three counts stay fixed. The validation paths are used
only for early stopping and for selection. They do not enter the test and
they do not enter the gate. The pilot starts from 5 network seeds.

Seed variance dominates when the sample variance of the seed-level mean
contrasts, divided by the number of seeds, exceeds the mean within-seed
path variance divided by the product of the number of seeds and the test
path count. If seed variance dominates, the pilot moves from 5 seeds to
10 seeds and then freezes the seed count. It does not change the path
counts.

## Controls

Negative control: `nu_H = nu_L`. Every gap interval covers 0. On the
ratio grid this is `nu_H = 1.5`.

Positive control, same regime: train and test in one regime. The
confidence interval covers 0, and the absolute gap is at most 0.10 of
that regime's entropic risk.

Positive control, wrong correlation: the sign of `rho` used in training
is the opposite of the sign used in testing. The lower confidence bound
of the gap is above 0.10 of the high-`nu` regime's entropic risk.

## Pinned, and still a draft

Eight settings that were open are now pinned. The list has eight items.
Pinning them does not freeze the file.

- P&L is per unit of the liability's time-0 premium, at `lambda = 0.1`,
  with the tail-weight rule above.
- The hedge step is one day, `1/252` of a trading year.
- The path counts are 100,000, 20,000 and 20,000, untouched. Validation
  is only for early stopping and selection.
- The bootstrap is the hierarchical paired percentile bootstrap above,
  with the 5-to-10 seed rule.
- Regime duration of 5 times the 21-day maturity, `Q5 = 2.3 Q`.
- Regime duration of 0.5 times the 21-day maturity, `Q0.5 = 23 Q`.
- The ratio grid is `nu_H` in `{1.5, 2.25, 3}` at `nu_L = 1.5`, outside
  the family. Ratio 1 is the negative control.
- H5 stays exploratory. Its statement is below.

## Pre-gate checks

These checks sit in front of the pilot. They are not members of the
family of three.

### Tail weight, one evaluation seed

The market is the pre-registered switching market: `H = 0.1`, `nu` 1.5
and 3, `rho = -0.4`, flat `xi0 = 0.04`, the primary generator, and
`p0 = (0.7, 0.3)`. The unit forward starts at 1. The books below are
written at time 0. The payoff is `max(Z_T - 1, 0)`.
`X = (premium - payoff + hedge - cost) / premium`. The unhedged book has
`hedge = 0` and `cost = 0`. The premium is the mean payoff on 65,536
paths from `run_seed(1, 0)`. The evaluation sample is 20,000 paths from
`run_seed(12, 2)`, one sample for every lambda on that book.

On the 30-trading-day call the premium is 0.02459316, with Monte Carlo
standard error 0.00014734. Minus the mean of the unhedged `X` is
0.004158. Both lambda rows share that number. At `lambda = 1` the
entropic risk is 17.596 and the top 200 paths carry exponential-weight
share 1.000. That fails the 10 percent bar. At `lambda = 0.1` the
entropic risk is 0.1495 and the share is 0.02277. That passes.

On the 21-trading-day call the premium is 0.02070205, with standard
error 0.00012343. At `lambda = 0.1` the unhedged share is 0.02337. That
also passes. Lambda stays 0.1. It is not halved.

The hedged book rebalances once a day. The hedge is the Black-Scholes
call delta. One column uses volatility `sqrt(v_t)`. The other uses the
flat volatility `sqrt(xi0) = 0.2`. The spot cost is `1e-4` of notional
on the turnover, including entry and the unwind. At `lambda = 0.1` the
top 1 percent shares on the 21-trading-day call are 0.01368 at
`sqrt(v_t)` and 0.01671 at the flat volatility. Those are the 1.4 percent
and 1.7 percent shares, to one decimal place. On the 30-trading-day call
the same two hedges give 0.01363 and 0.01581.

### Quote information

Prices enter the P&L only. They are not policy inputs. The three filters
are functions of the observation series: one-step log realised variance
`y`, the change in windowed log realised variance `y_rv`, and the latent
increment of `log v`. `regime_prob` is an output of those filters. The
variance-swap quote and `surface_at` read `regime_prob` and the forward
curve. They do not write a price back into the filter. The audit
recomputes the three probability arrays from `y`, `y_rv` and `log v` and
matches the arrays stored on the path.

### Variance-swap left-point leg

`VarianceSwapPricer` returns the remaining integral `int_t^T xi_t(u) du`,
a left Riemann sum on `xi_tenor`. It does not add the left-point model
variance. That leg is the sum of `v` at the left of each step times the
step. It is not the sum of squared log returns. At inception the leg is
zero, so the entry quote is the whole floating leg. The mark of a live
swap is the left-point model variance from the contract's start up to
`t`, plus that remaining integral, minus the strike times the contractual
variance period.

The pre-gate tests that mark. See the martingale section. The
deep-hedging repository still needs the same test on its own P&L. The
hedger is not here.

### Log-space mean of variance

For constant `nu`, the spot variance is `xi0` times an exponential
martingale whose compensator is `nu^2 t^{2H} / 2`. If the Volterra driver
has mean zero,

`E[log v_T] = log xi0 - nu^2 T^{2H} / 2`.

The threshold is the z-rule in `volsurface/mc/zrule.py`. On a sample of
`n` replication z-scores, `max |z|` is below `Phi^{-1}(1 - 0.001 / (2n))`
and `|mean z|` is below `3.5 / sqrt(n)`. At `n = 8` those bounds are
3.836 and 1.237.

Eight runs of 2,000 paths, `H = 0.1`, `T = 1`, 11 times, seeds
`run_seed(70 + r, 2)`. The log-space z-scores pass at constant `nu = 1.5`
and at constant `nu = 3`. The largest absolute z is 1.49. The driver is
linear in `nu` and the shocks are shared, so those two log-space draws
match. On the switching market the target is pathwise: `log xi0` minus
that path's compensator. The same eight seeds pass. The largest absolute
z is 1.74. The pre-gate is this log-space check.

### Instrument martingale

The spot on the 30-step daily grid is a martingale. Eight runs of 2,000
paths, pre-registered switching market, `H = 0.1`, seeds
`run_seed(40 + r, 2)`. The z-score of the terminal mean `E[Z_T] - 1`
passes the z-rule. The one-step P&L is `Z_i - Z_{i-1}`. Its
cross-sectional mean at each date is regressed on a constant and on the
left calendar time, weighted by the reciprocal of that date's sample
variance. The intercept z-scores and the slope z-scores pass the z-rule.
The largest absolute z on this draw is 1.20 for the terminal mean, 1.75
for the intercept, and 2.21 for the slope.

The variance-swap mark is tested on the daily episode at the gate
settings: `H = 0.1`, `nu` 1.5 and 3, `price_filter="rv"`, `n_sub = 21`.
The contract is entered at `t0 = 0.5`. Eight runs of 1,500 paths, seeds
`run_seed(40 + r, 2)`. The gap is the mark at maturity minus the mark at
`t0`. Its z-scores pass for the 21-trading-day swap and for the
30-trading-day swap. The one-step mark change is regressed on a constant
and on calendar time, weighted by the reciprocal of that date's sample
variance. The last step is left out of the regression: its remaining
integral is the diagonal `v`, so the mark change is numerical noise.
The intercept z-scores and the slope z-scores pass for both maturities.

I3, the call priced by `SurrogatePricer`, is the third instrument. It
stays untested until that surrogate exists.

A hold from time 0 to `T = 1`, eight runs of 1,500 paths, seeds
`run_seed(40 + r, 0)`, compares the left-point model variance with the
time-0 remaining integral. Constant `nu = 1.5` has mean ratio 1.004 and
passes. Constant `nu = 3` has mean ratio 0.965 and passes. The switching
market has mean ratio 0.975 and passes.

### Episode filter scores

The grid is daily from time 0 through `t0` plus 30 trading days. The
episode starts at index 126. Seed entropy 12, test split, `n_pilot = 1024`,
`n_sub = 21`. A path is switched at a date when its regime on the path up
to that date is not constant. The chain does not depend on `H` or on
`nu`. At `t0`, 600 paths split into 316 not switched and 284 switched.
4,000 paths split into 2,182 and 1,818.

Calibration error is the path-weighted mean absolute gap between the bin
hit rate and the mean predicted probability of the high regime. Five
bins. A bin with fewer than 40 paths is left out.

The windowed score assumes 21 intraday returns on each day. The windowed
level is left-point model variance `v` plus chi-squared noise on 21
degrees, up to the drift and the bridge. It is not integrated variance.
That is the realism limit of these daily-grid skills.

At the episode start, market-test settings, `H = 1/2`, `nu` 0.4 and 1.1,
600 paths. Not switched. Spot skill -0.0671, calibration error 0.0879.
Windowed skill -0.1659, calibration error 0.1806. Latent skill 0.9908,
calibration error 0.0188. Switched. Spot skill -0.7035, calibration
error 0.3504. Windowed skill -0.2716, calibration error 0.1877. Latent
skill 0.8371, calibration error 0.0054. The same settings, 4,000 paths.
Not switched: spot skill -0.1113, windowed skill -0.1723, latent skill
0.9814. Switched: spot skill -0.6900, windowed skill -0.2283, latent
skill 0.7668.

At the episode start, pre-registered settings, `H = 0.1`, `nu` 1.5 and 3,
600 paths. Not switched. Spot hit rate 0.8354, skill 0.4437, calibration
error 0.1081. Windowed hit rate 0.9873, skill 0.9365, calibration error
0.0433. Latent hit rate 0.9842, skill 0.9364, calibration error 0.0308.
Switched. Spot hit rate 0.7535, skill -0.0413, calibration error 0.1660.
Windowed hit rate 0.8979, skill 0.5761, calibration error 0.0319. Latent
hit rate 0.9155, skill 0.6502, calibration error 0.0299. The same
settings, 4,000 paths. Not switched: spot skill 0.4702, windowed skill
0.9214, latent skill 0.9439. Switched: spot skill -0.1317, windowed
skill 0.4921, latent skill 0.5688. The 4,000-path windowed confusions
are `[[1327, 34], [13, 808]]` on the paths that stayed and
`[[396, 100], [128, 1194]]` on the paths that switched.

Pooled over the episode's own dates, pre-registered settings, 4,000
paths. The 21-day episode has dates 126 through 146. Not switched:
windowed skill 0.9220, confusion `[[26578, 637], [293, 16276]]`.
Switched: windowed skill 0.5391, confusion
`[[9039, 2201], [2476, 26500]]`. The 30-day episode has dates 126 through
155. Not switched: windowed skill 0.9190, confusion
`[[37047, 874], [478, 22862]]`. Switched: windowed skill 0.5438,
confusion `[[13467, 3159], [3628, 38485]]`. Spot skill on the switched
paths is negative at both horizons. The windowed filter keeps skill near
0.54 on those paths.

## Not the gate

H4 is exploratory. Three Hamilton filters are defined. For this gate,
`pi_filt` is the differenced windowed filter, `price_filter="rv"`.
Prices, including the I2 variance swap and `surface_at`, read that
filter. The spot filter is stored and is not the gate price. The
library default remains `"spot"`. The latent-variance benchmark is not
a deployable policy and it is not `pi_oracle`. `pi_oracle` still sees
the latent regime label.

The spot filter reads one-step log realised variance,
`log((d log Z)^2 / dt)`. It is not run on a synthetic observation drawn
from the latent regime. The emission mean is the same in every regime,
because the return's variance is the spot variance and not `nu`. The
variance of `log(chi^2_1)` is `pi^2 / 2`. At `nu` of 0.4 and 1.1 that
term dwarfs the gap in `nu^2`, and the filter has no Brier skill. At
the pre-registered `nu` of 1.5 and 3 the same emission does have skill.
The claim depends on the settings. There is no single lift band.

The windowed level is `v` at the start of the step plus chi-squared
noise on 21 degrees, up to the drift and the bridge. It is not
integrated variance. The gate filter reads the change in that level.
The change at date `i` is emitted by the regime at date `i - 2`. The
emission table is fit on pilot paths that each stay in one regime.
That fit does not read `v`.

The latent-variance benchmark reads `log v_i - log v_{i-1}` through a
pilot table, at every `H`. At `H = 1/2` the table estimates
`Normal(-nu^2 dt / 2, nu^2 dt)`. At `H = 0.1` that Gaussian is not the
rough likelihood, so the benchmark does not use it and it is not an
oracle.

Six hundred paths, clock index 10, seed entropy 12, test split, 21 times
on `[0, 1]`. The generator, initial law `(0.7, 0.3)`, `rho = -0.4`, and
flat `xi0 = 0.04` are the pins above. The spot cost `1e-4` is not an
input to any filter. The chain does not depend on `H` or on `nu`: 277
paths in the low regime and 323 in the high regime. The base rate is
0.5383. Quoting it scores Brier 0.4971 and log score -0.6902. Brier
skill is `1 - Brier / Brier_base`. The episode scores, after the burn-in,
are in the pre-gate section.

Market-test settings, `H = 1/2`, `nu` 0.4 and 1.1, 600 paths. Spot: 287
of 600, skill -0.0505, calibration error 0.0713. Windowed: 345 of 600,
skill 0.0035, calibration error 0.0295. Latent benchmark: 487 of 600,
skill 0.4355, calibration error 0.0221. The same settings, 4,000 paths.
Spot skill -0.0292, calibration error 0.0591. Windowed skill 0.0094,
calibration error 0.0106. Latent skill 0.4785, calibration error 0.0066.
Implied-vol root-mean-square error on 35,032 finite nodes, 600 paths,
is 0.005348, 0.005232, and 0.003915. The call-bin accuracy 0.6600 on
this draw is the price bin, not a filter hit rate. The variance swap
does not move at `H = 1/2`.

Pre-registered settings, `H = 0.1`, `nu` 1.5 and 3, 600 paths. Spot: 410
of 600, hit rate 0.6833, skill 0.1565, calibration error 0.0632.
Windowed: 452 of 600, hit rate 0.7533, skill 0.2838, calibration error
0.0255. Latent benchmark: 464 of 600, skill 0.3330, calibration error
0.0487. The same settings, 4,000 paths. Spot hit rate 0.6608, skill
0.1178, calibration error 0.0527. Windowed hit rate 0.7048, skill
0.2071, calibration error 0.0319. Latent hit rate 0.7438, skill 0.2849,
calibration error 0.0338. The windowed skill at clocks 5, 10 and 20 on
that 4,000-path draw is 0.2079, 0.2071 and 0.2089. Implied-vol
root-mean-square error on 31,684 finite nodes, 600 paths, is 0.02069,
0.01906, and 0.01805.

Constant `nu`, eight runs of 500 paths, clock index 8. The mean forward
curve after the clock, divided by `xi0`, is 1.019 at `nu = 1.5` and
1.031 at `nu = 3`. Both pass the z-rule.

H5 is exploratory. It is not a gate. The link to minimal DYSANOS is the
tail diffusion in DYSANOS v3, arXiv:2608.12587, Appendix C, equations
(55) and (56). Those two equations were read from the paper's HTML.
This restatement does not claim a reading of the rest of the paper.
Minimal DYSANOS in this repository does not implement the diffusion.

Equation (55) draws the innovation from a scalar log-variance state `g`.
With `z` a standard normal vector independent of the scalar shock in
(56),

`epsilon_{n+1} = sqrt(rho exp(g_n - v_g / 2)) G_5' z_{n+1}`.

Equation (56) is the exact discrete transition of a stationary
Ornstein-Uhlenbeck diffusion, so a fractional time needs no Euler step:

`g_{n+1} = phi g_n + sqrt(v_g (1 - phi^2)) xi_{n+1}`.

Stationarity gives `g ~ Normal(0, v_g)`, and then
`E[rho exp(g - v_g / 2)] = rho`. The fit on 703 pre-2025 one-week
innovations is `(rho, phi, v_g) = (0.878, 0.139, 0.0649)`, from the
Kalman likelihood of equation (57),
`log(delta^2) = log rho + g - v_g / 2 + log(chi^2_200 / 200)`. That scale
is the paper's fit on its factor innovations. It is not fitted to the
variance in this repository. On 158 innovations from 2025, the log-radius
negative log likelihood falls from 2.453 to 0.125 against a
pre-2025-scaled Gaussian. Neither H4 nor H5 is frozen.
