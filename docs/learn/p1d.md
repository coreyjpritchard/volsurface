# P1d: five ways to differentiate the same call

## How to read this

Each panel is an error against the Black-Scholes delta (or gamma) on log-log
axes, so a power law is a straight line. Left: Monte Carlo estimators against
path count. Middle: a central difference of the closed form against the bump.
Right: a central difference of a Monte Carlo price against the bump. The
takeaway is that pathwise differentiation is the cheapest accurate delta, and
that bumping only works when the noise cannot blow up. On the formula the
error is truncation (falling like h²) until rounding takes over (rising like
1/h), so there is a best bump. On a Monte Carlo price with independent paths
the price noise is divided by h, so the delta error grows like 1/h and the
gamma error like 1/h². Shared paths cancel most of that noise. The caveat:
this is one at-the-money call under Black-Scholes, and the ranking can change
for digitals or barriers, where pathwise needs smoothing.

## What to watch

The figure `docs/figures/p1d_greeks.png` has three panels. The contract is an
at-the-money call on a forward of 100, volatility 0.2, expiry 1, except the
left panel, which uses nine strikes with log-moneyness from -0.2 to 0.2.

The left panel is the root-mean-square error of a forward delta against how
many paths were used. The reference is the Black-Scholes formula. Pathwise
and autodiff lie on top of each other: autodiff is that same derivative,
done by the library in float64. The likelihood ratio sits above them. The
grey line is slope -1/2. Six seeds are pooled with the nine strikes. The
strikes on a seed share the paths, so the effective sample is about six
seeds, not 54 points.

The middle panel is a central finite difference of the closed form. Large
bumps truncate. Tiny bumps round. On this call the smallest error is at a
bump of 3.2e-4, and the absolute error there is 5.8e-12.

The right panel is the same difference on the Monte Carlo price, 4,000
paths and eight seeds, bumps from 0.05 to 8. The shared-path delta stays
near 0.009 and ticks up at the right edge. The independent delta falls like
1/h, from about 3 to about 0.015, and is still above the shared delta at a
bump of 8. The shared gamma falls from about 0.006 to about 0.0004. The
independent gamma falls like 1/h², from about 260 to about 0.01. The black
lines are the analytic truncation, `h² |f'''| / 6` for the delta and
`h² |f''''| / 12` for the gamma. Both sit below the Monte Carlo error on
this grid. At a bump of 8 the delta bias is about 0.003 and the gamma bias
is about 0.0002.

A simulator whose caller omits `bump` uses 0.01 times the level, so a
forward of 100 is bumped by 1. That sits where the shared delta has already
flattened. A closed form still uses 1e-4 times the level.

## The definition

Pathwise holds the normals fixed and differentiates `df (F Z - K)+`. The
delta on a path is `df Z` when the call finishes in the money, and 0
otherwise. A gamma and a digital differentiate a corner, so that estimator
is 0 almost surely and is refused. Returning the 0 would be a wrong greek.
A model with the control variate on is refused too: the engine
differentiates the paths, not the controlled price. The likelihood ratio
and autodiff refuse that flag for the same reason.

The likelihood ratio leaves the payoff alone and multiplies by the score of
the lognormal density of the terminal forward. A digital is an ordinary
payoff for that method. The score of the forward divides by `sigma² T`. On
seeds 7 and 8, 30,000 paths, the cash-or-nothing delta's standard error at
0.05 year is 4.49 times the standard error at 1 year. The test requires
more than twice. A vanilla's standard error need not grow as expiry
shrinks, because the call payoff itself shrinks.

Autodiff replays the exact log update in torch, float64, on the same
normals. It matches the pathwise value to 1e-10. Torch is an optional
extra named `autodiff`. Importing the greeks package does not import it.

Finite differences bump the forward or `sigma`. Common random numbers reuse
the seed, and a `SeedSequence` is copied so a second call still draws child
0. On a simulator the standard error is the scatter of the per-batch
differences when the seed is shared, and the root-sum-square of the two
price errors when it is not. A closed form reports no standard error. On 40
seeds of 2,000 paths, bump 1 on a forward of 100, the variance without a
shared seed was 412 times the variance with one. The rough calculation for
these settings is about 300. The test requires more than 50. Above the
round-off floor, halving the bump cuts a first-derivative error by about
four. The slope of log error on log bump, from 1e-2 down to 1.25e-3, is 2
within 0.2.

## Hypotheses

These are not results.

- The Monte Carlo curves on the right have not turned up, apart from a
  small rise in the shared delta at a bump of 8. A wider bump, or fewer
  paths, would put the truncation above the noise. This grid does not show
  that crossing.
- The likelihood ratio's extra error shrinks like 1 over square root of
  paths once the average has settled. The left panel is six seeds, not a
  proof of the slope.
- The variance-reduction factor depends on the bump. At bump 1 it was 412
  on this sample of 40. A bump that moves the paths into a different region
  of the payoff keeps less of the cancellation.
- Spot and forward deltas agree here because the discount factor is held
  fixed. A spot bump that also moves the forward through a rate will not.

## Explain it back

1. Why do pathwise and autodiff land on the same dots, and why does the likelihood ratio not?
2. Why does the pathwise engine refuse a digital instead of returning zero?
3. On the middle panel, why does the error rise again once the bump is very small?
4. On the right panel, why does the shared-path delta stay flat while the independent delta falls?
