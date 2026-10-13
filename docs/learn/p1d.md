# P1d: five ways to differentiate the same call

## What to watch

The figure `docs/figures/p1d_greeks.png` has two panels.

The left panel is the root-mean-square error of a forward delta, against how
many paths were used, on log-log axes. The reference is the Black-Scholes
formula. Pathwise and autodiff lie on top of each other: autodiff is that
same derivative, done by the library in float64. The likelihood ratio sits
above them. It is the same price, with the derivative moved onto the density,
and that score is a noisier random variable. The grey line is slope -1/2,
which is what an average of independent paths does when the variance stays
put. Six seeds and nine strikes are pooled so one unlucky seed does not draw
the line.

The right panel is a central finite difference of the closed form, absolute
error against the bump. Large bumps truncate. Tiny bumps round. On this call
the valley is near a bump of 3e-4, where the error is about 6e-12.

## The definition

Pathwise holds the normals fixed and differentiates `df (F Z - K)+`. The
delta on a path is `df Z` when the call finishes in the money, and 0
otherwise. A gamma and a digital differentiate a corner, so that estimator is
0 almost surely and is refused. Returning the 0 would be a wrong greek.

The likelihood ratio leaves the payoff alone and multiplies by the score of
the lognormal density of the terminal forward. A digital is an ordinary
payoff for that method. The score of the forward divides by `sigma² T`, so a
short-dated digital has the larger standard error. A vanilla's standard error
need not, because the call payoff itself shrinks with expiry.

Autodiff replays the exact log update in torch, float64, on the same normals.
It matches the pathwise value to 1e-10. It does not grow a gamma out of the
kink.

Finite differences bump the forward or `sigma`. Common random numbers reuse
the seed. On 40 seeds of 2,000 paths, with a bump of 1 on a forward of 100,
the variance without a shared seed is more than five times the variance with
one. Above the round-off floor, halving the bump cuts a first-derivative
error by about four. The slope of log error on log bump, from 1e-2 down to
1.25e-3, is 2 within 0.2.

## Hypotheses

These are not results.

- The valley at a bump of about 3e-4 is this delta on a forward of 100. A
  gamma, or a Monte Carlo difference, bottoms out at another bump.
- The likelihood ratio's extra error is mostly the score, and it shrinks like
  1 over square root of paths once the paths are numerous enough for the
  average to settle. The left panel is six seeds, not a proof of the slope.
- Common-random-number gains depend on the bump. A bump that moves the paths
  into a different region of the payoff keeps less of the cancellation.
- Spot and forward deltas agree here because the discount factor is held
  fixed. A spot bump that also moves the forward through a rate will not.

## Explain it back

1. Why do pathwise and autodiff land on the same dots, and why does the likelihood ratio not?
2. Why does the pathwise engine refuse a digital instead of returning zero?
3. On the right-hand panel, why does the error rise again once the bump is very small?
