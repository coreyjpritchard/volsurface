# Session 00 Heston fit: the 14 October numbers and the 22 October numbers

*Note dated 22 October 2026.*

Two versions of the session 00 Heston fit exist. Both come from
`docs/figure_00.py` on the same committed chain
(`data/snapshots/SPX_2026-10-14.parquet`, SPXW mids, CBOE delayed).

| | 14 October (LOG, old README) | 22 October (current) |
|---|---|---|
| rho | -0.533 | -0.551 |
| v0, kappa, theta, xi | 0.012593, 8.805, 0.044344, 1.740 | 0.0100, 4.11, 0.0806, 2.000 |
| front skew, 10 hours (market 2.965) | 1.865 | 2.115 |
| skew at 1.42 days (market 1.989) | 1.785 | 2.101 |
| skew at 351 days (market 0.282) | 0.180 | 0.241 |
| reported cost | 0.7454 | 0.5288 |
| cost on the smooth loss below | 0.8375 | 0.5288 |

## What did not change

Between the 14 October commit (`3f3be72`) and the rerun on 22 October, the
script, the three starting points, the parameter box, the optimiser
(scipy `least_squares`, `trf`), the unweighted loss (ATM level and skew at
all 40 expiries) and the data file are identical. The parquet is
byte-for-byte the same. Running the 14 October commit itself with the
dependencies now pinned in `uv.lock` (numpy 2.4.6, scipy 1.17.1) gives
`rho = -0.699`, cost 0.6472: the same as the current code before the fix. The
dependency versions used on 14 October were not recorded.

## Why the numbers moved

The loss was discontinuous. The model skew at each expiry is a quadratic fit
to 9 model implied vols across the skew window. At the 10-hour expiry the
outer call strike prices near `4e-7` of the forward, and
`bs.implied_vol` at its default relative tolerance of `1e-12` converges on
some parameter sets and returns NaN on neighbouring ones. When it returns NaN
the strike drops out of the quadratic fit and the front skew jumps, by up to
about 0.2. A change of `2e-4` in `rho` was enough. A least-squares solver on such a
loss stops on its step tolerance (`xtol`) wherever it meets a jump, and which
jump it meets depends on rounding, so a change in numpy or scipy moves the
answer. Both the 14 October run (per the LOG) and the 22 October rerun
ended on an `xtol` stop.

## Reproducing the 14 October fit

The LOG prints the 14 October parameters to 4 or 5 significant figures.
Drawing 600 parameter sets uniformly inside that printed precision and
evaluating the old loss (default tolerance) gives costs from 0.628 to 0.855:
84% of draws land in [0.83, 0.84) and 13% in [0.74, 0.75). One draw,
`v0 = 0.0125933, kappa = 8.8045897, theta = 0.0443444, xi = 1.7402747,
rho = -0.5334303`, gives cost 0.7454 and skews of 1.866, 1.785, 1.181 and 0.180 at 10 hours,
1.42 days, 7.42 days and 351 days. The LOG has 0.7454 and 1.865, 1.785,
1.181 and 0.180.
`python scripts/heston_fit_old_vs_new.py` prints every number in this
section and the next.

So the 14 October fit is reproducible as a value of the loss as it was
defined, on the low side of a jump. It is not a local minimum of the
underlying smooth problem: with the strike kept (tolerance `1e-8`) the same
parameters cost 0.8375, the gradient is far from zero (46.8 in `v0`), and one
`least_squares` run from them goes to cost 0.5288 with `rho = -0.5515`, the
same basin as the current fit.

## What is defensible in each version

- The qualitative result is the same in both: on this chain the at-the-money
  skew follows `0.305 T^-0.333` (H about 0.17), and a Heston fit in the
  session box (`xi <= 2`) undershoots the 10-hour skew (1.87 or 2.12
  against 2.96) and, at one year, undershoots it too (0.18 or 0.24 against
  0.28). The power law does not depend on the Heston fit at all.
- The old numbers are what the 14 October code printed and they can be
  recovered from the same loss, as above. They are not the best fit in the
  box.
- The current numbers come from a smooth loss (`iv_tol = 1e-8`) and 35
  seeded starts. 27 of the 35 starts end within 1% of the best cost, with
  `rho` between -0.553 and -0.551. `v0` and `xi` end on the edge of the box,
  and `kappa` and `theta` are not identified (kappa from 2.0 to 7.1 across
  those starts), so only `rho`, the skews and the cost should be quoted.

## Update, later on 22 October: the box was binding

The 22 October fit above ends with `v0` and `xi` on the edge of the session
box, so its flat short end was a property of the box. `docs/figure_00.py`
now uses a wider box, `v0` in [1e-4, 0.15], `kappa` in [0.1, 50], `theta` in
[0.001, 0.3], `xi` in [0.05, 5], with the same loss and seeded design. One of
the 35 starts has non-finite residuals in this box and is skipped.

| | session box (`xi <= 2`) | `xi <= 5` (current figure) | `xi <= 10` (`XI_MAX=10`) |
|---|---|---|---|
| cost | 0.5288 | 0.0658 | 0.0197 |
| v0, kappa, theta | 0.0100, 4.11, 0.0806 | 0.00292, 10.15, 0.0563 | 0.00553, 20.76, 0.0670 |
| xi, rho | 2.000, -0.5515 | 5.000, -0.6218 | 10.000, -0.5584 |
| on the bound | v0, xi | xi | xi |
| Feller ratio | 0.17 | 0.046 | 0.028 |
| front skew (market 2.965) | 2.115 | 2.782 | 2.930 |
| skew at 351 days (market 0.282) | 0.241 | 0.290 | 0.263 |
| front ATM vol (market 0.151) | 0.099 | 0.056 | 0.070 |
| ATM rmse over 40 expiries | 0.0158 | 0.0365 | 0.0240 |

The cost keeps falling as the `xi` bound rises, so this loss has no interior
optimum in `xi`. Heston buys the short-end skew with a large vol-of-vol and a
small `v0` (the T to 0 skew limit is `rho xi / (4 sqrt(v0))`), and pays for it
in the short-dated at-the-money level, which tends to `sqrt(v0)`. The earlier
README sentence that calibrated Heston "goes flat below about 2 days" was a
property of the session box and is withdrawn. The power law and the 14 and
22 October explanations above are unaffected.
