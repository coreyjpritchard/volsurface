# Log

Dated entries, newest first. What was learned, what surprised, open questions.

**Current state:** Market paths switch `nu` on a continuous-time chain, price a variance swap from the forward curve, and attach a causal filter probability. **Next:** a minimal DYSANOS replication on a synthetic history, with the draft pre-registration.

## 2026-10-14 Library P3f: market paths and regimes

`xi_level` and `both` raise. `H = 0.1` with no plugin raises `RuntimeError` and does not build a substitute driver. A stand-in `driver_y` is called and is not labelled rBergomi. Train spawn key `(0,)` and test spawn key `(2,)` produce different first draws. The same seed repeats `Z`, the regime, and the filter probability.

The chain, 4,000 paths, seed entropy 1, test split, time 1. Generator rates 1.5 and 0.8, initial law `(0.7, 0.3)`. Counts 1,550 and 2,450. The law `p0 expm(Q t)` expects 1,532.5 and 2,467.5. Chi-square statistic `0.323`, p-value `0.570`. The bar in the test is p-value above `0.001`.

Terminal time 1, eight runs, 2,000 paths each, `nu` in `{0.4, 1.1}`, `rho = -0.4`, `xi0 = 0.04`, `H = 0.5`. The z-score of `E[Z] - 1` has maximum absolute value `1.193` and mean `-0.209`. The z-score of `E[v] - xi0` has maximum absolute value `1.838` and mean `-0.246`. Both sets pass the z-rule. This is the martingale hypothesis under switching, for the left-point log-Euler, not a proof for the continuous limit.

The variance swap at inception against realised left-point variance, eight runs, 1,500 paths. The z-score of the gap has maximum absolute value `1.819` and mean `-0.473`. It passes the z-rule.

The filter, 2,500 paths, 30 dates, observation means `-1` and `1`, variance `0.25`. Replacing the series after date 10 leaves the probabilities up to date 10 unchanged, to `1e-12`. Two bins at the last date have at least 40 paths. The largest gap between the hit rate and the mean predicted probability is `0.00108`. Both bins sit inside the 0.1 and 99.9 percent binomial quantiles.

Black-Scholes along one path, forward 100, strike 90, expiry 1, rate 0.01. The pricer matches the formula to `1e-12` at `t = 0` and at `t = 0.5`. At expiry and one step later the value is the payoff, the same number. A surrogate whose validation error is 10 bp against a bar of 5 bp raises. One whose error is 2 bp keeps that error. The figure is `docs/figures/p3f_paths.png`.

**Hypothesis:** the terminal mean of `Z` under this switch stays inside the z-rule at `H = 0.5`. The next specification is the same eight-run check when `Y` comes from `driver_y` at an `H` below one half. The quantity to report is the maximum absolute z-score, next to the `1.193` measured here.

## 2026-10-14 Library P3e: rough-volatility plug-in

`roughvol` does not import. The conformance test skips. `calibrate` calls `create("rbergomi")` and raises `KeyError` when that name is absent. It does not build another model.

On the known two-pillar mixture the log contract `VS(T) = -2 E[log Z_T]` matches `-2 q' log K + eta V` to `1e-12`, and the value at the midpoint maturity is the average of the two pillars to `1e-12`. On the three-expiry Heston book at `eta = 0.25` the same contract is nondecreasing on 13 maturities: every difference is at least `-1e-8`.

A test double, not labelled rBergomi, prices `sigma = 0.2 + 0.4 nu k`. With `H` fixed at 0.167 and seed 7 it recovers `nu = 0.3` to `1e-3`. A second call with the same seed returns the same parameters. `detail["rbergomi"]` is false and the label is `_MockVol`.

SPX, 40 expiries. The power law refit to the market skew is `0.30511 T^{-0.33283}`. The merged Heston parameters are `v0 = 0.007982`, `kappa = 13.466`, `theta = 0.03836`, `xi = 2.703`, `rho = -0.6508`. At `T = 0.00114` the market skew is `-2.9648` and the Heston skew is `-3.2744`. At `T = 0.02032` they are `-1.0565` and `-1.4822`. At `T = 0.4614` they are `-0.3817` and `-0.3881`. At `T = 0.96278` they are `-0.2815` and `-0.2387`. The front at-the-money level is `0.1511` in the market and `0.0879` in Heston. The figure is `docs/figures/p3e_skew.png`. No rBergomi curve is drawn.

**Hypothesis:** the merged Heston fit is steeper than this snapshot from the front out to about half a year, and flatter after that. The next specification is the same figure with `create("rbergomi")` at `H` free, `H = 0.167` and `H = 0.149`, once `roughvol` is installed. The size of the gap between those three curves and the power law is the quantity that comparison measures.

## 2026-10-14 Library P3d: discrete local volatility

A random nonnegative `Sigma` on 9 strikes, seed 1, step `0.2`. The columns of `Q` sum to 1 and each column has mean equal to its strike, both to `1e-12`. Torch float64 and NumPy agree to `1e-12` on seed 4, step `0.15`.

Round trip, seeds 2 and 3, step `0.25`. Calls go to `Sigma`, back to `Q`, and back to calls. The calls agree to `1e-10` (the measured gap on this draw is `1.1e-16`). The recovered interior `Sigma` differs from the input by at most `1.5e-14`. A source grid `(0.7, 1, 1.3)` split onto that target and then stepped keeps the mass at 1 and the mean at the source mean, to `1e-12`. Three expiries built from seeds 5, 6 and 7 pass `check_surface` at `1e-9` with an empty violation table.

SPX, the `eta = 0.25` surface, 39 steps, 37 log-moneyness nodes from `-0.25` to `0.2`. Median discrete local volatility `0.283`. The 99th percentile is `5.81`. The maximum is `212.3`. The root mean square of the second difference along strike is `15.7`. 5.4 percent of the cells are 0. The figure clips the colour scale at the 99th percentile and records the maximum in the title. The file is `docs/figures/p3d_dlv.png`.

**Hypothesis:** the cells above 5 are the ratio blowing up where the strike second difference is near 0, not a second mode of local volatility. The next specification is the same ratio on a grid that drops cells whose call convexity is below a stated floor. The median `0.283` is the level that specification should keep.

## 2026-10-14 Library P3c: SANOS at eta > 0

Theorem 3.1, fixed weights, atoms at 0.5, 1 and 1.5. The `eta = 0.25` call minus the `eta = 0` call is nonnegative at `T = 0.5`, `0.75` and `1`, to `1e-12`. On a 20,001-point grid the mixture density at each pillar has mass 1 and mean 1, to `1e-8`. The Heston book with cash half-spread 0.15, `eta = 0.25` and PCHIP in time, has no `check_surface` violation at `1e-9` and a positive minimum density.

On SPX, `T = 0.0203`, the finite-difference jump in `dC/dT` at `k = 0` is `0.00206` at step `1e-4`, `2.11e-4` at step `1e-5`, and `2.1e-5` at step `1e-6`. It scales with the step. At `k = 0.15` the jump stays at `-1.41e-4` when the step goes from `1e-4` to `1e-6`.

Market bar, `eta = 0.25`, `omega = 0`, PCHIP, `dx = 0.05`, interior point method. Grid: every pillar, every midpoint, and 71 log-moneyness nodes from `-0.4` to `0.3` (5,609 nodes), tolerance `1e-9`. Violations: 0. `in_spread = 1`, above 0.558. Objective `6.302e-05`. Wall time 149.8 s. Projection L1 `3.29e-13`. Minimum density on that grid `1.10e-38`, which is the wing of the shortest expiry. Every expiry's own in-spread fraction is 1.

Eta sweep, same solver and `omega = 0`. `eta = 0`: objective `1.633e-05`, 0 violations, 116.1 s, minimum density 0 because the call is piecewise linear. `eta = 0.0625`: objective `2.093e-05`, 0 violations, 146.1 s, minimum density `5.46e-142`. `eta = 0.25` is the bar above. The `eta = 0` objective here is the interior point value. The simplex value on the same book is the P3b entry, `1.752e-06`. Both have `in_spread = 1`.

`omega = 1` at `eta = 0.25` is labelled not guaranteed. Objective `5.529e-04`, `in_spread = 1`, wall time 171.2 s. On the same 5,609 nodes the report is not ok: 6 calendar violations, largest magnitude `4.5e-5`, all near `T = 0.10` and `k = 0.16` or `0.17`.

Shoot-out on that grid. Linear: `in_spread = 1`, 1,167 violations, minimum density 0, 0.030 s. Spline: `in_spread = 1`, 3,170 violations, minimum density `-197.0`, root mean square second difference of local variance `11.84`, 0.055 s. Raw SVI: `in_spread = 0.55823`, 2,244 violations, minimum density `-0.00330`, local-variance second-difference `64.52`, 8.56 s. That in-spread fraction is the same 0.558 as the earlier log entry. SANOS `eta = 0.25`: `in_spread = 1`, 0 violations, 149.8 s. The finite-difference local variance of SANOS on this sample has root mean square second difference `1.45e6`. That sample includes the front expiry, where the implied-vol step is unstable. It is not a smoothness of the mixture density.

**Hypothesis:** moving `eta` from 0 to 0.25 raises the objective from `1.633e-05` to `6.302e-05` and does not move the in-spread fraction off 1. The next specification is the discrete local variance of these marginals. The finite-difference Dupire ratio is not yet a stable object on the front of this book.

## 2026-10-14 Library P3b: SANOS at eta = 0

The call at a pillar is `sum q (K - x)+`. On a known two-point-free marginal (atoms at 0.5, 1 and 1.5) that formula and `SanosSurface.call` agree to `1e-12`.

Heston mids, `v0 = 0.04`, `kappa = 1.5`, `theta = 0.04`, `xi = 0.3`, `rho = -0.5`, three expiries, nine strikes, cash half-spread 0.15. The linear program objective is under `1e-6` and every two-sided quote is inside the spread. After projection each weight vector is nonnegative, the mass is 1 and the mean is 1, each to `1e-12`. On a grid of seven maturities that are not pillars and eleven log-moneyness nodes, `check_surface` reports no calendar violation. The objective and the pre-projection quote prices from `highs` and from `highs-ds` agree to `1e-9`.

SPX snapshot, 40 expiries, 8,973 quotes, `dx = 0.05`, `omega = 0`. HiGHS status optimal. Objective `1.752e-06`. Wall time 23.35 s. Projection L1 `1.978e-06`. `in_spread = 1`. The largest absolute residual is 0.669 half-spreads. The median is `3.39e-08` half-spreads. No expiry had its ATM total variance raised. Wing strikes used in the program: `K_min = 0.03193`, `K_max = 2.088`. The figure is `docs/figures/p3b_sanos_eta0.png`.

**Hypothesis:** the gap between this `in_spread` and the raw SVI value 0.558 on the same file is the linear model's freedom to kink at the quoted strikes. The next specification is `eta` in `{0.0625, 0.25}` with the same in-spread rule. The size of the drop, if there is one, is the quantity P3c measures.

## 2026-10-14 Library P3a: sparse quote linear program

The coefficients are the ones in section 6.2. The rows are a CSR matrix.
`dense=True` solves that same matrix as an array.

First 24 expiries, 4,398 quotes. Sparse and dense both return `ok` true,
objective 0, and an empty violation set. In a fresh process the sparse run
took 2.87 s with peak RSS 0.709 GB. The dense run took 8.56 s with peak
RSS 6.178 GB. The two shortest expiries, 278 quotes, agree the same way.
A planted ask of 0.1 at normalised strike 0.5 is short of the intrinsic
0.5 by 0.4.

All 40 expiries, 8,973 quotes. HiGHS status optimal, objective 0, `ok`
true, 0 violations. Fresh-process wall time 6.64 s, peak RSS 1.552 GB.
That is under the 2 GB bar. At 32 expiries, 6,189 quotes, the sparse peak
is 0.969 GB. The figure is `docs/figures/p3a_lp_memory.png`.

**Hypothesis:** the 15.6 GB kill was the dense allocation. These RSS
numbers are what that hypothesis predicted. The full book having objective
0 is a property of this snapshot and this program, not a claim about every
pre-open file.

## 2026-10-14 Library P3-0: Phase 3 interfaces

Sections 17 to 22 name the sparse linear program, the SANOS fit, discrete
local volatility, the rough-volatility entry point, market paths and
pricers, and the regime seed order. `MarketPaths`, `RegimeSpec`,
`SurfaceBatch`, `Pricer`, and `SupportsMarketSimulate` are stubs. No fit
was run. Rough Bergomi stays in the `roughvol` package. The hedger stays
in deep-hedging. Rough Heston is not specified.

## 2026-10-14 Library P2f: SABR, Hagan's formula, log-Euler

`beta = 1`, `nu = 0`, `alpha = 0.2`, `rho = -0.4`. Implied vol on 21 strikes
from `k = -1` to `1`, at three maturities, differs from `0.2` by `0`. The
cash price matches Black at that vol to `1e-12`. The at-the-money limit is
the closed form. For `alpha = 0.2`, `rho = -0.3`, `nu = 0.4`, `T = 1` it is
`0.20110666666666668`. A strike at log-moneyness `1e-8` is within `1e-8` of
that value.

Log-Euler against Hagan, `nu = 0.15`, `rho = -0.2`, `beta = 1`, so
`nu^2 T = 0.0225` at one year. Twelve independent nodes, `k` from `-0.12`
to `0.12`, maturities `0.5` and `1`, 8,000 paths, 32 steps, seeds `50` to
`61`. The z-rule bound at `n = 12` is max |z| below `3.93`. The grid
passes. Across seeds `10000` onward, 200 further grids: pass rate
`200/200`. Median max |z| `1.92`, 95th percentile `2.73`, largest `3.65`.
Mean of the per-grid mean z is `-0.027` (sample sd `0.29`).

With `beta = 1` and `rho = 0` the price is an average of Black prices over
the integrated variance of `alpha`. On the panel `alpha = 0.25`, `nu = 1`,
`T = 2` (`nu^2 T = 2`), that mixture at 100,000 paths and 1,000 steps
(seed 20) has implied vol `0.2733` at the money and `0.2913` at
`k = ±0.2`. The standard error is `7.6` to `7.9` bp. Hagan minus that
mixture is `+184` bp at the money and `+271` bp at `k = ±0.2`. The
20,000-path, 64-step log-Euler (seed 3) minus the mixture runs from
`-290` bp to `-187` bp. Monte Carlo minus Hagan, `-561` bp to `-380` bp,
is those two errors added. It is not the size of the formula's error.
The small-regime panel (`nu^2 T = 0.0225`, 8,000 paths, 32 steps, seed 11,
strikes sharing paths) still has a largest Monte Carlo minus Hagan gap of
`28` bp against standard errors of `26` to `90` bp.

`beta = 0.5`, `alpha = 0.8`, `nu = 0`, unit forward. The hitting
probability is `exp(-2 / 0.64) = 0.04394`. At 64 steps, 20,000 paths, seed
2, the estimate is `0.0433` (standard error `0.00144`, four standard
errors is `0.0058`). The terminal mean of `Z` is `1.0020` (standard error
`0.0057`). No absorbed path leaves 0. A level below `1e-12` is absorbed,
so the step does not leave a positive underflow. The 20,000-path test
does not see the step bias. At 200,000 paths and 64 steps, seed 2 is
`0.04551` (`z = +3.4`) and seed 3 is `0.04542` (`z = +3.2`). Seed 2 at
256 steps is `0.04420` (`z = +0.57`). Seeds 10000 to 10029 of the
20,000-path test have mean `z = +0.55`.

Synthetic book, `beta` fixed at 1, one fit per expiry, truth
`alpha = 0.22`, `rho = -0.45`, `nu = 0.35`. Four starts, 0.18 seconds.
Each of `T = 0.5`, `1` and `1.5` recovers the parameters. The largest
price residual is `4.6e-10`, on `T = 1.5`.

The same fitter on the thinned SPXW book (440 quotes, 40 expiries,
`beta = 1`, four starts, seed 0) takes 5.94 seconds. Median price RMSE
is `2.82` index points. The largest is `3.84`, at `T = 0.371`. The
expiry nearest one year (`T = 0.963`) has `alpha = 0.161`,
`rho = -0.669`, `nu = 0.932`, RMSE `1.62`. The front expiry has
`alpha = 0.161`, `rho = -0.999`, `nu = 4`, RMSE `0.37`. `nu` is on its
upper bound of 4 for the 15 expiries through `T = 0.056`, and `rho` is
on its lower bound of `-0.999` for the first three.

`check_surface` on Hagan prices, `alpha = 0.3`, `beta = 1`, `rho = -0.3`,
`nu = 1.5`, `T = 10`, `k` from `-1.5` to `0.5`. The report is not ok, and
it contains a butterfly. On the 81-point figure grid the density is
negative from `k = -1.5` to `-0.15` and from `0.325` to `0.50`, and
positive in between. The minimum is `-1.099` at `k = -0.325`. Implied vol
on that window stays between `0.662` and `2.065`. The preset
`data/presets/sabr.json` is 6,042 bytes.

## 2026-10-14 Library P2d: Dupire local variance and log-Euler

Flat Black-Scholes, sigma `0.2`, gives local volatility `0.2` with a gap of
`0` on the test grid. The tolerance asked for was `1e-4`. The expanded
denominator matches Durrleman's `g`. Outside the quoted domain the variance
is NaN.

The Vogt slice (Gatheral and Jacquier, example 3.1) has 61 cells with
`g <= 0` on `k` from `-0.2` to `1.4` (161 points). The minimum is
`-0.03286` at `k = 0.88`. The negative run is `k` from `0.65` to `1.25`.
Those cells stay negative. They are counted, not floored.

Log-Euler of that flat variance is the exact lognormal step, so the Euler
budget is 5 bp of implied vol. Twelve strikes, 6,000 paths, 4 steps, seeds
`20` to `31`. The z-rule at `n = 12` allows max |z| below `3.93`. This grid
has max |z| `1.99`, mean z `0.32`, and every node inside 2. The worst
implied vol is `0.2120` against a standard error of `0.0061` (120 bp). It
passes because the cap is the z-rule bound times that standard error, not
one standard error. Across seeds `10000` onward, 200 further grids: pass
rate `200/200`. Median max |z| `1.86`, 95th percentile `2.89`, largest
`3.71`. Mean of the per-grid mean z is `0.0097` (sample sd `0.29`).

Heston local variance by finite difference, then the same stepper. Parameters
`v0 = theta = 0.04`, `kappa = 2`, `xi = 0.15`, `rho = -0.3`. Four standard
deviations of `log Z` at `T = 1` is about `0.8`, and that grid has 3
negative cells and 1 exploding cell. The test uses `k` from `-0.5` to
`0.5` (21 points), the widest grid that stays positive, and `T` from
`0.15` to `1.2` (8 points). Queries past the edge use the edge cell.
Strikes 95, 100, 105 at `T = 1`, 200,000 paths, 48 steps, seed 4.
Implied-vol gaps (Monte Carlo minus Heston) are `-4.9`, `-2.2` and
`+0.1` bp. Standard errors are `8.5`, `7.1` and `6.2` bp. The budget is
15 bp plus three standard errors. 4,898 paths touch the edge. None hits
a non-positive cell. The same draw on `k` in `[-0.3, 0.3]` is unchanged
when `Domain` goes from `±0.4` to `±0.8`: the variance grids match and
the gaps are `-5.0`, `-2.2` and `+0.1` bp. An earlier 16,000-path run on
the narrow grid, before clipping, had gaps `-29.9`, `-18.6` and `-13.1`
bp against standard errors `28.7`, `23.6` and `20.2` bp.

SPXW mids, expiry nearest 23 calendar days (`T = 0.06427`, 6 November
2026), quoted `k` from `-0.673` to `0.104`. On `k` from `-0.55` to `0.30`
the spline of implied vol has maximum |local variance| `44.16` inside the
quotes and `3.18` outside them. Exploding cells inside the quotes: 3
(cap 5). Negative denominators inside the quotes: 20. On this window the
large spline values sit inside the quotes. Linear interpolation of the
call, differenced, reaches `6.193e6`, with 40 exploding cells and 43
negative denominators. Raw SVI on that window has maximum |local variance|
`1.059` inside the fit domain and `0.328` outside, with 0 negative
denominators and 0 exploding cells.

At `k = 0`, total variance is linear in time between expiries, so local
variance is one level on each open interval. SVI levels run from `0.003785`
to `0.03225`. The largest jump from one interval to the next is `0.01796`.
The spline's interval levels run from `-0.03046` to `0.2491`. The largest
jump is `0.2367`. A negative level is a denominator and a time slope of
opposite sign, not a clipped variance.

Heston (`v0 = theta = 0.04`, `kappa = 1.5`, `xi = 0.3`, `rho = -0.5`) at
`T = 1`: at-the-money implied vol `0.1907`, Dupire local variance at `k = 0`
is `0.03423`. No negative or exploding cell on `k` from `-0.4` to `0.4`.
The preset `data/presets/localvol.json` is 4,442 bytes.

## 2026-10-14 Library P2c: Heston cosine price, Euler, calibration

The published Fang and Oosterlee at-the-money one-year call is
`5.785155450`, nine digits. A 40-digit Lewis integral is `5.785155434376`.
The library Lewis is `5.785155434375`. COS with the term count chosen from
the interval is `5.785155412` (`2.2e-8` under the integral). Fixed
`N = 256` is `5.785155373` (`6.1e-8` under the integral). The old
`-7.7e-8` against `5.785155450` was mostly that published rounding
(`1.56e-8`). On the same parameters, `k` from `-1` to `1` and maturities
from one week to two years, the largest |Lewis minus COS| on the heatmap
is `2.2e-8`. The Feller
ratio there is `0.38`. A second violating set (ratio `0.32`) stays within
`1e-6`. With `v0 = theta` and `xi = 1e-4` the fourth cumulant is round-off,
`L` is raised until the price settles, and both prices sit within `5e-4`
of Black.

The stencil step is `f / sqrt(c2)`, not a fraction of `sqrt(c2)`. On the
Fang and Oosterlee set, `c4` is finite at one day and at one week (one-week
kurtosis `0.572`). A fourth cumulant above `80 c2^2` is kept, and `L`
still doubles until successive prices agree, because kurtosis near 1000
leaves a tail outside `L = 10`. Puts at `F = 100` and `k` in
`{-1, -0.4, 0, 0.3}`: largest |Lewis minus COS| is `2.5e-7` at `xi = 1.5`,
`kappa = 0.5`, `rho = -0.9`, `T = 0.25`, and `9.0e-8` at `T = 1`. At
`xi = 5`, `kappa = 1`, `rho = -0.7` it is `6.2e-6` at `T = 0.25` and
`5.6e-6` at `T = 1`. The corner `v0 = 6.582e-4`, `kappa = 2.1`,
`theta = 0.00999`, `xi = 3.56`, `rho = -0.54`, `T = 0.25` has kurtosis
1049 and a put gap of `1.0e-8`. A one-day slice with `v0 = 1e-4` and
`xi = 5` (kurtosis 1095) has a gap of `1.1e-5`. A `T = 1` draw with
kurtosis about 24,000 has a gap of `7.3e-6`. A `T = 3` slice with kurtosis
about 42,000 hits the 65536-term cap before two widths agree: the cosine
price warns, `converged` is false, and Lewis is the method to use there.

Euler full truncation, 8,000 paths, 64 steps, 24 independent nodes (six
strikes, four maturities). The z-rule bound at `n = 24` is max |z| below
`4.10`, fraction |z| < 2 inside `[0.792, 1]`, |mean z| below `0.71`. The
fixed seed in the test passes. Across 200 further seeds the pass rate is
`200/200`. The median max |z| is `2.23`. The mean of the per-grid mean z is
`-0.013` (sample sd `0.20`). Bias at 64 steps on 240,000 paths is `-0.035`,
`-0.023`, `-0.013` at strikes 90, 100, 110, against 8,000-path standard
errors of about `0.17`, `0.13`, `0.10`. On the Fang and Oosterlee
parameters (Feller ratio `0.38`), 64 steps, the bias at strikes 90, 100
and 110 averaged over four runs of 1,000,000 paths (seeds 0 to 3) is
`+0.0159`, `+0.0241` and `+0.0205` (sample sd `0.0067`, `0.0068`,
`0.0037`). The z-rule test subtracts that budget. The budget is smaller
than the 8,000-path standard error, and the bound on the mean z is 0.71,
so the test does not have power to detect it. It is a smoke test.

Synthetic recovery, four starts, vega-weighted prices: parameters return to
better than `1e-8` and the largest price residual is `9.6e-12`. Wall time
about `0.4` seconds. The Feller ratio is reported (`0.75` on that set) and
not enforced. `check_surface` on a Heston `ModelSurface`, five expiries by
nine strikes, returns no violations.

SPXW mids of 14 October 2026, thinned to `|k| <= 0.4` and at most 11 strikes
an expiry: 440 quotes, 40 expiries. Four starts, seed 0, vega-weighted
prices, 27.2 seconds on that run. The fit file does not store the clock.
`v0 = 0.007982`, `kappa = 13.466`, `theta = 0.03836`, `xi = 2.703`,
`rho = -0.6508`, Feller ratio `0.141`. Solver `xtol`, cost `6.912e-2`.
Root-mean-square price residual `5.48` index points, largest `20.68`, on a
spot of `7805.45`. ATM skew on the full quote set, not the thinned fit set.
At `T = 0.00114`: market `-2.965`, model `-3.050` (level `0.151` against
`0.087`). At `T = 0.00388`: market `-1.989`, model `-2.890`. At
`T = 0.0203`: market `-1.057`, model `-1.498`. At `T = 0.963`: market
`-0.282`, model `-0.239` (level `0.160` against `0.164`). The short-dated
model skew is steeper than the market. The long end is a little flat.

**Hypothesis:** that pattern is this loss and this thinning, not the only
thing Heston can do. The session 00 fit on this same file, ATM level and
skew with `kappa` capped at 20, misses the front the other way (model skew
`-1.865` against the market's `-2.965`). A fit aimed at the skew, or a cap
on `xi`, is the next specification.

Phase 1 follow-ups in the same branch. One-sided SVI quotes take the median
two-sided weight. A HiGHS failure sets `ok` to None and keeps the message.
Over seeds `10000` to `10199`, finite-difference standard-error ratios are
`1.073` (forward delta), `0.976` (sigma) and `1.020` (forward gamma), inside
`[0.9, 1.1]`, `[0.9, 1.1]` and `[0.8, 1.2]`. Heston `c2` and `c4` are pinned
to the mpmath values `0.031571152013` and `7.486782e-3`. SSVI and eSSVI are
listed as out of scope.

Library work below is not a session. Roadmap boxes stay unticked until Corey
works through the material.

## 2026-10-14 SPY snapshot replaced by SPX

The committed chain is now `data/snapshots/SPX_2026-10-14.parquet`. The file
`data/snapshots/SPY_2026-10-07.parquet` is removed. SPY options are American
and SPY pays dividends, so a European implied vol on that chain does not match
the contract. SPX options are European and cash-settled. The note is
`docs/notes/spx_not_spy.md`. The entries below this one that quote SPY counts
are that removed file. `docs/figures/p1b_surfaces.png` and `docs/00_the_smile.png`
were regenerated on the SPX chain.

Source: CBOE public delayed quotes,
`https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json`. The JSON
`timestamp` is `2026-10-14 10:01:12` with no timezone. The fetch clock matched
that stamp in UTC, so `quote_time` is that instant as UTC. The index object
has `current_price` 7805.4502, equal to `prev_day_close` 7805.4502, `close`
7805.45, `price_change` -1.7898, and `last_trade_time` `2026-10-13T16:14:59`
(no zone). At 10:01 UTC the US cash session is not open. The option rows still
have a bid and an ask.

Only the SPXW root is kept: 18,543 rows, 40 expiries, T from 0.001139 to
0.962783, strikes per expiry 134, 366.5, 1232 (min, median, max). The SPX root
is the AM-settled monthly and is dropped, including on dates where both roots
list the same calendar day. Every SPXW expiry with T > 0 inside moneyness
0.5 to 1.3 and price at least 0.03 is kept, dailies included. `rate` is NaN.
`price_source` is mid. No crossed market remains. Out-of-the-money implied
vols: 8,973 rows, from 0.0800 to 0.8571. Median half-spread 0.20 index points
(maximum 0.85); the median half-spread is 1.08% of the mid.

Put-call parity at each expiry, eight strikes nearest the crossing of C - P:
`df = -slope`, `F = intercept / df`. No dividend yield. The front expiry
(14 October, T = 0.001139) has F = 7823.9057 and df = 1.000357. The next
(15 October, T = 0.003879) has F = 7824.2157 and df = 1.001786. The last
(30 September 2027, T = 0.962783) has F = 8133.2863 and df = 0.953810, so
`-log(df) / T = 0.0491`. A discount factor above 1 on the two front expiries
is what the regression returned. **Hypothesis:** that is the bid-ask tilt on
a pre-open book, not a negative policy rate. An explicit `rate` still replaces
`df` with `exp(-r T)`.

At-the-money skew on this file: 2.965 at T = 0.001139 (about 10 hours), 1.989
at T = 0.003879 (1.42 days), 1.057 at T = 0.02032 (7.42 days), 0.282 at
T = 0.9628 (351 days). Power law on all 40 expiries:
`|ψ(T)| = 0.30511 T^(-0.33283)`, so `H = 0.167` if read as `1/2 - alpha`.
The 7 October SPY last-trade fit in the Session 00 entry was 0.30 and 0.26.
The date, the quote (last trade against mid), and the underlying all changed.
**Hypothesis:** the exponent moved. This pair of files does not say which
change did it.

Heston, joint least squares of at-the-money level and skew, three starts,
slider box with kappa up to 20 (`docs/figure_00.py`). Winning start, `xtol`,
cost 0.7454: `v0 = 0.012593`, `kappa = 8.805`, `theta = 0.044344`,
`xi = 1.740`, `rho = -0.5334`. Rho is inside [-0.99, 0.99]. RMSE 0.0128 in
the level and 0.193 in the skew. Model skew against the market: -1.865 against
-2.965 at the front, -1.785 against -1.989 at 1.42 days, -1.181 against
-1.057 at 7.42 days, -0.180 against -0.282 at 351 days. The T to 0 limit
`rho xi / (4 sqrt(v0))` is -2.068. The short end of this fit is flatter than
the market. The one-week point is steeper. The long end is flatter.

SVI on the same `MarketData`, vega weights from the half-spread: 40 slices,
wall 8.5 seconds, `in_spread = 0.558`. On `k` in [-0.5, 0.5], 24 of 40 slices
have `g < 0`. The smallest such `g` is -0.0923 at T = 0.1218 (27 November
2026). The plotted expiry is 30 October 2026, T = 0.044975, 466 strikes,
chosen because the cubic continued 0.12 past the quotes reaches implied vol
-24.95, the lowest such wing in the file. On that slice the weighted SVI has
`g` minimum 0.0578 on the same window, in-sample implied-vol RMS 0.00213, and
spline density negative on 125 of 263 points inside the quotes. Butterfly
nodes on the plotted grid: 129 of 361. The fitter's wider diagnostic grid,
half a unit of log-moneyness past the quoted strikes, is a different count:
minimum `g` is -5509.8 at T = 0.711 (30 June 2027) and -13054 at T = 0.963
(30 September 2027). Those wings are outside the figure's window.

The quote linear program on the two shortest expiries together: `ok` true,
278 quotes, total slack 0, 0.08 seconds. The first 24 expiries together:
`ok` true, 4,398 quotes, 0 violations, 9.0 seconds. The first 32, and all 40,
were killed at 15.6 GB before a status came back.

## 2026-10-14 Library P1d review: finite-difference errors, torch extra

A Monte Carlo finite difference now reports a standard error: the scatter of
the per-batch differences when the seed is shared, and the root-sum-square of
the price errors when it is not. A closed form still reports none. An omitted
bump is `1e-4` times the level on a closed form and `0.01` times the level on
a simulator. An explicit bump stays absolute.

On 40 seeds of 2,000 paths, bump 1 on a forward of 100, common random numbers
cut the variance by a factor of 412. The rough calculation for those settings
is about 300. The test requires more than 50. Two computes with the same
`SeedSequence` match, and the caller's spawn count stays 0.

The figure's right panel is that bump on 4,000 paths. The shared-path delta
stays near 0.009 from bump 0.05 to 8. The independent delta falls like `1/h`
and the independent gamma like `1/h²`. The analytic bias stays below both on
this grid. The middle panel's closed-form valley is unchanged: bump 3.2e-4,
error 5.8e-12. The left panel pools six seeds; the nine strikes share paths.

Pathwise, likelihood ratio, and autodiff refuse a model with the control
variate on. Torch is the optional extra `autodiff` and is imported when that
engine runs. The z-rule helper is what the greek tests call. On seeds 7 and
8 the short digital's standard error is 4.49 times the long one; the test
requires more than twice.

## 2026-10-14 Library P1c review: z-rule, calibration, out-of-the-money wings

The z-rule bounds come from `volsurface/mc/zrule.py` and scale with the node
count. A shared-path grid checks the maximum on one draw and the mean on 32
replications, using the observed standard deviation of those means. The
standard-error test is a calibration over 200 seeds, not a slope that holds
because `stderr = sd / sqrt(n)`. The implied-vol coverage grid is
standardised moneyness on `[-4, 4]`.

The figure prices puts for `k < 0` and calls otherwise. Two of 63 nodes have
no implied vol, both the one-week wings at `|k| = 0.10`, where the
out-of-the-money price on this seed is zero. The one-year curve sits about
0.45 volatility points below Black because the strikes share the paths.

## 2026-10-13 Library P1d: five greek engines, in review

Analytic Black-Scholes (vanilla and cash-or-nothing), central finite
differences, pathwise, likelihood ratio, and torch autodiff in float64.
Pathwise and autodiff agree to 1e-10 on the same paths, including a four-step
exact scheme. Both refuse a digital and a gamma. The likelihood ratio prices
the digital. Its forward-delta standard error on a cash-or-nothing call is
larger at 0.05 year than at 1 year, 30,000 paths.

Finite differences: from bumps 1e-2 down to 1.25e-3 the log error slopes at 2
within 0.2, and halving the bump multiplies the error by about 1/4. Common
random numbers, 40 seeds of 2,000 paths, bump 1 on a forward of 100, cut the
variance by more than a factor of five. The figure's right panel puts the
closed-form valley near a bump of 3e-4. The left panel is RMSE against paths:
pathwise and autodiff together, likelihood ratio above, a slope of -1/2 drawn
through the pathwise start.

Hypotheses, not findings: that valley moves if the greek or the noise changes;
the likelihood-ratio slope is the usual square root and six seeds do not prove
it; the variance-reduction factor depends on the bump; spot and forward
deltas agree only while the discount factor is held fixed.

## 2026-10-14 Library P1b review: checker, SVI diagnostics, two-sided quotes

`SVISurface` accepts list inputs, which is what `in_spread` passes. The floor
penalty in the SVI residual is always one row, so the Jacobian length stays
put. Each fitted slice reports `min_g`, Lee's bound, calendar order, and its
own `in_spread`. Spreads weight the fit when they exist. On the saved SPY
chain they do not, and 14 of 24 slices still have `g < 0` on `k` in
`[-0.5, 0.5]`. The figure says so, and shows the short-dated slice where `g`
reaches about -0.18.

The quote linear program carries a tail mass and moment instead of a strike
at 1.5 times the last quote. A normalised call of 0.8 at the money is
feasible. A call of 1 is not, up to a cap of `1e6` on the tail strike. An
empty bid-ask file returns `ok is None` and `n_checked == 0`. A flat Black
surface passes the default butterfly tolerance because its derivatives are
analytic. The mixin second-difference step is `1e-3`, with a `1e-8` floor
when the checker is still differencing.

No g-penalty and no SSVI fit. Both would change the 14 of 24 count instead
of reporting it.

## 2026-10-14 Library review: curves, seeds, and the z-rule

`ArbReport.ok` is None when the checker evaluated nothing, and `n_checked`
counts what it did evaluate. `as_seed_sequence` copies a `SeedSequence` so a
second spawn with common random numbers still draws child 0. The discount
curve is anchored at `df(0) = 1` and both curves extend the last carry past
the final pillar. The z-rule bounds scale with the node count; the helper is
`volsurface/mc/zrule.py`. Heston `c2` and `c4` are checked against a second
stencil, not against a copied closed form.

`Spot` stays a one-for-one forward bump with `df` fixed. That is the reading
in DESIGN section 8, and the greek tests lock `Spot == Forward`.

## 2026-10-13 Library P1c: geometric Brownian motion by Monte Carlo, in review

Exact and Euler stepping of the unit-mean martingale `Z`. One seed covers a
strike by expiry grid. The standard error is kept on `PriceResult`, off the
array `price` returns. Antithetic pairs are the second half of the batch. The
control is `Z` at expiry, mean 1.

Against Black, 200 nodes with independent seeds and 8,000 paths pass the
z-rule (max |z| below 4.5, the fraction inside 2 between 0.90 and 0.99, mean
z inside 0.5). From 10^3 to 10^6 paths the log standard error slopes at -0.5
within 0.05. One-step Euler at volatility 1 matches the normal formula and
misses Black, on 100,000 paths. Implied vol round trip: 85 of 124 nodes on the
closed-form grid have normalised vega at or below 1e-8 and are excluded. The
figure, 8,000 shared paths, has no implied vol at 6 of 63 nodes, all of them
one-week wings.

Hypotheses, not findings: the smooth shift of the one-year curve is the shared
seed; empty wings fill from more paths, not from a smoother interpolant; Euler
bias at volatility 0.2 hides inside a few thousand paths; the control helps
most at the money.

## 2026-10-13 Library P1b: surfaces and the checker, in review

Three interpolators of the normalised call: linear in the strike, a natural
cubic of implied volatility, and raw SVI with total variance linear in
expiry. The surface checker is pointwise (bounds, call spread, butterfly,
calendar). The quote checker is the feasibility linear program, not a fit.

On the saved SPY chain the figure uses the 30 September 2027 expiry. The
spline passes through the last-trade vols and still goes negative in density
between those strikes; past the lowest strike the same cubic crosses through
zero volatility. Raw SVI on that slice stays positive in density. Across all
24 expiries, 14 fitted slices have Durrleman's `g` negative somewhere on
`k` in `[-0.5, 0.5]`. The file has no bid or ask, so the quote checker drops
every row. `in_spread` on the SVI fit is NaN for that reason.

Hypotheses, not findings: flat extrapolation of the linear call is the SANOS
linear model; the SPY marks mix stale lasts, American premium, and real
convexity breaks; short-dated negative `g` is the least squares, not a
tradable wing; equal slack weights can book a calendar gap as a quote slack.

## 2026-10-13 Library P1a: interfaces, in review

`docs/DESIGN.md` is now the source of truth for coordinates (normalised call,
`k = log(K/F)`), the `Model` / `Surface` / `GreeksEngine` protocols, and the
pull-request stack. Black-Scholes and Heston implement it. `price` still
returns an array; a standard error lives on `PriceResult`. `state=None` means
the initial state, which is empty for Black-Scholes and the variance for Heston.

The Heston first cumulant matches `-1/2` times the integrated expected variance.
The second and fourth come from a fixed stencil on the log characteristic
function. The committed SPY snapshot still has no bid or ask.

Hypotheses, not findings: the SANOS HTML's normalised call matches the PDF;
marking SPY American and physical is the right contract label; Heston
cumulants from the stencil are accurate enough to place a cosine grid, which
is all Phase 1 asks of them.

## 2026-10-08 Session 00: the smile is the model's fingerprint

Built the package core (Black-Scholes, Heston, the `Model` interface, yfinance and IBKR
chain loaders) and the first app: a Heston implied vol surface with sliders, a Black-Scholes
switch and SPY market points on the same $(k, T)$ grid.

- Heston pricing matches Fang and Oosterlee (2008) to $2 \times 10^{-8}$ (5.785155434
  against 5.785155450 at $T = 1$) and converges to Black-Scholes as $\xi \to 0$.
- Snapshot `data/snapshots/SPY_2026-10-07.parquet`: 3,695 options, 24 expiries from 1 to 358
  days. Fetched before the US open, so prices are last trades of the 7 October session,
  priced against the close of 777.22. Rate 4.11% from the 13-week T-bill.
- SPY at-the-money skew: 1.56 at 1 day, 0.74 at 1 week, 0.28 at 1 year. A power law fits it
  well: $|\psi(T)| \approx 0.30\, T^{-0.26}$, so $H \approx 0.24$ if read through rough
  volatility.
- One Heston fit to all expiries' ATM vols and skews (scipy `least_squares`, three starts,
  box as in the app sliders with $\kappa \le 20$): $v_0 = 0.0098$, $\kappa = 4.26$,
  $\theta = 0.0373$, $\xi = 0.388$, $\rho = -0.99$ (at the bound). Skew 0.97 at 1 day
  against 1.56; 0.19 at 1 year against 0.28. Heston's skew is flat for $T \ll 1/\kappa$ and
  decays like $1/T$ beyond it; the market's does neither.
- Correction to the session brief: Heston can make a one-day skew steeper than the market's
  (6.3 at $\xi = 2$, $\rho = -0.99$). What it cannot do is match the short and long ends with
  one parameter set.

Open questions: why is the exponent 0.26 here and about 0.4 in the literature (window
choice, SPY against SPX, last trades against mids)? Does a mid-quote snapshot move it?
