# Log

Dated entries, newest first. What was learned, what surprised, open questions.

**Current state:** SPX snapshot of 14 October 2026 is the committed chain. Session 00 reads that chain. **Next:** Session 01, SVI on the SPX snapshot.

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
