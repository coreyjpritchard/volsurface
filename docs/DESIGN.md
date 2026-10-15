# volsurface design

Source of truth for interfaces, coordinates, and the way pull requests stack.
Code changes cite the section they implement. A change to an interface in this
file is a `design/<topic>` pull request of its own, reviewed before the code
that depends on it.

This document covers Phase 1 in sections 1 to 13, the Heston, local-volatility
and SABR additions in sections 14 to 16, and Phase 3 in sections 17 to 22.
The deep-hedging adapter is not part of this repository.

The normalised-call coordinate below follows the plan's reading of the SANOS
arXiv HTML (2601.11209). Equation numbers from the PDF are not in this file.
**Hypothesis:** the HTML summary is faithful. Nothing here should be quoted
as a paper result until the PDF has been read against it.

## 1. What a price is

Every model prices in the forward. The inputs are the forward `F` to the
option's expiry, the strike `K`, the year fraction `T`, and the discount
factor `df` to that expiry. Spot, the interest rate, and the dividend yield
appear only through

```
F = S * exp((r - q) T),    df = exp(-r T).
```

A model never sees `S`, `r`, or `q`. One call prices one maturity, so a
Monte Carlo model can reuse a single set of paths for every strike.

The returned price is the cash price: `df` times the expected payoff under
the forward measure. `price` always returns an `ndarray`. It does not carry
a standard error. Standard errors travel on `PriceResult`, from
`price_with_error` (section 4).

## 2. Coordinates

`F(T)` is today's forward curve for maturity `T`. `F_T` is the random
terminal value of that forward. Under proportional dividends `F_T = S_T`
at the expiry. `Z_T = F_T / F(T)` is a unit-mean martingale in the forward
measure. SANOS writes its quotes in that coordinate. We use the same one
on every surface.

The log-moneyness is `k = log(K / F(T))`, so the normalised strike is
`x = K / F(T) = exp(k)`.

The normalised call is the cash call divided by `df * F`:

```
C(T, k) = cash_call(T, K) / (df(T) * F(T)).
```

Black's formula then collapses to `C = N(d1) - exp(k) N(d2)`, with total
variance `w = sigma^2 T` and `d1 = -k / sqrt(w) + sqrt(w) / 2`,
`d2 = d1 - sqrt(w)`.

Evaluation shape, used everywhere a surface is sampled: `T` has shape
`(m,)`, `k` has shape `(n,)`, and the result has shape `(m, n)`. A scalar
is treated as length 1. There is no silent extrapolation. Values outside
the quoted domain are still returned, so plots can show the wing, and
`extrapolated(T, k)` is a boolean array of the same shape, true where the
value is not an interpolation of quotes.

## 3. Types

Defined in `volsurface/core/types.py`.

`State` is a mapping from a string name to an array, batched over paths
when a path is in hand. Black-Scholes has an empty state. Heston has
`{"v": v0}`. `state=None` on `price` means the model's initial state, so
existing callers keep working.

`Paths` holds a simulation:

- `times`, shape `(n + 1,)`, including `t = 0`.
- `Z`, shape `(m, n + 1,)`, the unit-mean martingale `S / F`.
- `state`, a dict of arrays, each shaped `(m, n + 1)` or
  `(m, n + 1, d)`.

`PriceResult` holds `value` (ndarray), `stderr` (ndarray or None when the
price is not a Monte Carlo estimate), `n_evals`, and `wall_time` in seconds.

`ParamSpec` is one parameter: `name`, `lower`, `upper`, `default`, and
`transform`. The transform is `"log"` (positive parameter), `"logit"`
(a bounded interval), or `"identity"`.

`Quote` is one option: `T`, `K`, `kind` (`"call"` or `"put"`), `bid`,
`ask`, `exercise` (`"european"` or `"american"`), `settlement`
(`"cash"` or `"physical"`), `volume`.

`MarketData` is a snapshot:

- `asof`: timezone-aware datetime.
- `quotes`: a dataframe. Required columns are the `Quote` fields. Loaders
  may add `price`, `iv`, `k`, `F`, `df`, `expiry`, `price_source`.
- `F(T)` and `df(T)`: the forward and the discount factor. They accept a
  scalar or an array and return the same shape.
- `F_err(T)`: an estimate of the uncertainty in `F`, or None.
- `day_count`: `"ACT/365"` for chains built by this package. Time is
  measured from the quote timestamp to 16:00 New York on the expiry date,
  divided by 365.

`Curves` is the pair `F`, `df` that a surface holds. It is the part of
`MarketData` the surface needs in order to turn a normalised call back
into a cash price.

`Domain` is the quoted rectangle `t_min`, `t_max`, `k_min`, `k_max`.

`FitResult` is what a fitter returns: `surface`, `residuals`, `in_spread`
(a fraction, or NaN when the quotes have no two-sided market),
`wall_time`, `solver_status`, and `slice_report` (a dataframe with one row
per fitted expiry, or None). The raw SVI fitter fills `min_g`, `lee_ok`,
`calendar_ok`, a per-slice `in_spread`, and `n_one_sided` (quotes on that
expiry with no two-sided market).

`ArbTol` is `price` (default `1e-9`, in normalised-call units) and
`spreads` (a half-spread in the same units, or None).

`ArbReport` is the checker's result:

- `violations`: a dataframe with columns `kind`, `T`, `k`, `magnitude`,
  `magnitude_in_spreads`. One row per violated node. Empty if nothing is
  flagged.
- `masks`: boolean arrays on the evaluation grid, keyed `"call_spread"`,
  `"butterfly"`, `"calendar"`, `"bounds"`. Quote checks use `"quote_bid"`,
  `"quote_ask"`, `"calendar"` and may leave the grid masks empty.
- `min_density`: the smallest risk-neutral density of `log(Z)` on the
  grid, or NaN if the grid has no density.
- `n_checked`: how many nodes entered the test. For a surface this is the
  grid size `m * n`. For quotes it is the number of quote rows that entered
  the linear program.
- `ok`: True when `violations` is empty and `n_checked > 0`, False when a
  violation was recorded, None when `n_checked == 0`. An empty quote file
  is None. It is not a pass.
- `note`: plain text. For `check_surface` it says that the test is
  pointwise on the supplied grid and says nothing between the nodes. When
  the butterfly comparison uses the finite-difference floor of section 6.1,
  the note says so.

`Vanilla(K, kind)` is a European call or put on the forward.
`Digital(K, kind)` is cash-or-nothing: it pays 1 when the forward finishes
in the money, and 0 otherwise. `Contract` pairs a payoff with an expiry `T`.

`Spot`, `Forward`, and `Param(name)` are risk factors. `Forward` is
`d/dF` with `df` held fixed. `Spot` is the same bump under the reading
that `F` moves one-for-one with spot (`r = q`, or a futures-style bump).
The two differ only when a later adapter supplies a rate. `Param("sigma")`
is Black-Scholes vega. `Param("v0")` would be a Heston vega. A number
called "vega" without a factor name is not a result this library returns.

`GreeksResult` holds `value` with shape `(n_contracts, n_factors)`,
`stderr` of that shape or None, `n_evals`, `wall_time`, and `method`
(a short string such as `"pathwise"`).

## 4. Protocols

Defined in `volsurface/core/protocols.py`. They are runtime-checkable, so
a plugin does not have to subclass anything. `capabilities(model)` returns
the set of names whose protocols the object satisfies. The dashboard should
use that helper rather than a hand-written flag.

```
class Model(Protocol):
    def price(self, F, K, T, df=1.0, kind="call", *, state=None) -> ndarray
    def initial_state(self) -> State

class SupportsSimulate(Protocol):
    def simulate(self, n_paths, times, *, seed, state0=None) -> Paths

class SupportsCharFunc(Protocol):
    def char_func(self, u, T, *, state=None) -> ndarray
        # E[exp(i u X_T)], X_T = log(F_T / F(T)). No extra drift term.
    def cumulants(self, T, *, state=None) -> tuple[float, float, float]
        # c1, c2, c4 of X_T, for the range of a cosine expansion.

class Parametric(Protocol):
    def to_unconstrained(self) -> ndarray
    def from_unconstrained(cls, x) -> Self          # classmethod
    def param_spec(cls) -> list[ParamSpec]          # classmethod

class Surface(Protocol):
    market: Curves
    domain: Domain
    def call(self, T, k) -> ndarray                 # the only required method

class Fitter(Protocol):
    def fit(self, data: MarketData) -> FitResult

class GreeksEngine(Protocol):
    def supports(self, model, contract, order: int) -> bool
    def compute(self, model, contracts, wrt, *, order=1,
                state=None, n_paths=None, seed=None) -> GreeksResult
```

`price_with_error(model, F, K, T, df=1.0, kind="call", *, state=None, n_paths=None, seed=None)`
returns a `PriceResult`. A closed-form model reports `stderr=None` and
`n_evals=1`. A Monte Carlo model reports the standard error of the
estimator it actually used, including antithetic pairing and a control
variate when those are switched on.

Black-Scholes and Heston implement `Model`, `Parametric`, and
`SupportsCharFunc`. Heston's characteristic function is the trap-free
Gatheral form already in `volsurface/heston.py`. Its first cumulant is the
closed form `-1/2 * integral E[v_t] dt`. The second and fourth are
derivatives of `log phi` at the origin, read from a fixed polynomial
stencil (see the method docstring). Heston also implements
`SupportsSimulate`. The scheme is Euler full truncation on an arbitrary
time grid. Draw order is in the class docstring: one forward normal and
one variance normal per step. `price` stays the Lewis integral.
`cos_price` is the Fang and Oosterlee cosine expansion of section 14.
The function `heston.simulate` keeps its old signature; the protocol
method is the one on the class.

`volsurface/plugins.py` holds an in-tree registry, `@register("name")`,
and resolves `volsurface.models` entry points the same way. A plugin in
another repository implements the protocols and does not import volsurface.

## 5. Surfaces

Implemented in `volsurface/surface/`. `SurfaceMixin` derives the quantities
below from `call`. A surface overrides a method when it has an analytic
form, so a comparison of interpolators is not a comparison of finite
differences.

Let `C` be the normalised call.

- `iv(T, k)`: Black implied volatility of `C`. NaN where the price has no
  volatility information.
- `total_variance(T, k)`: `iv^2 * T`.
- `dcall_dk`, `d2call_dk2`, `dcall_dT`: derivatives. The mixin uses a
  central step of `1e-4` for a first derivative and `1e-3` for a second
  derivative. A second difference at `1e-4` sits in round-off: the noise
  on a price of order 1 is about `1e-8`, larger than `ArbTol.price`.
  Overrides replace the finite difference. `ModelSurface` around
  Black-Scholes overrides them, so a flat Black surface is judged by the
  analytic derivative.
- `density(T, k)`: the risk-neutral density of `k_log = log(Z_T)`,

```
phi(k) = exp(-k) * (d²C/dk² - dC/dk).
```

  Equivalently `phi(k) = exp(k) * d²C/dx²` with `x = exp(k)`. For a total
  variance `w(k)` this is Durrleman's formula

```
g(k) = (1 - k w' / (2 w))^2 - (w')^2 / 4 * (1/w + 1/4) + w'' / 2,
phi(k) = g(k) / sqrt(2 pi w) * exp(-d2(k)^2 / 2),
```

  with `d2 = -k / sqrt(w) - sqrt(w) / 2`. Flat Black-Scholes has `g = 1`
  and `phi` equal to the normal density of `N(-w/2, w)`. A negative value
  is a butterfly arbitrage.

- `local_var(T, k)`: Dupire local variance in total-variance form,
  `(dw/dT) / g(k)`. This is a variance, not a volatility, so a negative
  number stays negative and visible. It is NaN where `g` is zero or `w`
  is not positive.

`ModelSurface(model, market, domain)` samples `model.price` and divides by
`df * F`.

### 5.1 Linear

`LinearSurface` interpolates the normalised call linearly in the normalised
strike `x = exp(k)` at each quoted expiry, and linearly in `T` at fixed `k`
between expiries. Outside the quoted strikes the call is held flat at the
edge value. Outside the quoted expiries it is held flat in `T`. Both
regions are marked extrapolated.

Between nodes the second derivative is zero, so a grid that misses the
nodes cannot see a butterfly. At a quoted node the surface reports a second
derivative that makes the checker's strike convexity equal to the jump in
slope `dC/dx` across that node (right slope minus left slope). The jump is
the discrete butterfly: convex call prices have an increasing slope, and
the violation magnitude is how far the slope falls.

This is the `eta = 0` limit described for SANOS: linear interpolation of
the call in the strike. **Hypothesis:** the paper's linear model is this
object, including the flat extrapolation past the last strike. The PDF
should confirm the extrapolation.

### 5.2 Cubic spline

`SplineSurface` fits a natural cubic spline of implied volatility against
`k` at each expiry, and interpolates total variance linearly in `T` at
fixed `k` between expiries. `iv` returns the spline, which is allowed to
be negative outside the quotes. `call` uses Black's formula where the
spline volatility is positive and the intrinsic value `(1 - exp(k))+`
otherwise. Derivatives, density, and local variance use the Durrleman
formulas where total variance is positive.

The cubic extrapolates as a cubic polynomial. That is the wing the demo
is about: the implied volatility oscillates or crosses zero, and the
density turns negative. Those points are marked extrapolated. The checker
still evaluates them, because the wing is the object under study.

### 5.3 Raw SVI

One expiry has total variance

```
w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2)),
```

with `b > 0`, `|rho| < 1`, `sigma > 0`, and `a + b * sigma * sqrt(1 - rho^2) >= 0`
so that `w` stays non-negative. `SVISurface` holds one parameter set per
quoted expiry and interpolates `w` linearly in `T` at fixed `k`. All
derivatives, `g`, the density, and local variance are analytic.

The fitter minimises squared error in total variance, in the unconstrained
coordinates of `param_spec`. It is a least-squares fit, not a guarantee of
no arbitrage.

The arbitrageable slice used in tests is Example 3.1 of Gatheral and
Jacquier (2014), the parameters Axel Vogt posted: `(a, b, m, rho, sigma) = (-0.0410, 0.1331, 0.3586, 0.3060, 0.4153)` at `T = 1`. The tuple order is
the one printed in the paper. Read that way, the SVI-JW parameters match
the values in their Example 5.1 (`v = 0.01742625` and the rest). Durrleman's
`g` is negative somewhere on that slice.

## 6. Arbitrage checker

Implemented in `volsurface/arbitrage/`.

### 6.1 `check_surface(surface, T, k, *, tol=ArbTol())`

Pointwise on the grid `T` (shape `(m,)`) and `k` (shape `(n,)`). At each
node, with `x = exp(k)` and `C` the normalised call:

| kind | quantity | violation when |
|---|---|---|
| bounds | `C` against `((1 - x)+ , 1)` | the price leaves the interval by more than `tol.price` |
| call_spread | slope `s = dC/dx = exp(-k) * dC/dk` | `s` leaves `[-1, 0]` by more than `tol.price` |
| butterfly | convexity `gamma = d²C/dx² = exp(-2k) * (d²C/dk² - dC/dk)` | `gamma < -tol.price`, or `gamma < -1e-8` when `d2call_dk2` is the mixin's finite difference |
| calendar | `dC/dT` at fixed `k` | `dC/dT < -tol.price` |

`magnitude` is the amount by which the quantity leaves the allowed region,
in normalised-call units (a slope gap, for the linear surface's discrete
butterfly). `magnitude_in_spreads` is `magnitude / tol.spreads` when a
spread is supplied, otherwise NaN.

The derivatives are the surface's own methods, so SVI is judged by
Durrleman's `g` and not by a finite difference. `min_density` is the
minimum of `density(T, k)` on the grid.

When `type(surface).d2call_dk2` is `SurfaceMixin.d2call_dk2`, the butterfly
comparison uses `max(tol.price, 1e-8)`. That floor is the finite-difference
noise left after the `1e-3` step, and the report's note records it. An
analytic second derivative, including `ModelSurface` around Black-Scholes,
is judged at `tol.price`. A flat Black surface therefore passes
`check_surface` at the default tolerance.

The report's `note` states that passing on the nodes does not constrain
the surface between the nodes. `n_checked` is `T.size * k.size`. An empty
grid returns `ok is None` and `n_checked == 0`.

### 6.2 `check_quotes(data, *, side="bidask")`

An LP, solved with `scipy.optimize.linprog` method `"highs"`. This is the
feasibility form of the SANOS `eta = 0` problem, not the full fit (that fit
is a later phase).

Puts are turned into calls by put-call parity, `cash_call = cash_put + df * (F - K)`,
and then divided by `df * F`. For each expiry the unknowns are probabilities
`q_j >= 0` on `{0}` union the quoted normalised strikes, plus a tail pair
`(Q, M)`. `Q >= 0` is the mass beyond the largest quoted strike `x_max`, and
`M` is that tail's contribution to the mean, with `M >= x_max * Q`. The
constraints are `sum q_j + Q = 1` and `sum q_j * x_j + M = 1`. The model
call at a quoted strike `x <= x_max` is `sum q_j * (x_j - x)+ + (M - x * Q)`.
The tail term is the call of an atom at `M / Q` when `Q > 0`. It is valid
only for strikes at or below `x_max`, which is why calendar rows (below) are
written only at knots that both expiries actually quote.

`side="bidask"` requires the model call to lie between the normalised bid
and ask. `side="mid"` uses the quote's working price as both bid and ask.
Slack variables absorb violations. The objective minimises the sum of
slacks. A quote whose slack exceeds `tol.price` is a row in `violations`,
kind `"quote_bid"` or `"quote_ask"`, magnitude equal to the slack.

Calendar: at a shared moneyness grid, the longer expiry's model call must
dominate the shorter one's, again up to a slack. A positive slack is kind
`"calendar"`.

Quotes with a missing bid or ask are dropped from a `"bidask"` check. The
note records how many were dropped. The committed SPX file is two-sided
mids on SPXW (see section 10), so a bid-ask check has quotes to constrain.
Checked alone, the two shortest expiries are feasible together: 278 quotes,
objective 0, `ok` true. The first 24 expiries together, 4,398 quotes, are
also feasible, in 9.0 seconds. The program on the first 32, and on all 40,
was killed at a 15.6 GB memory limit before HiGHS returned. That is a limit
of this run, not a measured arbitrage. **Hypothesis:** a slack on this book
would be a failure of the quoted mids to sit on one martingale, or a stale
pre-open quote. It would not be an American early-exercise premium.

`ok` is True when every slack is within `tol.price` and at least one quote
entered the program. `n_checked` is that quote count. When every quote was
dropped, `ok` is None and `n_checked` is 0. A solver that returns no point sets `ok` to None and copies the solver
message into `note`. That is not a recorded violation and not a pass.

## 7. Monte Carlo and the z-rule

Implemented in `volsurface/mc/` and `volsurface/models/gbm.py`.

`GBM` is geometric Brownian motion with constant volatility `sigma`.
Fields `n_paths`, `n_steps`, `seed`, `antithetic`, `control_variate`, and
`scheme` (`"exact"` or `"euler"`) are part of the model, so the same fields
produce the same prices.

- `"exact"` draws one normal increment per time interval and updates the
  log forward. For geometric Brownian motion that transition is exact.
- `"euler"` updates `Z` with Euler-Maruyama, `Z + sigma Z dW`. The
  discretisation is biased. The bias does not shrink when the path count
  grows, only when the step count grows.

`simulate(n_paths, times, *, seed, state0=None)` returns `Paths`. `times`
includes 0. One `SeedSequence` child is spawned for the whole run, not one
per strike or expiry. Draw order, exact scheme: for each time step after
0, a vector of `n_paths` standard normals, path index fastest. If
`antithetic` is on, the paired path uses the negated normal and is stored
as the second half of the batch (the batch size doubles). The control
variate, when on, is `Z_T` itself, whose mean is 1. The coefficient is the
sample regression coefficient of the payoff on `Z_T`. The reported standard
error is the sample standard deviation of the controlled, antithetic-averaged
payoffs, divided by `sqrt(n_batches)`.

`price` prices every strike at one maturity from one simulation.
`price_grid(model, F, strikes, maturities, df, kind, *, n_paths, seed)`
simulates once on the union of the maturities and prices the whole grid
from those paths. That is the grid whose implied volatility the demo plots.

`implied_vol` in `volsurface/bs.py` returns NaN when the safeguarded Newton
residual misses `tol`, as well as outside the arbitrage bounds. A finite
implied vol is one the formula reprices. Section 12 records why that edit
sits in the function module.

Inversion, on a grid of cash prices (Monte Carlo or closed form):

- Price to implied vol to price recovers the price to a relative error of
  `1e-12` wherever the implied vol is finite. Once a failed inversion is
  NaN, this holds by construction on those nodes: the price is rebuilt from
  the volatility that was just solved. It does not test the solver.
- The informative check is coverage. Implied vol to price to implied vol
  recovers the vol to `1e-8` where the normalised vega `n(d1) * sqrt(T)`
  exceeds `1e-8`. The test reports the fraction of the grid that clears
  that threshold and inverts, on a grid built in standardised moneyness
  `k / (sigma sqrt(T))`. It does not lock an absolute count of excluded
  nodes.

Put-call parity for Monte Carlo is pathwise, on the same paths:
`call_payoff - put_payoff = df * F * (Z - K/F)`, to `1e-12`. The averaged
prices satisfy parity with the averaged `Z`, which equals `df (F - K)` only
in expectation.

### The z-rule

A test that demands `|price - reference| < 3 * stderr` at every node of a
large grid fails by chance. For about 200 independent normal nodes the
probability that the maximum exceeds 3 is a few tens of percent. Fixed
caps have the same problem at the other end: `|mean z| < 0.5` is about two
standard errors when `n = 20`, and a fraction cap of 0.99 rejects a small
sample in which every `|z|` is below 2.

`volsurface/mc/zrule.py` is the one implementation. For `n` nodes, with
`z = (estimate - reference) / stderr`, and `p = 2 Phi(2) - 1`:

1. `max |z| < Phi^{-1}(1 - 0.001 / (2 n))`. One normal exceeds the bound
   with probability `0.001 / n`, so a union bound keeps the chance of any
   exceedance at or below 0.1 percent. The test states `n`.
2. The fraction of nodes with `|z| < 2` lies between the 0.1 percent and
   99.9 percent quantiles of `Binomial(n, p)`, divided by `n`.
3. `|mean z| < 3.5 / sqrt(n_eff)`. For independent nodes `n_eff = n`.

Nodes that share paths are one draw. The fraction band does not apply, and
`n_eff` is not `n`. On that grid the test keeps the max bound (it is valid
as a union bound, and dependence only makes it looser) and checks the mean
on `R` independent replications of the whole grid:
`|mean(m)| < 3.5 * sd(m) / sqrt(R)`, where `m` is the per-replication mean
of `z` and `sd` uses `ddof=1`. The test states `R` and the observed `sd`.

The standard-error check is a calibration, not a slope fitted to
`stderr = sd / sqrt(n)`, which holds by construction. Over `R` seeds the
sample standard deviation of the estimates, divided by the mean reported
standard error, lies in `[0.9, 1.1]`. The test states `R`.

## 8. Greeks

Implemented in `volsurface/greeks/`. Five engines, each a `GreeksEngine`.

The analytic engine supports Black-Scholes vanillas and cash-or-nothing
digitals, orders 1 and 2, for `Spot`, `Forward`, and `Param("sigma")`.
With `df` held fixed and `Spot` read as a one-for-one forward bump, spot
delta equals forward delta. Cash-or-nothing call price is `df * N(d2)`.

The finite-difference engine reprices with a central bump. It supports
any model whose `price` covers the contract, orders 1 and 2. `crn=True`
reuses the seed (common random numbers). `crn=False` uses an independent
seed on the bumped price. When `bump` is omitted the step is `1e-4` times
the level on a closed form and `0.01` times the level on a simulator. The
level is `F` for `Spot` and `Forward`, and the parameter's current value
otherwise. An explicit `bump` is an absolute step, which is what the order
test passes.

A simulator reports a standard error. With common random numbers it is the
sample standard deviation of the per-batch payoff differences, antithetic
pairs averaged first and a control variate applied to each bumped payoff
when the model has one, divided by `sqrt(n_batches)`. A forward bump
simulates once. A parameter bump resimulates each stencil node on that
seed. Without common random numbers the standard error is the
root-sum-square of the bumped prices' standard errors: order 1 divides by
`2h`, and order 2 divides `sqrt(se_up^2 + 4 se_mid^2 + se_down^2)` by
`h^2`. A closed form reports `stderr=None`.

The order test: for bump sizes above the round-off floor, halving the bump
cuts the error in a first derivative by about 4, and the slope of log error
against log bump is `2` within `0.2`. The common-random-number test states
the variance-reduction factor it requires and the theoretical factor for
those settings. The figure shows the closed-form U-curve and, at a fixed
path count, Monte Carlo root-mean-square error against the bump for delta
and gamma, with common random numbers on and off. The analytic truncation
`|h^2 f''' / 6|` is drawn on the delta, and `|h^2 f'''' / 12|` on the
gamma. Neither curve is a pass/fail threshold.

The pathwise engine differentiates the discounted payoff with the normal
increments held fixed. For a vanilla call on geometric Brownian motion the
forward delta on a path is `df * Z * 1_{F Z > K}`, and the vega follows by
differentiating `Z` with respect to `sigma`. The estimator is unbiased for
Lipschitz payoffs. It is zero almost surely for a digital and for gamma,
because those derivatives hit a kink or a corner. `supports` returns false
for digitals and for order 2. Returning a number that is silently zero
would be a wrong greek. It also returns false, and `compute` raises, when
`control_variate` is true: the engine differentiates the paths, not the
controlled price.

The likelihood-ratio engine writes the derivative onto the lognormal
density of the terminal forward and multiplies the undifferentiated payoff
by the score. It supports digitals and order 1 and 2 for geometric
Brownian motion. The score for the forward is `u / (F * w)` with
`w = sigma^2 T` and `u = log Z + w / 2`, so the weight divides by
`sigma^2 T`. A vanilla price shrinks with `T` along with that weight. A
cash-or-nothing digital pays 1 and does not, which is why the short-dated
comparison uses a digital. The test requires the short-dated standard error
to exceed twice the long-dated one at equal path count. The theoretical
ratio for an at-the-money digital delta is several times that.

The autodiff engine differentiates the same simulation the pathwise engine
differentiates, in torch, dtype float64. Torch is the optional extra
`autodiff`. Importing the greeks package does not import torch; the engine
imports it when `compute` runs. The estimator matches the pathwise value to
`1e-10` and does not repair gamma or digitals. `supports` agrees with the
pathwise engine, including the control-variate refusal. The likelihood-ratio
engine refuses that flag for the same reason.

`compute` accepts a sequence of contracts and a sequence of risk factors.
`value[i, j]` is contract `i` against factor `j`.

Cost in the demo is `n_evals` (paths times repricings) against RMSE versus
the analytic greek, on log-log axes.

## 9. Random numbers

Every draw goes through `np.random.default_rng` on an `int` or a
`numpy.random.SeedSequence`. `volsurface/mc/rng.py` turns an int into a
`SeedSequence`. A `SeedSequence` argument is copied (`entropy`, `spawn_key`,
`pool_size`) before use, because `spawn` advances `n_children_spawned` on
the object it is called on. Two repricings with common random numbers and
the same `SeedSequence` therefore both draw child 0. `spawn` produces one
child per independent stream.

A grid of strikes and expiries shares one child for the whole model run.
Draw order is part of each model's docstring. Two implementations produce
the same paths for the same seed only when they consume draws in the same
order. Tests that need identical paths assert that order. Tests that need
accuracy use the z-rule instead.

## 10. Data

`market_data_from_snapshot` builds a `MarketData` from a tidy chain
(`volsurface/chains.py`). Forwards and discount factors are the ones
`market_implied_vols` already infers, at the pillars. Between pillars `F`
is linear in calendar time and `df` is log-linear. If the first pillar is
after 0, `log df` is anchored at `(0, 0)`, so `df(0) = 1`. Past the last
pillar the last carry is extended: `log(F_n / F_{n-1}) / (T_n - T_{n-1})`
for the forward, and the slope of `log df` for the discount factor. The
forward is extended backward with the first carry as well. One pillar has
no second point, so both curves hold their level past that pillar (the
discount anchor still fills `[0, T_1]` when `T_1 > 0`). `F_err` is the
standard deviation of the near-the-money parity forwards, interpolated
flat outside the pillars. `day_count` is `"ACT/365"`.

Exercise and settlement are not in the parquet file. The loader fills
`"american"` and `"physical"` when the underlying is `SPY`, and
`"european"` and `"cash"` when it is `SPX` or `XSP`. Any other underlying
requires the caller to pass them. **Hypothesis:** that mapping matches the
contracts. It is an assumption of the loader, not a field we observed.

The committed file is `data/snapshots/SPX_2026-10-14.parquet`. It is SPXW
only, PM-settled, two-sided mids from CBOE's delayed quotes. Forwards and
discount factors come from put-call parity at each expiry. The note
`docs/notes/spx_not_spy.md` says why this replaced the SPY last-trade file.
Surface demos use this file. SANOS fits, later, want European quotes, which
is what this file is.

## 11. Tests that Phase 1 must keep

Existing tests stay green: Black-Scholes parity and implied-vol round trip,
the Fang and Oosterlee Heston references, the chain schema, the committed
snapshot. New tests are those named in sections 5 to 8. A test marked
`slow` is allowed to be heavy. `pytest -m "not slow"` is the bar for a
pull request, together with `ruff check`.

Statistical tests state the tolerance in standard errors and the sample
size, or they use the z-rule and state `n`.

## 12. Pull requests

Stack, each branch based on the previous one:

| logical name | branch slug | depends on | owns |
|---|---|---|---|
| P1a | `p1a-design-foundations` | `main` | `docs/DESIGN.md`, `volsurface/core/`, `volsurface/mc/rng.py`, `volsurface/mc/zrule.py`, `volsurface/mc/pricing.py`, `volsurface/plugins.py`, `volsurface/models/` as a package, the Black-Scholes and Heston classes, and the failed-residual NaN in `volsurface/bs.py` |
| P1b | `p1b-surfaces-checker` | P1a | `volsurface/surface/`, `volsurface/arbitrage/`, `volsurface/fit/` |
| P1c | `p1c-gbm-mc` | P1b | `volsurface/models/gbm.py`, `volsurface/mc/grid.py` |
| P1d | `p1d-greeks` | P1c | `volsurface/greeks/` |

`volsurface/bs.py`, `volsurface/heston.py`, `volsurface/chains.py`, and
`volsurface/smile.py` stay the function modules the existing tests import.
The classes moved into `volsurface/models/` call those functions.

One exception. This design revision changes `implied_vol` so a Newton residual
that misses `tol` returns NaN. A finite volatility is then one the formula
reprices, and the price round trip in section 7 holds by construction on
those nodes. P1c's Monte Carlo grid is where the coverage fraction is tested.
The change is a few lines at the end of the iteration. A separate pull request
for those lines was considered and not opened.

Rules:

- One pull request, one branch. Review fixes go to that branch.
- Title `[P1b] Surfaces and arbitrage checker`. The body has two paragraphs
  of plain English, the DESIGN sections implemented, the acceptance tests
  and their results, the demo figure, the hypotheses, the LOG note, and
  which pull request it depends on.
- Draft until Corey merges. This stack does not merge itself.
- Do not tick boxes in `ROADMAP.md` and do not add a session tag. Corey
  does that after working through the material.
- Labels, when a human adds them: `stack`, `needs-claude-review`,
  `needs-corey`. `blocked-on-data` and `cut-list` are for later phases.
- Two pull requests do not edit the same module. `core/protocols.py`
  changes only in a design pull request.
- A number reaches the site, the research statement, or the question bank
  only after the pull request that produced it has been merged and written
  into `LOG.md`.

Branch names in git for the cloud run that opened this stack carry a
`cursor/` prefix and an `-aa44` suffix, because that run's branch policy
requires them. The logical name in the table is the name to rebase and to
cite.

## 13. Out of scope for Phase 1

SANOS stages S1 to S5 beyond the feasibility LP in `check_quotes`, Heston
calibration and the cosine pricer, Dupire simulation, SABR, rough Bergomi,
the deep-hedging adapter, and the dashboard JSON. `check_quotes` is the LP
those stages will call, not a fit to the market.

SSVI and eSSVI are out of scope as well. The fitter in `volsurface/fit/svi.py`
is raw SVI, one parameter set per expiry. A surface SVI would tie the slices
together and change the arbitrage counts this phase reports.

## 14. Heston cosine price, Euler scheme, calibration

`Heston.cos_price` expands the put in a Fourier cosine series on the
log-forward and recovers the call by put-call parity. The put payoff is
bounded. The call payoff grows like the forward, and its cosine coefficients
decay slowly, so the same truncation that is accurate for the put is not
accurate for the call. Parity uses `E[F_T] = F`, which the characteristic
function satisfies, and does not add a second truncation.

The interval is `[c1 - L sqrt(c2 + sqrt(c4)), c1 + L sqrt(c2 + sqrt(c4))]`
with `L = 10` when `c4` is usable. `c1` is the integrated expected
variance. `c2` and `c4` come from a degree-8 fit of `log phi(-i t)`. The
step is `f / sqrt(c2)` with `f` in `0.001`, `0.002` and `0.005`, halved
while `phi(-i t)` is not finite. A first pass uses `sqrt(-2 c1)` as the
scale and a second pass uses the `c2` that pass kept. `c4` is kept only
when two neighbouring steps agree to `1e-3` relative, and the finest such
pair is the one kept. Steps of `0.02 / sqrt(c2)` and wider return a
negative `c4` on a draw with kurtosis near 1000, so they are not used.
When those three factors do not agree, the fit is repeated at `0.0005`,
`0.001` and `0.002`. On the Fang and Oosterlee parameters the kept values
remain `c2 = 0.031571152013` and `c4 = 0.007486782` to the protocol
tolerances, and `c4` is now finite at one day and at one week.

`L` then doubles and `N` is chosen again until two successive prices agree
to `1e-7`, or the next width would pass `L = 160` or 65536 terms. The
price kept is the narrower of the two that agreed. If they never agree,
the widest width whose last doubling of `N` moved the price by at most
`2.5e-4` is kept, a `RuntimeWarning` is emitted, and `converged` on the
result is false. Long-dated, very-high-kurtosis slices can hit the term
cap; use Lewis there. A fixed multiple of `sqrt(c2)` is not that test.
Kurtosis near 1000 still leaves a tail outside `L = 10`, so the doubling
runs when `c4` is usable as well. A large positive `c4` stays inside the
formula. `N` is at least 256 and grows with `(b - a) / sqrt(c2)`. Puts
are integrated with the Fang and Oosterlee `chi` and `psi` weights.
`k = 0` in the sum is halved.

The cosine price is checked against the library Lewis price. On the
Fang and Oosterlee at-the-money one-year call the two differ by `2.2e-8`.
Puts at the kurtosis-1049 corner (`v0 = 6.582e-4`, `kappa = 2.1`,
`theta = 0.00999`, `xi = 3.56`, `rho = -0.54`, `T = 0.25`) differ by
`1.0e-8`. A one-day slice with `v0 = 1e-4` and `xi = 5` (kurtosis about
1095) differs by `1.1e-5`. A `T = 1` draw with kurtosis about 24,000
differs by `7.3e-6`. A seed-0 sample of 12 draws and four maturities, 48
prices, has largest gap `8.6e-6`. `c4` is non-finite on 8 of those 48.
The largest kurtosis in that sample is about 6700.

`Heston.simulate` is Euler full truncation (Lord, Koekkoek and van Dijk
2010). The variance used in the drift and the diffusion is `max(v, 0)`.
The updated variance is allowed to go negative and is stored that way.
`Z` is the unit-mean forward. One `SeedSequence` child covers the run.
Each step draws the forward normal, then the variance normal.

`HestonFitter` minimises a vega-weighted price residual, or the distance
outside the bid-ask spread when `loss="in_spread"`. The search is in the
unconstrained coordinates of `param_spec`, with bounds taken from that
spec. Several starts are tried. The Feller ratio `2 kappa theta / xi^2`
is reported and not constrained. The returned surface is a `ModelSurface`
around the fitted model, so `check_surface` can read it.

## 15. Dupire local variance and log-Euler

Local variance is `(dw/dT) / g(k)`, with `g` the denominator in section 5.
In expanded form

```
g = 1 - (k/w) w_k + (1/4) (-1/4 - 1/w + k^2/w^2) (w_k)^2 + (1/2) w_kk.
```

`dupire` uses analytic `variance_slopes` when the surface returns them
(raw SVI, the implied-vol spline, flat Black-Scholes). `None` means there
is no closed form, and the mixin steps are used. A `TypeError` is not
that signal. `SVISurface.local_var` and `SplineSurface.local_var` call
`dupire` for the ratio and then apply their own NaN mask (no calendar,
non-positive `w`, or a tiny `g`). Cells outside `extrapolated` are NaN
on the `dupire` grid. A denominator `g <= 0` is `bad_denominator` and is
counted. A finite |local variance| above 5 is `exploding` and is counted.
Neither is clipped.

`LocalVol` simulates `Z` by log-Euler. Local variance is bilinear in
`(t, k)` with `k = log Z`. A query outside the grid is clipped to the
edge. `state["clipped"]` counts those samples. `state["invalid"]` counts
cells that are not strictly positive, and those add no shock. One normal
per step, forward shock only. For a flat variance the step is the exact
lognormal transition, so the Euler bias is zero. On a variance that
depends on `k` the bias remains as the path count grows. The Heston test
uses `k` in `[-0.5, 0.5]`, the widest finite-difference grid that stays
positive for those parameters, 200,000 paths and 48 steps. The budget is
15 bp of implied vol plus three standard errors. Widening `Domain` while
the grid stays inside it does not change that variance grid.

## 16. SABR, Hagan's formula, and log-Euler

`SABR.price` is Black's formula at the Hagan, Kumar, Lesniewski and
Woodward (2002) lognormal implied volatility. `beta` is a parameter. The
at-the-money strike uses the closed limit of `z / x(z)`, which is 1, so
the implementation does not divide zero by zero. `|z| <= 1e-6` uses
`1 - rho z / 2`. `nu = 0` and `beta = 1` is a flat smile at `alpha`.

`simulate` is log-Euler on `Z`. Draw order, one child of the seed: at each
time after 0, the forward normals, then the volatility normals. The
volatility shock is `rho Z_F + sqrt(1 - rho^2) Z_vol`. Both updates read
the volatility at the left edge of the step. `euler_price` starts that
volatility at `alpha * F^(beta - 1)`. For `beta < 1` a normal Euler
predictor that is not strictly positive sets `Z` to 0, and later steps
leave it there. A level below `1e-12`, or a non-finite power, absorbs
as well. `beta = 1` does not absorb.

The formula is an expansion in small `nu^2 T`. The Monte Carlo test uses
`nu = 0.15` (so `nu^2 T = 0.0225` at one year) and the z-rule. At
`beta = 1` and `rho = 0` the price is the average of Black prices over
the integrated variance of `alpha`. On the panel with `nu^2 T = 2` that
mixture is the reference: Hagan is about 184 to 271 bp high, and the
20,000-path log-Euler is about 187 to 290 bp low. With `beta = 0.5`,
`alpha = 0.8` and `nu = 0`, the exact absorption probability is 0.04394.
At 64 steps and 200,000 paths the estimate is 0.04551 (seed 2) and
0.04542 (seed 3). At 256 steps, seed 2, it is 0.04420. `SABRFitter` holds
`beta` fixed and fits `alpha`, `rho` and `nu` once per expiry. On the
thinned SPXW book that fit has median price RMSE 2.82 index points. The
expiry nearest one year has `alpha = 0.161`, `rho = -0.669`, `nu = 0.932`.
Through `T = 0.056`, `nu` sits on its upper bound of 4.

`check_surface` on Hagan prices at `T = 10`, `nu = 1.5`, `beta = 1` reports
a butterfly. The implied vol on that window stays positive. The negative
number is the density.

## 17. Sparse linear program

`check_quotes` in section 6.2 builds every row as a dense vector and stacks
those rows. On the committed SPX book that allocation is what the 15.6 GB
kill measured. The constraint coefficients are unchanged. The storage is
what changes.

`volsurface/fit/_lp.py` is the shared core. It accepts the constraint matrix
as a SciPy sparse matrix, or builds one from row, column, and value
triplets, and calls `scipy.optimize.linprog` with `method="highs"`. HiGHS
accepts a sparse matrix. A `dense=True` flag materialises the same
coefficients for the equivalence test. Production code does not set it.

`check_quotes` calls that core. The name `linprog` stays in
`volsurface.arbitrage.check`, so a test can still replace the solver. A
solver that returns no point still sets `ok` to None and copies the message.

The equivalence test compares sparse and dense on the same coefficients.
The first 24 expiries are 4,398 quotes, objective 0, `ok` true, and an
empty violation set. The two shortest expiries are 278 quotes, with the
same outcome. Seeded bid, ask, and calendar violations stay on the node and
at the magnitude the existing tests name.

The scale test runs all 40 expiries, requires a HiGHS status, and requires
peak resident memory below 2 GB. The figure plots peak memory and wall time
against the number of expiries, dense against sparse. Dense stops at the
expiry count whose matrix would pass the memory budget of that run. The
logged numbers are the ones the script measured.

## 18. SANOS

SANOS (arXiv:2601.11209) builds the normalised call as a mixture of Black
calls. At a model expiry `T_j`, with weights `q_j` on model strikes `K^i`
and an ATM total variance `V_j`,

```
C_j(K) = sum_i q_j^i Call(K^i, K, eta * V_j).
```

`Call(s, k, v)` is the Black call with forward `s`, strike `k`, total
variance `v`, and unit discount, equation (1) of the paper. `eta` lies in
`[0, 1)`. `eta = 0` is the linear call of section 5.1, because
`Call(s, k, 0) = (s - k)+`. The production statement is equation (11), and
the `eta` factor is the one in section 4.2. Theorem 3.1 says that at fixed
weights the smooth call is at least the linear call. A refitted smooth
surface is a different object: linear interpolation through the quotes is
the most expensive interpolation, so the refitted smooth surface sits below
it between quotes.

The fit is one linear program, "(MDL)" in section 4.2. The unknowns are
`q_j >= 0` with `1' q_j = 1` and `K' q_j = 1`. The calendar rows are
`U_j q_j >= R_j q_{j-1}` at the model strikes. `omega = 0` makes `U` and
`R` the undiscounted payoffs `(K^i - K^l)+`, which is the choice the paper
says guarantees no calendar arbitrage. `omega = 1` uses Black calls with
variance `eta * V` in those rows. That variant is reported and labelled
not guaranteed. It is not the default.

The objective is Remark 4.1 item 3. With mid `c`, bid `B`, and ask `A`,

```
sum w (eps |C - c| + max(C - A, 0) + max(B - C, 0)),
```

`eps = 1e-8`, and `w = 1 / (A - B)`. Only out-of-the-money quotes enter. A
call and a put at the same normalised strike become one row, and the row
keeps the tighter bid and the tighter ask.

`V_j` is the at-the-money bid total variance. A cumulative maximum makes
the sequence increasing, and each expiry that was raised is logged.
Boundary strikes follow Remark 2.7, footnote 9: `K_min` is one tenth of
the smallest wing strike the footnote constructs, and `K_max` is 1.5 times
the largest. An interior strike is inserted wherever a gap exceeds `dx`.
`dx` is an argument of the fitter and is logged.

After HiGHS returns, each `q_j` is clipped at 0 and tilted so that the mass
is 1 and the mean is 1. The L1 size of that correction is stored on
`FitResult.detail`. The objective and the quote prices are what a
determinism test compares. The weights `q` are not.

Between expiries the call is the blend in equation (4). The weight `alpha`
comes from a PCHIP of at-the-money total variance, Remark 2.13, through
`scipy.interpolate.PchipInterpolator`. At `eta = 0` the same blend with a
linear `alpha` is the linear-in-time surface. `SanosSurface` implements
`call`, `dcall_dk`, `d2call_dk2`, `density`, and `dcall_dT` from the
mixture. `dcall_dT` uses `alpha'(T)`. Values outside the boundary strikes
or outside the first and last model expiry are marked extrapolated.

The market bar on the committed SPX snapshot, with `omega = 0`, is zero
static-arbitrage violations on a dense grid, and an in-spread fraction
strictly above 0.558. That 0.558 is the raw SVI in-spread already in
`LOG.md` for this file. The paper's 91.4 percent is a different book and
is context only.

A second bar holds strikes out of the same snapshot. In each expiry, about
20 percent of the strikes are drawn without replacement by
`numpy.random.default_rng` with a fixed seed, and those quotes are left
out of the fit. The bar is that the share of held-out quotes whose model
call lies inside that quote's bid and ask is at least the raw SVI share
on the same held-out quotes, and that the fitted SANOS surface still has
zero static-arbitrage violations on the dense grid. Corey has ratified
this bar: SANOS held-out in-spread at least raw SVI held-out in-spread,
with zero static-arbitrage violations, ratified 14 Oct 2026.

## 19. Discrete local volatility

On one strike grid `K^1 < ... < K^N`, a nonnegative vector `Sigma` builds a
transition matrix `Q` by Theorem 3.5 of the ar5iv HTML (the plan's PDF v4
numbers this as Theorem 3.4, equations (19) and (20)). The inverse is
tridiagonal. Remark 3.6 of that HTML (the plan's Remark 3.5) writes it as

```
Q^{-1} = E + Omega Sigma^2 dT,
```

where the product with `Sigma^2` is column by column. For interior column
`i` the three entries are, with `dT = T_j - T_{j-1}`,

```
w^{i-} = Sigma_i^2 dT (K^i)^2 / ((K^{i+1} - K^{i-1}) (K^i - K^{i-1})),
w^{i+} = Sigma_i^2 dT (K^i)^2 / ((K^{i+1} - K^{i-1}) (K^{i+1} - K^i)).
```

Row `i - 1` of that column is `-w^{i-}`, the diagonal is `1 + w^{i-} + w^{i+}`,
and row `i + 1` is `-w^{i+}`. The first and last columns are unit columns
(`Sigma` there is 0). `Q` is column-stochastic and a martingale:
`1' Q = 1'` and `K' Q = K'`. The same functions exist in NumPy and in
torch float64.

The implied discrete local volatility is equation (16) of that HTML (the
plan's Theorem 3.7). With `0 / 0` read as 0,

```
Theta = (C_j - C_{j-1}) / dT,
Gamma = (dC^i - dC^{i-1}) / (0.5 (K^{i+1} - K^{i-1})),
Sigma = sqrt(2 Theta / (K^2 Gamma)).
```

`dC` is the slope in equation (2), with `dC` equal to `-1` on the left of
the first strike and `0` on the right of the last. The round trip is
calls to `Sigma` to `Q` to calls.

The HTML points at Buehler and Ryskin (2015) for strikes that differ by
expiry. The operator implemented here composes two maps. The source
density is sent to the target grid by the mean-preserving split onto the
two bracketing strikes, and the homogeneous operator then runs on the
target grid. Columns stay probabilities and the mean is the source strike.

## 20. Rough volatility plug-in

Rough Bergomi lives in the public `rough-volatility` package `roughvol`.
That package does not import volsurface. It returns its own frozen paths
object with `times`, `Z`, and `state`, and it registers

```
[project.entry-points."volsurface.models"]
rbergomi = "roughvol.rbergomi:RoughBergomi"
```

`create("rbergomi")` resolves that entry point. The conformance test
imports `roughvol` and skips when the import fails. A mock registered in a
unit test is not that plugin, and a figure does not label the mock as
rBergomi.

`volsurface/fit/rbergomi.py` calibrates `(H, nu, rho)` through
`create("rbergomi")`, with common random numbers. The three settings are
`H` free, `H` fixed at 0.167, and `H` fixed at 0.149. The forward variance
`xi0` is read from a SANOS surface by the log contract,
`VS(T) = -2 E[log Z_T]`, integrated against the surface density. Calendar
order makes `VS` increasing in `T`. Rough Heston is not implemented.

The vol-of-vol parameter is named `nu`. `eta` remains the SANOS smoothing
weight. The plugin's `param_spec` enforces `rho <= 0`. Conditional Monte
Carlo is the estimator the plugin uses for a call. This repository does
not reimplement the hybrid scheme.

## 21. Market paths and pricers

`MarketPaths`, `RegimeSpec`, and `SurfaceBatch` are section 3 types.
`Pricer` and `SupportsMarketSimulate` are section 4 protocols.
`capabilities` adds `"market_simulate"` when the second protocol matches.

`price_along` returns cash prices of shape `(m, n + 1, c)`. At grid time
`t` before expiry `T`, the forward into the model is `Z_t * F0(T)` and the
discount from `t` to `T` is `df(T) / df(t)`. The model's own `price` is
called with the time left `T - t`. At and after expiry the entry is the
payoff at `T`, held constant. It is not grown at the interest rate.

`ClosedFormPricer` does that for a model whose `price` is a formula
(Black-Scholes, Heston, local volatility). `VarianceSwapPricer` reads
`state["xi"]` on a tenor grid and integrates it from `t` to `T`. The
result is a total variance, which is the unit the realised-variance check
uses. `SurrogatePricer` prices from a table or a network. It stores its
validation error in basis points and raises when that error is above the
bar it was given.

`regime` on `MarketPaths` is never copied into a default feature set.
`regime_prob` is the causal feature.

## 22. Regimes, seeds, and minimal DYSANOS

`RegimeSpec.switch` is `"nu"`, `"xi_level"`, or `"both"`. The market
implemented in this phase switches `nu` only. The other two values raise.
The chain is a continuous-time Markov chain, independent of the Brownian
drivers.

A switch of `nu` acts on increments only. After a switch, the new `nu`
multiplies new Volterra increments. It does not rescale the integral
already accumulated, so `v` has no jump at a switch. With `nu_s` the vol
of vol in force at time `s`,

```
I_t(u) = int_0^t nu_s (u - s)^{H - 1/2} dW_s,
v_t = xi0(t) exp(sqrt(2H) I_t(t) - H int_0^t nu_s^2 (t - s)^{2H - 1} ds).
```

On a discrete grid the increment over `(t_{i-1}, t_i]` uses the `nu` of
the regime at `t_{i-1}`. The regime recorded at `t_i` applies to later
increments. At `H = 1/2` the kernel is 1, and the step is
`v_i = v_{i-1} exp(nu_{i-1} dW_i - nu_{i-1}^2 dt_i / 2)`, with
`v_0 = xi0`. Constant `nu` reduces to the compensator `nu^2 t^{2H} / 2`.
There is no separate plan file in this repository. This paragraph is the
specification of that increment rule.

`H = 1/2` is drawn in this module, as Brownian increments. Any other `H`
calls `volterra_components` on the rBergomi plugin of section 20. The
method returns the hybrid components `(dW, I_1, ..., I_kappa)` of shape
`(n_paths, n_steps, kappa + 1)` on a uniform grid, drawn as the plugin's
volatility child with no further spawn. `kappa` is the plugin's, 1 or 2.
This module multiplies each step's whole vector by the `nu` in force on
that step. `I_k` lands at lag `k`. The tail uses the plugin's
power-kernel weights, whose first `kappa` entries are 0. `dW` alone is
not enough: it is the regression of the kernel onto the Wiener increment,
and at `H = 0.1` that regression keeps `5/9` of `Var(I_1)`, so `E[v]`
falls short of `xi0`. The compensator stays the exact one,
`sum_j nu_j^2 ((t - t_j)^{2H} - (t - t_{j+1})^{2H}) / 2`. The Cholesky
of the hybrid scheme stays in the plugin. If the method is missing, this
module raises. It does not build a substitute rough driver, and it does
not silently replace the requested `H` with `1/2`.

`VarianceSwapPricer` integrates `state["xi"]`. At `H = 1/2` that curve
is `v` on and after the clock, so the filter weights do not enter. At
any other `H` it is the conditional expectation of the simulated hybrid
variance given the `nu` in force on each step already finished and the
hybrid components of those steps. That is the `nu` path the scheme used
to build `v`. A future step is left out. At lag 1 the near-field factor
has conditional mean 1. A tail lag uses the power-kernel weight, so the
factor mean is not exactly 1. Given the true `nu` path, the mark is a
martingale up to that tail-weight bias, about -0.1 percent of the quote
on the gate grid. The clock entry is `v_t`.

`surface_at` prices the call on `state["xi"]` at the clock. Each Monte
Carlo copy starts from the true regime, `paths.regime` at that clock,
and runs the chain forward with the same generator, so `nu` switches
inside the copy. Holding today's `nu` fixed for the remaining life is
not that price. The kernel is drawn again from the clock. In gate mode
(`price_filter="rv"` and `H` other than 1/2) a missing `state["xi"]`
raises. Otherwise, when the curve is absent, the variance level is the
restarted curve of the regime at the clock.
At `H = 1/2` the curve is flat at `v_t` and the step is the exponential
martingale. A Black call on the expected variance would be flat in strike
and independent of `nu`. The plugin is not called at that `H`. Any other
`H` restarts `volterra_components` from the clock. The past Volterra
integral sits in the curve and is not drawn again. The call curve is the
conditional curve of the path whose past `nu` switched, because the
variance swap uses that curve and the call uses it too. The grid from
the clock to the last expiry has to be equally spaced. A non-uniform
grid raises. The call is a martingale by construction up to Monte Carlo
error. Each copy replicates the simulator's law. The tail-weight bias
does not enter the call. `regime_prob` is not a
weight. The same shocks drive a Black path whose variance is the
deterministic left forward curve. The control is that Black price. The
copies are split in half, and the coefficient on each half is the other
half's `cov / var`, so the residual variance is not the in-sample
minimum. The known mean is the formula. One terminal is used for every
strike. `n_mc` defaults to 1024. The seed is the path entropy and
`spawn_key + (4, 0)`. The clock is not in the key, so every hedge
date draws the same shocks. With two or more regimes the regime child
draws the forward chain and the other child is split into the vol and
spot streams. A one-regime spec does not draw that chain.
`SurfaceBatch.stderr` is the residual
standard deviation divided by `sqrt(n_mc)`. The absolute price still
carries that Monte Carlo error.
`total_variance` is the controlled log contract `-2 E[log Z]`, the
variance-swap strike of those paths. On this curve it matches the
integral the variance swap prices, up to Monte Carlo error. Whether `Z`
stays a martingale when `nu` switches is a hypothesis. The test is the
z-rule. It is not a theorem in this file.

Three Hamilton filters are defined. The hedger reads one of them. The
call and the variance swap do not.

`price_filter` defaults to `"spot"`. `regime_prob` is then that filter.
`surface_at` does not mix it. The variance swap integrates `state["xi"]`.
At `H = 1/2` that curve is `v` on and after the clock. At any other `H`
it is the hybrid curve of the `nu` path, and the filter weights are not
in it. Replacing `regime_prob` after the draw does not rebuild `xi` and
does not move the call. Set `price_filter` to `"rv"` to store the
windowed filter as the hedger's probability. The latent-variance
benchmark is not a price in either case. Neither
`state["regime_prob_rv"]` nor `state["regime_prob_latent"]` is read by
`surface_at`.

The spot filter reads one-step log realised variance of the simulated
spot, `log((log Z_i - log Z_{i-1})^2 / dt)`. It does not read a synthetic
side signal, and it does not read `v`. There is no observation at
`t = 0`, and the filter probability there is `p0`. The emission is
Gaussian. The mean is `log(xi0) - gamma - log(2)` in every regime, the
mean of `log(xi0 * chi^2_1)`, because the return's variance is the spot
variance and not `nu`. The variance in regime `j` is `pi^2 / 2 + nu_j^2`.
`pi^2 / 2` is about 4.93. For `nu` of 0.4 and 1.1 the extra term is 0.16
and 1.21, so the two emissions are close. The probabilities sit in
`state["regime_prob_spot"]`.

The windowed realised-variance filter uses the returns inside each clock
step. The coarse Brownian increment is refined to `n_sub` sub-steps,
default 21, by a bridge that sums to the stored shock, so `Z` and `v`
do not depend on `n_sub`. Variance is frozen at the start of the step.
The level `y_i = log(sum_k (d log Z_k)^2 / dt)` is `log v` at that
start plus the log of a chi-squared on 21 degrees, scaled by 21, up to
the drift and the bridge constraint. It is not the log of integrated
variance. `v` does not move inside the step. The filter does not read
the level. The same level at every date would count one piece of
information again, and the skill would fall as the clock moved. The
observation at date `i >= 2` is the change `y_i - y_{i-1}`. That change
is emitted by the regime two steps back. The emission is a table, one
row per change and one column per regime, fit on pilot paths that each
stay in one regime. The pilot seed is
`SeedSequence(entropy=101, spawn_key=(7, regime))`. It is not a market
seed. The fitted statistic is the change in `y`. The pilot does not read
`v`. A sample variance below `1e-8` is floored there. The probabilities
sit in `state["regime_prob_rv"]`.

The latent-variance benchmark reads `log v_i - log v_{i-1}`. Its
emission table is fit on the same pilots, and that fit does read `v`.
The same table is used at every `H`. At `H = 1/2` the increment is
`Normal(-nu^2 dt / 2, nu^2 dt)` and the pilot estimates that law. At
any other `H` that Gaussian is not the rough likelihood, so the
benchmark does not use it and it is not an oracle. The probabilities
sit in `state["regime_prob_latent"]`.

The bar is the Brier skill against the constant forecast that quotes
the sample share of each regime, `1 - Brier / Brier_base`. It is scored
at each setting on its own. There is no single lift band.

On 600 paths, 21 times, clock index 10, seed entropy 12, test split, the
generator is `[[-1.5, 1.5], [0.8, -0.8]]`, the initial law is
`(0.7, 0.3)`, `rho = -0.4`, and `xi0` is flat at 0.04. The spot hedge
cost `1e-4` is not an input to any filter. The chain does not depend on
`H` or on `nu`, so both settings below have the same labels at the
clock: 277 paths in the low regime and 323 in the high regime. The base
rate is 0.5383. The base-rate forecast scores Brier 0.4971 and log
score -0.6902. Confusion matrices have the true regime in the row and
the modal regime in the column, low `nu` then high `nu`.

Market-test settings, `H = 1/2`, `nu` of 0.4 and 1.1, 600 paths. The
spot filter is right on 287 paths, confusion `[[129, 148], [165, 158]]`,
Brier 0.5222, log score -0.7185, Brier skill -0.0505, calibration error
0.0713. It has no skill at these settings. The windowed filter is right
on 345 paths, confusion `[[109, 168], [87, 236]]`, Brier 0.4953, log
score -0.6893, Brier skill 0.0035, calibration error 0.0295. The latent
benchmark is right on 487 paths, confusion `[[223, 54], [59, 264]]`,
Brier 0.2806, log score -0.4412, Brier skill 0.4355, calibration error
0.0221. With 1,024 copies, the mean absolute call gap against the
true-regime mixture is 0.0002898 (spot), 0.0002834 (windowed), and
0.0001549 (latent). The log-contract gaps are 0.0004526, 0.0004396, and
0.0002434. On the 35,032 nodes where all four implied vols are finite,
out of 37,800, the implied-vol root-mean-square error is 0.005348,
0.005232, and 0.003915.

The same market-test settings on 4,000 paths, same seed and same clock.
Base rate 0.5383, from 1,847 and 2,153. Spot: hit rate 0.5023, confusion
`[[894, 953], [1038, 1115]]`, Brier skill -0.0292, calibration error
0.0591. Windowed: hit rate 0.5543, confusion `[[677, 1170], [613, 1540]]`,
Brier skill 0.0094, calibration error 0.0106. Latent: hit rate 0.8295,
confusion `[[1561, 286], [396, 1757]]`, Brier skill 0.4785, calibration
error 0.0066.

Pre-registered settings, `H = 0.1`, `nu` of 1.5 and 3, 600 paths. The
spot filter is right on 410 paths, confusion `[[176, 101], [89, 234]]`,
Brier 0.4193, log score -0.6157, Brier skill 0.1565, calibration error
0.0632. The hit rate is 0.6833. The windowed filter is right on 452
paths, confusion `[[207, 70], [78, 245]]`, Brier 0.3560, log score
-0.5406, Brier skill 0.2838, calibration error 0.0255. The
latent-variance benchmark is right on 464 paths, confusion
`[[207, 70], [66, 257]]`, Brier 0.3316, log score -0.5099, Brier skill
0.3330, calibration error 0.0487. With 1,024 copies the mean absolute
call gaps are 0.001650, 0.001452, and 0.001310. The log-contract gaps
are 0.002470, 0.001941, and 0.001788. On the 31,684 nodes where all
four implied vols are finite, the implied-vol root-mean-square errors
are 0.02069, 0.01906, and 0.01805. Prices still read the spot filter.

The same pre-registered settings on 4,000 paths. Spot: hit rate 0.6608,
confusion `[[1174, 673], [684, 1469]]`, Brier skill 0.1178, calibration
error 0.0527. Windowed: hit rate 0.7048, confusion
`[[1336, 511], [670, 1483]]`, Brier skill 0.2071, calibration error
0.0319. Latent: hit rate 0.7438, confusion `[[1394, 453], [572, 1581]]`,
Brier skill 0.2849, calibration error 0.0338. The windowed skill at
clocks 5, 10 and 20 is 0.2079, 0.2071 and 0.2089. It does not fall with
the clock.

Calibration error is the path-weighted mean absolute gap between the
bin hit rate and the mean predicted probability of the high regime.
Five bins, and a bin with fewer than 40 paths is left out. The score is
at clock index 10 except where a clock is named. The differenced
windowed filter is also inside binomial bands at clocks 5, 10 and 20 on
a 1,600-path pre-registered draw.

Constant `nu`, `H = 0.1`, flat `xi0 = 0.04`, eight runs of 500 paths,
clock index 8 on 17 times. The mean of the forward curve on the dates
after the clock, divided by `xi0`, is 1.019 at `nu = 1.5` and 1.031 at
`nu = 3`. Both sets pass the z-rule. The diagonal is `v_t` and is not
in that average.

The same market-test paths, read from eight quantile bins, give a call
modal accuracy of 0.6600 with 1,024 Monte Carlo copies (lift 0.1217)
and a variance-swap modal accuracy of 0.6600 (lift 0.1217). That 0.6600
is the price bin. It is not a filter hit rate. Both lifts sit between
0.05 and 0.35. The call bin is 0.6600 under the spot weights and under
the windowed weights. On one path with `v = 0.04`, `rho = -0.4`,
32 steps and 8,192 copies, seed entropy 7, test split, the implied-vol
span at expiry 1 is 0.000383 when `nu = 0.01` and 0.04358 when
`nu = 1.5`. At `nu = 1.5` the implied vol is 0.2107 at log-moneyness
-0.2 and 0.1676 at log-moneyness 0.2. On one path with `nu = 1.1` and
1,024 copies the at-the-money relative standard error at expiry 1 is
0.0207. At `H = 0.1`, eight paths, clock 0.5, maturity 1, 4,096 copies,
seed entropy 21, test split, the flat anchor `v_t (T - t)` misses the
variance swap by a median relative gap of 0.281. The surface log
contract stays inside 4 standard errors of that swap on every expiry.
The one-step expiry has a numerical residual. The largest absolute
z-score on a later expiry is 1.80. The out-of-the-money strip on
log-moneyness from -0.8 to 0.8 stays inside 4 standard errors on the
last expiry. Its largest absolute z-score is 1.56.
Reliability of the filter on a hand-built Gaussian series is checked
against binomial bands.

Seeds use `numpy.random.SeedSequence(entropy=E, spawn_key=(split,))`.
`split` is 0 for train, 1 for validation, and 2 for test. Children of a
run are spawned in this order: 0 the regime chain, 1 the vol drivers, 2
the spot-orthogonal driver, 3 surrogate noise. That order is part of the
model docstring. Two splits have disjoint spawn keys, so they do not share
a stream.

`volsurface/dynamic/dysanos.py` is minimal DYSANOS. The full v3 baseline
is future work: a fitted decoder, a spot copula, the paper's kappa units
and expiry grid, and joint training of the decoder and the transition.
Paper statements in that module, its note, and its log entry are marked
UNVERIFIED: the design was written from the plan's citations, not from a
full reading of arXiv:2608.12587. The decoder maps a
latent to nonnegative discrete local volatilities and increasing variances,
then to a SANOS surface. The default grid is seven expiries (2, 5, 10, 20,
40, 126, and 252 business days) and 20 to 50 strikes. The 18 by 200 grid is
a flag and is not the default. Compression is affine, `u = b + B h`, with
`d = 20` by default. The transition is the AR(1)

```
h_{t+D} = m + A_D (h_t - m) + G_D' z,    A_D = exp(-kappa D),
kappa = L L' + S + kappa_min I,    kappa_min = 2e-4.
```

`S` is antisymmetric, so the eigenvalues of `A_D` have modulus below 1 by
construction. The fit is two-stage: per-date surfaces, then PCA, then
Gaussian maximum likelihood. That is a stated deviation from the paper's
joint training. The spot is drawn from the front-expiry marginal, which
adds no physical drift. A scalar OU multiplier on the innovations is
optional. The only history available is synthetic, generated by Heston or
by the regime market above. There is no daily SPX file in this repository.

The regime experiment itself, including the hedger, lives in
`deep-hedging`. `experiments/regime/PREREG.md` records hypotheses H1, H2,
and H3. A day there is `1/252` of a trading year, and the episode starts
at `t0 = 0.5` with the earlier path visible to the filter. The file is a
pinned draft. It is to be frozen before the pilot ends. The pilot is a
CPU run. No GPU is assumed.

Phase 3 pull requests, each based on the previous branch:

| logical name | branch slug | depends on | owns |
|---|---|---|---|
| P3-0 | `design/phase3` | `main` | sections 17 to 22, the stubs in `core/types.py` and `core/protocols.py` |
| P3a | `p3a-sparse-lp` | P3-0 | `fit/_lp.py`, the sparse `check_quotes` |
| P3b | `p3b-sanos-s1s2` | P3a | `surface/sanos.py`, `fit/sanos.py` at `eta = 0` |
| P3c | `p3c-sanos-s3s4` | P3b | the same two files at `eta > 0`, the shoot-out |
| P3d | `p3d-sanos-dlv` | P3c | `surface/sanos_dlv.py` |
| P3e | `p3e-rough-plugin` | P3d | `fit/rbergomi.py`, the conformance test |
| P3f | `p3f-market-paths` | P3e | `markets/` |
| P3g | `p3g-dysanos-min` | P3f | `dynamic/dysanos.py`, the draft preregistration |

Branch names in git for the cloud run that opens this stack carry a
`cursor/` prefix and an `-faa7` suffix, because that run's branch policy
requires them. The logical name in the table is the name to rebase and to
cite. Rough Heston is not in the table.
